"""
PolicyRunner — carga una policy PPO entrenada y la ejecuta en inferencia dentro
del supervisor, replicando exactamente el preprocesamiento de observaciones del
entrenamiento (VecNormalize sobre `velocity`).

Lo usa `supervisor_controller.py` para que la Robot Window del supervisor pueda
cargar un modelo de `models/<run_id>/final_model.zip` y ponerlo a correr en
Webots sin un cliente de entrenamiento conectado.

Si el modelo fue entrenado con VecFrameStack (n_stack>1), el runner mantiene su
propio buffer de frames y reproduce EXACTAMENTE el apilado de SB3:
  - normaliza cada frame (VecNormalize esta ADENTRO de VecFrameStack en training)
    y recien despues apila;
  - el frame mas nuevo va ultimo (orden viejo->nuevo);
  - al inicio del episodio los frames faltantes se rellenan con CEROS (no se repite
    el primero), igual que SB3.
El n_stack se infiere del propio modelo, asi anda con modelos stackeados o no.

Si el modelo es recurrente (LSTM, sb3-contrib RecurrentPPO), el runner lo detecta
del .zip, lo carga con RecurrentPPO y ARRASTRA el hidden state entre predict()
(reseteandolo en reset_stack() con episode_start=True). Sin esto el LSTM correria
con memoria en cero cada step y se comportaria como un feedforward sin memoria.

Interfaz consumida por el supervisor:
    runner = PolicyRunner()
    runner.load(model_path)            # tambien intenta cargar vecnormalize.pkl
    runner.unload()
    runner.predict(obs_dict) -> list   # obs = {"velocity": [...], "image": ...}
    runner.reset_stack()               # limpiar el buffer al reiniciar un episodio
    runner.status_dict() -> dict
    runner.enabled / .deterministic    # flags mutables
    runner.loaded / .model_path / .last_action / .last_error
"""

import os
import pickle
import zipfile
from collections import deque

import numpy as np
from stable_baselines3 import PPO

from helpers.image_obs import decode_image_observation
from helpers.robot_bridge import log_supervisor

# RecurrentPPO (sb3-contrib) es opcional: solo hace falta para modelos LSTM. Import
# perezoso para no romper si no esta instalado y el modelo es feedforward.
try:
    from sb3_contrib import RecurrentPPO
except Exception:  # pragma: no cover - depende de si sb3-contrib esta instalado
    RecurrentPPO = None

# Dimension de 'velocity' por frame (env define [forward, yaw_rate]); se usa para
# inferir n_stack desde el observation_space del modelo.
BASE_VELOCITY_DIM = 2

# Raiz del proyecto (carpeta que contiene helpers/, models/, controllers/...).
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# Claves de observacion que el entrenamiento normaliza con VecNormalize.
# Debe coincidir con `norm_obs_keys=["velocity"]` en rl/trainer.py.
NORMALIZED_OBS_KEYS = ("velocity",)


class PolicyRunner:
    """Envuelve un modelo PPO + VecNormalize para inferencia en el supervisor."""

    def __init__(self):
        self.model = None
        self.vecnormalize = None
        self.model_path = None
        self.vecnormalize_path = None
        self.loaded = False
        self.enabled = False
        self.deterministic = True
        self.step_count = 0
        self.last_action = None
        self.last_error = None
        # Frame stacking (se infiere del modelo en load()).
        self.n_stack = 1
        self._stack_buffer = None
        # Si el modelo fue entrenado con goal_direction en la obs (se infiere en load()).
        self.include_goal_direction = False
        # Si el modelo fue entrenado con el parche de cobertura en la obs (se infiere en load()).
        self.include_coverage_patch = False
        # Si el modelo fue entrenado condicionado (image-goal): lleva target_image en la obs.
        self.include_target_image = False
        # Recurrencia (LSTM). Si el modelo es RecurrentPPO hay que cargarlo con esa clase
        # y ARRASTRAR el hidden state entre steps; si no, el LSTM corre con memoria en
        # cero todos los steps (= feedforward lobotomizado, da vueltas). Se infiere en load().
        self.is_recurrent = False
        self._lstm_states = None
        self._episode_start = True

    # ------------------------------------------------------------------
    # Carga / descarga
    # ------------------------------------------------------------------

    def load(self, model_path):
        """
        Carga un modelo PPO desde un .zip y, si existe, el vecnormalize.pkl
        asociado (mismo directorio o el directorio padre del run).
        """
        resolved_model_path = self._resolve_model_path(model_path)

        # Detectar si el modelo es recurrente (LSTM) ANTES de cargar, para elegir la
        # clase correcta. RecurrentPPO maneja el hidden state en predict(); PPO no.
        is_recurrent = self._detect_recurrent(resolved_model_path)
        if is_recurrent:
            if RecurrentPPO is None:
                raise RuntimeError(
                    "El modelo es recurrente (LSTM) pero sb3-contrib no esta instalado. "
                    "Instalalo con: pip install sb3-contrib"
                )
            model = RecurrentPPO.load(resolved_model_path, device="cpu")
        else:
            model = PPO.load(resolved_model_path, device="cpu")

        vecnormalize_path = self._find_vecnormalize_path(resolved_model_path)
        vecnormalize = (
            self._load_vecnormalize(vecnormalize_path)
            if vecnormalize_path
            else None
        )

        # Commit del estado solo si todo cargo bien.
        self.model = model
        self.vecnormalize = vecnormalize
        self.model_path = resolved_model_path
        self.vecnormalize_path = vecnormalize_path
        self.loaded = True
        self.step_count = 0
        self.last_action = None
        self.last_error = None
        # Inferir n_stack del modelo y arrancar con el buffer limpio.
        self.n_stack = self._infer_n_stack(model)
        self._stack_buffer = None
        # Estado recurrente: arranca limpio (primer step = inicio de episodio).
        self.is_recurrent = is_recurrent
        self._lstm_states = None
        self._episode_start = True
        # El modelo dicta si la obs lleva goal_direction (no el .env): asi el playback
        # coincide con como fue entrenado.
        try:
            self.include_goal_direction = "goal_direction" in model.observation_space.spaces
        except Exception:
            self.include_goal_direction = False
        # Idem para el parche de cobertura: lo dicta el modelo. Asi en la composicion el
        # explore (entrenado CON parche) lo recibe y el exploit (SIN parche) lo ignora,
        # cada uno arma su obs por keys propias sobre la MISMA observacion del supervisor.
        try:
            self.include_coverage_patch = "coverage_patch" in model.observation_space.spaces
        except Exception:
            self.include_coverage_patch = False
        # Image-goal: el modelo lleva la referencia del target en la obs (se reconstruye local
        # desde el nombre del color que manda el supervisor).
        try:
            self.include_target_image = "target_image" in model.observation_space.spaces
        except Exception:
            self.include_target_image = False
        return resolved_model_path

    def unload(self):
        """Descarga el modelo y desactiva la policy."""
        self.model = None
        self.vecnormalize = None
        self.model_path = None
        self.vecnormalize_path = None
        self.loaded = False
        self.enabled = False
        self.step_count = 0
        self.last_action = None
        self.last_error = None
        self.n_stack = 1
        self._stack_buffer = None
        self.include_goal_direction = False
        self.include_coverage_patch = False
        self.include_target_image = False
        self.is_recurrent = False
        self._lstm_states = None
        self._episode_start = True

    def reset_stack(self):
        """
        Limpia el buffer de frames y el hidden state del LSTM. Llamar al reiniciar
        un episodio (reset del robot) para no arrastrar memoria del episodio anterior:
        ni frames apilados ni el estado recurrente. El proximo predict marca inicio
        de episodio (episode_start=True) para que el LSTM reinicie su hidden a cero.
        """
        self._stack_buffer = None
        self._lstm_states = None
        self._episode_start = True

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predict(self, observation):
        """
        Predice una accion [left, right] a partir de la observacion del supervisor.

        `observation` es el dict de `_build_nav_observation`:
            {"velocity": [forward, yaw_rate], "image": ...}

        Devuelve una lista de 2 floats (velocidades de rueda) lista para enviar
        al robot via JSON.
        """
        if not self.loaded or self.model is None:
            raise RuntimeError("No hay ninguna policy cargada para predecir.")

        # 1) frame unico normalizado (como en training, ANTES de apilar);
        # 2) apilado replicando VecFrameStack (no-op si n_stack <= 1).
        frame = self._prepare_single_frame(observation)
        model_obs = self._apply_stack(frame)

        if self.is_recurrent:
            # RecurrentPPO: hay que pasarle y recibir el hidden state en cada step, y
            # marcar episode_start en el primer step post-reset (ahi reinicia el hidden
            # a cero). Sin esto, el LSTM no acumula nada => no usa memoria.
            action, self._lstm_states = self.model.predict(
                model_obs,
                state=self._lstm_states,
                episode_start=np.array([self._episode_start]),
                deterministic=bool(self.deterministic),
            )
            self._episode_start = False
        else:
            action, _ = self.model.predict(
                model_obs, deterministic=bool(self.deterministic)
            )

        action_list = [float(value) for value in np.asarray(action).reshape(-1)]
        self.last_action = action_list
        self.step_count += 1
        self.last_error = None
        return action_list

    # ------------------------------------------------------------------
    # Estado para la UI
    # ------------------------------------------------------------------

    def status_dict(self):
        return {
            "loaded": self.loaded,
            "enabled": self.enabled,
            "deterministic": self.deterministic,
            "step_count": self.step_count,
            "model_path": self.model_path,
            "n_stack": self.n_stack,
            "is_recurrent": self.is_recurrent,
            "last_action": self.last_action,
            "last_error": self.last_error,
        }

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _prepare_single_frame(self, observation):
        """
        Convierte el dict del supervisor a UN frame con el formato que espera el
        modelo y aplica la misma normalizacion (VecNormalize sobre `velocity`) que
        en training. Devuelve un dict de 1 frame (sin apilar).
        """
        if not isinstance(observation, dict):
            raise RuntimeError(
                "La observacion para la policy debe ser un dict {velocity, image}."
            )

        model_obs = {
            "velocity": np.asarray(
                observation.get("velocity", [0.0, 0.0]), dtype=np.float32
            ).reshape(-1),
            "image": decode_image_observation(observation.get("image")),
        }
        if self.include_goal_direction:
            model_obs["goal_direction"] = np.asarray(
                observation.get("goal_direction", [0.0, 0.0]), dtype=np.float32
            ).reshape(-1)

        if self.vecnormalize is not None:
            # normalize_obs no usa el venv, asi que es seguro llamarlo con el
            # objeto VecNormalize cargado desde el pickle sin set_venv.
            model_obs = self.vecnormalize.normalize_obs(model_obs)

        return model_obs

    def _detect_recurrent(self, model_path):
        """
        Detecta si un .zip de SB3 contiene una policy recurrente (LSTM) leyendo el
        miembro 'data' (JSON serializado) y buscando la firma del policy_class de
        sb3-contrib. No carga el modelo: solo decide con que clase cargarlo despues.
        """
        try:
            with zipfile.ZipFile(model_path) as archive:
                data = archive.read("data").decode("utf-8", "ignore")
        except Exception:
            return False
        return ("Recurrent" in data) or ("lstm_hidden_size" in data)

    def _infer_n_stack(self, model):
        """
        Infiere n_stack del observation_space del modelo: cuantos frames de
        'velocity' (base = BASE_VELOCITY_DIM) entran en la dimension apilada.
        """
        try:
            velocity_space = model.observation_space.spaces["velocity"]
            return max(1, int(velocity_space.shape[0]) // BASE_VELOCITY_DIM)
        except Exception:
            return 1

    def _apply_stack(self, frame):
        """
        Apila el frame actual con los anteriores replicando VecFrameStack:
        frames en orden viejo->nuevo (el nuevo ultimo); al inicio rellena con
        CEROS los frames faltantes. No-op si n_stack <= 1.
        """
        if self.n_stack <= 1:
            return frame

        if self._stack_buffer is None:
            # Reset: buffer de ceros + el frame actual en la ultima posicion.
            zero_frame = {key: np.zeros_like(value) for key, value in frame.items()}
            self._stack_buffer = deque(
                (zero_frame for _ in range(self.n_stack - 1)), maxlen=self.n_stack
            )
            self._stack_buffer.append(frame)
        else:
            self._stack_buffer.append(frame)  # el maxlen descarta el mas viejo

        # Concatenar viejo->nuevo: imagen sobre el eje de canales (0), vectores 1D
        # sobre su unico eje. Coincide con el layout de VecFrameStack.
        stacked = {}
        for key in frame.keys():
            stacked[key] = np.concatenate(
                [buffered[key] for buffered in self._stack_buffer], axis=0
            )
        return stacked

    def _resolve_model_path(self, model_path):
        raw = str(model_path or "").strip()
        if not raw:
            raise RuntimeError("Ruta de modelo vacia.")

        candidates = []
        if os.path.isabs(raw):
            candidates.append(raw)
        else:
            candidates.append(os.path.abspath(raw))
            candidates.append(os.path.join(PROJECT_ROOT, raw))

        # Permitir indicar el directorio del run en lugar del .zip directo.
        extra = []
        for candidate in candidates:
            if os.path.isdir(candidate):
                extra.append(os.path.join(candidate, "final_model.zip"))
            elif not candidate.endswith(".zip"):
                extra.append(f"{candidate}.zip")
        candidates.extend(extra)

        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

        raise FileNotFoundError(
            f"No se encontro el modelo PPO en ninguna ruta candidata: {model_path}"
        )

    def _find_vecnormalize_path(self, model_path):
        model_dir = os.path.dirname(model_path)
        candidates = [
            os.path.join(model_dir, "vecnormalize.pkl"),
            os.path.join(os.path.dirname(model_dir), "vecnormalize.pkl"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return None

    def _load_vecnormalize(self, vecnormalize_path):
        """
        Carga el VecNormalize directamente del pickle (sin venv). Solo se usa
        para `normalize_obs`, que depende unicamente de obs_rms / clip_obs /
        epsilon, no del entorno vectorizado.
        """
        with open(vecnormalize_path, "rb") as file_handler:
            vecnormalize = pickle.load(file_handler)
        # Inferencia: no actualizar estadisticas ni tocar el reward.
        vecnormalize.training = False
        vecnormalize.norm_reward = False
        return vecnormalize