# Ejecuta con 
#  python -m rl.trainer --total-timesteps 100000 --n-stack 4

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
    unwrap_vec_normalize,
)
from launch_webots import launch_webots, launch_webots_instances

from rl.env import NavEnv


class RLMetricsCallback(BaseCallback):
    """
    Loguea metricas de seguimiento de carril en TensorBoard (namespace custom/).

    Un episodio se cierra por:
      - off-track (pasto en el centro de la vista) -> terminated
      - carril perdido N steps                      -> terminated (term_reason=line_lost)
      - timeout (max steps)                         -> truncated (TimeLimit.truncated)
    Ademas promedia por rollout las features visuales del reward (clearance,
    |offset|, blanco en el centro) y el reward por step, desde info["reward_breakdown"].
    """

    def __init__(self):
        super().__init__()
        # Contadores acumulados durante toda la corrida.
        self.total_episodes = 0
        self.total_laps = 0
        self.total_offtrack = 0
        self.total_line_lost = 0
        self.total_wrong_way = 0
        self.total_timeouts = 0
        # Acumuladores por rollout (se resetean en cada _on_rollout_end).
        self._rollout_episodes = 0
        self._rollout_laps = 0
        self._rollout_offtrack = 0
        self._rollout_line_lost = 0
        self._rollout_wrong_way = 0
        self._rollout_timeouts = 0
        self._continuous = defaultdict(list)
        self._ep_progress = []  # progreso acumulado por env en el episodio en curso

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # Acumulador de progreso por env (lazily; n_envs no se conoce de antemano).
        if len(self._ep_progress) != len(infos):
            self._ep_progress = [0.0] * len(infos)

        # Metricas continuas (todos los steps), desde el desglose de reward.
        for i, info in enumerate(infos):
            breakdown = info.get("reward_breakdown")
            if not isinstance(breakdown, dict):
                continue
            for key in ("reward", "center_clearance", "white_center"):
                value = breakdown.get(key)
                if value is not None:
                    self._continuous[key].append(float(value))
            offset = breakdown.get("offset")
            if offset is not None:
                self._continuous["abs_offset"].append(abs(float(offset)))
            progress = breakdown.get("progress")
            if progress is not None:
                self._ep_progress[i] += float(progress)

        # Cierre de episodio: timeout (truncado) vs terminacion (off-track / carril perdido).
        for i, (done, info) in enumerate(zip(dones, infos)):
            if not done:
                continue
            # Progreso NETO recorrido en el episodio (arc-length sobre el circuito) -> curva
            # de aprendizaje continua y robusta cuando las vueltas completas son raras.
            self._continuous["ep_progress"].append(self._ep_progress[i])
            self._ep_progress[i] = 0.0
            self.total_episodes += 1
            self._rollout_episodes += 1
            if info.get("TimeLimit.truncated", False):
                self.total_timeouts += 1
                self._rollout_timeouts += 1
                continue
            breakdown = info.get("reward_breakdown")
            reason = breakdown.get("term_reason") if isinstance(breakdown, dict) else None
            if reason == "lap_complete":  # EXITO: completo la vuelta
                self.total_laps += 1
                self._rollout_laps += 1
            elif reason == "wrong_way":
                self.total_wrong_way += 1
                self._rollout_wrong_way += 1
            elif reason == "line_lost":
                self.total_line_lost += 1
                self._rollout_line_lost += 1
            else:  # offtrack_grass (u otra terminacion)
                self.total_offtrack += 1
                self._rollout_offtrack += 1

        return True

    def _on_rollout_end(self):
        # Conteos acumulados (monotonos).
        self.logger.record("custom/episodes_total", self.total_episodes)
        self.logger.record("custom/laps_total", self.total_laps)
        self.logger.record("custom/offtrack_total", self.total_offtrack)
        self.logger.record("custom/line_lost_total", self.total_line_lost)
        self.logger.record("custom/wrong_way_total", self.total_wrong_way)
        self.logger.record("custom/timeouts_total", self.total_timeouts)

        # Tasas por rollout.
        if self._rollout_episodes > 0:
            n = self._rollout_episodes
            self.logger.record("custom/lap_rate", self._rollout_laps / n)
            self.logger.record("custom/offtrack_rate", self._rollout_offtrack / n)
            self.logger.record("custom/line_lost_rate", self._rollout_line_lost / n)
            self.logger.record("custom/wrong_way_rate", self._rollout_wrong_way / n)
            self.logger.record("custom/timeout_rate", self._rollout_timeouts / n)
            self.logger.record("custom/episodes_this_rollout", n)

        # Promedios continuos del rollout.
        for name, values in self._continuous.items():
            if values:
                self.logger.record(f"custom/{name}_mean", float(np.mean(values)))

        # Reset de acumuladores por rollout.
        self._rollout_episodes = 0
        self._rollout_laps = 0
        self._rollout_offtrack = 0
        self._rollout_line_lost = 0
        self._rollout_wrong_way = 0
        self._rollout_timeouts = 0
        self._continuous.clear()

    def print_summary(self):
        """Resumen por consola de los conteos acumulados de toda la corrida."""
        total = self.total_episodes
        pct = (lambda n: f"{100.0 * n / total:5.1f}%" if total > 0 else "  n/a")
        print("=" * 52)
        print("RESUMEN DE EPISODIOS")
        print(f"  episodios totales : {total}")
        print(f"  VUELTAS completas : {self.total_laps:5d}  ({pct(self.total_laps)})")
        print(f"  off-track (pasto) : {self.total_offtrack:5d}  ({pct(self.total_offtrack)})")
        print(f"  carril perdido    : {self.total_line_lost:5d}  ({pct(self.total_line_lost)})")
        print(f"  contramano        : {self.total_wrong_way:5d}  ({pct(self.total_wrong_way)})")
        print(f"  timeout           : {self.total_timeouts:5d}  ({pct(self.total_timeouts)})")
        print("=" * 52)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenador PPO para line following en Webots."
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host del supervisor de Webots."
    )
    parser.add_argument(
        "--port", type=int, default=10001, help="Puerto del supervisor."
    )
    parser.add_argument(
        "--save-path",
        default="final_model",
        help=(
            "Nombre del modelo final dentro de la carpeta de la corrida; "
            "Stable-Baselines3 agrega .zip si falta."
        ),
    )
    parser.add_argument(
        "--tensorboard-log",
        default="tensorboard",
        help=(
            "Subdirectorio para logs de TensorBoard dentro de la carpeta "
            "de la corrida actual."
        ),
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Cantidad total de timesteps para entrenar.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-4, help="Learning rate inicial de PPO."
    )
    parser.add_argument("--n-steps", type=int, default=1024, help="n_steps de PPO.")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.995,
        help=(
            "Factor de descuento de PPO. En reward SPARSE conviene alto: el bonus "
            "terminal se propaga hacia atras como gamma^N; con N grande, 0.99 lo "
            "apaga (0.99^300~0.05) y el agente no 'siente' el goal desde lejos."
        ),
    )
    parser.add_argument(
        "--norm-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalizar el reward (VecNormalize). Default True: estabiliza el critico con "
            "retornos grandes (~500/episodio). Apagar con --no-norm-reward."
        ),
    )
    parser.add_argument(
        "--n-stack",
        type=int,
        default=1,
        help=(
            "Cantidad de frames a apilar (VecFrameStack). En la rama vision_lstm el default "
            "es 1: la RECURRENCIA (LSTM) aporta la memoria temporal, no el stacking."
        ),
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=256,
        help="Tamanio del estado oculto del LSTM (RecurrentPPO). Default 256.",
    )
    parser.add_argument("--n-epochs", type=int, default=5, help="n_epochs de PPO.")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch_size de PPO."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed usada por PPO.")
    parser.add_argument(
        "--target-kl", type=float, default=0.02, help="Cuanto actualizar."
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5,
        help="Peso del critico en la loss total de PPO (default 0.5).",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.02,
        help="Coeficiente de entropia de PPO (default 0.02).",
    )
    parser.add_argument(
        "--clip-range", type=float, default=0.2,
        help="Clip range inicial de PPO (default 0.2).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo para PyTorch/Stable-Baselines3, por ejemplo cpu o cuda.",
    )
    parser.add_argument(
        "--skip-check-env",
        action="store_true",
        help="Omitir la validacion inicial de check_env.",
    )
    parser.add_argument(
        "--resume-model-path",
        default=None,
        help=(
            "Ruta a un modelo PPO ya entrenado (.zip) para continuar el "
            "entrenamiento desde esa policy."
        ),
    )
    parser.add_argument(
        "--resume-vecnormalize-path",
        default=None,
        help=(
            "Ruta a un vecnormalize.pkl previo. Si no se indica, se intenta "
            "inferir desde --resume-model-path."
        ),
    )
    parser.add_argument(
        "--reset-timesteps-on-resume",
        action="store_true",
        help=(
            "Si se reanuda un modelo, reinicia el contador interno de "
            "timesteps en lugar de continuarlo."
        ),
    )
    parser.add_argument(
        "--webots-world",
        default="worlds/track1.wbt",
        help="World de Webots a lanzar automaticamente.",
    )
    parser.add_argument(
        "--webots-executable",
        default=r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe",
        help="Ruta al ejecutable de Webots.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help=(
            "Cantidad de entornos Webots EN PARALELO (SubprocVecEnv). 1 = comportamiento "
            "original (DummyVecEnv, un solo Webots). >1 lanza N instancias headless, cada "
            "una en su puerto (base-port + i), y PPO colecta de todas a la vez (~lineal en "
            "wall-clock hasta saturar CPU/GPU). Con N>1 conviene bajar --n-steps."
        ),
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=None,
        help="Puerto base para los N supervisores (default: --port). Usa base, base+1, ...",
    )
    parser.add_argument(
        "--no-webots-launch",
        action="store_true",
        help="No lanzar Webots automaticamente.",
    )
    parser.add_argument(
        "--no-discord",
        action="store_true",
        help="No enviar notificacion a Discord al terminar.",
    )

    return parser.parse_args()


def build_env(host, port, monitor=False, reset_options=None):
    env = NavEnv(host=host, port=port)
    env.set_default_reset_options(reset_options)
    if monitor:
        env = Monitor(env)
    return env


def resolve_saved_model_path(save_path):
    if save_path.endswith(".zip"):
        return save_path
    return f"{save_path}.zip"


def resolve_output_path(path):
    resolved = resolve_saved_model_path(str(path).strip())
    return os.path.abspath(os.path.normpath(resolved))


def resolve_existing_path(path, description):
    candidate = (
        resolve_output_path(path) if description == "modelo" else str(path).strip()
    )
    normalized = os.path.abspath(os.path.normpath(candidate))
    if not os.path.exists(normalized):
        raise FileNotFoundError(f"No se encontro el {description}: {path}")
    return normalized


def infer_resume_vecnormalize_path(model_path):
    model_dir = os.path.dirname(model_path)
    candidate_paths = [
        os.path.join(model_dir, "vecnormalize.pkl"),
        os.path.join(os.path.dirname(model_dir), "vecnormalize.pkl"),
    ]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def build_training_run_paths():
    base_run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = base_run_id
    run_dir = os.path.abspath(os.path.join("models", run_id))
    suffix = 1
    while os.path.exists(run_dir):
        run_id = f"{base_run_id}_{suffix:02d}"
        run_dir = os.path.abspath(os.path.join("models", run_id))
        suffix += 1
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "metadata_path": os.path.join(run_dir, "run_metadata.json"),
    }


def resolve_run_subpath(run_dir, raw_path, default_name):
    normalized = os.path.normpath(str(raw_path).strip()) if raw_path is not None else ""
    if not normalized or normalized in {".", ""}:
        normalized = default_name
    subpath = normalized.lstrip("\\/")
    drive, tail = os.path.splitdrive(subpath)
    if drive:
        subpath = tail.lstrip("\\/")
    parts = [
        part
        for part in subpath.replace("\\", "/").split("/")
        if part not in {"", ".", ".."}
    ]
    if not parts:
        parts = [default_name]
    return os.path.abspath(os.path.join(run_dir, *parts))


def populate_run_artifact_paths(run_paths, args):
    run_dir = run_paths["run_dir"]
    run_paths["tensorboard_log"] = resolve_run_subpath(
        run_dir, args.tensorboard_log, "tensorboard"
    )
    run_paths["final_model_path"] = resolve_run_subpath(
        run_dir, args.save_path, "final_model"
    )
    run_paths["vecnormalize_path"] = os.path.join(run_dir, "vecnormalize.pkl")
    run_paths["curriculum_path"] = os.path.join(run_dir, "curriculum_schedule.json")
    return run_paths


def print_device_summary(requested_device, actual_device):
    normalized_requested = str(requested_device).strip().lower()
    actual_device = str(actual_device)
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0

    print(
        "Dispositivo PyTorch:",
        {
            "requested": requested_device,
            "actual": actual_device,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
        },
    )

    if "cuda" in normalized_requested and not actual_device.startswith("cuda"):
        print(
            "ADVERTENCIA: se solicito GPU/CUDA pero el modelo no quedo en CUDA. "
            "Revisar instalacion de PyTorch con soporte GPU."
        )



def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value



def resolve_torch_device(requested_device):
    """Resuelve 'auto' a cuda si esta disponible, si no cpu. Devuelve string."""
    requested = str(requested_device).strip().lower()
    if requested in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def build_vec_env(
    host,
    ports,
    resume_vecnormalize_path=None,
    initial_reset_options=None,
    n_stack=1,
    norm_reward=False,
    device="cpu",
):
    # Permitir pasar un solo puerto (int) o una lista de puertos.
    if isinstance(ports, int):
        ports = [ports]

    # Shape de la imagen de UN frame (antes de stackear): RND opera sobre el frame mas
    # nuevo, asi que necesita esta shape para construir sus redes. La leo de un env
    # "sonda" construido sin conectar (el observation_space sale del .env, no del sim).
    probe_env = build_env(host, ports[0], monitor=False)
    base_image_shape = tuple(probe_env.observation_space.spaces["image"].shape)
    probe_env.close()

    # Una factory por puerto. Cada NavEnv recibe SU puerto como argumento (no por env var),
    # asi cada worker se conecta a la instancia de Webots correcta.
    env_fns = [
        (lambda p=p: build_env(host, p, monitor=True, reset_options=initial_reset_options))
        for p in ports
    ]
    # DummyVecEnv para 1 env (sin overhead de subprocesos); SubprocVecEnv para N en
    # paralelo. En Windows el start_method DEBE ser "spawn" (no hay fork).
    if len(env_fns) == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(vec_env)
    if resume_vecnormalize_path:
        vec_env = VecNormalize.load(resume_vecnormalize_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = bool(norm_reward)
    else:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=bool(norm_reward),
            norm_obs_keys=["velocity"],
        )

    # Frame stacking: apila los ultimos n_stack frames para dar info TEMPORAL.
    # Sin esto, un solo frame no codifica velocidad -> no se pueden anticipar
    # obstaculos dinamicos (ni distinguir bien "trabado" de "avanzando").
    # Se aplica DESPUES de VecNormalize (la normalizacion opera sobre el frame sin
    # apilar). VecFrameStack apila TODAS las keys del Dict: la imagen sobre el eje
    # de canales, y velocity/goal_direction como historia temporal de esos vectores
    # (lo cual ademas suma). El NavExtractor se adapta solo a las nuevas shapes.
    if n_stack and n_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)

    return vec_env




def load_and_freeze_cnn(model, checkpoint_path):
    """
    Inicializa la CNN (rama imagen del NavExtractor) de `model` desde un checkpoint de
    homing y la CONGELA (requires_grad=False). Descomposicion explorar/explotar: la policy
    de busqueda reusa el backbone visual ya entrenado con el target en vez de aprenderlo de
    cero; solo entrena LSTM + cabezas.

    Transfiere SOLO la rama imagen (cnn + cnn_head); la vector_head y el resto quedan como
    estan. Aplica a todos los features extractors de la policy (shared o pi/vf separados).
    Falla con mensaje claro si los canales no matchean (n_stack/resolucion distinta).
    """
    from stable_baselines3.common.save_util import load_from_zip_file

    _, params, _ = load_from_zip_file(checkpoint_path, device="cpu")
    src = params.get("policy", {})

    # Tomar las keys de la rama imagen del extractor fuente, sin el prefijo del extractor.
    prefix = "features_extractor."
    cnn_state = {
        key[len(prefix):]: value
        for key, value in src.items()
        if key.startswith(prefix + "cnn.") or key.startswith(prefix + "cnn_head.")
    }
    if not cnn_state:
        raise RuntimeError(
            f"El checkpoint {checkpoint_path} no tiene features_extractor.cnn* "
            "(¿no fue entrenado con NavExtractor?)."
        )

    extractor_attrs = ["features_extractor", "pi_features_extractor", "vf_features_extractor"]
    frozen = 0
    for attr in extractor_attrs:
        extractor = getattr(model.policy, attr, None)
        if extractor is None:
            continue
        try:
            extractor.load_state_dict(cnn_state, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                f"No se pudo cargar la CNN de {checkpoint_path}: {exc}. "
                "Probablemente el n_stack/resolucion no coincide (la CNN espera otros canales)."
            ) from exc
        for module in (extractor.cnn, extractor.cnn_head):
            for param in module.parameters():
                param.requires_grad = False
                frozen += param.numel()
    return frozen


def build_model(args, vec_env, run_paths):
    learning_rate = args.learning_rate
    clip_range = args.clip_range
    model_kwargs = {
        "learning_rate": learning_rate,
        "n_steps": args.n_steps,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "tensorboard_log": run_paths["tensorboard_log"],
        "seed": args.seed,
        "device": args.device,
        "target_kl": args.target_kl,
        "clip_range": clip_range,
        "vf_coef": args.vf_coef,
        "ent_coef": args.ent_coef,
        "max_grad_norm": 0.5,
    }

    # Rama vision_lstm: RecurrentPPO con LSTM en la policy (MultiInputLstmPolicy) en vez del
    # PPO feedforward. La recurrencia aporta memoria temporal -> se entrena con --n-stack 1
    # (la recurrencia REEMPLAZA al frame stacking). Comparacion del ablation: vision (1 frame)
    # / vision apilada (4 frames) / vision LSTM (1 frame + recurrencia).
    model = RecurrentPPO(
            "MultiInputLstmPolicy",
            vec_env,
            verbose=1,
            policy_kwargs={"lstm_hidden_size": args.lstm_hidden_size},
            **model_kwargs,
    )
    return model


def resolve_resume_context(args):
    if not args.resume_model_path:
        return {
            "resume_model_path": None,
            "resume_vecnormalize_path": None,
            "reset_num_timesteps": True,
        }

    resume_model_path = resolve_existing_path(args.resume_model_path, "modelo")
    resume_vecnormalize_path = None
    if args.resume_vecnormalize_path:
        resume_vecnormalize_path = resolve_existing_path(
            args.resume_vecnormalize_path, "vecnormalize"
        )
    else:
        resume_vecnormalize_path = infer_resume_vecnormalize_path(resume_model_path)

    if resume_vecnormalize_path is None:
        print(
            "ADVERTENCIA: no se encontro vecnormalize.pkl para el modelo a reanudar. "
            "Se iniciaran estadisticas nuevas de normalizacion."
        )

    return {
        "resume_model_path": resume_model_path,
        "resume_vecnormalize_path": resume_vecnormalize_path,
        "reset_num_timesteps": bool(args.reset_timesteps_on_resume),
    }


def write_training_run_metadata(args, model, run_paths):
    run_metadata = {
        "run_id": run_paths["run_id"],
        "run_dir": os.path.abspath(run_paths["run_dir"]),
        "requested_device": args.device,
        "actual_device": str(model.device),
        "artifacts": {
            "final_model_path": resolve_output_path(run_paths["final_model_path"]),
            "tensorboard_log": os.path.abspath(run_paths["tensorboard_log"]),
            "vecnormalize_path": os.path.abspath(run_paths["vecnormalize_path"]),
        },
        "hyperparameters": {
            "total_timesteps": int(args.total_timesteps),
            "learning_rate": float(args.learning_rate),
            "vf_coef": float(args.vf_coef),
            "ent_coef": float(args.ent_coef),
            "clip_range": float(args.clip_range),
            "n_steps": int(args.n_steps),
            "n_envs": int(max(1, args.n_envs)),
            "gamma": float(args.gamma),
            "norm_reward": bool(args.norm_reward),
            "n_stack": int(args.n_stack),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "seed": int(args.seed),
            "host": args.host,
            "port": int(args.port),
            # Marca de la rama vision_lstm: el eval debe cargar con RecurrentPPO y manejar
            # el estado del LSTM en predict().
            "recurrent": True,
            "lstm_hidden_size": int(args.lstm_hidden_size),
        },
    }

    with open(run_paths["metadata_path"], "w", encoding="utf-8") as metadata_file:
        json.dump(run_metadata, metadata_file, indent=2, ensure_ascii=True)

    return os.path.abspath(run_paths["metadata_path"])




def run_training(args):
    """
    Ejecuta una sesion de entrenamiento completa. Devuelve rutas clave al terminar con exito.
    """
    webots_processes = []

    # Puertos: base, base+1, ... uno por entorno paralelo.
    n_envs = max(1, int(args.n_envs))
    base_port = int(args.base_port) if args.base_port is not None else int(args.port)
    ports = [base_port + i for i in range(n_envs)]

    if not args.no_webots_launch:
        if n_envs == 1:
            webots_processes = [launch_webots(args)]
        else:
            webots_processes = launch_webots_instances(args, ports)

    run_paths = build_training_run_paths()
    populate_run_artifact_paths(run_paths, args)
    os.makedirs(run_paths["run_dir"], exist_ok=False)
    os.makedirs(run_paths["tensorboard_log"], exist_ok=True)

    if not args.skip_check_env:
        env_for_check = build_env(
            args.host,
            ports[0],
            monitor=False,
        )
        try:
            check_env(env_for_check, warn=True)
        finally:
            env_for_check.close()

    vec_env = build_vec_env(
        args.host,
        ports,
        n_stack=args.n_stack,
        norm_reward=args.norm_reward,
        device=args.device,
    )

    final_model_path = resolve_output_path(run_paths["final_model_path"])
    os.makedirs(
        os.path.dirname(final_model_path) or run_paths["run_dir"], exist_ok=True
    )

    model = build_model(args, vec_env, run_paths)
    print_device_summary(args.device, model.device)

    callbacks = [RLMetricsCallback()]
    metrics_callback = callbacks[0]
    callback = CallbackList(callbacks)
    metadata_path = None

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=True,
            callback=callback,
        )
        model.save(final_model_path)
        # Con VecFrameStack afuera, el save de VecNormalize va sobre la capa interna.
        unwrap_vec_normalize(vec_env).save(run_paths["vecnormalize_path"])
        metadata_path = write_training_run_metadata(args, model, run_paths)
        metrics_callback.print_summary()
        print(f"Carpeta de corrida: {run_paths['run_dir']}")
        print(f"Modelo final guardado en: {final_model_path}")
        print(f"TensorBoard en: {os.path.abspath(run_paths['tensorboard_log'])}")
        print(f"VecNormalize en: {os.path.abspath(run_paths['vecnormalize_path'])}")
        print(f"Metadata del entrenamiento en: {metadata_path}")
    finally:
        vec_env.close()
        for process in webots_processes:
            try:
                process.kill()
            except Exception:
                pass

    train_metrics = {
        "run_id": run_paths["run_id"],
        "run_dir": os.path.abspath(run_paths["run_dir"]),
        "final_model_path": os.path.abspath(final_model_path),
        "metadata_path": os.path.abspath(metadata_path) if metadata_path else None,
        "vecnormalize_path": os.path.abspath(run_paths["vecnormalize_path"]),
        "total_timesteps": int(args.total_timesteps),
        "ending_timesteps": int(model.num_timesteps),
        "requested_device": args.device,
        "actual_device": str(model.device),
        "learning_rate": float(args.learning_rate),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
    }
    if not args.no_discord:
        print("Enviando notificacion a Discord...")
        

    return train_metrics


def main():
    run_training(parse_args())


if __name__ == "__main__":
    main()
