import os
import sys
# El supervisor corre desde controllers/supervisor_controller/ — agregar project root
# para que los imports de helpers.* resuelvan a la carpeta global helpers/.
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import json
import math
import numpy as np
from controller import Supervisor
from helpers.robot_bridge import (log_supervisor, RobotBridge, summarize_robot_message)
from helpers.read_env_value import read_env_value
from helpers.training_server import TrainingServer
from helpers.policy_runner import PolicyRunner
from helpers.image_obs import blank_image_payload
from helpers.lane_vision import decode_rgb_hwc, detect_lane, road_band_offsets
from helpers.track_progress import build_loop, project_s, signed_delta
from helpers.geom_obs import GEOM_BANDS, GEOM_BOUND, blank_geom, geom_vector_from_rgb


# Domain randomization de la pose del robot al inicio de cada episodio: jitter de
# rotacion/traslacion SOBRE el spawn elegido, para mejorar la generalizacion.
# Activado por defecto; todo configurable por .env.
DOMAIN_RANDOMIZATION_ENABLED = bool(read_env_value("DOMAIN_RANDOMIZATION_ENABLED", 1, int))
# Probabilidad de aplicar el jitter en un episodio dado.
DOMAIN_RANDOMIZATION_PROBABILITY = read_env_value("DOMAIN_RANDOMIZATION_PROBABILITY", 0.5)
# Perturbacion maxima de rotacion (en GRADOS, +/-) — se convierte a radianes.
DOMAIN_RANDOMIZATION_MAX_ROTATION = math.radians(
    read_env_value("DOMAIN_RANDOMIZATION_MAX_ROTATION_DEG", 15.0)
)
# Perturbacion maxima de traslacion en x e y (en METROS, +/-).
DOMAIN_RANDOMIZATION_MAX_TRANSLATION = read_env_value("DOMAIN_RANDOMIZATION_MAX_TRANSLATION", 0.02)

# Domain randomization VISUAL del fondo: randomiza el color de las PAREDES de la arena por
# episodio, para que la CNN NO se sesgue a la textura/color del fondo (ver analysis/
# cnn_activations: la vision-RL fijaba la atencion en el muro/horizonte). Flag PROPIO: el
# eval lo apaga para una corrida determinista.
BACKGROUND_RANDOMIZATION_ENABLED = bool(read_env_value("BACKGROUND_RANDOMIZATION_ENABLED", 1, int))
BACKGROUND_RANDOMIZATION_PROBABILITY = read_env_value("BACKGROUND_RANDOMIZATION_PROBABILITY", 0.8)
# Carpeta con el pool de texturas de pared (PNGs) que se swapean por episodio (como el piso).
WALL_TEXTURE_DIR = read_env_value("WALL_TEXTURE_DIR", "worlds/wall_textures", str)
# Skyboxes que trae Webots por defecto (campo `texture` de TexturedBackground); se rotan por
# episodio para que el fondo lejano (montañas/edificios) tampoco sea una señal fija.
SKYBOX_TEXTURES = [
    "mountains", "noon_cloudy_countryside", "noon_park_empty", "factory",
    "entrance_hall", "empty_office", "dawn_cloudy_empty", "morning_cloudy_empty",
    "noon_building_overlook", "twilight_cloudy_empty", "dusk", "mars",
]

# ----------------------------------------------------------------------------
# Constantes por defecto.
# La estructura viene de otro proyecto; estos valores son defaults razonables
# para que el controlador arranque sin errores. Se pueden sobreescribir por .env.
# ----------------------------------------------------------------------------
# Pasos maximos a esperar una respuesta del robot antes de declarar timeout.
MAX_WAIT_STEPS = read_env_value("MAX_WAIT_STEPS", 1000)
# Pasos maximos por episodio de entrenamiento.
MAX_EPISODE_STEPS = read_env_value("MAX_EPISODE_STEPS", 1000)
# Pasos de simulacion que se dejan "asentar" tras un reset de pose.
RESET_SETTLE_STEPS = read_env_value("RESET_SETTLE_STEPS", 10)
# Cantidad de ticks que se repite una misma accion (frame-skip).
ACTION_REPEAT = read_env_value("ACTION_REPEAT", 5)
# Cada cuantos steps se manda un heartbeat de estado a la UI cuando no hay cliente.
UI_HEARTBEAT_PERIOD = read_env_value("UI_HEARTBEAT_PERIOD", 200)
# Bonus terminal al alcanzar el target.
ARRIVAL_BONUS = read_env_value("ARRIVAL_BONUS", 1.0)
# Velocidades maximas REALES (para normalizar la velocidad a ~[-1, 1]). Es el
# DIVISOR de normalizacion: debe ser ~la vmax real del robot para que "a fondo"
# mapee a ~1.0. e-puck: WHEEL_MAX_SPEED * radio_rueda = 5.0 * 0.02 = 0.1 m/s.
MAX_LINEAR_SPEED = read_env_value("MAX_LINEAR_SPEED", 0.1)
MAX_ANGULAR_SPEED = read_env_value("MAX_ANGULAR_SPEED", 2.0 * math.pi)
# Radio por defecto del target cuando no se puede resolver del nodo.
TARGET_DEFAULT_RADIUS = read_env_value("TARGET_DEFAULT_RADIUS", 0.05)
# Composicion explorar/explotar: ruta del modelo de homing. Vacio => deshabilitado.
EXPLOIT_MODEL_PATH = read_env_value("EXPLOIT_MODEL_PATH", "") or None
# Gate de visibilidad del target para la composicion de policies.
TARGET_VISIBLE_MIN_PIXELS = read_env_value("TARGET_VISIBLE_MIN_PIXELS", 50)
GATE_HYSTERESIS_STEPS = read_env_value("GATE_HYSTERESIS_STEPS", 5)
# Nombres de color validos para forzar el target via reset-options.
COLOR_NAMES = ("red", "green", "blue", "yellow")

# ----------------------------------------------------------------------------
# Reward de seguimiento de carril (opcion A: vision-pura desde la camara frontal).
# Pesos y umbrales tuneables por .env. El detector de carril vive en
# helpers/lane_vision; aca solo se componen los terminos del reward.
# ----------------------------------------------------------------------------
# Estructura: reward = clearance * (REWARD_BASE + REWARD_SPEED_W * speed) + penas.
# La velocidad es el termino PRINCIPAL y el centrado (clearance) es un GATE que
# multiplica todo: solo se cobra fuerte yendo RAPIDO y CENTRADO. Como el robot ya
# no puede frenar ni retroceder, "rapido + centrado" == "recorrer la pista rapido".
# Base por ir en el carril (multiplicada por clearance): mantiene el reward NETO
# POSITIVO portandose bien, asi salirse (terminal -1) nunca es "conveniente".
REWARD_BASE = read_env_value("REWARD_BASE", 0.1)
# Velocidad: termino principal. speed_norm en ~[0.15, 1.0] (ver MAX_LINEAR_SPEED).
REWARD_SPEED_W = read_env_value("REWARD_SPEED_W", 1.0)
# Direccion: penaliza el offset lateral firmado (rompe la simetria del centrado).
REWARD_OFFSET_W = read_env_value("REWARD_OFFSET_W", 0.3)
# Borde: penaliza tener blanco (linea externa) en la banda central.
REWARD_WHITE_W = read_env_value("REWARD_WHITE_W", 0.5)
# Costo por step. OFF por defecto: el robot ya no puede quedarse quieto, asi que
# no hace falta, y en neto negativo incentivaria chocar para terminar el episodio.
REWARD_STEP_COST = read_env_value("REWARD_STEP_COST", 0.0)
# Off-track: fraccion de pasto en la banda central para declarar que se fue.
OFFTRACK_GREEN_FRAC = read_env_value("OFFTRACK_GREEN_FRAC", 0.4)
# Gracia de off-track: steps CONSECUTIVOS sobre pasto antes de terminar. No se corta apenas
# se detecta verde (podria ser una curva muy cerrada o un desvio momentaneo del que se
# recupera); recien se termina si sigue afuera OFFTRACK_GRACE_STEPS pasos seguidos.
OFFTRACK_GRACE_STEPS = read_env_value("OFFTRACK_GRACE_STEPS", 6, int)
# Reward terminal al salirse de la pista.
OFFTRACK_PENALTY = read_env_value("OFFTRACK_PENALTY", -1.0)
# Penalizacion por step sin ver ninguna marca (carril perdido).
LINE_LOST_PENALTY = read_env_value("LINE_LOST_PENALTY", -0.3)
# Steps consecutivos sin ver el carril antes de terminar el episodio.
LOST_LINE_MAX_STEPS = read_env_value("LOST_LINE_MAX_STEPS", 20, int)

# ----------------------------------------------------------------------------
# Progreso por gates (si el track tiene "gates" en spawns.json). Reemplaza al
# termino de velocidad como motor de avance: premia avanzar SOBRE el circuito
# (Δarc-length), no ir rapido en cualquier direccion (mata loop/reversa/corte).
# ----------------------------------------------------------------------------
# Peso del progreso. Se multiplica por un Δs normalizado (~[-1, 1]) y por clearance.
REWARD_PROGRESS_W = read_env_value("REWARD_PROGRESS_W", 1.0)
# Bonus terminal al completar una vuelta (progreso neto >= perimetro).
LAP_BONUS = read_env_value("LAP_BONUS", 5.0)
# Penalizacion terminal por ir en contramano demasiados steps.
WRONG_WAY_PENALTY = read_env_value("WRONG_WAY_PENALTY", -1.0)
# Steps consecutivos de progreso negativo (contramano) antes de terminar.
WRONG_WAY_MAX_STEPS = read_env_value("WRONG_WAY_MAX_STEPS", 30, int)

# ----------------------------------------------------------------------------
# Linea de largada/meta UNICA (fallback de deteccion de vuelta cuando el track NO
# tiene gates). La linea pasa por el punto del spawn, perpendicular al rumbo del
# spawn; el SENTIDO valido es el del spawn (rumbo +). Una vuelta = el robot, tras
# alejarse hacia adelante, vuelve y cruza la linea en el sentido del spawn.
# ----------------------------------------------------------------------------
# Semi-ancho lateral (m) de la linea: el cruce solo cuenta si el robot pasa a menos
# de esta distancia del punto del spawn (asi solo dispara cerca de la largada, no en
# el otro extremo del circuito que cae en el mismo semiplano).
START_LINE_HALF_WIDTH = read_env_value("START_LINE_HALF_WIDTH", 0.5)
# Distancia (m) hacia adelante que el robot debe alejarse de la linea para "armar"
# la deteccion (evita contar la largada inicial como vuelta).
START_LINE_ARM_DISTANCE = read_env_value("START_LINE_ARM_DISTANCE", 0.5)


class SupervisorController:
    """
    Controlador del supervisor.
    Crea el bridge con el robot
    """

    def __init__(self):
        #
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.timestep)
        self.emitter.setChannel(1)  # Asegurar que el emisor use el mismo canal que el robot
        self.bridge = RobotBridge(
            supervisor=self.supervisor,
            emitter=self.emitter,
            receiver=self.receiver,
            timestep=self.timestep,
        )
        # 
        self.training_server = TrainingServer()
        self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
        # DEFINICION DE LOS ELEMENTOS DEL AMBIENTE
        self.arena_node = self.supervisor.getFromDef("ARENA")
        # Textura del piso (DEF FLOOR_TEX en el .wbt): se swapea por episodio para
        # entrenar en distintos tracks. None si el .wbt no tiene el DEF (no swapea).
        self.floor_texture_node = self.supervisor.getFromDef("FLOOR_TEX")
        self.floor_texture_url_field = (
            self.floor_texture_node.getField("url") if self.floor_texture_node else None
        )
        # Domain randomization VISUAL del fondo (best-effort; None si el .wbt no tiene el DEF):
        #  - WALL_TEX: textura de la pared, se swapea desde un pool por episodio (como el piso).
        #  - WALL_APP.baseColor: tinte suave sobre la textura, para mas variedad.
        #  - SKYBOX.texture: rota entre los skyboxes que trae Webots.
        wall_app_node = self.supervisor.getFromDef("WALL_APP")
        self.wall_basecolor_field = (
            wall_app_node.getField("baseColor") if wall_app_node else None
        )
        wall_tex_node = self.supervisor.getFromDef("WALL_TEX")
        self.wall_texture_url_field = (
            wall_tex_node.getField("url") if wall_tex_node else None
        )
        skybox_node = self.supervisor.getFromDef("SKYBOX")
        self.skybox_texture_field = (
            skybox_node.getField("texture") if skybox_node else None
        )
        self.wall_texture_pool = self._load_wall_texture_pool()
        # DEFINICION DEL ROBOT E-PUCK
        self.epuck_robot = self.supervisor.getFromDef("EPUCK")
        self.epuck_translation_field = self.epuck_robot.getField("translation")
        self.epuck_rotation_field = self.epuck_robot.getField("rotation")
        self.epuck_robot_initial_translation = list(self.epuck_translation_field.getSFVec3f())
        self.epuck_robot_initial_rotation = list(self.epuck_rotation_field.getSFRotation())
        self.last_action_label = "idle"
        self.last_robot_message = None
        self.last_camera_frame = None
        self.last_observation_image = None
        self.debug_frame_mode = "full"
        self.step_count = 0
        self.robot_request_counter = 0
        self.episode_id = 0
        self.episode_step = 0
        self.lost_line_steps = 0
        self.offtrack_steps = 0
        self.policy_runner = PolicyRunner()
        self.policy_status = self.policy_runner.status_dict()
        self.policy_debug_snapshot = None
        self.reset_rng = np.random.default_rng()
        # Mapa track->spawns (spawns.json). None => textura/pose por defecto del .wbt.
        self.track_spawns = self._load_track_spawns()
        # Catalogo para el selector de pista de la UI: TODAS las worlds/*.png (incl. eval)
        # + pose de spawn por textura (para teleportar al cambiar de pista a mano).
        self.ui_track_list, self.ui_track_poses = self._load_ui_track_catalog()
        # Track del episodio en curso (para metricas/telemetria). None = default del .wbt.
        self.current_track_texture = None
        # Seleccion actual de fondo para la UI (dropdowns de pared / skybox). Arrancan en el
        # default del .wbt (primer item del pool y primer skybox de la lista).
        self.current_wall_texture = self.wall_texture_pool[0] if self.wall_texture_pool else None
        self.current_skybox = SKYBOX_TEXTURES[0] if SKYBOX_TEXTURES else None
        # Estado de progreso por gates del episodio en curso (None = track sin gates).
        self.current_loop = None
        self.progress_s = 0.0
        self.cumulative_progress = 0.0
        self.wrong_way_steps = 0
        # Spawn del episodio en curso (translation, rotation) para la linea de meta.
        self.current_spawn = None
        # Linea de largada/meta unica (fallback sin gates). None = no aplica.
        self.start_line = None
        self.start_line_armed = False
        self.start_line_prev_dfwd = 0.0
        # Maximo avance esperado por step, para normalizar Δs a ~[-1, 1].
        self._max_step_progress = max(
            1e-6, MAX_LINEAR_SPEED * (self.timestep / 1000.0) * ACTION_REPEAT
        )
        self.current_map_config = None
        self.current_spawn_pose_index = None
        self.episodes_on_current_map = 0
        self.last_commanded_wheel_state = [0.0, 0.0]
        self.episode_max_steps = MAX_EPISODE_STEPS
        self.ui_heartbeat_period = UI_HEARTBEAT_PERIOD
        self.current_recording_path = None
        self.policy_step_counter = 0
        # Ultimo desglose de reward computado (para mostrarlo en la Robot Window).
        self.last_reward_breakdown = None
        # RESET DEL ROBOT A LA POSE INICIAL
        self._reset_robot_pose()
        # PONGO EL WEBOTS EN MODO RAPIDO PARA QUE LA SIMULACION CORRA A MAXIMA VELOCIDAD NORMAL DESDE EL INICIO
        self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

    def _next_robot_request_id(self, prefix="robot"):
        """
        Genera un ID único para un request del robot.
        """
        self.robot_request_counter += 1
        return f"{prefix}-{self.robot_request_counter}"

    def _step_world(self, process_passive=False):
        """
        Step del mundo. Simula el paso de tiempo del ambiente.
        """
        if self.supervisor.step(self.timestep) == -1:
            return False

        self.step_count += 1
        self.bridge.drain_robot_messages()

        if process_passive:
            self._handle_training_requests()
            self._run_policy_step()
            self._handle_ui_messages()
            self._handle_unmatched_robot_messages()

            if (
                not self.training_server.is_client_connected()
                and self.step_count % self.ui_heartbeat_period == 0
            ):
                self._send_ui_state(
                    "Esperando acciones manuales o requests de entrenamiento."
                )

        return True

    def _advance_simulation(self, steps):
        """
        Avanza la simulación del ambiente por un número de steps.
        """
        for _ in range(steps):
            if not self._step_world(process_passive=False):
                raise RuntimeError(
                    "La simulacion de Webots se detuvo durante un avance interno."
                )

    def _reset_robot_pose(self, translation=None, rotation=None):
        """
        Resetea la pose del robot a una posición y rotación específica o a la pose inicial.
        """
        target_translation = list(translation or self.epuck_robot_initial_translation)
        target_rotation = list(rotation or self.epuck_robot_initial_rotation)
        self.epuck_translation_field.setSFVec3f(target_translation)
        self.epuck_rotation_field.setSFRotation(target_rotation)
        self.epuck_robot.resetPhysics()
        self.last_commanded_wheel_state = [0.0, 0.0]
        self.previous_line_error = None
        self.previous_line_visible = False
        self.last_action_label = "reset_pose"
        log_supervisor(
            f"[Supervisor] pose restaurada: translation={target_translation}, rotation={target_rotation}"
        )

    def _apply_domain_randomization(self):
        """
        Aplica domain randomization al robot: rotacion suave y traslacion suave con cierta probabilidad.

        Se aplica una pequeña perturbacion aleatoria a la pose actual del robot para mejorar
        la generalizacion durante el entrenamiento. Los parametros se controlan via variables
        de entorno o constantes (DOMAIN_RANDOMIZATION_*).
        """
        if not DOMAIN_RANDOMIZATION_ENABLED:
            return

        # Decidir si aplicar domain randomization esta vez
        if self.reset_rng.random() > DOMAIN_RANDOMIZATION_PROBABILITY:
            return

        # Obtener pose actual
        current_translation = list(self.epuck_translation_field.getSFVec3f())
        current_rotation = list(self.epuck_rotation_field.getSFRotation())

        # Aplicar rotacion suave: perturbacion pequeña del angulo alrededor del eje Z
        # El formato de rotacion es (axis_x, axis_y, axis_z, angle_rad)
        # Para un robot en el plano, solo modificamos el angulo manteniendo el eje Z
        random_rotation_delta = self.reset_rng.uniform(
            -DOMAIN_RANDOMIZATION_MAX_ROTATION, DOMAIN_RANDOMIZATION_MAX_ROTATION
        )
        perturbed_rotation = list(current_rotation)
        perturbed_rotation[3] = float(current_rotation[3] + random_rotation_delta)

        # Aplicar traslacion suave: perturbacion pequeña en x e y (no en z)
        random_translation_x = self.reset_rng.uniform(
            -DOMAIN_RANDOMIZATION_MAX_TRANSLATION, DOMAIN_RANDOMIZATION_MAX_TRANSLATION
        )
        random_translation_y = self.reset_rng.uniform(
            -DOMAIN_RANDOMIZATION_MAX_TRANSLATION, DOMAIN_RANDOMIZATION_MAX_TRANSLATION
        )
        perturbed_translation = [
            float(current_translation[0] + random_translation_x),
            float(current_translation[1] + random_translation_y),
            float(current_translation[2]),  # Z no se modifica
        ]
        # Aplicar la nueva pose y resetear la fisica (que no arrastre estado del jitter).
        self.epuck_translation_field.setSFVec3f(perturbed_translation)
        self.epuck_rotation_field.setSFRotation(perturbed_rotation)
        self.epuck_robot.resetPhysics()

    def _load_wall_texture_pool(self):
        """URLs (relativas al world) de las texturas de pared en WALL_TEXTURE_DIR. [] si no hay."""
        wall_dir = os.path.join(PROJECT_ROOT, WALL_TEXTURE_DIR)
        worlds_dir = os.path.join(PROJECT_ROOT, "worlds")
        try:
            files = sorted(f for f in os.listdir(wall_dir) if f.lower().endswith(".png"))
        except OSError:
            return []
        return [os.path.relpath(os.path.join(wall_dir, f), worlds_dir).replace("\\", "/")
                for f in files]

    def _randomize_background(self):
        """
        Domain randomization VISUAL del fondo por episodio (con probabilidad): swapea la
        TEXTURA de la pared desde el pool (como el piso), rota el SKYBOX entre los que trae
        Webots, y aplica un tinte suave al color de la pared. Si el fondo es ruido respecto a
        la accion correcta, la CNN no puede usarlo de atajo -> obligada a mirar la calzada.
        Best-effort (cada pieza guardada por su handle). El eval apaga BACKGROUND_RANDOMIZATION_
        ENABLED (corrida determinista).
        """
        if not BACKGROUND_RANDOMIZATION_ENABLED:
            return
        if self.reset_rng.random() > BACKGROUND_RANDOMIZATION_PROBABILITY:
            return
        # 1) Textura de pared desde el pool.
        if self.wall_texture_url_field is not None and self.wall_texture_pool:
            tex = self.wall_texture_pool[int(self.reset_rng.integers(len(self.wall_texture_pool)))]
            self.wall_texture_url_field.setMFString(0, tex)
        # 2) Tinte suave sobre la textura (no oscurecer de mas: [0.55, 1.0]).
        if self.wall_basecolor_field is not None:
            self.wall_basecolor_field.setSFColor(
                [float(self.reset_rng.uniform(0.55, 1.0)) for _ in range(3)])
        # 3) Skybox de Webots (fondo lejano: montañas/edificios/etc.).
        if self.skybox_texture_field is not None:
            sky = SKYBOX_TEXTURES[int(self.reset_rng.integers(len(SKYBOX_TEXTURES)))]
            self.skybox_texture_field.setSFString(sky)

    def _configure_episode_start(self, seed=None, options=None):
        """
        Configura el inicio de un episodio.
        """
        pass

    def _request_robot(
        self, message_type, expected_type, max_steps=MAX_WAIT_STEPS, **payload
    ):
        """
        Envía un mensaje al robot y espera una respuesta.
        """
        request_id = self._next_robot_request_id()
        message = {"type": message_type, "request_id": request_id, **payload}
        self.bridge.send_to_robot(message)
        response = self._wait_for_robot_message(
            expected_type, request_id, max_steps=max_steps
        )
        if response is None:
            raise RuntimeError(
                f"Timeout esperando '{expected_type}' del robot para request_id={request_id}"
            )
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "Error desconocido del robot"))
        self._record_robot_message(response)
        return response

    def _wait_for_robot_message(self, expected_type, request_id, max_steps):
        """
        Espera una respuesta del robot para un request específico.
        """
        response = self._pop_response_for_request(expected_type, request_id)
        if response is not None:
            return response

        for _ in range(max_steps):
            if not self._step_world(process_passive=False):
                return None
            response = self._pop_response_for_request(expected_type, request_id)
            if response is not None:
                return response

        return None

    def _pop_response_for_request(self, expected_type, request_id):
        """
        Extrae una respuesta del buffer de mensajes del robot para un request específico.
        """
        error_message = self.bridge.pop_error(request_id=request_id)
        if error_message is not None:
            return error_message
        return self.bridge.pop_message(
            expected_type=expected_type, request_id=request_id
        )

    def _record_robot_message(self, message):
        """
        Registra un mensaje del robot.
        """
        self.last_robot_message = summarize_robot_message(message)
        wheel_state = message.get("wheel_state")
        if isinstance(wheel_state, list) and len(wheel_state) >= 2:
            self.last_commanded_wheel_state = [
                float(wheel_state[0]),
                float(wheel_state[1]),
            ]
        if message.get("type") == "debug_frame":
            self.last_camera_frame = message.get("camera")
            self.debug_frame_mode = message.get("mode", self.debug_frame_mode)
        # (Sin log per-mensaje: corria ~2x por step en training construyendo el string
        # aunque SUPERVISOR_VERBOSE_MESSAGES este en False. last_robot_message igual queda
        # para la UI.)

    def _send_ui_message(self, payload):
        """

        Envia un mensaje a la UI.
        """
        self.supervisor.wwiSendText(json.dumps(payload))

    def _send_ui_state(self, status):
        """
        Envia el estado actual de la simulación a la UI.
        """
        robot_pos  = list(self.epuck_robot.getPosition())
        features = self._lane_features()
        payload = {
            "type": "status",
            "status": status,
            "last_action": self.last_action_label,
            "last_robot_message": self.last_robot_message,
            "camera_frame": self.last_camera_frame,
            "step_count": self.step_count,
            "episode_id": self.episode_id,
            "episode_step": self.episode_step,
            "training_client_connected": self.training_server.is_client_connected(),
            "policy_status": self.policy_status,
            "policy_debug_snapshot": self.policy_debug_snapshot,
            # Estado de carril (vision-pura) + ultimo desglose de reward.
            "lane": {
                "offset": features.get("offset"),
                "center_clearance": round(features.get("center_clearance", 0.0), 4),
                "green_center": round(features.get("center_green", 0.0), 4),
                "white_center": round(features.get("center_white", 0.0), 4),
                # RGB crudo de la banda central (para calibrar el umbral de off-track).
                "center_rgb": features.get("center_rgb"),
                "line_visible": features.get("line_visible", False),
            },
            "reward_breakdown": self.last_reward_breakdown,
            "robot_pos": [round(robot_pos[0], 4), round(robot_pos[1], 4)],
            # Selector de pista (UI): catalogo + pista actual.
            "available_tracks": self.ui_track_list,
            "current_track": self.current_track_texture,
            # Selectores de fondo (UI): pool de texturas de pared + skyboxes de Webots.
            "available_wall_textures": self.wall_texture_pool,
            "current_wall_texture": self.current_wall_texture,
            "available_skyboxes": SKYBOX_TEXTURES,
            "current_skybox": self.current_skybox,
        }
        self._send_ui_message(payload)

    def _handle_ui_messages(self):
        """
        Maneja los mensajes de la UI.
        """
        while True:
            raw_message = self.supervisor.wwiReceiveText()
            if not raw_message:
                break

            log_supervisor(f"[Supervisor] mensaje desde UI: {raw_message}")
            if raw_message.startswith("configure "):
                continue

            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                log_supervisor("[Supervisor] mensaje de UI no JSON ignorado")
                continue

            message_type = message.get("type")

            try:
                if message_type == "action":
                    label = message.get("label", "manual_action")
                    action = self._normalize_action(
                        message.get("action", [0.0, 0.0])
                    ).tolist()
                    self._handle_manual_action(label, action)
                elif message_type == "capture_frame":
                    mode = message.get("mode", self.debug_frame_mode)
                    self.debug_frame_mode = mode
                    self._capture_debug_frame(mode=mode)
                    self._send_ui_state(f"Frame de debug actualizado ({mode}).")
                elif message_type == "request_policy_debug":
                    self._handle_policy_debug_request()
                    self._send_ui_state("Snapshot de policy actualizado.")
                elif message_type == "save_camera_sample":
                    log_supervisor(
                        "[Supervisor] UI solicito guardar captura manual del robot.",
                        force=True,
                    )
                    saved_paths = self._handle_save_camera_sample()
                    self._send_ui_state(
                        "Captura guardada en carpeta: " f"{saved_paths['capture_dir']}"
                    )
                elif message_type == "configure_policy":
                    self._handle_policy_configuration(message)
                elif message_type == "set_track":
                    self._handle_set_track(message)
                elif message_type == "set_wall_texture":
                    self._handle_set_wall_texture(message)
                elif message_type == "set_skybox":
                    self._handle_set_skybox(message)
                elif message_type == "reset_pose":
                    self._handle_manual_reset()
                    self._send_ui_state("Robot reseteado a la pose inicial.")
                elif message_type == "request_state":
                    self._refresh_policy_status()
                    self._send_ui_state("Estado actualizado.")
            except RuntimeError as error:
                self._send_ui_state(f"Error: {error}")

    def _handle_manual_action(self, label, action):
        """
        Maneja una acción manual.
        """
        self._request_robot(
            "apply_action", "action_applied", label=label, action=action
        )
        self.last_commanded_wheel_state = [float(action[0]), float(action[1])]
        self.last_action_label = label
        self._capture_debug_frame(mode=self.debug_frame_mode)
        self._send_ui_state(f"Accion manual enviada: {label}")

    def _load_ui_track_catalog(self):
        """
        Catalogo para el selector de pista de la UI. Devuelve (texturas, poses):
          - texturas: lista de todas las worlds/*.png (incl. las eval-only), para elegir.
          - poses: {textura: (translation, rotation)} del primer spawn (de spawns.json),
                   para teleportar el robot al cambiar de pista a mano. Best-effort.
        """
        worlds_dir = os.path.join(PROJECT_ROOT, "worlds")
        try:
            textures = sorted(f for f in os.listdir(worlds_dir) if f.lower().endswith(".png"))
        except OSError:
            textures = []
        poses = {}
        path = os.environ.get("SPAWNS_JSON") or os.path.join(PROJECT_ROOT, "spawns.json")
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            for track in (data.get("tracks") or []):
                texture = track.get("texture") if isinstance(track, dict) else None
                spawns = [ss for ss in track.get("spawns", []) if self._valid_spawn(ss)]
                if isinstance(texture, str) and spawns:
                    poses[texture] = (list(spawns[0]["translation"]),
                                      list(spawns[0]["rotation"]))
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            pass
        return textures, poses

    def _handle_set_track(self, message):
        """
        Cambia la textura del piso a la pista elegida desde la UI (para probar/calibrar a
        mano). Si se conoce un spawn de esa pista, teleporta el robot ahi; si no, deja el
        robot donde esta. Avanza unos pasos para que la nueva textura ya se renderice.
        """
        texture = message.get("texture")
        if not texture or (self.ui_track_list and texture not in self.ui_track_list):
            self._send_ui_state(f"Track invalido: {texture}")
            return
        if self.floor_texture_url_field is None:
            self._send_ui_state("Este mundo no tiene DEF FLOOR_TEX; no se puede cambiar la pista.")
            return
        self.floor_texture_url_field.setMFString(0, texture)
        self.current_track_texture = texture
        pose = self.ui_track_poses.get(texture)
        if pose is not None:
            translation, rotation = pose
            self.current_spawn = (translation, rotation)
            self._reset_robot_pose(translation=translation, rotation=rotation)
        self._advance_simulation(RESET_SETTLE_STEPS)
        self._refresh_robot_observation()
        self._send_ui_state(f"Pista cambiada a {texture}.")

    def _handle_set_wall_texture(self, message):
        """Cambia la textura de la PARED (DEF WALL_TEX) a la elegida desde la UI."""
        texture = message.get("texture")
        if not texture or (self.wall_texture_pool and texture not in self.wall_texture_pool):
            self._send_ui_state(f"Textura de pared invalida: {texture}")
            return
        if self.wall_texture_url_field is None:
            self._send_ui_state("Este mundo no tiene DEF WALL_TEX; no se puede cambiar la pared.")
            return
        self.wall_texture_url_field.setMFString(0, texture)
        self.current_wall_texture = texture
        self._advance_simulation(RESET_SETTLE_STEPS)
        self._refresh_robot_observation()
        self._send_ui_state(f"Textura de pared cambiada a {texture}.")

    def _handle_set_skybox(self, message):
        """Cambia el SKYBOX (DEF SKYBOX.texture) al elegido desde la UI."""
        skybox = message.get("skybox")
        if not skybox or (SKYBOX_TEXTURES and skybox not in SKYBOX_TEXTURES):
            self._send_ui_state(f"Skybox invalido: {skybox}")
            return
        if self.skybox_texture_field is None:
            self._send_ui_state("Este mundo no tiene DEF SKYBOX; no se puede cambiar el fondo.")
            return
        self.skybox_texture_field.setSFString(skybox)
        self.current_skybox = skybox
        self._advance_simulation(RESET_SETTLE_STEPS)
        self._refresh_robot_observation()
        self._send_ui_state(f"Skybox cambiado a {skybox}.")

    def _handle_manual_reset(self):
        """
        Maneja el reseteo manual del robot.
        """
        self._reset_robot_pose()
        self._request_robot("reset_robot", "reset_done")
        self._advance_simulation(RESET_SETTLE_STEPS)
        self._capture_debug_frame(mode=self.debug_frame_mode)
        self.last_commanded_wheel_state = [0.0, 0.0]
        self.last_action_label = "manual_reset"
        # Nuevo episodio: limpiar el buffer de frames del stacking de la policy y el
        # estado de carril (si no, contamina el siguiente episodio).
        self.policy_runner.reset_stack()
        self.lost_line_steps = 0
        self.offtrack_steps = 0
        self.last_reward_breakdown = None

    def _capture_debug_frame(self, mode=None):
        """
        Solicita al robot una captura de frame de debug.
        """
        pass

    def _handle_policy_configuration(self, message):
        """
        Carga, activa, desactiva o descarga la policy localmente en el supervisor.
        """
        enabled = message.get("enabled")
        if enabled and self.training_server.is_client_connected():
            raise RuntimeError(
                "No se puede activar la politica mientras el cliente de entrenamiento esta conectado."
            )

        model_path = message.get("model_path")
        if isinstance(model_path, str):
            model_path = model_path.strip() or None

        unload = bool(message.get("unload", False))
        deterministic = message.get("deterministic")

        try:
            if unload:
                self.policy_runner.unload()
            elif model_path:
                self.policy_runner.load(model_path)

            if enabled is not None:
                self.policy_runner.enabled = bool(enabled)
            if deterministic is not None:
                self.policy_runner.deterministic = bool(deterministic)

        except Exception as exc:
            self.policy_runner.last_error = str(exc)
            self.policy_status = self.policy_runner.status_dict()
            raise RuntimeError(f"Error al configurar policy: {exc}") from exc

        self.policy_status = self.policy_runner.status_dict()

        if self.policy_runner.enabled:
            # Pasar a tiempo real para poder VER al robot ejecutar la policy.
            # (En modo FAST Webots no renderiza el 3D).
            self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_REAL_TIME)
            self.last_action_label = "policy_running"
            self._send_ui_state("Politica cargada y ejecutandose.")
        elif self.policy_runner.loaded:
            self.last_action_label = "policy_loaded"
            self._send_ui_state("Politica cargada (sin activar).")
        else:
            self.last_action_label = "policy_disabled"
            self._send_ui_state("Politica desactivada.")

    def _refresh_policy_status(self):
        self.policy_status = self.policy_runner.status_dict()

    def _handle_policy_debug_request(self):
        self._refresh_robot_observation()
        obs  = self._build_inference_observation()
        rp   = self.epuck_robot.getPosition()

        predicted_action = None
        if self.policy_runner.loaded:
            try:
                predicted_action = self.policy_runner.predict(obs)
            except Exception as exc:
                predicted_action = None
                self.policy_runner.last_error = str(exc)

        # Reward REAL del estado actual: mismo desglose que el step de entrenamiento.
        # Usa la accion que la policy tomaria ahora (o la ultima aplicada).
        debug_action = predicted_action or self.policy_runner.last_action or [0.0, 0.0]
        breakdown = self._compute_reward_breakdown(debug_action, obs["velocity"])
        self.last_reward_breakdown = breakdown

        self.policy_debug_snapshot = {
            "policy_loaded":      self.policy_runner.loaded,
            "policy_enabled":     self.policy_runner.enabled,
            "policy_deterministic": self.policy_runner.deterministic,
            "model_path":         self.policy_runner.model_path,
            "velocity":           obs["velocity"],
            "robot_pos":          [round(rp[0], 4), round(rp[1], 4)],
            "last_action":        self.policy_runner.last_action,
            "predicted_action":   predicted_action,
            "reward_inputs":      breakdown,
        }
        self._refresh_policy_status()


    def _run_policy_step(self):
        """
        Si la policy esta activa y no hay cliente de entrenamiento, ejecuta un paso
        de inferencia y envia la accion al robot (fire-and-forget cada ACTION_REPEAT ticks).
        """
        if not self.policy_runner.enabled or not self.policy_runner.loaded:
            return
        if self.training_server.is_client_connected():
            return
        self.policy_step_counter += 1
        if self.policy_step_counter % ACTION_REPEAT != 0:
            return

        try:
            self._refresh_robot_observation()
            obs = self._build_inference_observation()
            action = self.policy_runner.predict(obs)
        except Exception as exc:
            log_supervisor(f"[Supervisor] error al predecir accion: {exc}", force=True)
            self.policy_runner.last_error = str(exc)
            return

        self.bridge.send_to_robot({
            "type": "apply_action",
            "label": "policy",
            "action": action,
        })
        self.last_action_label = "policy_step"
        self.policy_status = self.policy_runner.status_dict()

    def _handle_training_requests(self):
        """
        Maneja las solicitudes de entrenamiento.
        """
        requests = self.training_server.poll_requests()
        for request in requests:
            response = self._dispatch_training_request(request)
            self.training_server.send_response(response)

    def _dispatch_training_request(self, request):
        """
        Despacha una solicitud de entrenamiento.
        """
        request_type = request.get("type")
        request_id = request.get("request_id")

        try:
            if request_type == "reset_env":
                self._disable_policy_for_training_if_needed()
                return self._handle_reset_env_request(request, request_id)
            if request_type == "step_env":
                self._disable_policy_for_training_if_needed()
                return self._handle_step_env_request(request, request_id)
            if request_type == "run_evaluation":
                self._disable_policy_for_training_if_needed()
                return self._handle_run_evaluation_request(request, request_id)
            if request_type == "start_recording":
                return self._handle_start_recording_request(request, request_id)
            if request_type == "stop_recording":
                return self._handle_stop_recording_request(request, request_id)
            if request_type == "request_debug_state":
                return self._build_debug_state_response(request_id)
            return {
                "type": "error",
                "request_id": request_id,
                "message": f"Unsupported training request type: {request_type}",
            }
        except Exception as error:  # noqa: BLE001
            log_supervisor(
                f"[Supervisor] error procesando request externa: {error}", force=True
            )
            return {
                "type": "error",
                "request_id": request_id,
                "message": str(error),
            }

    def _disable_policy_for_training_if_needed(self):
        if not self.policy_runner.enabled:
            return
        self.policy_runner.enabled = False
        self.policy_status = self.policy_runner.status_dict()
        self.last_action_label = "policy_paused_for_training"

    def _load_track_spawns(self):
        """
        Carga el mapa track->spawns (spawns.json en PROJECT_ROOT, o ruta en la env
        SPAWNS_JSON). Formato:
            {
              "tracks": [
                {
                  "texture": "track1.png",
                  "spawns": [
                    {"translation": [x, y, z], "rotation": [x, y, z, angle]},
                    ...
                  ]
                },
                ...
              ]
            }
        Devuelve la lista de tracks validos (cada uno con >=1 spawn), o None si no
        existe / es invalido / esta vacio. En ese caso el reset usa la textura y la
        pose por defecto del .wbt (comportamiento de un solo track).
        """
        path = os.environ.get("SPAWNS_JSON") or os.path.join(PROJECT_ROOT, "spawns.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            log_supervisor(
                f"[Supervisor] spawns.json no cargado ({exc}); track/pose por defecto.",
                force=True,
            )
            return None

        worlds_dir = os.path.join(PROJECT_ROOT, "worlds")
        tracks = data.get("tracks") if isinstance(data, dict) else None
        valid = []
        for track in tracks or []:
            texture = track.get("texture") if isinstance(track, dict) else None
            # Tracks marcados "eval": true son SOLO para evaluacion (rl/evaluate.py los
            # fuerza por su cuenta); se excluyen del pool de entrenamiento.
            if isinstance(track, dict) and track.get("eval"):
                log_supervisor(
                    f"[Supervisor] track '{texture}' es eval-only; excluido del training.",
                    force=True,
                )
                continue
            spawns = [s for s in track.get("spawns", []) if self._valid_spawn(s)]
            if not (isinstance(texture, str) and texture and spawns):
                continue
            # El PNG debe existir en worlds/ (si no, en headless entrenarias contra
            # una pista negra sin enterarte). Tracks sin textura presente se ignoran.
            if not os.path.exists(os.path.join(worlds_dir, texture)):
                log_supervisor(
                    f"[Supervisor] track '{texture}' no existe en worlds/; se ignora.",
                    force=True,
                )
                continue
            # Gates ordenados (opcional) -> loop para el reward de progreso. None si
            # no hay gates / son invalidos (ese track cae al reward de velocidad vision).
            loop = None
            gates = track.get("gates")
            if isinstance(gates, list) and len(gates) >= 2:
                try:
                    loop = build_loop([[float(g[0]), float(g[1])] for g in gates])
                except (TypeError, ValueError, IndexError):
                    log_supervisor(
                        f"[Supervisor] gates de '{texture}' invalidos; sin progreso.",
                        force=True,
                    )
            valid.append({"texture": texture, "spawns": spawns, "loop": loop})
        if not valid:
            log_supervisor(
                "[Supervisor] spawns.json sin tracks validos; track/pose por defecto.",
                force=True,
            )
            return None
        total_spawns = sum(len(t["spawns"]) for t in valid)
        log_supervisor(
            f"[Supervisor] spawns.json: {len(valid)} tracks, {total_spawns} spawns.",
            force=True,
        )
        return valid

    @staticmethod
    def _valid_spawn(spawn):
        """Un spawn valido = {translation:[x,y,z], rotation:[x,y,z,angle]}."""
        if not isinstance(spawn, dict):
            return False
        translation = spawn.get("translation")
        rotation = spawn.get("rotation")
        return (
            isinstance(translation, list) and len(translation) == 3
            and isinstance(rotation, list) and len(rotation) == 4
        )

    def _select_random_track_spawn(self):
        """
        Elige un track random y un spawn random de ese track, setea la textura del
        piso y devuelve (translation, rotation) del spawn. Si no hay spawns.json (o
        el .wbt no tiene DEF FLOOR_TEX) devuelve (None, None): el caller usa la pose
        inicial y no cambia la textura.
        """
        if not self.track_spawns:
            return None, None
        track = self.track_spawns[int(self.reset_rng.integers(len(self.track_spawns)))]
        spawn = track["spawns"][int(self.reset_rng.integers(len(track["spawns"])))]
        if self.floor_texture_url_field is not None:
            self.floor_texture_url_field.setMFString(0, track["texture"])
        self.current_track_texture = track["texture"]
        self.current_loop = track.get("loop")
        log_supervisor(f"[Supervisor] episodio {self.episode_id + 1} en track '{track['texture']}'")
        return list(spawn["translation"]), list(spawn["rotation"])

    def _init_progress(self):
        """Resetea el progreso del episodio; fija el s inicial desde la pose del spawn."""
        self.cumulative_progress = 0.0
        self.wrong_way_steps = 0
        if self.current_loop is not None:
            pos = self.epuck_robot.getPosition()
            self.progress_s = project_s(self.current_loop, (pos[0], pos[1]))
        else:
            self.progress_s = 0.0
        self._init_start_line()

    def _init_start_line(self):
        """
        Arma la linea de largada/meta unica desde la pose del spawn. Solo aplica cuando
        el track NO tiene gates y hay un spawn definido: la linea pasa por el punto del
        spawn, perpendicular a su rumbo, y el sentido valido de cruce es el del spawn.
        """
        self.start_line = None
        self.start_line_armed = False
        self.start_line_prev_dfwd = 0.0
        if self.current_loop is not None or self.current_spawn is None:
            return
        translation, rotation = self.current_spawn
        if translation is None or rotation is None or len(rotation) < 4:
            return
        # Rumbo del spawn: el robot mira a +x local; rotado theta sobre z -> (cos, sin).
        # rotation = [ax, ay, az, angle]; az=+1 gira +theta, az=-1 gira -theta.
        theta = float(rotation[3]) * (1.0 if float(rotation[2]) >= 0.0 else -1.0)
        forward = (math.cos(theta), math.sin(theta))
        lateral = (-math.sin(theta), math.cos(theta))
        p0 = (float(translation[0]), float(translation[1]))
        self.start_line = {"p0": p0, "forward": forward, "lateral": lateral}
        # d_forward inicial desde la pose ya asentada, para no contar un cruce espurio.
        pos = self.epuck_robot.getPosition()
        self.start_line_prev_dfwd = (
            (pos[0] - p0[0]) * forward[0] + (pos[1] - p0[1]) * forward[1]
        )

    def _update_start_line(self):
        """
        Deteccion de vuelta por cruce de la linea de meta unica (sin gates). Devuelve
        True el step en que el robot, ya alejado hacia adelante (armado), vuelve y cruza
        la linea cerca del punto del spawn y en el sentido del spawn (d_forward de - a +).
        """
        if self.start_line is None:
            return False
        line = self.start_line
        pos = self.epuck_robot.getPosition()
        rel = (pos[0] - line["p0"][0], pos[1] - line["p0"][1])
        d_fwd = rel[0] * line["forward"][0] + rel[1] * line["forward"][1]
        d_lat = rel[0] * line["lateral"][0] + rel[1] * line["lateral"][1]
        # Armar una vez que se alejo bien hacia adelante (deja atras la largada inicial).
        if not self.start_line_armed and d_fwd > START_LINE_ARM_DISTANCE:
            self.start_line_armed = True
        lap = False
        if (
            self.start_line_armed
            and abs(d_lat) < START_LINE_HALF_WIDTH
            and self.start_line_prev_dfwd < 0.0
            and d_fwd >= 0.0
        ):
            lap = True
            self.start_line_armed = False  # rearmar para una eventual proxima vuelta
        self.start_line_prev_dfwd = d_fwd
        return lap

    def _update_progress(self):
        """
        Avanza el progreso sobre el circuito. Devuelve (progress_delta, lap_done).
        progress_delta = None si el track no tiene gates (el reward cae a velocidad).
        """
        if self.current_loop is None:
            return None, False
        total = self.current_loop[2]
        pos = self.epuck_robot.getPosition()
        s_cur = project_s(self.current_loop, (pos[0], pos[1]))
        delta = signed_delta(self.progress_s, s_cur, total)
        self.progress_s = s_cur
        self.cumulative_progress += delta
        # Contramano: contar steps de retroceso (umbral para ignorar jitter chico).
        if delta < -0.25 * self._max_step_progress:
            self.wrong_way_steps += 1
        else:
            self.wrong_way_steps = 0
        lap_done = self.cumulative_progress >= total
        return delta, lap_done

    def _handle_reset_env_request(self, request, request_id):
        self._request_robot("reset_robot", "reset_done")
        # Nuevo episodio: track + spawn random (si hay spawns.json). Se setea la
        # textura ANTES del settle para que la imagen post-reset ya sea del track nuevo.
        translation, rotation = self._select_random_track_spawn()
        # Guardar el spawn para la linea de meta (fallback sin gates).
        self.current_spawn = (translation, rotation) if translation is not None else None
        self._reset_robot_pose(translation=translation, rotation=rotation)
        # Jitter de pose SOBRE el spawn elegido (si DR esta activo).
        self._apply_domain_randomization()
        # DR visual del fondo: color de pared aleatorio por episodio (si esta activo).
        self._randomize_background()
        self._advance_simulation(RESET_SETTLE_STEPS)
        self._refresh_robot_observation()
        self.episode_id += 1
        self.episode_step = 0
        # Nuevo episodio: limpiar estado de carril, progreso y el stack de la policy.
        self.lost_line_steps = 0
        self.offtrack_steps = 0
        self._init_progress()
        self.last_reward_breakdown = None
        self.policy_runner.reset_stack()
        return {
            "type": "env_reset",
            "request_id": request_id,
            "observation": self._build_nav_observation(),
            "info": {"track": self.current_track_texture},
        }

    def _lane_features(self):
        """
        Decodifica la ultima imagen de camara y corre el detector de carril.
        Devuelve el dict de features (ver helpers/lane_vision.detect_lane) o {} si
        no hay imagen valida / falla la deteccion (el caller usa un breakdown neutro).
        """
        try:
            rgb = decode_rgb_hwc(self.last_observation_image)
            if rgb is None:
                return {}
            return detect_lane(rgb)
        except Exception as exc:  # noqa: BLE001
            log_supervisor(f"[Supervisor] error en deteccion de carril: {exc}", force=True)
            return {}

    def _compute_reward_breakdown(self, action, velocity, progress_delta=None, lap_done=False):
        """
        Calcula el reward y su desglose. UNICA fuente de verdad (step de entrenamiento
        + panel de debug). Si el track tiene gates, el AVANCE lo da el PROGRESO sobre el
        circuito (Δarc-length) en vez de la velocidad cruda -> no premia loopear ni ir
        en contramano. clearance (calzada en el centro) gatea el avance; pasto en el
        centro = off-track (terminal); completar la vuelta = exito terminal con bonus.
        """
        features = self._lane_features()

        forward = float(velocity[0]) if velocity else 0.0
        speed_fwd = max(0.0, forward)  # solo cuenta avanzar, no retroceder

        clearance = float(features.get("center_clearance", 0.0))
        green_center = float(features.get("center_green", 0.0))
        white_center = float(features.get("center_white", 0.0))
        offset = features.get("offset")
        line_visible = bool(features.get("line_visible", False))
        has_features = bool(features)

        # --- Vuelta completa => exito terminal con bonus alto. ---
        if lap_done:
            return self._reward_dict(
                reward=float(LAP_BONUS), terminated=True, term_reason="lap_complete",
                clearance=clearance, offset=offset, speed=speed_fwd, progress=progress_delta,
                green_center=green_center, white_center=white_center, line_visible=line_visible,
            )

        # --- Off-track: pasto en el centro. Solo SEÑAL, no termina aca: la terminacion
        #     (con gracia de OFFTRACK_GRACE_STEPS) la decide el step handler, por si es una
        #     curva muy cerrada o un desvio del que se recupera. ---
        offtrack = bool(has_features and green_center >= OFFTRACK_GREEN_FRAC)

        # --- Avance: PROGRESO sobre el circuito (si hay gates) o velocidad vision. ---
        # Gateado por clearance: solo cobra fuerte avanzando Y sobre la calzada.
        if progress_delta is not None:
            progress_norm = max(-1.0, min(1.0, progress_delta / self._max_step_progress))
            r_forward = REWARD_PROGRESS_W * progress_norm * clearance
        else:
            r_forward = REWARD_SPEED_W * speed_fwd * clearance
        r_base = REWARD_BASE * clearance
        r_offset = -REWARD_OFFSET_W * abs(offset) if offset is not None else 0.0
        r_white = -REWARD_WHITE_W * white_center
        r_lost = LINE_LOST_PENALTY if (has_features and not line_visible) else 0.0
        r_step = -REWARD_STEP_COST
        # Sin features (imagen invalida) no premiamos ni penalizamos.
        if not has_features:
            r_base = r_forward = r_offset = r_white = r_lost = 0.0

        reward = r_base + r_forward + r_offset + r_white + r_lost + r_step

        return self._reward_dict(
            reward=float(reward), terminated=False, term_reason=None,
            clearance=clearance, offset=offset, speed=speed_fwd, progress=progress_delta,
            green_center=green_center, white_center=white_center, line_visible=line_visible,
            offtrack=offtrack,
            r_drive=r_base + r_forward, r_offset=r_offset, r_white=r_white,
            r_lost=r_lost, r_step=r_step,
        )

    @staticmethod
    def _reward_dict(reward, terminated, term_reason, clearance, offset,
                     green_center, white_center, line_visible, speed=0.0,
                     progress=None, offtrack=False, r_drive=0.0, r_offset=0.0, r_white=0.0,
                     r_lost=0.0, r_step=0.0):
        """Estructura uniforme del desglose de reward (misma forma siempre)."""
        return {
            "reward": round(float(reward), 4),
            "terminated": bool(terminated),
            "term_reason": term_reason,
            # Señal de off-track (pasto en el centro). La terminacion con gracia la maneja
            # el step handler; aca es solo la deteccion por-frame.
            "offtrack": bool(offtrack),
            # Features observadas
            "center_clearance": round(clearance, 4),
            "offset": round(offset, 4) if offset is not None else None,
            "speed": round(speed, 4),
            "progress": round(progress, 5) if progress is not None else None,
            "green_center": round(green_center, 4),
            "white_center": round(white_center, 4),
            "line_visible": bool(line_visible),
            # Terminos del reward
            "r_drive": round(r_drive, 4),
            "r_offset": round(r_offset, 4),
            "r_white": round(r_white, 4),
            "r_lost": round(r_lost, 4),
            "r_step": round(r_step, 4),
        }

    def _handle_step_env_request(self, request, request_id):
        action = request.get("action", [0.0, 0.0])
        self._request_robot("apply_action", "action_applied", label="train", action=action)
        self._advance_simulation(ACTION_REPEAT)
        self._refresh_robot_observation()
        self.episode_step += 1

        # Progreso sobre el circuito (si el track tiene gates). La obs del agente es
        # percepcion local; el progreso/contramano son info PRIVILEGIADA del supervisor
        # (no entra en la observacion) -> el reward queda identico al de la rama vision.
        progress_delta, lap_done = self._update_progress()
        # Sin gates: deteccion de vuelta por cruce de la linea de meta unica.
        if not lap_done and self.current_loop is None:
            lap_done = self._update_start_line()

        velocity = self._compute_velocity()
        breakdown = self._compute_reward_breakdown(
            action, velocity, progress_delta=progress_delta, lap_done=lap_done
        )
        self.last_reward_breakdown = breakdown

        # Carril perdido: contador para terminar si el robot se desorienta.
        if breakdown["line_visible"]:
            self.lost_line_steps = 0
        else:
            self.lost_line_steps += 1

        # Off-track con gracia: contar pasos CONSECUTIVOS sobre pasto. No se corta apenas se
        # detecta (curva cerrada / desvio momentaneo); recien tras OFFTRACK_GRACE_STEPS. Se
        # evalua ANTES que line_lost para que ese sea el motivo de corte cuando aplica.
        if breakdown.get("offtrack"):
            self.offtrack_steps += 1
        else:
            self.offtrack_steps = 0

        terminated = breakdown["terminated"]
        if not terminated and self.offtrack_steps >= OFFTRACK_GRACE_STEPS:
            terminated = True
            breakdown["term_reason"] = "offtrack_grass"
            breakdown["reward"] = round(float(OFFTRACK_PENALTY), 4)
        if not terminated and self.lost_line_steps >= LOST_LINE_MAX_STEPS:
            terminated = True
            breakdown["term_reason"] = "line_lost"
        # Contramano sostenido => terminar (el reset lo deja encarado bien de nuevo).
        if not terminated and self.wrong_way_steps >= WRONG_WAY_MAX_STEPS:
            terminated = True
            breakdown["term_reason"] = "wrong_way"
            breakdown["reward"] = round(float(WRONG_WAY_PENALTY), 4)
        truncated = (not terminated) and self.episode_step >= self.episode_max_steps

        return {
            "type": "env_step",
            "request_id": request_id,
            "observation": self._build_nav_observation(),
            "reward": breakdown["reward"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": {
                "reward_breakdown": breakdown,
                "episode_step": self.episode_step,
                "lost_line_steps": self.lost_line_steps,
                "offtrack_steps": self.offtrack_steps,
                "track": self.current_track_texture,
                "cumulative_progress": round(self.cumulative_progress, 4),
            },
        }

    def _refresh_robot_observation(self):
        """
        Pide al robot su observacion sensorial actual (imagen de camara) y la guarda.
        Se llama cuando la simulacion ya esta en el estado que queremos observar
        (despues de aplicar la accion / settle de reset), asi la imagen y el reward
        derivado de ella corresponden al mismo instante.
        """
        response = self._request_robot("request_observation", "observation")
        observation = response.get("observation", {})
        image = observation.get("image")
        if isinstance(image, dict) and isinstance(
            image.get("data_bytes"), (bytes, bytearray)
        ):
            self.last_observation_image = image
        return self.last_observation_image

    def _build_nav_observation(self) -> dict:
        # Rama geometrica: la obs es un VECTOR de features (sin imagen).
        return {"geometry": self._compute_geometry_obs()}

    def _build_inference_observation(self) -> dict:
        """
        Observacion SUPERSET para inferencia en vivo (robot window / policy step). Trae las
        tres representaciones desde los sensores crudos, para que el policy_runner pueda
        correr CUALQUIER modelo (geometrico o de vision) desde esta rama: el runner elige el
        subconjunto segun el observation_space del modelo. La geometria se reconstruye con la
        MISMA logica del training (helpers.geom_obs.geom_vector_from_rgb).
        """
        velocity = self._compute_velocity()
        rgb = decode_rgb_hwc(self.last_observation_image)
        return {
            "velocity": velocity,
            "image": self.last_observation_image or blank_image_payload(),
            "geometry": geom_vector_from_rgb(rgb, velocity),
        }

    def _perception_features(self):
        """
        Features LOCALES de la imagen de camara de este timestep (helpers/lane_vision).
        Devuelve el dict de detect_lane + 'band_offsets' (look-ahead en la imagen), o None
        si no hay frame valido. NO usa geometria global del track -> track-agnostico.
        """
        try:
            rgb = decode_rgb_hwc(self.last_observation_image)
            if rgb is None:
                return None
            feats = detect_lane(rgb)
            feats["band_offsets"] = road_band_offsets(rgb, GEOM_BANDS)
            return feats
        except Exception as exc:  # noqa: BLE001
            log_supervisor(f"[Supervisor] error en features de percepcion: {exc}", force=True)
            return None

    def _compute_geometry_obs(self):
        """Vector de observacion = metricas derivadas de la imagen (ver helpers/geom_obs)."""
        vel = self._compute_velocity()  # [forward, yaw_rate], ya ~[-1, 1] (proprioceptivo)
        feats = self._perception_features()
        if feats is None:
            return blank_geom()
        b = GEOM_BOUND
        clip = lambda v: max(-b, min(b, float(v)))
        obs = [
            clip(vel[0]), clip(vel[1]),
            clip(feats.get("road_frac", 0.0)),
            clip(feats.get("center_green", 0.0)),
        ]
        for off in feats.get("band_offsets", []):
            obs.append(clip(off))
        # Asegurar el tamanio exacto (por si band_offsets viniera corto).
        target = 4 + GEOM_BANDS
        obs += [0.0] * (target - len(obs))
        return obs[:target]

    

    def _detect_collision(self) -> bool:
        """
        Deteccion de colision.
        """
        return False    

    def _compute_velocity(self) -> list:
        """
        Velocidad del cuerpo del robot en FRAME LOCAL, normalizada a ~[-1, 1]:
            [forward, yaw_rate]
        - forward : velocidad lineal a lo largo de la nariz (+x local). ~0 si esta
                    trabado aunque comande full -> senal clave de "stuck".
        - yaw_rate: velocidad angular alrededor del eje vertical. Distingue
                    "trabado empujando" (yaw~0) de "girando para escapar" (yaw!=0).

        Se computa con getVelocity() (ground-truth del sim = propriocepcion exacta;
        un robot real la estima con odometria + IMU, misma magnitud).
        """
        velocity = self.epuck_robot.getVelocity()  # [vx,vy,vz, wx,wy,wz] en mundo
        vx, vy = velocity[0], velocity[1]
        wx, wy, wz = velocity[3], velocity[4], velocity[5]
        rot = self.epuck_robot.getOrientation()  # row-major; columnas = ejes locales

        # Proyectar la velocidad lineal sobre el eje adelante (+x local) y la
        # angular sobre el eje vertical (+z local) del robot.
        forward  = vx * rot[0] + vy * rot[3]
        yaw_rate = wx * rot[2] + wy * rot[5] + wz * rot[8]

        forward_norm = forward / MAX_LINEAR_SPEED if MAX_LINEAR_SPEED > 0 else 0.0
        yaw_norm     = yaw_rate / MAX_ANGULAR_SPEED if MAX_ANGULAR_SPEED > 0 else 0.0

        forward_norm = max(-1.0, min(1.0, forward_norm))
        yaw_norm     = max(-1.0, min(1.0, yaw_norm))
        return [float(forward_norm), float(yaw_norm)]

    def _handle_run_evaluation_request(self, request, request_id):
        # TODO: implementar evaluacion de politica completa
        return {
            "type": "error",
            "request_id": request_id,
            "message": "run_evaluation no implementado aun",
        }

    def _handle_start_recording_request(self, request, request_id):
        # TODO: implementar grabacion de episodios
        return {
            "type": "error",
            "request_id": request_id,
            "message": "start_recording no implementado aun",
        }

    def _handle_stop_recording_request(self, request, request_id):
        # TODO: implementar grabacion de episodios
        return {
            "type": "error",
            "request_id": request_id,
            "message": "stop_recording no implementado aun",
        }

    def _handle_save_camera_sample(self):
        """
        Guarda una captura manual de la camara del robot. Stub: no implementado.
        El caller (_handle_ui_messages) captura RuntimeError y lo reporta a la UI.
        """
        raise RuntimeError("save_camera_sample no implementado aun")

    def _build_debug_state_response(self, request_id):
        """
        Construye una respuesta de estado de depuración.
        Sirve para obtener el estado actual de la simulación y del robot y enviarlo a la UI.
        """
        return {
            "type": "debug_state",
            "request_id": request_id,
            "episode_id": self.episode_id,
            "episode_step": self.episode_step,
            "step_count": self.step_count,
            "training_client_connected": self.training_server.is_client_connected(),
            "pose": self._current_pose(),
            "last_robot_message": self.last_robot_message,
        }


    def _normalize_action(self, action):
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != 2:
            raise RuntimeError(
                f"Accion invalida: se esperaban 2 valores y llegaron {action}"
            )
        return np.clip(action_array, -1.0, 1.0)



    def _handle_unmatched_robot_messages(self):
        """
        Maneja mensajes del robot que no coinciden con ningún handler.
        """
        while True:
            message = self.bridge.pop_next_message()
            if message is None:
                break
            self._record_robot_message(message)

    def _current_pose(self):
        """
        Obtiene la pose actual del robot.
        """
        return {
            "translation": list(self.epuck_translation_field.getSFVec3f()),
            "rotation": list(self.epuck_rotation_field.getSFRotation()),
        }

    
    def run(self):
        while self._step_world(process_passive=True):
            pass
            
            


controller = SupervisorController()
controller.run()