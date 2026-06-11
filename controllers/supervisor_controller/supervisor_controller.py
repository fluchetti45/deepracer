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
from helpers.lane_vision import decode_rgb_hwc, detect_lane


DOMAIN_RANDOMIZATION_ENABLED = True
DOMAIN_RANDOMIZATION_PROBABILITY = 0.5
DOMAIN_RANDOMIZATION_MAX_ROTATION=math.radians(30)  # perturbacion de hasta +/-30 grados
DOMAIN_RANDOMIZATION_MAX_TRANSLATION=0.05  # perturbacion de hasta +/-5 cm

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
# Velocidades maximas (para normalizar la observacion de velocidad a ~[-1, 1]).
MAX_LINEAR_SPEED = read_env_value("MAX_LINEAR_SPEED", 1.0)
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
# Centrado: premia que la banda central de la vista sea asfalto limpio (~1 centrado).
REWARD_CENTER_W = read_env_value("REWARD_CENTER_W", 1.0)
# Direccion: penaliza el offset lateral firmado (rompe la simetria del centrado).
REWARD_OFFSET_W = read_env_value("REWARD_OFFSET_W", 0.3)
# Borde: penaliza tener blanco (linea externa) en la banda central.
REWARD_WHITE_W = read_env_value("REWARD_WHITE_W", 0.5)
# Velocidad: premia avanzar, pero SOLO si va centrado (speed * clearance).
REWARD_SPEED_W = read_env_value("REWARD_SPEED_W", 0.3)
# Costo por step: empuja a no quedarse quieto.
REWARD_STEP_COST = read_env_value("REWARD_STEP_COST", 0.01)
# Off-track: fraccion de pasto en la banda central para declarar que se fue.
OFFTRACK_GREEN_FRAC = read_env_value("OFFTRACK_GREEN_FRAC", 0.4)
# Reward terminal al salirse de la pista.
OFFTRACK_PENALTY = read_env_value("OFFTRACK_PENALTY", -1.0)
# Penalizacion por step sin ver ninguna marca (carril perdido).
LINE_LOST_PENALTY = read_env_value("LINE_LOST_PENALTY", -0.3)
# Steps consecutivos sin ver el carril antes de terminar el episodio.
LOST_LINE_MAX_STEPS = read_env_value("LOST_LINE_MAX_STEPS", 20, int)


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
        self.policy_runner = PolicyRunner()
        self.policy_status = self.policy_runner.status_dict()
        self.policy_debug_snapshot = None
        self.reset_rng = np.random.default_rng()
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
        # Aplicar la nueva pose
        self.epuck_translation_field.setSFVec3f(perturbed_translation)
        self.epuck_rotation_field.setSFRotation(perturbed_rotation)

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
                "line_visible": features.get("line_visible", False),
            },
            "reward_breakdown": self.last_reward_breakdown,
            "robot_pos": [round(robot_pos[0], 4), round(robot_pos[1], 4)],
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
        obs  = self._build_nav_observation()
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
            obs = self._build_nav_observation()
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

    def _handle_reset_env_request(self, request, request_id):
        self._request_robot("reset_robot", "reset_done")
        self._reset_robot_pose()
        self._advance_simulation(RESET_SETTLE_STEPS)
        self._refresh_robot_observation()
        self.episode_id += 1
        self.episode_step = 0
        # Nuevo episodio: limpiar estado de carril y el stack de frames de la policy.
        self.lost_line_steps = 0
        self.last_reward_breakdown = None
        self.policy_runner.reset_stack()
        return {
            "type": "env_reset",
            "request_id": request_id,
            "observation": self._build_nav_observation(),
            "info": {},
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

    def _compute_reward_breakdown(self, action, velocity):
        """
        Calcula el reward y su desglose por componente desde la imagen de la camara
        frontal (vision-pura, opcion A). UNICA fuente de verdad del reward: la usan
        tanto el step de entrenamiento (_handle_step_env_request) como el panel de
        debug de la Robot Window, asi la window muestra EXACTAMENTE el mismo reward
        que ve el agente.

        El agente navega su carril (corredor entre la linea amarilla del centro y la
        blanca del borde). Mantener la banda central de la vista como asfalto limpio
        => va centrado. Pasto en el centro => se fue de la pista (termina).
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

        # --- Terminacion: el frente del robot esta sobre pasto => fuera de pista. ---
        terminated = has_features and green_center >= OFFTRACK_GREEN_FRAC
        if terminated:
            return self._reward_dict(
                reward=float(OFFTRACK_PENALTY),
                terminated=True,
                term_reason="offtrack_grass",
                clearance=clearance,
                offset=offset,
                green_center=green_center,
                white_center=white_center,
                line_visible=line_visible,
            )

        # --- Reward denso por componentes. ---
        r_center = REWARD_CENTER_W * clearance
        r_offset = -REWARD_OFFSET_W * abs(offset) if offset is not None else 0.0
        r_white = -REWARD_WHITE_W * white_center
        r_speed = REWARD_SPEED_W * speed_fwd * clearance
        r_lost = LINE_LOST_PENALTY if (has_features and not line_visible) else 0.0
        r_step = -REWARD_STEP_COST
        # Sin features (imagen invalida) no premiamos ni penalizamos el centrado.
        if not has_features:
            r_center = r_offset = r_white = r_speed = r_lost = 0.0

        reward = r_center + r_offset + r_white + r_speed + r_lost + r_step

        return self._reward_dict(
            reward=float(reward),
            terminated=False,
            term_reason=None,
            clearance=clearance,
            offset=offset,
            green_center=green_center,
            white_center=white_center,
            line_visible=line_visible,
            r_center=r_center,
            r_offset=r_offset,
            r_white=r_white,
            r_speed=r_speed,
            r_lost=r_lost,
            r_step=r_step,
        )

    @staticmethod
    def _reward_dict(reward, terminated, term_reason, clearance, offset,
                     green_center, white_center, line_visible,
                     r_center=0.0, r_offset=0.0, r_white=0.0,
                     r_speed=0.0, r_lost=0.0, r_step=0.0):
        """Estructura uniforme del desglose de reward (misma forma siempre)."""
        return {
            "reward": round(float(reward), 4),
            "terminated": bool(terminated),
            "term_reason": term_reason,
            # Features observadas
            "center_clearance": round(clearance, 4),
            "offset": round(offset, 4) if offset is not None else None,
            "green_center": round(green_center, 4),
            "white_center": round(white_center, 4),
            "line_visible": bool(line_visible),
            # Terminos del reward
            "r_center": round(r_center, 4),
            "r_offset": round(r_offset, 4),
            "r_white": round(r_white, 4),
            "r_speed": round(r_speed, 4),
            "r_lost": round(r_lost, 4),
            "r_step": round(r_step, 4),
        }

    def _handle_step_env_request(self, request, request_id):
        action = request.get("action", [0.0, 0.0])
        self._request_robot("apply_action", "action_applied", label="train", action=action)
        self._advance_simulation(ACTION_REPEAT)
        self._refresh_robot_observation()
        self.episode_step += 1

        velocity = self._compute_velocity()
        breakdown = self._compute_reward_breakdown(action, velocity)
        self.last_reward_breakdown = breakdown

        # Carril perdido: contador para terminar si el robot se desorienta.
        if breakdown["line_visible"]:
            self.lost_line_steps = 0
        else:
            self.lost_line_steps += 1

        terminated = breakdown["terminated"]
        if not terminated and self.lost_line_steps >= LOST_LINE_MAX_STEPS:
            terminated = True
            breakdown["term_reason"] = "line_lost"
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
        # Vision-pura: la obs es {image, velocity}.
        observation = {
            "velocity": self._compute_velocity(),
            "image": self.last_observation_image or blank_image_payload(),
        }
        return observation

    

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