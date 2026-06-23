import struct
import gymnasium as gym
import numpy as np
from helpers.read_env_value import read_env_value
from helpers.supervisor_socket_bridge import SupervisorSocketBridge
from helpers.geom_obs import build_geom_space, GEOM_OBS_SIZE, GEOM_BOUND
#
DEFAULT_HOST = read_env_value("DEFAULT_HOST", "127.0.0.1", str)
DEFAULT_PORT = read_env_value("DEFAULT_PORT", 10001, int)
PACKET_HEADER_FORMAT = read_env_value("SUPERVISOR_PACKET_HEADER_FORMAT", ">II", str)
PACKET_HEADER_SIZE = struct.calcsize(PACKET_HEADER_FORMAT)
MAX_SPEED = read_env_value("MAX_SPEED", 6.28, float)
MIN_SPEED = read_env_value("MIN_SPEED", -6.28, float)

class NavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        bridge=None,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
    ):
        self.bridge = bridge or SupervisorSocketBridge(host=host, port=port)
        self.wheel_count = 2
        self.velocity_size = 2  # [forward, yaw_rate] del cuerpo en frame local
        # La accion del gym es NORMALIZADA en [-1, 1] (fraccion de la velocidad
        # maxima de la rueda). El escalado a rad/s reales lo hace el robot_controller
        # con su propia maxVelocity y el margen MOTOR_SPEED_MARGIN. Mantener la accion
        # en [-1, 1] hace que la policy gaussiana (sigma ~1) explore todo el rango.
        self.action_low = -1.0
        self.action_high = 1.0
        self.default_reset_options = {}

        # Espacio de observación: VECTOR GEOMETRICO (sin imagen). El supervisor lo
        # calcula desde la centerline de gates + la pose: velocidad, error lateral,
        # error de rumbo y look-ahead local. Full observable -> no se pierde el sentido.
        self.observation_space = build_geom_space()
        # Espacio de acción.
        # Vector de 2 elementos (rueda izq/der) NORMALIZADO en [-1.0, 1.0].
        # -1 = full reversa, +1 = full adelante; el robot lo escala a rad/s.
        self.action_space = gym.spaces.Box(
            low=self.action_low,
            high=self.action_high,
            shape=(self.wheel_count,),
            dtype=np.float32,
        )

    def _decode_observation(self, observation_payload):
        if not isinstance(observation_payload, dict):
            raise RuntimeError(
                "La observacion del supervisor no tiene el formato esperado."
            )

        geometry = observation_payload.get("geometry")
        observation = np.asarray(geometry, dtype=np.float32)
        if observation.shape != (GEOM_OBS_SIZE,):
            raise RuntimeError(f"geometry invalido: {geometry}")
        # Clip defensivo a la cota del Box (el supervisor ya clampea, pero por las dudas).
        observation = np.clip(observation, -GEOM_BOUND, GEOM_BOUND).astype(np.float32)
        self._validate_observation(observation)
        return observation

    

    def _validate_observation(self, observation):
        if not self.observation_space.contains(observation):
            raise RuntimeError(
                "La observacion recibida no coincide con observation_space."
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        effective_options = dict(self.default_reset_options)
        if options:
            effective_options.update(options)

        response = self.bridge.request(
            {
                "type": "reset_env",
                "seed": seed,
                "options": effective_options,
            }
        )

        observation = self._decode_observation(response["observation"])
        info = response.get("info", {})
        return observation, info

    def set_default_reset_options(self, options=None):
        self.default_reset_options = dict(options or {})
        return dict(self.default_reset_options)

    def get_default_reset_options(self):
        return dict(self.default_reset_options)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self.wheel_count,):
            raise RuntimeError(f"Accion invalida: {action}")
        # Clipping de la acción al rango normalizado [-1, 1].
        action = np.clip(action, self.action_low, self.action_high)

        response = self.bridge.request(
            {
                "type": "step_env",
                "action": action.tolist(),
            }
        )

        observation = self._decode_observation(response["observation"])
        reward = float(response["reward"])
        terminated = bool(response["terminated"])
        truncated = bool(response["truncated"])
        info = response.get("info", {})

        return observation, reward, terminated, truncated, info

    def close(self):
        close_method = getattr(self.bridge, "close", None)
        if callable(close_method):
            close_method()

    def run_policy_evaluation(
        self,
        model_path,
        deterministic=True,
        max_steps=300,
        video_path=None,
        timeout=None,
        seed=None,
        map_name=None,
    ):
        request_payload = {
            "type": "run_evaluation",
            "model_path": model_path,
            "deterministic": bool(deterministic),
            "max_steps": int(max_steps),
            "video_path": video_path,
        }
        if seed is not None:
            request_payload["seed"] = int(seed)
        if map_name:
            request_payload["map_name"] = str(map_name).strip()

        response = self.bridge.request(request_payload, timeout=timeout)
        return response
