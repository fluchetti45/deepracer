import struct
import gymnasium as gym
import numpy as np
from helpers.read_env_value import read_env_value
from helpers.supervisor_socket_bridge import SupervisorSocketBridge
from helpers.image_obs import build_image_space, decode_image_observation
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
        # Accion estilo DeepRacer: [steering, speed] (2 valores). El controller mapea
        # steering -> angulo de direccion y speed -> velocidad de ruedas traseras.
        self.wheel_count = 2
        self.velocity_size = 2  # [forward, yaw_rate] del cuerpo en frame local
        # La accion del gym es NORMALIZADA en [-1, 1]. El escalado real (angulo en rad,
        # velocidad en rad/s) lo hace el agent_controller. Mantener la accion en [-1, 1]
        # hace que la policy gaussiana (sigma ~1) explore todo el rango.
        self.action_low = -1.0
        self.action_high = 1.0
        self.default_reset_options = {}

        # Espacio de observación.
        # La idea es que el agente aprenda a navegar hacia la meta.
        # Solo necesia información sobre el estado de las ruedas y la dirección hacia la meta (respecto a si mismo).
        observation_spaces = {
            # Velocidad del cuerpo en frame local, normalizada: [forward, yaw_rate].
            # forward~0 con accion grande => trabado; yaw distingue girar de empujar.
            "velocity": gym.spaces.Box(
                low=np.full(self.velocity_size, -1.0, dtype=np.float32),
                high=np.full(self.velocity_size, 1.0, dtype=np.float32),
                shape=(self.velocity_size,),
                dtype=np.float32,
            ),
            # Imagen de la camara frontal (RGB, channel-first) — Level 3+.
            "image": build_image_space(),
        }
        self.observation_space = gym.spaces.Dict(observation_spaces)
        # Espacio de acción.
        # Vector de 2 elementos [steering, speed] NORMALIZADO en [-1.0, 1.0].
        # steering: -1 derecha / +1 izquierda (angulo). speed: -1 minima / +1 maxima
        # (siempre positiva). El agent_controller lo escala a rad y rad/s.
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

        velocity_payload = observation_payload.get("velocity", [0.0] * self.velocity_size)

        velocity = np.asarray(velocity_payload, dtype=np.float32)
        if velocity.shape != (self.velocity_size,):
            raise RuntimeError(f"velocity invalido: {velocity_payload}")

        image = decode_image_observation(observation_payload.get("image"))

        observation = {
            "velocity": velocity,
            "image": image,
        }
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
