import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import struct
import numpy as np
from controller import Robot
from helpers.read_env_value import read_env_value

HEADER_FMT = read_env_value("ROBOT_PACKET_HEADER_FORMAT", "!I", str)
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Mapeo de accion por rueda: [-1, 1] -> [WHEEL_MIN_SPEED, WHEEL_MAX_SPEED] rad/s.
# Ambas POSITIVAS: el robot NUNCA retrocede ni frena del todo (estilo DeepRacer).
#   accion -1 -> WHEEL_MIN_SPEED (velocidad minima),  +1 -> WHEEL_MAX_SPEED.
# WHEEL_MAX_SPEED queda clampeado por la maxVelocity del motor en el .wbt (6.28).
WHEEL_MIN_SPEED = read_env_value("WHEEL_MIN_SPEED", 0.75, float)  # rad/s
WHEEL_MAX_SPEED = read_env_value("WHEEL_MAX_SPEED", 5.0, float)   # rad/s


class EpuckController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.left_encoder = self.robot.getDevice("left wheel sensor")
        self.right_encoder = self.robot.getDevice("right wheel sensor")
        self.left_encoder.enable(self.timestep)
        self.right_encoder.enable(self.timestep)

        self.receiver = self.robot.getDevice("receiver")
        self.emitter = self.robot.getDevice("emitter")
        self.receiver.enable(self.timestep)

    # ------------------------------------------------------------------
    # Comunicacion con el supervisor — binary framing: [4-byte size][JSON]
    # ------------------------------------------------------------------

    def _send(self, data: dict, binary: bytes = b""):
        header_bytes = json.dumps(data).encode("utf-8")
        packet = struct.pack(HEADER_FMT, len(header_bytes)) + header_bytes + binary
        self.emitter.send(packet)

    def _recv(self) -> dict | None:
        if self.receiver.getQueueLength() == 0:
            return None
        raw = bytes(self.receiver.getBytes())
        self.receiver.nextPacket()
        if len(raw) < HEADER_SIZE:
            return None
        (header_size,) = struct.unpack(HEADER_FMT, raw[:HEADER_SIZE])
        return json.loads(raw[HEADER_SIZE : HEADER_SIZE + header_size].decode("utf-8"))

    # ------------------------------------------------------------------
    # Accion y observacion (placeholders)
    # ------------------------------------------------------------------

    def _apply_action(self, action: list):
        # La accion llega NORMALIZADA en [-1, 1] por rueda. Se remapea a
        # [WHEEL_MIN_SPEED, WHEEL_MAX_SPEED] (ambas positivas): -1 -> minima,
        # +1 -> maxima. Asi el robot nunca retrocede ni frena del todo.
        left_n  = float(np.clip(action[0], -1.0, 1.0))
        right_n = float(np.clip(action[1], -1.0, 1.0))
        span = WHEEL_MAX_SPEED - WHEEL_MIN_SPEED
        left  = WHEEL_MIN_SPEED + (left_n  + 1.0) * 0.5 * span
        right = WHEEL_MIN_SPEED + (right_n + 1.0) * 0.5 * span
        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    def _stop_motors(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def _read_wheel_state(self) -> list:
        # TODO: leer velocidades de rueda y normalizar
        return [0.0, 0.0]

    def _camera_image_payload(self):
        """
        Lee el frame actual de la camara y lo devuelve como (header_dict, bytes_rgb).

        Webots entrega la imagen en BGRA (4 bytes/pixel, row-major HWC). La pasamos
        a RGB compacto (3 bytes/pixel) para mandarla en la seccion binaria del packet.
        """
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        raw = self.camera.getImage()
        if not raw:
            return None, b""

        bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
        # Canales 2,1,0 de BGRA -> R,G,B (descartando alpha).
        rgb = np.ascontiguousarray(bgra[:, :, 2::-1])
        header = {
            "width": width,
            "height": height,
            "channels": 3,
            "encoding": "rgb",
            "layout": "hwc",
        }
        return header, rgb.tobytes()

    # ------------------------------------------------------------------
    # Handlers de mensajes del supervisor
    # ------------------------------------------------------------------

    def _handle_act(self, msg: dict):
        self._apply_action(msg["action"])
        self._send({"type": "obs", "wheel_state": self._read_wheel_state()})

    def _handle_apply_action(self, msg: dict):
        self._apply_action(msg["action"])
        self._send({
            "type": "action_applied",
            "request_id": msg.get("request_id"),
            "wheel_state": self._read_wheel_state(),
        })

    def _handle_reset_robot(self, msg: dict):
        self._stop_motors()
        self._send({
            "type": "reset_done",
            "request_id": msg.get("request_id"),
        })

    def _handle_request_observation(self, msg: dict):
        """
        Devuelve la observacion sensorial actual del robot: imagen de la camara
        (en la seccion binaria) + estado de ruedas.
        """
        header, binary = self._camera_image_payload()
        observation = {"wheel_state": self._read_wheel_state()}
        if header is not None:
            observation["image"] = header
        self._send(
            {
                "type": "observation",
                "request_id": msg.get("request_id"),
                "observation": observation,
            },
            binary=binary,
        )

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def run(self):
        handlers = {
            "act":                 self._handle_act,
            "apply_action":        self._handle_apply_action,
            "reset_robot":         self._handle_reset_robot,
            "request_observation": self._handle_request_observation,
        }
        while self.robot.step(self.timestep) != -1:
            msg = self._recv()
            if msg is None:
                continue
            handler = handlers.get(msg.get("type"))
            if handler:
                handler(msg)


EpuckController().run()
