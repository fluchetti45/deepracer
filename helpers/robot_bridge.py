import json
import struct
from helpers.read_env_value import read_env_value

ROBOT_PACKET_HEADER_FORMAT = read_env_value("ROBOT_PACKET_HEADER_FORMAT", "!I", str)
ROBOT_PACKET_HEADER_SIZE = struct.calcsize(ROBOT_PACKET_HEADER_FORMAT)

SUPERVISOR_VERBOSE_MESSAGES = False


def log_supervisor(message, force=False):
    if force or SUPERVISOR_VERBOSE_MESSAGES:
        print(message)


def summarize_robot_message(message: dict) -> dict:
    summary = dict(message)
    camera = summary.get("camera")
    if isinstance(camera, dict):
        summary["camera"] = f"<image {camera.get('width')}x{camera.get('height')}>"
    observation = summary.get("observation")
    if isinstance(observation, dict):
        obs_copy = dict(observation)
        if isinstance(obs_copy.get("image"), dict):
            img = obs_copy["image"]
            obs_copy["image"] = f"<image {img.get('width')}x{img.get('height')}>"
        summary["observation"] = obs_copy
    return summary


class RobotBridge:
    def __init__(self, supervisor, emitter, receiver, timestep):
        self.supervisor = supervisor
        self.emitter = emitter
        self.receiver = receiver
        self.timestep = timestep
        self.receiver.enable(self.timestep)
        self.pending_messages = []
        log_supervisor(
            f"[Supervisor] bridge listo: emitter={self.emitter is not None}, "
            f"receiver={self.receiver is not None}, timestep={self.timestep}",
            force=True,
        )

    def send_to_robot(self, payload, binary_payload=b""):
        header_bytes = json.dumps(payload).encode("utf-8")
        packet = (
            struct.pack(ROBOT_PACKET_HEADER_FORMAT, len(header_bytes))
            + header_bytes
            + binary_payload
        )
        self.emitter.send(packet)
        log_supervisor(f"[Supervisor] enviado al robot: {payload}")

    def drain_robot_messages(self):
        while self.receiver.getQueueLength() > 0:
            raw = bytes(self.receiver.getBytes())
            self.receiver.nextPacket()
            if len(raw) < ROBOT_PACKET_HEADER_SIZE:
                log_supervisor("[Supervisor] packet del robot truncado, descartado.")
                continue
            (header_size,) = struct.unpack(
                ROBOT_PACKET_HEADER_FORMAT, raw[:ROBOT_PACKET_HEADER_SIZE]
            )
            header_end = ROBOT_PACKET_HEADER_SIZE + header_size
            header_bytes = raw[ROBOT_PACKET_HEADER_SIZE:header_end]
            binary_payload = bytes(raw[header_end:])
            message = json.loads(header_bytes.decode("utf-8"))
            if binary_payload:
                observation = message.get("observation")
                if isinstance(observation, dict):
                    image = observation.get("image")
                    if isinstance(image, dict):
                        image["data_bytes"] = binary_payload
            self.pending_messages.append(message)
            log_supervisor(
                f"[Supervisor] recibido del robot: {summarize_robot_message(message)}"
            )

    def pop_message(self, expected_type=None, request_id=None):
        for index, message in enumerate(self.pending_messages):
            if request_id is not None and message.get("request_id") != request_id:
                continue
            if expected_type is not None and message.get("type") != expected_type:
                continue
            return self.pending_messages.pop(index)
        return None

    def pop_error(self, request_id=None):
        return self.pop_message(expected_type="error", request_id=request_id)

    def pop_next_message(self):
        if not self.pending_messages:
            return None
        return self.pending_messages.pop(0)
