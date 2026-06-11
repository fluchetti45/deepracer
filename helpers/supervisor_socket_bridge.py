from helpers.read_env_value import read_env_value
import socket
import struct
import json
import time

DEFAULT_HOST = read_env_value("DEFAULT_HOST", "127.0.0.1", str)
DEFAULT_PORT = read_env_value("DEFAULT_PORT", 10001, int)
DEFAULT_TIMEOUT = read_env_value("DEFAULT_TIMEOUT", 30.0, float)
PACKET_HEADER_FORMAT = read_env_value("SUPERVISOR_PACKET_HEADER_FORMAT", ">II", str)
PACKET_HEADER_SIZE = struct.calcsize(PACKET_HEADER_FORMAT)



class SupervisorSocketBridge:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=DEFAULT_TIMEOUT):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.request_counter = 0

    def connect(self, max_wait: float = 60.0, retry_interval: float = 1.0):
        if self.socket is not None:
            return

        deadline = time.monotonic() + max_wait
        attempt  = 0
        while True:
            try:
                self.socket = socket.create_connection(
                    (self.host, self.port), timeout=2.0
                )
                if attempt > 0:
                    print()
                print(f"[Bridge] Conectado al supervisor en {self.host}:{self.port}")
                return
            except (ConnectionRefusedError, OSError):
                if time.monotonic() >= deadline:
                    raise ConnectionRefusedError(
                        f"No se pudo conectar al supervisor en {self.host}:{self.port} "
                        f"despues de {max_wait:.0f} seg. "
                        f"Asegurate de que Webots este corriendo con el world file."
                    )
                if attempt == 0:
                    print(f"[Bridge] Esperando al supervisor en {self.host}:{self.port} ", end="", flush=True)
                else:
                    print(".", end="", flush=True)
                attempt += 1
                time.sleep(retry_interval)

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _send_packet(self, payload, binary_payload=b""):
        header_bytes = json.dumps(payload).encode("utf-8")
        packet = (
            struct.pack(PACKET_HEADER_FORMAT, len(header_bytes), len(binary_payload))
            + header_bytes
            + binary_payload
        )
        self.socket.sendall(packet)

    def _recv_exact(self, size):
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self.socket.recv(size - len(chunks))
            if not chunk:
                raise RuntimeError(
                    "El supervisor cerro la conexion mientras esperaba una respuesta."
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def _receive_packet(self):
        header_prefix = self._recv_exact(PACKET_HEADER_SIZE)
        header_size, binary_size = struct.unpack(PACKET_HEADER_FORMAT, header_prefix)
        header_bytes = self._recv_exact(header_size)
        binary_payload = self._recv_exact(binary_size) if binary_size else b""
        response = json.loads(header_bytes.decode("utf-8"))

        if binary_payload:
            observation = response.get("observation")
            if isinstance(observation, dict):
                image = observation.get("image")
                if isinstance(image, dict):
                    response = dict(response)
                    observation = dict(observation)
                    image = dict(image)
                    image["data_bytes"] = binary_payload
                    observation["image"] = image
                    response["observation"] = observation

        return response

    def request(self, payload, timeout=None):
        self.connect()
        effective_timeout = self.timeout if timeout is None else float(timeout)
        self.socket.settimeout(effective_timeout)
        try:
            self.request_counter += 1
            request_id = payload.get("request_id") or f"env-{self.request_counter}"
            request_payload = dict(payload)
            request_payload["request_id"] = request_id

            self._send_packet(request_payload)
            response = self._receive_packet()
            if response.get("request_id") != request_id:
                raise RuntimeError(
                    f"Respuesta desalineada del supervisor: request_id={response.get('request_id')}"
                )
            if response.get("type") == "error":
                raise RuntimeError(
                    response.get("message", "Error desconocido del supervisor")
                )

            return response
        finally:
            self.socket.settimeout(self.timeout)