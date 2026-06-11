import socket
import struct
import json
import base64
from helpers.read_env_value import read_env_value 
SUPERVISOR_VERBOSE_MESSAGES = False
def log_supervisor(message, force=False):
    if force or SUPERVISOR_VERBOSE_MESSAGES:
        print(message)


TRAIN_SERVER_HOST = read_env_value("TRAIN_SERVER_HOST", "127.0.0.1", str)
TRAIN_SERVER_PORT = read_env_value("TRAIN_SERVER_PORT", 10001, int)
PACKET_HEADER_FORMAT = read_env_value("PACKET_HEADER_FORMAT", ">II", str)
PACKET_HEADER_SIZE = struct.calcsize(PACKET_HEADER_FORMAT)

class TrainingServer:
    """
    Clase que se encarga de la comunicación entre el supervisor y el robot.

    """

    def __init__(self, host = TRAIN_SERVER_HOST , port = TRAIN_SERVER_PORT):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.setblocking(False)
        self.client_socket = None
        self.client_address = None
        self.receive_buffer = b""
        log_supervisor(
            f"[Supervisor] servidor de entrenamiento escuchando en {self.host}:{self.port}",
            force=True,
        )

    def is_client_connected(self):
        return self.client_socket is not None

    def poll_requests(self):
        self._accept_pending_clients()
        if self.client_socket is None:
            return []

        chunks = []
        while True:
            try:
                data = self.client_socket.recv(65536)
            except BlockingIOError:
                break
            except ConnectionResetError:
                self._disconnect_client()
                return []

            if not data:
                self._disconnect_client()
                return []

            chunks.append(data)

        if not chunks:
            return []

        self.receive_buffer += b"".join(chunks)
        requests = []
        while True:
            try:
                decoded = self._decode_next_packet()
            except json.JSONDecodeError:
                log_supervisor("[Supervisor] request externa invalida ignorada")
                self.receive_buffer = b""
                break

            if decoded is None:
                break

            payload, binary_payload = decoded
            if binary_payload:
                payload["_binary_payload"] = binary_payload
            requests.append(payload)

        return requests

    def send_response(self, payload):
        if self.client_socket is None:
            return False

        packet = self._encode_response_packet(payload)
        try:
            self.client_socket.setblocking(True)
            self.client_socket.sendall(packet)
            self.client_socket.setblocking(False)
            return True
        except OSError:
            self._disconnect_client()
            return False

    def _decode_next_packet(self):
        if len(self.receive_buffer) < PACKET_HEADER_SIZE:
            return None

        header_size, binary_size = struct.unpack(
            PACKET_HEADER_FORMAT, self.receive_buffer[:PACKET_HEADER_SIZE]
        )
        packet_size = PACKET_HEADER_SIZE + header_size + binary_size
        if len(self.receive_buffer) < packet_size:
            return None

        header_start = PACKET_HEADER_SIZE
        header_end = header_start + header_size
        binary_end = header_end + binary_size
        header_bytes = self.receive_buffer[header_start:header_end]
        binary_payload = self.receive_buffer[header_end:binary_end]
        self.receive_buffer = self.receive_buffer[binary_end:]
        payload = json.loads(header_bytes.decode("utf-8"))
        return payload, bytes(binary_payload)

    def _encode_response_packet(self, payload):
        """
        Codifica la respuesta al trainer. Si la imagen ya viene como bytes
        crudos (formato nuevo del robot), pasa directo al binary section
        del packet TCP. Cae a base64 solo por compatibilidad si alguien
        publica data_b64 (camino obsoleto).
        """
        response_payload = dict(payload)
        binary_payload = b""

        observation = response_payload.get("observation")
        if isinstance(observation, dict):
            image = observation.get("image")
            if isinstance(image, dict):
                image_bytes = image.get("data_bytes")
                if isinstance(image_bytes, (bytes, bytearray)):
                    binary_payload = bytes(image_bytes)
                    image_copy = {
                        key: value
                        for key, value in image.items()
                        if key not in ("data_bytes", "data_b64")
                    }
                    observation_copy = dict(observation)
                    observation_copy["image"] = image_copy
                    response_payload["observation"] = observation_copy
                elif image.get("data_b64"):
                    binary_payload = base64.b64decode(image["data_b64"].encode("ascii"))
                    image_copy = dict(image)
                    image_copy.pop("data_b64", None)
                    observation_copy = dict(observation)
                    observation_copy["image"] = image_copy
                    response_payload["observation"] = observation_copy

        header_bytes = json.dumps(response_payload).encode("utf-8")
        return (
            struct.pack(PACKET_HEADER_FORMAT, len(header_bytes), len(binary_payload))
            + header_bytes
            + binary_payload
        )

    def _accept_pending_clients(self):
        """
        Acepta clientes pendientes.
        """
        while True:
            try:
                client_socket, client_address = self.server_socket.accept()
            except BlockingIOError:
                break

            client_socket.setblocking(False)
            if self.client_socket is not None:
                client_socket.close()
                continue

            self.client_socket = client_socket
            self.client_address = client_address
            self.receive_buffer = b""
            log_supervisor(
                f"[Supervisor] cliente de entrenamiento conectado: {client_address}",
                force=True,
            )

    def _disconnect_client(self):
        """
        Desconecta un cliente.
        """
        if self.client_socket is None:
            return

        try:
            self.client_socket.close()
        finally:
            log_supervisor(
                f"[Supervisor] cliente de entrenamiento desconectado: {self.client_address}",
                force=True,
            )
            self.client_socket = None
            self.client_address = None
            self.receive_buffer = b""