"""
Formato y decodificacion compartida de la observacion de imagen (camara del e-puck).

Tanto el entorno de entrenamiento (`rl/env.py`, lado trainer) como la inferencia
en el supervisor (`helpers/policy_runner.py`) deben producir EXACTAMENTE el mismo
tensor de imagen, asi que la logica vive en un solo lugar.

Convencion:
  - En el cable la imagen viaja como bytes crudos RGB en layout HWC (row-major),
    acompaniada de metadata {width, height, channels} en el header JSON.
  - El modelo (SB3) espera la imagen channel-first uint8 [0, 255] con shape
    (C, H, W). SB3 se encarga de dividir por 255 (normalize_images=True).
"""

import gymnasium as gym
import numpy as np

from helpers.read_env_value import read_env_value

# Dimensiones de la camara del e-puck en worlds/nav1.wbt (width=52, height=39).
IMAGE_WIDTH = read_env_value("CAMERA_WIDTH", 84, int)
IMAGE_HEIGHT = read_env_value("CAMERA_HEIGHT", 84, int)
IMAGE_CHANNELS = read_env_value("CAMERA_CHANNELS", 3, int)

# Shape channel-first que consume el modelo.
IMAGE_SHAPE = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)


def build_image_space():
    """Espacio de observacion de la imagen para el observation_space del env."""
    return gym.spaces.Box(low=0, high=255, shape=IMAGE_SHAPE, dtype=np.uint8)


def blank_image_array():
    """Imagen negra (C, H, W) uint8 — fallback cuando todavia no hay frame."""
    return np.zeros(IMAGE_SHAPE, dtype=np.uint8)


def blank_image_payload():
    """
    Dict de imagen 'vacio' que el supervisor mete en la observacion cuando aun no
    recibio ningun frame del robot. Mantiene la misma estructura que el frame real
    (incluido data_bytes) para que el pipeline no tenga ramas especiales.
    """
    return {
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "channels": IMAGE_CHANNELS,
        "encoding": "rgb",
        "layout": "hwc",
        "data_bytes": b"\x00" * (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS),
    }


def decode_image_observation(image_payload):
    """
    Convierte el dict de imagen recibido ({width, height, channels, data_bytes})
    en un array uint8 channel-first (C, H, W). Si falta data o no matchea el tamanio
    esperado, devuelve una imagen negra del shape correcto (nunca rompe la obs).
    """
    if not isinstance(image_payload, dict):
        return blank_image_array()

    data = image_payload.get("data_bytes")
    if not isinstance(data, (bytes, bytearray)):
        return blank_image_array()

    height = int(image_payload.get("height", IMAGE_HEIGHT))
    width = int(image_payload.get("width", IMAGE_WIDTH))
    channels = int(image_payload.get("channels", IMAGE_CHANNELS))

    buffer = bytes(data)
    if len(buffer) != height * width * channels:
        return blank_image_array()

    hwc = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, channels)
    chw = np.transpose(hwc, (2, 0, 1))  # HWC -> CHW
    return np.ascontiguousarray(chw, dtype=np.uint8)
