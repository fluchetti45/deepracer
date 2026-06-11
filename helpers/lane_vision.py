"""
Deteccion de carril vision-pura para el reward del supervisor.

La idea (opcion A): el reward se computa desde la MISMA imagen de camara frontal
que va en la observacion. No hay waypoints ni geometria de la pista: solo se
clasifican pixeles por color en la franja inferior del frame (la pista cercana) y
se derivan un par de escalares:

  - center_clearance : que tan "limpia" (asfalto, sin marcas ni pasto) esta la
                       banda central de la vista -> ~1 cuando el robot esta
                       centrado en su carril, baja al acercarse a un borde.
  - offset           : desplazamiento lateral firmado del centro del carril
                       (punto medio entre la linea amarilla y la blanca) respecto
                       al centro de la imagen. Da la DIRECCION de correccion.
  - green_center     : fraccion de pasto en la banda central -> se fue de la pista.
  - white_center     : fraccion de blanco en la banda central -> pisando el borde.

Todo en numpy puro (los frames son chicos, ~52x39); sin dependencia de OpenCV.
Convencion de imagen: igual que helpers/image_obs -> RGB, layout HWC, uint8.
"""

import numpy as np

from helpers.read_env_value import read_env_value

# Franja inferior del frame que se analiza (saltea cielo/horizonte/montanias).
ROI_TOP_FRAC = read_env_value("LANE_ROI_TOP_FRAC", 0.45)
# Ancho (fraccion) de la banda central de columnas que deberia estar limpia
# (asfalto) cuando el robot va centrado en su carril.
CENTER_BAND_FRAC = read_env_value("LANE_CENTER_BAND_FRAC", 0.34)

# Umbrales de color RGB (0-255). Pensados para: amarillo (linea central),
# blanco (borde externo), verde (pasto = fuera de pista).
YELLOW_R_MIN = read_env_value("LANE_YELLOW_R_MIN", 120, int)
YELLOW_G_MIN = read_env_value("LANE_YELLOW_G_MIN", 120, int)
YELLOW_B_MAX = read_env_value("LANE_YELLOW_B_MAX", 100, int)
WHITE_MIN = read_env_value("LANE_WHITE_MIN", 175, int)
GREEN_G_MIN = read_env_value("LANE_GREEN_G_MIN", 60, int)
GREEN_MARGIN = read_env_value("LANE_GREEN_MARGIN", 25, int)


def decode_rgb_hwc(image_payload):
    """
    Convierte el dict de imagen del robot ({width, height, channels, data_bytes})
    en un array uint8 HWC (alto, ancho, 3) RGB. Devuelve None si no se puede
    (sin data, tamanio inconsistente, no-RGB) -> el caller usa un breakdown neutro.
    """
    if not isinstance(image_payload, dict):
        return None
    data = image_payload.get("data_bytes")
    if not isinstance(data, (bytes, bytearray)):
        return None
    height = int(image_payload.get("height", 0))
    width = int(image_payload.get("width", 0))
    channels = int(image_payload.get("channels", 3))
    if channels != 3 or height <= 0 or width <= 0:
        return None
    buffer = bytes(data)
    if len(buffer) != height * width * channels:
        return None
    return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)


def _color_masks(rgb):
    """Mascaras booleanas (amarillo, blanco, verde) sobre una imagen RGB HWC."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    yellow = (r >= YELLOW_R_MIN) & (g >= YELLOW_G_MIN) & (b <= YELLOW_B_MAX)
    white = (r >= WHITE_MIN) & (g >= WHITE_MIN) & (b >= WHITE_MIN)
    green = (g >= GREEN_G_MIN) & (g - r >= GREEN_MARGIN) & (g - b >= GREEN_MARGIN)
    return yellow, white, green


def _centroid_col(mask, width):
    """Columna centroide (ponderada por cantidad de pixeles) de una mascara, o None."""
    col_counts = mask.sum(axis=0)
    total = float(col_counts.sum())
    if total <= 0.0:
        return None
    return float((np.arange(width) * col_counts).sum() / total)


def detect_lane(rgb):
    """
    Extrae las features de carril de un frame RGB HWC. Ver el docstring del modulo
    para el significado de cada campo. Nunca lanza: si algo falla devuelve features
    neutras (line_visible=False).
    """
    height, width, _ = rgb.shape
    top = int(ROI_TOP_FRAC * height)
    top = min(max(top, 0), max(height - 1, 0))
    roi = rgb[top:, :, :]
    yellow, white, green = _color_masks(roi)

    roi_h, roi_w = yellow.shape
    roi_area = float(roi_h * roi_w) or 1.0
    yellow_frac = float(yellow.sum()) / roi_area
    white_frac = float(white.sum()) / roi_area
    green_frac = float(green.sum()) / roi_area

    # Banda central de columnas.
    band = max(1, int(CENTER_BAND_FRAC * roi_w))
    c0 = (roi_w - band) // 2
    c1 = c0 + band
    band_area = float(roi_h * band) or 1.0
    center_yellow = float(yellow[:, c0:c1].sum()) / band_area
    center_white = float(white[:, c0:c1].sum()) / band_area
    center_green = float(green[:, c0:c1].sum()) / band_area
    center_clearance = max(0.0, 1.0 - (center_yellow + center_white + center_green))

    # Offset firmado: punto medio entre amarillo y blanco respecto al centro.
    # Solo es confiable con ambas marcas visibles; si no, queda None (la simetria
    # de center_clearance igual sostiene el termino de centrado).
    yc = _centroid_col(yellow, roi_w)
    wc = _centroid_col(white, roi_w)
    if yc is not None and wc is not None and roi_w > 0:
        lane_center = 0.5 * (yc + wc)
        offset = (lane_center - roi_w / 2.0) / (roi_w / 2.0)
        offset = float(max(-1.0, min(1.0, offset)))
    else:
        offset = None

    line_visible = (yc is not None) or (wc is not None)

    return {
        "yellow_frac": yellow_frac,
        "white_frac": white_frac,
        "green_frac": green_frac,
        "center_yellow": center_yellow,
        "center_white": center_white,
        "center_green": center_green,
        "center_clearance": center_clearance,
        "offset": offset,
        "line_visible": bool(line_visible),
    }
