"""
Deteccion de PISTA vision-pura para el reward del supervisor.

Modelo DeepRacer: la pista es TODA la calzada entre los bordes blancos (no hay
sub-carril que respetar; la linea amarilla es solo decorativa y se puede pisar).
El reward se computa desde la MISMA imagen de camara frontal que va en la obs. No
hay waypoints ni geometria: se clasifican pixeles por color en la franja inferior
del frame y se derivan un par de escalares:

  - road = asfalto + amarillo = todo lo que NO es borde blanco ni pasto verde.
  - center_clearance : fraccion de CALZADA en la banda central de la vista (~1
                       cuando la pista llena el centro, baja al acercarse a un borde
                       o al pasto). NO penaliza la linea amarilla.
  - offset           : desplazamiento firmado del centroide de la CALZADA respecto
                       al centro de la imagen -> hacia donde dobla la pista (steer).
  - green_center     : fraccion de pasto en la banda central -> se fue de la pista.
  - white_center     : fraccion de blanco (borde) en la banda central -> pisando el borde.

Todo en numpy puro (frames chicos); sin dependencia de OpenCV.
Convencion de imagen: igual que helpers/image_obs -> RGB, layout HWC, uint8.
"""

import numpy as np

from helpers.read_env_value import read_env_value

# Franja inferior del frame que se analiza (saltea cielo/horizonte/montanias).
ROI_TOP_FRAC = read_env_value("LANE_ROI_TOP_FRAC", 0.45)
# Ancho (fraccion) de la banda central de columnas que deberia ser calzada cuando
# el robot va bien encarado sobre la pista.
CENTER_BAND_FRAC = read_env_value("LANE_CENTER_BAND_FRAC", 0.34)

# Umbrales de color RGB (0-255). Solo necesitamos distinguir borde blanco y pasto
# verde; el resto (asfalto gris + linea amarilla) es calzada drivable.
WHITE_MIN = read_env_value("LANE_WHITE_MIN", 175, int)
GREEN_G_MIN = read_env_value("LANE_GREEN_G_MIN", 60, int)
GREEN_MARGIN = read_env_value("LANE_GREEN_MARGIN", 25, int)
GREEN_BLUE_SLACK = read_env_value("LANE_GREEN_BLUE_SLACK", 40, int)


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


def _edge_masks(rgb):
    """Mascaras booleanas (blanco=borde, verde=pasto) sobre una imagen RGB HWC."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    white = (r >= WHITE_MIN) & (g >= WHITE_MIN) & (b >= WHITE_MIN)
    # Pasto: verde brillante, domina al rojo, y el azul no lo supera por mas de SLACK
    # (cubre tanto el verde-amarillo viejo como el teal PMS 3395 C; excluye azul puro).
    green = (
        (g >= GREEN_G_MIN)
        & (g - r >= GREEN_MARGIN)
        & (b - g <= GREEN_BLUE_SLACK)
    )
    return white, green


def _centroid_col(mask, width):
    """Columna centroide (ponderada por cantidad de pixeles) de una mascara, o None."""
    col_counts = mask.sum(axis=0)
    total = float(col_counts.sum())
    if total <= 0.0:
        return None
    return float((np.arange(width) * col_counts).sum() / total)


def road_band_offsets(rgb, n_bands):
    """
    Offset horizontal del centroide de CALZADA en `n_bands` franjas horizontales de la
    ROI, ordenadas de CERCA (abajo de la imagen) a LEJOS (arriba). Cada valor en [-1, 1]
    (centro de la imagen = 0; <0 izquierda, >0 derecha). Si una franja no tiene calzada,
    devuelve 0.0. La secuencia traza HACIA DONDE va la pista adelante (curvatura), todo
    derivado de la imagen de este timestep -> track-agnostico.
    """
    height, width, _ = rgb.shape
    top = int(ROI_TOP_FRAC * height)
    top = min(max(top, 0), max(height - 1, 0))
    roi = rgb[top:, :, :]
    white, green = _edge_masks(roi)
    road = ~(white | green)
    roi_h = road.shape[0]
    band_h = max(1, roi_h // max(1, n_bands))
    half_w = width / 2.0 or 1.0
    offsets = []
    for b in range(n_bands):
        # Banda b desde ABAJO (cerca): filas mas altas del array = parte baja de la imagen.
        hi = roi_h - b * band_h
        lo = max(0, roi_h - (b + 1) * band_h)
        col = _centroid_col(road[lo:hi, :], width)
        if col is None:
            offsets.append(0.0)
        else:
            offsets.append(float(max(-1.0, min(1.0, (col - half_w) / half_w))))
    return offsets


def detect_lane(rgb):
    """
    Extrae las features de PISTA de un frame RGB HWC (ver docstring del modulo).
    road = ~(blanco | verde) = asfalto + amarillo = calzada drivable.
    Nunca lanza: si algo falla devuelve features neutras (line_visible=False).
    """
    height, width, _ = rgb.shape
    top = int(ROI_TOP_FRAC * height)
    top = min(max(top, 0), max(height - 1, 0))
    roi = rgb[top:, :, :]
    white, green = _edge_masks(roi)
    road = ~(white | green)  # asfalto + amarillo = calzada

    roi_h, roi_w = white.shape
    roi_area = float(roi_h * roi_w) or 1.0
    road_frac = float(road.sum()) / roi_area
    white_frac = float(white.sum()) / roi_area
    green_frac = float(green.sum()) / roi_area

    # Banda central de columnas: cuanto del centro de la vista es calzada.
    band = max(1, int(CENTER_BAND_FRAC * roi_w))
    c0 = (roi_w - band) // 2
    c1 = c0 + band
    band_area = float(roi_h * band) or 1.0
    center_white = float(white[:, c0:c1].sum()) / band_area
    center_green = float(green[:, c0:c1].sum()) / band_area
    center_road = max(0.0, 1.0 - center_white - center_green)

    # Offset firmado: hacia donde esta la CALZADA (centroide del road respecto al
    # centro). En una curva el road se corre hacia un lado -> indica el steer.
    rc = _centroid_col(road, roi_w)
    if rc is not None and roi_w > 0:
        offset = (rc - roi_w / 2.0) / (roi_w / 2.0)
        offset = float(max(-1.0, min(1.0, offset)))
    else:
        offset = None

    road_visible = road_frac > 0.05

    return {
        "road_frac": road_frac,
        "white_frac": white_frac,
        "green_frac": green_frac,
        # center_clearance ahora = fraccion de CALZADA en el centro (no penaliza amarillo)
        "center_clearance": center_road,
        "center_white": center_white,
        "center_green": center_green,
        "offset": offset,
        # line_visible ahora = hay calzada visible (queda el mismo nombre de campo)
        "line_visible": bool(road_visible),
    }
