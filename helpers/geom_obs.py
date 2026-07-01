"""
Observacion del agente DERIVADA DE LA CAMARA (rama `geometrica`, redefinida): la policy
NO recibe pixeles, recibe METRICAS escalares calculadas LOCALMENTE sobre la imagen de la
camara en ESTE timestep (helpers/lane_vision). Es track-agnostica: depende de lo que el
agente PERCIBE, no de geometria global del circuito.

Vector (float32):
  [ forward, yaw_rate,        # velocidad propia (proprioceptiva)
    road_frac,                # fraccion de calzada visible en la ROI
    green_center,             # pasto en la banda central (proximidad a salirse)
    off_0, off_1, ..., off_{B-1} ]   # offset horizontal del centroide de calzada en B
                                     # franjas de CERCA->LEJOS (traza la pista adelante)

La logica de tamanio/escala vive en un solo lugar (como helpers/image_obs) para que el
env y el supervisor coincidan exactamente.
"""

import gymnasium as gym
import numpy as np

from helpers.read_env_value import read_env_value

# Cantidad de franjas horizontales (look-ahead EN LA IMAGEN, cerca->lejos).
GEOM_BANDS = read_env_value("GEOM_BANDS", 5, int)
# Cota del Box (las features van clampeadas a +-esto; deja margen para VecNormalize).
GEOM_BOUND = read_env_value("GEOM_OBS_BOUND", 5.0, float)

# Tamanio: velocidad(2) + road_frac(1) + green_center(1) + band_offsets(B).
GEOM_OBS_SIZE = 4 + GEOM_BANDS


def build_geom_space():
    """Espacio de observacion (Box plano) de features de percepcion."""
    return gym.spaces.Box(
        low=-GEOM_BOUND, high=GEOM_BOUND, shape=(GEOM_OBS_SIZE,), dtype=np.float32
    )


def blank_geom():
    """Vector de ceros (fallback cuando no hay frame de camara valido)."""
    return [0.0] * GEOM_OBS_SIZE


def geom_vector_from_rgb(rgb, velocity):
    """
    Vector geometrico Box(GEOM_OBS_SIZE) desde una imagen RGB HWC + velocity
    [forward, yaw_rate]. UNICA fuente de la logica: la usa el supervisor de la variante
    geometrica (obs de training) y el policy_runner para INFERENCIA cross-variante (correr
    un modelo geometrico desde cualquier rama, reconstruyendo las features desde la camara).
    Devuelve blank_geom() si no hay imagen valida.
    """
    # Import perezoso: road_band_offsets vive en lane_vision (evita ciclos y no rompe la
    # carga del modulo si una rama todavia no lo tuviera).
    from helpers.lane_vision import detect_lane, road_band_offsets

    if rgb is None:
        return blank_geom()
    try:
        feats = detect_lane(rgb)
        band_offsets = road_band_offsets(rgb, GEOM_BANDS)
    except Exception:
        return blank_geom()

    b = GEOM_BOUND
    clip = lambda v: max(-b, min(b, float(v)))  # noqa: E731
    obs = [
        clip(velocity[0]), clip(velocity[1]),
        clip(feats.get("road_frac", 0.0)),
        clip(feats.get("center_green", 0.0)),
    ]
    obs += [clip(o) for o in band_offsets]
    obs += [0.0] * (GEOM_OBS_SIZE - len(obs))  # padding defensivo
    return obs[:GEOM_OBS_SIZE]
