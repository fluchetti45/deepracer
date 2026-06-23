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
