"""
Geometria de colision auto<->obstaculo (compartida entre el supervisor y el monitor de
test, para que midan EXACTAMENTE igual).

El auto NO es un punto: es un rectangulo alargado (largo != ancho) que ademas rota con el
yaw. Medir distancia centro-a-centro falla cuando la caja se toca con una esquina o una
rueda descentrada. Aca el auto se modela como un RECTANGULO ORIENTADO (OBB) y la caja como
un punto; se devuelve la distancia de la caja al BORDE del rectangulo (0 si cae adentro).
El caller declara choque si esa distancia <= margen (radio efectivo de la caja).
"""

import math


def car_obstacle_clearance(car_pos, car_rot, obstacle_xy, half_len, half_width):
    """
    Distancia planar (m) del centro de la caja al borde del rectangulo orientado del auto.

    car_pos : getPosition() del auto [x, y, z] (mundo).
    car_rot : getOrientation() del auto, 9 floats row-major. Las componentes planas del
              eje local +x del auto en el mundo son (rot[0], rot[3]); las del +y, (rot[1], rot[4]).
    obstacle_xy : (x, y) del centro de la caja (mundo).
    half_len   : media-longitud del auto (eje local +x).
    half_width : medio-ancho del auto (eje local +y).

    Devuelve 0.0 si el centro de la caja cae DENTRO del rectangulo del auto.
    """
    dx = obstacle_xy[0] - car_pos[0]
    dy = obstacle_xy[1] - car_pos[1]
    # Proyectar el delta (mundo) sobre los ejes locales del auto -> coords en el frame del auto.
    local_x = dx * car_rot[0] + dy * car_rot[3]
    local_y = dx * car_rot[1] + dy * car_rot[4]
    # Distancia del punto al rectangulo [-half_len, half_len] x [-half_width, half_width].
    over_x = max(0.0, abs(local_x) - half_len)
    over_y = max(0.0, abs(local_y) - half_width)
    return math.hypot(over_x, over_y)
