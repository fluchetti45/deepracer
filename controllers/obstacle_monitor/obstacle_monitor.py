"""
Monitor de COLISION para test manual (worlds/obstacle_test.wbt).

Supervisor que lee la pose del auto (DEF EPUCK) y de las cajas (DEF OBSTACLE_*) y, en cada
step, calcula la distancia planar centro-a-centro auto<->caja EXACTAMENTE como lo hace
supervisor_controller._check_obstacle_hit (mismo umbral OBSTACLE_HIT_DIST del .env). Imprime:
  - la distancia minima en vivo (throttled), con la caja mas cercana;
  - un evento "COLISION" cuando la distancia cruza el umbral (entrada) y cuando sale.

Asi se puede manejar el auto a mano hacia una caja y medir a que distancia salta la
deteccion vs. cuando el auto la TOCA visualmente, para calibrar OBSTACLE_HIT_DIST.
"""

import os
import sys

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import math

from controller import Supervisor
from helpers.read_env_value import read_env_value
from helpers.obstacle_geom import car_obstacle_clearance

# Mismos parametros que el supervisor (de .env) para medir IDENTICO.
HIT_DIST = read_env_value("OBSTACLE_HIT_DIST", 0.07, float)
CAR_HALF_LENGTH = read_env_value("CAR_HALF_LENGTH", 0.11, float)
CAR_HALF_WIDTH = read_env_value("CAR_HALF_WIDTH", 0.075, float)
# Una caja se considera ACTIVA (en juego) si esta cerca del origen; las parkeadas
# (XY ~100) quedan fuera.
ACTIVE_RADIUS = 20.0
PRINT_EVERY = 8  # cada cuantos steps imprimir la distancia en vivo


def planar_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    car = supervisor.getFromDef("EPUCK")
    if car is None:
        print("[obstacle_monitor] ERROR: no encuentro DEF EPUCK en el world.")
        return

    # Recolectar el pool de cajas.
    obstacles = []
    i = 0
    while True:
        node = supervisor.getFromDef(f"OBSTACLE_{i}")
        if node is None:
            break
        obstacles.append((f"obstacle_{i}", node))
        i += 1

    print("=" * 60)
    print(f"[obstacle_monitor] umbral de choque OBSTACLE_HIT_DIST = {HIT_DIST:.3f} m "
          f"(clearance al rectangulo del auto)")
    print(f"[obstacle_monitor] footprint auto: half_len={CAR_HALF_LENGTH:.3f} "
          f"half_width={CAR_HALF_WIDTH:.3f} m")
    print(f"[obstacle_monitor] {len(obstacles)} cajas en el pool. Activas (cerca del origen):")
    for name, node in obstacles:
        pos = node.getPosition()
        if planar_dist(pos, [0.0, 0.0]) <= ACTIVE_RADIUS:
            print(f"    {name}: x={pos[0]:+.3f} y={pos[1]:+.3f}")
    print("[obstacle_monitor] Maneja el auto contra una caja y observa la distancia.")
    print("=" * 60)

    in_collision = False
    step = 0
    while supervisor.step(timestep) != -1:
        step += 1
        car_pos = car.getPosition()
        car_rot = car.getOrientation()

        # Caja activa mas cercana (clearance al rectangulo orientado del auto).
        closest_name = None
        closest_dist = float("inf")
        for name, node in obstacles:
            pos = node.getPosition()
            if planar_dist(pos, [0.0, 0.0]) > ACTIVE_RADIUS:
                continue  # parkeada
            d = car_obstacle_clearance(
                car_pos, car_rot, (pos[0], pos[1]), CAR_HALF_LENGTH, CAR_HALF_WIDTH
            )
            if d < closest_dist:
                closest_dist = d
                closest_name = name

        if closest_name is None:
            continue

        hit = closest_dist <= HIT_DIST

        # Transiciones de estado (entrada/salida de colision).
        if hit and not in_collision:
            print(f">>> COLISION con {closest_name}  dist={closest_dist:.3f} m  "
                  f"(umbral {HIT_DIST:.3f})")
            in_collision = True
        elif not hit and in_collision:
            print(f"<<< fuera de colision ({closest_name})  dist={closest_dist:.3f} m")
            in_collision = False

        # Distancia en vivo (throttled) para ver el acercamiento.
        if step % PRINT_EVERY == 0:
            flag = "  <== CHOQUE" if hit else ""
            print(f"min dist -> {closest_name}: {closest_dist:.3f} m{flag}")


if __name__ == "__main__":
    main()
