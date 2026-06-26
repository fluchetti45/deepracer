"""
Controlador MANUAL (teclado) del auto Ackermann, para testear a mano la deteccion de
colision con los obstaculos (ver worlds/obstacle_test.wbt + controllers/obstacle_monitor).

Teclas (la ventana 3D de Webots tiene que tener el foco):
  Flecha ARRIBA  -> avanzar     Flecha ABAJO -> retroceder
  Flecha IZQ     -> doblar izq  Flecha DER   -> doblar der
  (sin tecla = sin traccion / ruedas al frente)

Usa el MISMO mapeo Ackermann que agent_controller, asi el comportamiento es identico al
del agente. No habla con el supervisor: solo lee el teclado y comanda los motores.
"""

import math

from controller import Robot, Keyboard

# Mismos valores que agent_controller / .env (copiados para no importar agent_controller,
# que se auto-ejecuta al importarse).
MAX_STEER_ANGLE = math.radians(30.0)   # rad
DRIVE_SPEED = 3.0                       # rad/s de las ruedas traseras al avanzar
WHEELBASE = 0.16                        # m
TRACK_WIDTH = 0.13                      # m


def ackermann_angles(delta):
    """Angulos (izq, der) de las ruedas delanteras para un delta central (bicicleta)."""
    if abs(delta) < 1e-4:
        return 0.0, 0.0
    radius = WHEELBASE / math.tan(delta)
    left = math.atan(WHEELBASE / (radius - TRACK_WIDTH / 2.0))
    right = math.atan(WHEELBASE / (radius + TRACK_WIDTH / 2.0))
    return left, right


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    left_rear = robot.getDevice("left rear motor")
    right_rear = robot.getDevice("right rear motor")
    for motor in (left_rear, right_rear):
        motor.setPosition(float("inf"))  # modo velocidad
        motor.setVelocity(0.0)
    left_steer = robot.getDevice("left steer motor")
    right_steer = robot.getDevice("right steer motor")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    print("[keyboard_drive] Auto manual listo. Foco en la ventana 3D y maneja con las flechas.")
    print("[keyboard_drive]   ARRIBA/ABAJO = avanzar/retroceder | IZQ/DER = doblar")

    while robot.step(timestep) != -1:
        # Drenar todas las teclas presionadas este step.
        keys = set()
        key = keyboard.getKey()
        while key != -1:
            keys.add(key & 0x7FFF)  # sin modificadores (shift/ctrl)
            key = keyboard.getKey()

        speed = 0.0
        if Keyboard.UP in keys:
            speed = DRIVE_SPEED
        elif Keyboard.DOWN in keys:
            speed = -DRIVE_SPEED

        delta = 0.0
        if Keyboard.LEFT in keys:
            delta = MAX_STEER_ANGLE
        elif Keyboard.RIGHT in keys:
            delta = -MAX_STEER_ANGLE

        left_angle, right_angle = ackermann_angles(delta)
        left_steer.setPosition(left_angle)
        right_steer.setPosition(right_angle)
        left_rear.setVelocity(speed)
        right_rear.setVelocity(speed)


if __name__ == "__main__":
    main()
