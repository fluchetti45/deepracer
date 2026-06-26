# Lanza Webots en modo GUI (manejo MANUAL) con el Python del venv del proyecto cargado
# para los controladores -> dotenv/numpy/etc. disponibles (a diferencia de abrir Webots a
# mano, que usa el Python del sistema). Pensado para tests manuales, p. ej. la deteccion
# de colision con obstaculos (worlds/obstacle_test.wbt + controllers/obstacle_monitor).
#
# Uso:
#   python launch_manual.py                                  # default: obstacle_test.wbt
#   python launch_manual.py --world worlds/ackermann.wbt     # cualquier otro world
#
# Los prints de los controladores salen al MISMO terminal (--stdout/--stderr de Webots).

import argparse
import os
import subprocess


def parse_args():
    p = argparse.ArgumentParser(
        description="Lanza Webots GUI con el venv del proyecto para test manual."
    )
    p.add_argument(
        "--world",
        default="worlds/obstacle_test.wbt",
        help="World a abrir (default: worlds/obstacle_test.wbt).",
    )
    p.add_argument(
        "--webots-executable",
        default=r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe",
        help="Ruta al ejecutable de Webots.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    world_path = os.path.abspath(args.world)
    if not os.path.exists(world_path):
        raise SystemExit(f"No existe el world: {world_path}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_root, "env", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        print(
            f"ADVERTENCIA: no encuentro el venv en {venv_python}; los controladores "
            "usaran el Python del sistema (puede faltar dotenv/numpy)."
        )

    # Mismo mecanismo que launch_webots.py: apuntar el Python de los controladores al venv.
    env = os.environ.copy()
    env["WEBOTS_PYTHON"] = venv_python
    env["PYTHONEXECUTABLE"] = venv_python
    env["PATH"] = os.path.dirname(venv_python) + os.pathsep + env.get("PATH", "")

    # GUI con rendering (para manejar) y salida de los controladores al terminal.
    command = [args.webots_executable, "--stdout", "--stderr", world_path]
    print(f"Lanzando Webots (GUI, venv): {command}")
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
