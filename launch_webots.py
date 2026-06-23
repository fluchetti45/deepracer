# Lanzo webots configurado para que levante el python de este env.
import time
import os
import atexit
import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenador PPO para line following en Webots."
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host del supervisor de Webots."
    )
    parser.add_argument(
        "--port", type=int, default=10001, help="Puerto del supervisor."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo para PyTorch/Stable-Baselines3, por ejemplo cpu o cuda.",
    )
    parser.add_argument(
        "--webots-world",
        default="worlds/ackermann.wbt",
        help="World de Webots a lanzar automaticamente.",
    )
    parser.add_argument(
        "--webots-executable",
        default=r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe",
        help="Ruta al ejecutable de Webots.",
    )
    return parser.parse_args()


def _spawn_webots(args, port=None):
    """
    Lanza UNA instancia de Webots (headless) y devuelve el proceso. Si se pasa `port`,
    lo inyecta como TRAIN_SERVER_PORT/DEFAULT_PORT SOLO en el entorno de ESE proceso
    (override por-proceso: dotenv no pisa os.environ, asi cada supervisor bindea su
    propio puerto sin tocar el .env global). NO duerme: el caller maneja la espera.
    """
    world_path = os.path.abspath(args.webots_world)
    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_root, "env", "Scripts", "python.exe")

    env = os.environ.copy()
    env["WEBOTS_PYTHON"] = venv_python
    env["PYTHONEXECUTABLE"] = venv_python
    env["PATH"] = os.path.dirname(venv_python) + os.pathsep + env["PATH"]
    if port is not None:
        env["TRAIN_SERVER_PORT"] = str(port)
        env["DEFAULT_PORT"] = str(port)

    command = [
        args.webots_executable,
        "--batch",
        "--minimize",
        "--no-rendering",
        world_path,
    ]
    print(f"Lanzando Webots{'' if port is None else f' (puerto {port})'}: {command}")

    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    atexit.register(process.kill)
    return process


def launch_webots(args):
    """Lanza UNA instancia (comportamiento original) y espera su inicializacion."""
    process = _spawn_webots(args)
    print("Esperando inicializacion de Webots...")
    time.sleep(10)
    return process


def launch_webots_instances(args, ports):
    """
    Lanza N instancias de Webots EN PARALELO (una por puerto) y espera a que arranquen.
    Cada una corre su propio supervisor bindeando su puerto. Devuelve la lista de procesos.
    """
    processes = [_spawn_webots(args, port=port) for port in ports]
    print(f"Esperando inicializacion de {len(processes)} instancias de Webots...")
    # Un poco mas de espera con varias instancias (contienden CPU al bootear).
    time.sleep(10 + 2 * max(0, len(processes) - 1))
    return processes


if __name__ == "__main__":
    process = launch_webots(parse_args())
    process.wait()
