"""
Recoleccion del dataset de destilacion: corre el agente GEOMETRICO (maestro) en Webots y
graba pares (imagen de camara, velocity, accion del maestro) para entrenar la variante
vision_distill (ver rl/distill).

Por cada step: reconstruye la geometria desde la MISMA camara con geom_vector_from_rgb, la
normaliza con el VecNormalize del maestro, pide su accion determinista, MANEJA con esa
accion (el experto se queda en pista -> estados buenos) y graba (imagen, velocity, accion).
Resetea entre episodios: el supervisor rota los tracks de training (eval:false) -> fondos
variados, lo que ademas ayuda a que el estudiante ignore el fondo.

Uso (Webots se lanza solo):
  python -m rl.collect_teacher --teacher models/20260702171446 --episodes 40 --out data/teacher
  python -m rl.collect_teacher --teacher models/<id> --episodes 40 --no-webots-launch  # Webots ya abierto
"""

import argparse
import os
import pickle
import time

import numpy as np
from stable_baselines3 import PPO

from launch_webots import launch_webots
from rl.env import NavEnv
from helpers.geom_obs import geom_vector_from_rgb, GEOM_OBS_SIZE


def load_teacher(model_dir, device):
    """Carga el maestro (PPO geometrico) + su VecNormalize (para normalizar la geometria)."""
    teacher = PPO.load(os.path.join(model_dir, "final_model.zip"), device=device)
    if teacher.observation_space.shape != (GEOM_OBS_SIZE,):
        raise SystemExit(
            f"El maestro espera obs {teacher.observation_space.shape} pero geom_vector_from_rgb "
            f"produce ({GEOM_OBS_SIZE},). Revisa GEOM_BANDS/.env."
        )
    vn_path = os.path.join(model_dir, "vecnormalize.pkl")
    if not os.path.exists(vn_path):
        print("ADVERTENCIA: el maestro no tiene vecnormalize.pkl; uso la geometria SIN normalizar.")
        return teacher, None
    with open(vn_path, "rb") as fh:
        vn = pickle.load(fh)  # VecNormalize se despickla sin venv; solo uso obs_rms
    return teacher, vn


def teacher_action(teacher, vn, rgb_hwc, velocity):
    """Accion determinista del maestro para un frame: geom -> normalizar -> predict."""
    geom = np.asarray(geom_vector_from_rgb(rgb_hwc, velocity), dtype=np.float32)
    if vn is not None:
        rms = vn.obs_rms
        geom = np.clip((geom - rms.mean) / np.sqrt(rms.var + vn.epsilon),
                       -vn.clip_obs, vn.clip_obs).astype(np.float32)
    action, _ = teacher.predict(geom, deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def parse_args():
    p = argparse.ArgumentParser(description="Recolecta pares (imagen, accion) del maestro geometrico.")
    p.add_argument("--teacher", required=True, help="run_dir del modelo geometrico maestro.")
    p.add_argument("--out", default="data/teacher", help="Carpeta destino de los .npz.")
    p.add_argument("--episodes", type=int, default=40, help="Episodios a recolectar.")
    p.add_argument("--max-steps", type=int, default=1500, help="Cap de steps por episodio.")
    p.add_argument("--device", default="cpu")
    # --- Noise injection (DART, Laskey 2017): robustez / cobertura de recuperacion ---
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="Desvio del ruido gaussiano AGREGADO a la accion que se EJECUTA (no a "
                        "la que se graba). El auto se desvia a estados off-center y queda "
                        "grabada la accion LIMPIA del maestro ahi -> cobertura de recuperacion. "
                        "0 = off. Tipico 0.1-0.2; mucho mas eyecta al maestro de la pista.")
    p.add_argument("--seed", type=int, default=0, help="Semilla del ruido (reproducibilidad).")
    p.add_argument("--tag", default="",
                   help="Prefijo de los .npz para que varias corridas convivan en la misma "
                        "carpeta sin pisarse (ej. limpia + ruidosa). Si falta, se deriva del "
                        "ruido+seed: 'n015s0', 'n000s0', etc.")
    # --- Webots ---
    p.add_argument("--webots-world", default="worlds/track1.wbt",
                   help="World a lanzar (el supervisor rota los tracks de training).")
    p.add_argument("--webots-executable",
                   default=r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe")
    p.add_argument("--no-webots-launch", action="store_true",
                   help="No lanzar Webots (ya esta abierto con el world cargado).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=10001)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    teacher, vn = load_teacher(args.teacher, args.device)
    teacher_id = os.path.basename(args.teacher.rstrip("/\\"))
    rng = np.random.default_rng(args.seed)
    # Tag por corrida: evita que varias colectas (limpia/ruidosa) se pisen en la misma carpeta.
    tag = args.tag or f"n{int(round(args.noise_std * 100)):03d}s{args.seed}"
    if args.noise_std > 0.0:
        print(f"Noise injection ON (std={args.noise_std}): se graba la accion LIMPIA, "
              f"se ejecuta con ruido.")
    print(f"Tag de esta corrida: '{tag}' (prefijo de los .npz).")

    webots_process = None
    env = None
    total_pairs = 0
    try:
        if not args.no_webots_launch:
            webots_process = launch_webots(args)
        env = NavEnv(host=args.host, port=args.port)

        for episode in range(args.episodes):
            obs, info = env.reset()
            images, velocities, actions = [], [], []
            for _ in range(args.max_steps):
                rgb = np.asarray(obs["image"]).transpose(1, 2, 0)  # CHW -> HWC
                vel = np.asarray(obs["velocity"], dtype=np.float32)
                action = teacher_action(teacher, vn, rgb, vel)  # accion LIMPIA = label

                images.append(np.asarray(obs["image"], dtype=np.uint8))
                velocities.append(vel)
                actions.append(action)

                # Se EJECUTA con ruido (DART): el auto visita estados de recuperacion,
                # pero arriba quedo grabada la accion limpia del maestro para ese estado.
                exec_action = action
                if args.noise_std > 0.0:
                    exec_action = np.clip(
                        action + rng.normal(0.0, args.noise_std, size=action.shape),
                        -1.0, 1.0).astype(np.float32)

                obs, _r, terminated, truncated, info = env.step(exec_action)
                if terminated or truncated:
                    break

            if not images:
                print(f"  ep {episode:3d}: sin pares (reset fallo?), salteo.")
                continue
            track = str(info.get("track", "track")).replace(".png", "")
            path = os.path.join(args.out, f"{tag}_ep{episode:03d}_{track}.npz")
            np.savez_compressed(
                path,
                images=np.stack(images).astype(np.uint8),
                velocities=np.stack(velocities).astype(np.float32),
                actions=np.stack(actions).astype(np.float32),
            )
            total_pairs += len(images)
            print(f"  ep {episode:3d}: {len(images):4d} pares ({track}) -> {os.path.basename(path)}")
    finally:
        if env is not None:
            env.close()
        if webots_process is not None:
            try:
                webots_process.kill()
            except Exception:
                pass

    print("=" * 56)
    print(f"Total: {total_pairs} pares en {args.out}/  (maestro {teacher_id})")
    print(f"Ahora: python -m rl.distill --data {args.out} --seeds 0 1 2 3 4 "
          f"--teacher-id {teacher_id}")
    print("=" * 56)


if __name__ == "__main__":
    main()
