# Orquesta train + eval para MULTIPLES seeds de la VARIANTE actual (rama git + n_stack).
#
# Por cada seed: entrena (el trainer lanza su propio Webots), detecta el run_dir nuevo en
# models/, y luego evalua ese modelo sobre las pistas held-out (evaluate lanza su Webots).
# El eval guarda <run_dir>/eval_results_<timestamp>.json; ademas se escribe un manifest
# models/experiment_<timestamp>.json que mapea seed -> run_dir para agregar despues.
#
# La VARIANTE (geometrica / vision / vision apilada) la define la RAMA git activa + n_stack;
# este script NO cambia de rama. Workflow: checkout <rama>, correr este script.
#
# Uso tipico:
#   python -m rl.run_experiment --seeds 0 1 2 3 4 --total-timesteps 300000 --n-stack 4 \
#       --episodes 20
#
# (--episodes => eval en modo tasa de exito; sin el, cae a --laps.)

import argparse
import glob
import json
import os
import subprocess
import sys
import time


def list_run_dirs():
    """Conjunto de carpetas de corrida actuales bajo models/."""
    return {p for p in glob.glob(os.path.join("models", "*")) if os.path.isdir(p)}


def detect_new_run_dir(before):
    """El run_dir creado por el ultimo train: el mas nuevo del set diferencia."""
    new = list_run_dirs() - before
    if not new:
        return None
    return max(new, key=os.path.getmtime)


def run_cmd(cmd):
    """Corre un subproceso heredando stdout/stderr; devuelve el return code."""
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def parse_args():
    p = argparse.ArgumentParser(
        description="Train + eval multi-seed de la variante actual (rama git + n_stack)."
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", required=True,
        help="Seeds a entrenar/evaluar, p. ej. --seeds 0 1 2 3 4.",
    )
    # --- Train ---
    p.add_argument("--total-timesteps", type=int, default=100000)
    p.add_argument(
        "--n-stack", type=int, default=4,
        help="Frames apilados. 1 = vision/geometrica; 4 = vision apilada.",
    )
    p.add_argument("--webots-world", default="worlds/track1.wbt")
    p.add_argument(
        "--n-envs", type=int, default=1,
        help="Webots en PARALELO por seed (SubprocVecEnv). >1 entrena mas rapido cada agente.",
    )
    p.add_argument(
        "--n-steps", type=int, default=None,
        help="n_steps de PPO. Con --n-envs >1 conviene bajarlo (el rollout es n_steps*n_envs).",
    )
    p.add_argument(
        "--base-port", type=int, default=None,
        help="Puerto base para los N Webots del train (default: 10001).",
    )
    p.add_argument("--device", default="auto", help="Dispositivo de PyTorch para el train.")
    p.add_argument(
        "--camera-only", action="store_true",
        help="Ablacion CAMERA-ONLY: la obs es SOLO la imagen (se descarta la propiocepcion de "
             "velocidad que comparten las demas variantes). Misma CNN que vision 1-frame, sin la "
             "rama de velocidad. Se aplica a train y eval de todas las seeds.",
    )
    # --- Eval ---
    p.add_argument(
        "--episodes", type=int, default=None,
        help="Modo tasa de exito: N episodios fijos/track en eval. Si falta, usa --laps.",
    )
    p.add_argument("--laps", type=int, default=3, help="Vueltas/track si NO se usa --episodes.")
    p.add_argument("--eval-device", default="cpu", help="Dispositivo de PyTorch para el eval.")
    p.add_argument(
        "--no-eval", action="store_true",
        help="Solo entrenar las seeds (sin evaluar).",
    )
    return p.parse_args()


def train_one(args, seed):
    """Entrena UNA seed y devuelve el run_dir detectado (o None si fallo)."""
    before = list_run_dirs()
    cmd = [
        sys.executable, "-m", "rl.trainer",
        "--seed", str(seed),
        "--total-timesteps", str(args.total_timesteps),
        "--n-stack", str(args.n_stack),
        "--webots-world", args.webots_world,
        "--n-envs", str(args.n_envs),
        "--device", args.device,
        "--no-discord",
    ]
    if args.n_steps is not None:
        cmd += ["--n-steps", str(args.n_steps)]
    if args.base_port is not None:
        cmd += ["--base-port", str(args.base_port)]
    if run_cmd(cmd) != 0:
        return None
    return detect_new_run_dir(before)


def eval_one(args, run_dir):
    """Evalua un run_dir sobre las pistas held-out (n_stack se lee de su metadata)."""
    cmd = [
        sys.executable, "-m", "rl.evaluate",
        "--model", run_dir,
        "--device", args.eval_device,
    ]
    if args.episodes is not None:
        cmd += ["--episodes", str(args.episodes)]
    else:
        cmd += ["--laps", str(args.laps)]
    return run_cmd(cmd) == 0


def main():
    args = parse_args()

    # CAMERA-ONLY: se propaga por variable de entorno; subprocess.run hereda os.environ,
    # asi que el trainer, sus workers (SubprocVecEnv) y el evaluate quedan todos camera-only.
    if args.camera_only:
        os.environ["CAMERA_ONLY"] = "1"
        print("[CAMERA-ONLY] Observacion = SOLO imagen (sin propiocepcion de velocidad). "
              "Train y eval de todas las seeds heredan el flag.")

    runs = []  # [{seed, run_dir, status}]

    for seed in args.seeds:
        print("=" * 64)
        print(f"SEED {seed}: TRAIN  (n_stack={args.n_stack}, "
              f"timesteps={args.total_timesteps})")
        print("=" * 64)
        run_dir = train_one(args, seed)
        if run_dir is None:
            print(f"[seed {seed}] TRAIN fallo o no se detecto run_dir; salteo eval.")
            runs.append({"seed": seed, "run_dir": None, "status": "train_failed"})
            continue
        print(f"[seed {seed}] run_dir: {run_dir}")

        if args.no_eval:
            runs.append({"seed": seed, "run_dir": run_dir, "status": "trained"})
            continue

        print("=" * 64)
        print(f"SEED {seed}: EVAL  ({run_dir})")
        print("=" * 64)
        ok = eval_one(args, run_dir)
        runs.append({
            "seed": seed,
            "run_dir": os.path.abspath(run_dir),
            "status": "ok" if ok else "eval_failed",
        })

    # Manifest del experimento (para agregar entre seeds despues).
    stamp = time.strftime("%Y%m%d%H%M%S")
    manifest_path = os.path.abspath(os.path.join("models", f"experiment_{stamp}.json"))
    manifest = {
        "created": stamp,
        "n_stack": int(args.n_stack),
        "camera_only": bool(args.camera_only),
        "total_timesteps": int(args.total_timesteps),
        "webots_world": args.webots_world,
        "eval_mode": (
            f"{args.episodes} episodios/track" if args.episodes is not None
            else f"{args.laps} vueltas/track"
        ),
        "runs": runs,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    print("=" * 64)
    print("RESUMEN DEL EXPERIMENTO")
    for r in runs:
        print(f"  seed {r['seed']:>3}  {r['status']:<13}  {r['run_dir']}")
    print(f"Manifest: {manifest_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
