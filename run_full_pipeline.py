# Pipeline COMPLETO del paper en DOS FASES, sin intervencion manual:
#   FASE 1 (TRAIN): entrena TODAS las seeds de TODAS las variantes RL, cada una en su rama
#                   (git checkout -> rl.trainer por seed). Captura el run_dir creado por cada
#                   (variante, seed).
#   FASE 2 (EVAL):  cuando TERMINO todo el entrenamiento, evalua ESAS MISMAS run_dirs (cada una
#                   en su rama -> rl.evaluate) para sacar las metricas finales.
# Escribe models/pipeline_<ts>.json con el mapeo (variante, seed) -> run_dir + estado.
#
# Une run_all_experiments.py (train multi-rama) + run_all_evals.py (eval multi-rama), pero
# separando en fases: primero se entrena TODO y recien despues se evalua TODO (asi el eval
# final corre con los mismos flags para todos: p.ej. fondo aleatorio + seed reproducible).
#
# NO cubre la variante destilada (vision_distill): esa no se entrena con RL, se produce con
# rl.collect_teacher + rl.distill aparte.
#
# IMPORTANTE: corre desde la raiz del repo, con el working tree LIMPIO (hace git checkout).
#
# Uso:
#   python run_full_pipeline.py --seeds 0 1 2 3 4 --total-timesteps 1000000 --n-envs 4 \
#       --n-steps 256 --episodes 20 --randomize-background --eval-seed 0
#   python run_full_pipeline.py --seeds 0 --only geometrica vision_lstm --episodes 20

import argparse
import glob
import json
import os
import subprocess
import sys
import time

# (rama, n_stack, etiqueta) -- mismas 4 variantes RL que run_all_experiments.py
VARIANTS = [
    ("master",      1, "vision_1frame"),
    ("master",      4, "vision_stacked"),
    ("geometrica",  1, "geometrica"),
    ("vision_lstm", 1, "vision_lstm"),
]


def sh(cmd, check=True):
    print(">>", " ".join(cmd), flush=True)
    code = subprocess.run(cmd).returncode
    if check and code != 0:
        raise SystemExit(f"Comando fallo (code {code}): {' '.join(cmd)}")
    return code


def current_branch():
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    ).decode().strip()


def tracked_changes():
    out = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ).decode().strip()
    return out != ""


def list_run_dirs(models_dir):
    return {p for p in glob.glob(os.path.join(models_dir, "*")) if os.path.isdir(p)}


def detect_new_run_dir(before, models_dir):
    new = list_run_dirs(models_dir) - before
    return max(new, key=os.path.getmtime) if new else None


def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline train+eval en 2 fases (entrena todo, luego evalua todo)."
    )
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--only", nargs="+", default=None,
                   help="Subconjunto de variantes: vision_1frame vision_stacked geometrica vision_lstm.")
    p.add_argument("--models-dir", default="models")
    # --- Train (a rl.trainer) ---
    p.add_argument("--total-timesteps", type=int, default=1000000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--base-port", type=int, default=None)
    p.add_argument("--webots-world", default="worlds/track1.wbt")
    p.add_argument("--device", default="auto", help="Dispositivo del train.")
    # --- Eval (a rl.evaluate) ---
    p.add_argument("--episodes", type=int, default=None,
                   help="Modo tasa de exito: N episodios/track. Si falta, usa --laps.")
    p.add_argument("--laps", type=int, default=3)
    p.add_argument("--eval-device", default="cpu")
    p.add_argument("--randomize-background", action="store_true",
                   help="Eval con fondo aleatorio por episodio (test de robustez).")
    p.add_argument("--eval-seed", type=int, default=None,
                   help="Seed de reset (spawn+fondo) para que todas las seeds evaluen igual.")
    p.add_argument("--skip-eval", action="store_true", help="Solo FASE 1 (entrenar).")
    p.add_argument("--camera-only", action="store_true",
                   help="Ablacion vision pura: obs = SOLO imagen (sin propiocepcion de "
                        "velocidad). Fuerza la unica variante vision_camonly (rama master, "
                        "n_stack 1) y propaga CAMERA_ONLY=1 al train Y al eval por env var.")
    return p.parse_args()


def train_cmd(args, seed, n_stack):
    cmd = [
        sys.executable, "-m", "rl.trainer",
        "--seed", str(seed),
        "--total-timesteps", str(args.total_timesteps),
        "--n-stack", str(n_stack),
        "--webots-world", args.webots_world,
        "--n-envs", str(args.n_envs),
        "--device", args.device,
        "--no-discord",
    ]
    if args.n_steps is not None:
        cmd += ["--n-steps", str(args.n_steps)]
    if args.base_port is not None:
        cmd += ["--base-port", str(args.base_port)]
    return cmd


def eval_cmd(args, run_dir):
    cmd = [sys.executable, "-m", "rl.evaluate", "--model", run_dir, "--device", args.eval_device]
    if args.episodes is not None:
        cmd += ["--episodes", str(args.episodes)]
    else:
        cmd += ["--laps", str(args.laps)]
    if args.randomize_background:
        cmd += ["--randomize-background"]
    if args.eval_seed is not None:
        cmd += ["--eval-seed", str(args.eval_seed)]
    return cmd


def write_manifest(path, args, trained):
    manifest = {
        "created": time.strftime("%Y%m%d%H%M%S"),
        "total_timesteps": args.total_timesteps,
        "seeds": args.seeds,
        "eval_mode": (f"{args.episodes} episodios/track" if args.episodes is not None
                      else f"{args.laps} vueltas/track"),
        "randomized_background": bool(args.randomize_background),
        "eval_seed": args.eval_seed,
        "runs": trained,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def main():
    args = parse_args()
    if tracked_changes():
        raise SystemExit(
            "Hay cambios en archivos trackeados -> commitea/limpia antes de lanzar "
            "(el pipeline hace git checkout entre ramas)."
        )

    if args.camera_only:
        # Ablacion vision pura: se propaga por env var; subprocess.run hereda os.environ,
        # asi que rl.trainer (y sus workers) y rl.evaluate quedan todos camera-only.
        os.environ["CAMERA_ONLY"] = "1"
        print("[CAMERA-ONLY] obs = SOLO imagen (sin velocidad). Variante unica: "
              "vision_camonly (rama master, n_stack 1). Train y eval heredan el flag.")
        variants = [("master", 1, "vision_camonly")]
    elif args.only:
        variants = [v for v in VARIANTS if v[2] in args.only]
        if not variants:
            raise SystemExit(f"--only no matchea. Validas: {[v[2] for v in VARIANTS]}")
    else:
        variants = VARIANTS

    origin = current_branch()
    manifest_path = os.path.join(
        args.models_dir, f"pipeline_{time.strftime('%Y%m%d%H%M%S')}.json")
    trained = []  # [{variant, branch, n_stack, seed, run_dir, train_status, eval_status}]

    try:
        # ------------------------- FASE 1: TRAIN TODO -------------------------
        for branch, n_stack, label in variants:
            print("#" * 72)
            print(f"# [TRAIN] {label}  (rama {branch}, n_stack {n_stack})")
            print("#" * 72, flush=True)
            sh(["git", "checkout", branch])
            for seed in args.seeds:
                print(f"\n----- TRAIN {label} seed {seed} -----", flush=True)
                before = list_run_dirs(args.models_dir)
                code = sh(train_cmd(args, seed, n_stack), check=False)
                run_dir = detect_new_run_dir(before, args.models_dir) if code == 0 else None
                entry = {
                    "variant": label, "branch": branch, "n_stack": n_stack, "seed": seed,
                    "run_dir": run_dir,
                    "train_status": "ok" if (code == 0 and run_dir) else f"fallo({code})",
                    "eval_status": "pendiente",
                }
                trained.append(entry)
                write_manifest(manifest_path, args, trained)  # incremental

        # ------------------------- FASE 2: EVAL TODO --------------------------
        if not args.skip_eval:
            # Agrupar por rama para minimizar checkouts, preservando orden.
            order = []
            for e in trained:
                if e["run_dir"] and e["branch"] not in order:
                    order.append(e["branch"])
            for branch in order:
                print("#" * 72)
                print(f"# [EVAL] rama {branch}")
                print("#" * 72, flush=True)
                sh(["git", "checkout", branch])
                for e in trained:
                    if e["branch"] != branch or not e["run_dir"]:
                        continue
                    print(f"\n----- EVAL {e['variant']} seed {e['seed']}  "
                          f"({os.path.basename(e['run_dir'])}) -----", flush=True)
                    code = sh(eval_cmd(args, e["run_dir"]), check=False)
                    e["eval_status"] = "ok" if code == 0 else f"fallo({code})"
                    write_manifest(manifest_path, args, trained)  # incremental
    finally:
        print(f"\nVolviendo a la rama original: {origin}")
        sh(["git", "checkout", origin], check=False)
        write_manifest(manifest_path, args, trained)

    print("#" * 72)
    print("RESUMEN DEL PIPELINE")
    for e in trained:
        rd = os.path.basename(e["run_dir"]) if e["run_dir"] else "-"
        print(f"  {e['variant']:<16} seed {e['seed']}  {rd:<16} "
              f"train={e['train_status']:<10} eval={e['eval_status']}")
    print(f"\nManifest: {manifest_path}")
    print("#" * 72)


if __name__ == "__main__":
    main()
