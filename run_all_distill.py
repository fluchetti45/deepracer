# Destilacion APAREADA multi-seed: por cada seed, colecta un dataset (limpio + DART) del
# geometrico maestro de ESA seed y destila un estudiante de vision a partir de ese dataset.
# Asi las 5 seeds destiladas son corridas INDEPENDIENTES (maestro + colecta + destilacion
# distintos por seed) -> error bars honestos y comparables con las 5 seeds de las RL.
#
# Maestro por seed: se auto-descubre en models/ (variant=="geometrica", total_timesteps ==
# --teacher-timesteps), tomando el run_dir MAS NUEVO de cada seed. Se puede fijar a mano con
# --teachers seed=run_dir ...
#
# Por seed hace, en orden:
#   1) rl.collect_teacher --noise-std 0.0   (limpio) -> data/teacher_s<seed>
#   2) rl.collect_teacher --noise-std 0.15  (DART)   -> data/teacher_s<seed>   (--no-dart lo saltea)
#   3) rl.distill --data data/teacher_s<seed> --seeds <seed>                    (estudiante)
#   4) (opcional --eval) rl.evaluate del estudiante con --randomize-background --eval-seed
#
# NO cambia de rama: corre todo en la rama vision_distill (collect_teacher calcula las
# features geometricas del maestro desde la RGB, y distill es supervisado).
#
# IMPORTANTE: corre desde la raiz del repo. collect_teacher y evaluate levantan Webots.
#
# Uso:
#   python run_all_distill.py --seeds 0 1 2 3 4 --episodes 40
#   python run_all_distill.py --seeds 0 1 2 3 4 --episodes 40 --eval --eval-seed 0
#   python run_all_distill.py --seeds 0 --no-dart            # solo limpio, una seed

import argparse
import glob
import json
import os
import subprocess
import sys
import time


def sh(cmd, check=True):
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    code = subprocess.run([str(c) for c in cmd]).returncode
    if check and code != 0:
        raise SystemExit(f"Comando fallo (code {code}): {' '.join(str(c) for c in cmd)}")
    return code


def list_run_dirs(models_dir):
    return {p for p in glob.glob(os.path.join(models_dir, "*")) if os.path.isdir(p)}


def detect_new_run_dir(before, models_dir):
    new = list_run_dirs(models_dir) - before
    return max(new, key=os.path.getmtime) if new else None


def discover_geometric_teachers(models_dir, teacher_timesteps):
    """{seed: run_dir} del geometrico mas nuevo de cada seed (variant geometrica)."""
    by_seed = {}
    for d in glob.glob(os.path.join(models_dir, "*")):
        mf = os.path.join(d, "run_metadata.json")
        if not os.path.isdir(d) or not os.path.exists(mf):
            continue
        try:
            m = json.load(open(mf, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("variant") != "geometrica":
            continue
        hp = m.get("hyperparameters") or {}
        if teacher_timesteps is not None and hp.get("total_timesteps") != teacher_timesteps:
            continue
        seed = hp.get("seed")
        if seed is None:
            continue
        if seed not in by_seed or os.path.getmtime(d) > os.path.getmtime(by_seed[seed]):
            by_seed[seed] = d
    return by_seed


def parse_teacher_overrides(items):
    """--teachers 0=models/xxx 1=models/yyy -> {0: 'models/xxx', ...}"""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--teachers espera seed=run_dir, recibi: {it}")
        k, v = it.split("=", 1)
        out[int(k)] = v
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Destilacion apareada multi-seed (colecta + destila por maestro geometrico)."
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--models-dir", default="models")
    p.add_argument("--data-root", default="data",
                   help="Los datasets van a <data-root>/teacher_s<seed>.")
    p.add_argument("--teacher-timesteps", type=int, default=1000000,
                   help="Filtro de timesteps del maestro geometrico a auto-descubrir.")
    p.add_argument("--teachers", nargs="+", default=None,
                   help="Override manual: seed=run_dir (ej: 0=models/20260706221752).")
    # --- Colecta ---
    p.add_argument("--episodes", type=int, default=40, help="Episodios por colecta.")
    p.add_argument("--clean-noise", type=float, default=0.0)
    p.add_argument("--dart-noise", type=float, default=0.15)
    p.add_argument("--no-dart", action="store_true", help="Solo colecta limpia (sin DART).")
    # --- Destilacion ---
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--variant", default="vision_distill")
    p.add_argument("--device", default="auto", help="Device de la destilacion.")
    # --- Eval opcional ---
    p.add_argument("--eval", action="store_true",
                   help="Tras destilar cada estudiante, evaluarlo (levanta Webots).")
    p.add_argument("--episodes-eval", type=int, default=20)
    p.add_argument("--randomize-background", action="store_true", default=True,
                   help="Eval con fondo aleatorio (default True; --no-randomize-background para apagar).")
    p.add_argument("--no-randomize-background", dest="randomize_background", action="store_false")
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--eval-device", default="cpu")
    # --- Webots (colecta/eval) ---
    p.add_argument("--webots-world", default="worlds/track1.wbt")
    p.add_argument("--webots-executable", default=None)
    return p.parse_args()


def collect_cmd(args, teacher_dir, data_dir, noise, seed):
    cmd = [sys.executable, "-m", "rl.collect_teacher",
           "--teacher", teacher_dir, "--out", data_dir,
           "--episodes", args.episodes, "--noise-std", noise, "--seed", seed,
           "--webots-world", args.webots_world]
    if args.webots_executable:
        cmd += ["--webots-executable", args.webots_executable]
    return cmd


def distill_cmd(args, data_dir, seed, teacher_dir):
    return [sys.executable, "-m", "rl.distill",
            "--data", data_dir, "--seeds", seed,
            "--epochs", args.epochs, "--batch-size", args.batch_size,
            "--variant", args.variant, "--teacher-id", os.path.basename(teacher_dir),
            "--device", args.device]


def eval_cmd(args, student_dir):
    cmd = [sys.executable, "-m", "rl.evaluate", "--model", student_dir,
           "--device", args.eval_device, "--episodes", args.episodes_eval,
           "--eval-seed", args.eval_seed]
    if args.randomize_background:
        cmd += ["--randomize-background"]
    return cmd


def main():
    args = parse_args()

    teachers = discover_geometric_teachers(args.models_dir, args.teacher_timesteps)
    teachers.update(parse_teacher_overrides(args.teachers))

    missing = [s for s in args.seeds if s not in teachers]
    if missing:
        raise SystemExit(
            f"No encuentro maestro geometrico para seed(s) {missing}. "
            f"Descubiertos: { {k: os.path.basename(v) for k, v in sorted(teachers.items())} }. "
            f"Fijalos con --teachers seed=run_dir."
        )

    print("Maestros por seed:")
    for s in args.seeds:
        print(f"  seed {s} -> {os.path.basename(teachers[s])}")

    results = []
    for seed in args.seeds:
        teacher_dir = teachers[seed]
        data_dir = os.path.join(args.data_root, f"teacher_s{seed}")
        print("#" * 72)
        print(f"# SEED {seed}  (maestro {os.path.basename(teacher_dir)}) -> {data_dir}")
        print("#" * 72, flush=True)

        # 1) colecta limpia
        sh(collect_cmd(args, teacher_dir, data_dir, args.clean_noise, seed))
        # 2) colecta DART
        if not args.no_dart:
            sh(collect_cmd(args, teacher_dir, data_dir, args.dart_noise, seed))
        # 3) destilar (capturar el run_dir nuevo)
        before = list_run_dirs(args.models_dir)
        sh(distill_cmd(args, data_dir, seed, teacher_dir))
        student_dir = detect_new_run_dir(before, args.models_dir)
        entry = {"seed": seed, "teacher": os.path.basename(teacher_dir),
                 "data_dir": data_dir,
                 "student": os.path.basename(student_dir) if student_dir else None,
                 "eval": "no"}
        # 4) eval opcional
        if args.eval and student_dir:
            code = sh(eval_cmd(args, student_dir), check=False)
            entry["eval"] = "ok" if code == 0 else f"fallo({code})"
        results.append(entry)

    print("#" * 72)
    print("RESUMEN DESTILACION APAREADA")
    for e in results:
        st = e["student"] or "-"
        print(f"  seed {e['seed']}  maestro {e['teacher']:<16} -> estudiante {st:<16} eval={e['eval']}")
    print("#" * 72)


if __name__ == "__main__":
    main()
