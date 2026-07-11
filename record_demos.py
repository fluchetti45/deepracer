# Graba videos demo de los modelos, UNO DETRAS DE OTRO, cada uno en la rama que le
# corresponde (switch de git automatico), igual que run_all_evals pero grabando el
# viewport 3D de Webots en vez de solo medir. Por cada modelo graba las DOS pistas de
# test (track9, track10) con --laps vueltas cada una; los mp4 quedan en --record-dir
# con nombre <variante>_s<seed>_<track>_<laps>laps.mp4.
#
# Requiere: Webots (se lanza CON render), working tree LIMPIO (hace git checkout), y que
# la rama de cada modelo tenga el soporte de grabacion (movie handlers + --record-movie).
# NO pisa el eval del paper (--record-movie implica --no-save-results).
#
# Uso:
#   python record_demos.py --discover                    # 5 variantes, seed 0, ambos mapas
#   python record_demos.py --discover --seed 0 --laps 3
#   python record_demos.py --models 20260706164309 20260706221752 ...
#   python record_demos.py --discover --randomize-background   # fondo aleatorio (robustez)

import argparse
import glob
import json
import os
import subprocess
import sys

VARIANT_BRANCH = {
    "vision_1frame": "master",
    "vision_stacked": "master",
    "geometrica": "geometrica",
    "vision_lstm": "vision_lstm",
    "vision_distill": "vision_distill",
    "vision_distill_dart": "vision_distill",
}


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


def resolve_model_dir(arg, models_dir):
    for cand in (arg, os.path.join(models_dir, arg)):
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    raise SystemExit(f"No encuentro el run_dir del modelo: {arg}")


def resolve_branch(model_dir):
    meta_path = os.path.join(model_dir, "run_metadata.json")
    branch, variant = None, None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        branch = (meta.get("git_branch") or "").strip() or None
        variant = meta.get("variant")
    except (OSError, ValueError):
        pass
    if branch:
        return branch, variant
    if variant in VARIANT_BRANCH:
        return VARIANT_BRANCH[variant], variant
    raise SystemExit(f"No puedo determinar la rama de {model_dir}.")


def discover_models(models_dir, variants, seed):
    """El run_dir mas nuevo por variante para la seed pedida (una seed por variante)."""
    best = {}  # variant -> (run_dir, mtime)
    for d in glob.glob(os.path.join(models_dir, "*")):
        mf = os.path.join(d, "run_metadata.json")
        if not os.path.isdir(d) or not os.path.exists(mf):
            continue
        try:
            m = json.load(open(mf, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        v = m.get("variant")
        if v not in variants:
            continue
        if (m.get("hyperparameters") or {}).get("seed") != seed:
            continue
        mt = os.path.getmtime(d)
        if v not in best or mt > best[v][1]:
            best[v] = (d, mt)
    missing = [v for v in variants if v not in best]
    if missing:
        print(f"[discover] OJO: sin modelo seed {seed} para: {', '.join(missing)}")
    return [os.path.normpath(best[v][0]) for v in variants if v in best]


def parse_args():
    p = argparse.ArgumentParser(
        description="Graba videos demo de los modelos, cada uno en su rama (switch automatico)."
    )
    p.add_argument("--models", nargs="+", default=None,
                   help="run_dirs o ids a grabar. Si se omite, usar --discover.")
    p.add_argument("--discover", action="store_true",
                   help="Auto-descubrir 1 modelo por variante (seed --seed).")
    p.add_argument("--variants", nargs="+",
                   default=["vision_1frame", "vision_stacked", "vision_lstm",
                            "geometrica", "vision_distill"],
                   help="Variantes a grabar con --discover.")
    p.add_argument("--seed", type=int, default=0, help="Seed a grabar (--discover).")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--record-dir", default="videos",
                   help="Carpeta de salida de los mp4.")
    p.add_argument("--laps", type=int, default=3, help="Vueltas por mapa (default 3).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-seed", type=int, default=0,
                   help="Seed de reset (spawn+fondo) para que todos arranquen igual. Default 0.")
    p.add_argument("--randomize-background", action="store_true",
                   help="Fondo aleatorio por episodio (test de robustez). Default: fondo fijo.")
    return p.parse_args()


def record_cmd(args, model_dir):
    cmd = [sys.executable, "-m", "rl.evaluate", "--model", model_dir,
           "--laps", str(args.laps), "--record-movie", args.record_dir,
           "--device", args.device, "--eval-seed", str(args.eval_seed)]
    if args.randomize_background:
        cmd += ["--randomize-background"]
    return cmd


def main():
    args = parse_args()
    if tracked_changes():
        raise SystemExit(
            "Hay cambios en archivos trackeados -> commitea/limpia antes de lanzar "
            "(el script hace git checkout entre ramas)."
        )

    models = args.models
    if not models:
        if not args.discover:
            raise SystemExit("Pasa --models <ids> o --discover.")
        models = discover_models(args.models_dir, args.variants, args.seed)
        if not models:
            raise SystemExit("--discover no encontro modelos.")
        print(f"[discover] {len(models)} modelos (seed {args.seed}):")
        for mm in models:
            print(f"    {os.path.basename(mm)}")

    # Resolver cada modelo -> (dir, rama) y agrupar por rama preservando el orden.
    order, groups, labels = [], {}, {}
    for m in models:
        d = resolve_model_dir(m, args.models_dir)
        branch, variant = resolve_branch(d)
        if branch not in groups:
            groups[branch] = []
            order.append(branch)
        groups[branch].append(d)
        labels[d] = variant or os.path.basename(d)

    os.makedirs(args.record_dir, exist_ok=True)
    origin = current_branch()
    results = []
    try:
        for branch in order:
            print("#" * 72)
            print(f"# RAMA {branch}  ({len(groups[branch])} modelo/s)")
            print("#" * 72, flush=True)
            sh(["git", "checkout", branch])
            for model_dir in groups[branch]:
                print(f"\n----- grabando {labels[model_dir]}  "
                      f"({os.path.basename(model_dir)}) -----", flush=True)
                code = sh(record_cmd(args, model_dir), check=False)
                results.append((labels[model_dir], os.path.basename(model_dir), branch,
                                "ok" if code == 0 else f"fallo({code})"))
    finally:
        print(f"\nVolviendo a la rama original: {origin}")
        sh(["git", "checkout", origin], check=False)

    print("#" * 72)
    print("RESUMEN DE GRABACION")
    for variant, run_id, branch, status in results:
        print(f"  {variant:<20} {run_id:<16} {branch:<14} {status}")
    print(f"\nVideos en: {os.path.abspath(args.record_dir)}")
    print("#" * 72)


if __name__ == "__main__":
    main()
