# Graba videos demo de los modelos, UNO DETRAS DE OTRO, cada uno en la rama que le
# corresponde (switch de git automatico), igual que run_all_evals pero grabando el
# viewport 3D de Webots en vez de solo medir. Por defecto graba UNA vuelta en UNA pista
# (track9) por modelo: cuando completa la vuelta, cierra el mp4 y pasa al siguiente. Los
# mp4 quedan en --record-dir con nombre <variante>_s<seed>_<track>_<laps>laps.mp4.
#
# Requiere: Webots (se lanza CON render), working tree LIMPIO (hace git checkout), y que
# la rama de cada modelo tenga el soporte de grabacion (movie handlers + --record-movie).
# NO pisa el eval del paper (--record-movie implica --no-save-results).
#
# Uso:
#   python record_demos.py --discover                        # 6 variantes, seed 0, 1 vuelta en track9
#   python record_demos.py --discover --track track10.png    # una vuelta en la otra pista
#   python record_demos.py --discover --track all --laps 3   # las 2 pistas, 3 vueltas c/u
#   python record_demos.py --models 20260713120404 20260714162646 ...

import argparse
import glob
import json
import os
import subprocess
import sys

VARIANT_BRANCH = {
    "vision_1frame": "master",
    "vision_camonly": "master",
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
                   default=["vision_1frame", "vision_camonly", "vision_stacked",
                            "vision_lstm", "geometrica", "vision_distill"],
                   help="Variantes a grabar con --discover.")
    p.add_argument("--seed", type=int, default=0, help="Seed a grabar (--discover).")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--record-dir", default="videos",
                   help="Carpeta de salida de los mp4.")
    p.add_argument("--track", default="track9.png",
                   help="Pista a grabar (una sola). Con '' o 'all' graba TODAS las de eval. "
                        "Default: track9.png (una vuelta en una pista por modelo).")
    p.add_argument("--laps", type=int, default=1, help="Vueltas a grabar por modelo (default 1).")
    p.add_argument("--speed", type=int, default=6,
                   help="Aceleracion CONSTANTE del video (Nx la sim, fija). Default 6.")
    p.add_argument("--cam-height", type=float, default=13.0,
                   help="Altura de la camara cenital sobre la pista. Default 13.")
    p.add_argument("--no-topdown", action="store_true",
                   help="Grabar con la vista 3D por defecto en vez de la camara cenital.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-seed", type=int, default=0,
                   help="Seed de reset (spawn+fondo) para que todos arranquen igual. Default 0.")
    p.add_argument("--randomize-background", action="store_true",
                   help="Fondo aleatorio por episodio (test de robustez). Default: fondo fijo.")
    p.add_argument("--checkout", action="store_true",
                   help="Cambiar de rama por modelo (comportamiento viejo). Por defecto graba "
                        "TODO en la rama actual (master) cargando cada policy; el geometrico se "
                        "saltea porque su obs de features no se calcula en master.")
    return p.parse_args()


def record_cmd(args, model_dir):
    cmd = [sys.executable, "-m", "rl.evaluate", "--model", model_dir,
           "--laps", str(args.laps), "--record-movie", args.record_dir,
           "--device", args.device, "--eval-seed", str(args.eval_seed),
           "--record-speed", str(args.speed), "--record-height", str(args.cam_height)]
    if args.no_topdown:
        cmd += ["--no-record-topdown"]
    # Una sola pista por modelo (default): graba una vuelta y pasa al siguiente. Con
    # --track '' o 'all' se omite y evaluate graba TODAS las pistas de eval.
    if args.track and args.track.lower() != "all":
        cmd += ["--track", args.track]
    if args.randomize_background:
        cmd += ["--randomize-background"]
    return cmd


def run_record(args, model_dir, variant):
    """Graba un modelo. Para vision_camonly setea CAMERA_ONLY=1 (obs = solo imagen) en el
    subprocess; subprocess.run hereda os.environ. Restaura el valor previo al terminar."""
    prev = os.environ.get("CAMERA_ONLY")
    if variant == "vision_camonly":
        os.environ["CAMERA_ONLY"] = "1"
    else:
        os.environ.pop("CAMERA_ONLY", None)
    try:
        return sh(record_cmd(args, model_dir), check=False)
    finally:
        if prev is None:
            os.environ.pop("CAMERA_ONLY", None)
        else:
            os.environ["CAMERA_ONLY"] = prev


def main():
    args = parse_args()
    # El working tree limpio solo hace falta si vamos a cambiar de rama (--checkout).
    if args.checkout and tracked_changes():
        raise SystemExit(
            "Hay cambios en archivos trackeados -> commitea/limpia antes de lanzar "
            "con --checkout (hace git checkout entre ramas)."
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

    os.makedirs(args.record_dir, exist_ok=True)
    origin = current_branch()
    resolved = [(resolve_model_dir(m, args.models_dir),) for m in models]
    resolved = [(d, resolve_branch(d)[1]) for (d,) in resolved]  # (dir, variant)
    results = []

    if not args.checkout:
        # ---- DEFAULT: todo en la rama actual (master), sin git checkout ----
        # Los modelos de vision (imagen; incl. LSTM y camera-only) cargan y corren aca.
        # El geometrico NO: su obs de 9 features no la produce el env de master.
        print(f"# Grabando en la rama actual ({origin}) sin cambiar de rama.")
        for model_dir, variant in resolved:
            rid = os.path.basename(model_dir)
            if variant == "geometrica" and origin != "geometrica":
                print(f"\n[skip] {variant} ({rid}): su obs de features no existe en "
                      f"'{origin}'. Grabalo con --checkout, o parado en la rama geometrica.")
                results.append((variant, rid, origin, "salteado (obs no disponible)"))
                continue
            print(f"\n----- grabando {variant}  ({rid}) -----", flush=True)
            code = run_record(args, model_dir, variant)
            results.append((variant, rid, origin, "ok" if code == 0 else f"fallo({code})"))
    else:
        # ---- Viejo: cada modelo en su rama (git checkout), agrupado por rama ----
        groups, vmap = {}, {}
        for model_dir, variant in resolved:
            branch = resolve_branch(model_dir)[0]
            groups.setdefault(branch, []).append(model_dir)
            vmap[model_dir] = variant
        order = list(dict.fromkeys(resolve_branch(d)[0] for d, _ in resolved))
        try:
            for branch in order:
                print("#" * 72)
                print(f"# RAMA {branch}  ({len(groups[branch])} modelo/s)")
                print("#" * 72, flush=True)
                sh(["git", "checkout", branch])
                for model_dir in groups[branch]:
                    variant = vmap[model_dir]; rid = os.path.basename(model_dir)
                    print(f"\n----- grabando {variant}  ({rid}) -----", flush=True)
                    code = run_record(args, model_dir, variant)
                    results.append((variant, rid, branch, "ok" if code == 0 else f"fallo({code})"))
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
