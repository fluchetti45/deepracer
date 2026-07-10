# Evalua MULTIPLES modelos, cada uno en la rama que le corresponde, sin intervencion manual.
# Espejo de run_all_experiments.py pero para EVAL en vez de train: por modelo hace
# git checkout <su rama> -> rl.evaluate. Agrupa por rama para minimizar checkouts y al
# terminar vuelve a la rama original.
#
# Por que hace falta la rama: el env de eval es ESPECIFICO de cada rama (vision -> obs de
# imagen Dict; geometrica -> vector de features). El VecNormalize del modelo solo casa con
# el observation_space de SU rama, asi que hay que estar parado ahi para evaluarlo.
#
# La rama de cada modelo se lee de models/<id>/run_metadata.json ("git_branch"); si falta,
# se infiere de "variant" con VARIANT_BRANCH.
#
# IMPORTANTE: corre desde la raiz del repo, con el working tree LIMPIO (hace git checkout).
# El proceso queda en memoria, asi que cambiar de rama no lo afecta.
#
# Uso:
#   python run_all_evals.py --models 20260706164309 20260706192903 20260707005917 \
#       20260708125635 20260706221752 --episodes 20 --randomize-background --eval-seed 0
#   python run_all_evals.py --models <id1> <id2> --laps 3            # modo vueltas
#   python run_all_evals.py --models <id>                            # fondo fijo, sin seed

import argparse
import glob
import json
import os
import subprocess
import sys

# Fallback variant -> rama, por si run_metadata no trae "git_branch".
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
    """True si hay cambios en archivos TRACKEADOS (lo que bloquearia un checkout)."""
    out = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ).decode().strip()
    return out != ""


def resolve_model_dir(arg, models_dir):
    """Acepta un run_dir (models/<id>), una ruta absoluta, o un id suelto (<id>)."""
    for cand in (arg, os.path.join(models_dir, arg)):
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    raise SystemExit(f"No encuentro el run_dir del modelo: {arg}")


def resolve_branch(model_dir):
    """Rama del modelo: run_metadata['git_branch'] o, si falta, VARIANT_BRANCH[variant]."""
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
    raise SystemExit(
        f"No puedo determinar la rama de {model_dir} "
        f"(sin git_branch ni variant conocido en run_metadata.json)."
    )


def discover_models(models_dir, variants, timesteps):
    """run_dir mas nuevo por (variant, seed) de las variantes pedidas. RL filtra timesteps;
    las destiladas (regime distill) se toman sin importar timesteps."""
    best = {}  # (variant, seed) -> (run_dir, mtime)
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
        hp = m.get("hyperparameters") or {}
        is_distill = m.get("training_regime") == "distill"
        if not is_distill and timesteps is not None and hp.get("total_timesteps") != timesteps:
            continue
        seed = hp.get("seed")
        key = (v, seed)
        mt = os.path.getmtime(d)
        if key not in best or mt > best[key][1]:
            best[key] = (d, mt)
    # ordenar por variante (orden de --variants) y luego seed
    ordered = sorted(best.items(), key=lambda kv: (variants.index(kv[0][0]),
                                                    (kv[0][1] if kv[0][1] is not None else -1)))
    return [os.path.normpath(run_dir) for (_key, (run_dir, _mt)) in ordered]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evalua varios modelos, cada uno en su rama git (switcheo automatico)."
    )
    p.add_argument("--models", nargs="+", default=None,
                   help="run_dirs o ids bajo models/ a evaluar (ej: 20260706164309 ...). "
                        "Si se omite, usar --discover.")
    p.add_argument("--discover", action="store_true",
                   help="Descubrir automaticamente los modelos a evaluar (5 variantes finales, "
                        "el run_dir mas nuevo por seed). RL filtrados por --timesteps.")
    p.add_argument("--variants", nargs="+",
                   default=["vision_1frame", "vision_stacked", "vision_lstm",
                            "geometrica", "vision_distill"],
                   help="Variantes a descubrir con --discover.")
    p.add_argument("--timesteps", type=int, default=1000000,
                   help="Filtro de total_timesteps para las variantes RL en --discover "
                        "(las destiladas se toman sin importar timesteps).")
    p.add_argument("--models-dir", default="models")
    # --- Pasan tal cual a rl.evaluate ---
    p.add_argument("--episodes", type=int, default=None,
                   help="Modo tasa de exito: N episodios/track. Si falta, usa --laps.")
    p.add_argument("--laps", type=int, default=3, help="Vueltas/track si NO se usa --episodes.")
    p.add_argument("--device", default="cpu", help="Dispositivo de PyTorch para el eval.")
    p.add_argument("--randomize-background", action="store_true",
                   help="Rota pared+skybox por episodio en eval (test de robustez).")
    p.add_argument("--eval-seed", type=int, default=None,
                   help="Seed de reset (spawn+fondo) para que todos evaluen la misma secuencia.")
    return p.parse_args()


def eval_cmd(args, model_dir):
    cmd = [sys.executable, "-m", "rl.evaluate", "--model", model_dir, "--device", args.device]
    if args.episodes is not None:
        cmd += ["--episodes", str(args.episodes)]
    else:
        cmd += ["--laps", str(args.laps)]
    if args.randomize_background:
        cmd += ["--randomize-background"]
    if args.eval_seed is not None:
        cmd += ["--eval-seed", str(args.eval_seed)]
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
            raise SystemExit("Pasa --models <ids> o --discover para auto-descubrir.")
        models = discover_models(args.models_dir, args.variants, args.timesteps)
        if not models:
            raise SystemExit("--discover no encontro modelos para las variantes pedidas.")
        print(f"[discover] {len(models)} modelos:")
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

    origin = current_branch()
    results = []
    try:
        for branch in order:
            print("#" * 72)
            print(f"# RAMA {branch}  ({len(groups[branch])} modelo/s)")
            print("#" * 72, flush=True)
            sh(["git", "checkout", branch])
            for model_dir in groups[branch]:
                print(f"\n----- eval {labels[model_dir]}  ({os.path.basename(model_dir)}) -----",
                      flush=True)
                code = sh(eval_cmd(args, model_dir), check=False)
                results.append((labels[model_dir], os.path.basename(model_dir), branch,
                                "ok" if code == 0 else f"fallo({code})"))
    finally:
        print(f"\nVolviendo a la rama original: {origin}")
        sh(["git", "checkout", origin], check=False)

    print("#" * 72)
    print("RESUMEN DE EVAL")
    for variant, run_id, branch, status in results:
        print(f"  {variant:<20} {run_id:<16} {branch:<14} {status}")
    print("#" * 72)


if __name__ == "__main__":
    main()
