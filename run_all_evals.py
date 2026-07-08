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


def parse_args():
    p = argparse.ArgumentParser(
        description="Evalua varios modelos, cada uno en su rama git (switcheo automatico)."
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="run_dirs o ids bajo models/ a evaluar (ej: 20260706164309 ...).")
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

    # Resolver cada modelo -> (dir, rama) y agrupar por rama preservando el orden.
    order, groups, labels = [], {}, {}
    for m in args.models:
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
