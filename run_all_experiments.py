# Lanza el experimento COMPLETO del paper: las 4 variantes, cada una en su rama, sin
# intervencion manual. Por variante: git checkout <rama> -> rl.run_experiment (que entrena
# las N seeds y evalua cada una). Al terminar vuelve a la rama original.
#
# Las 4 variantes:
#   vision_1frame   -> master,      --n-stack 1
#   vision_stacked  -> master,      --n-stack 4
#   geometrica      -> geometrica,  --n-stack 1
#   vision_lstm     -> vision_lstm, --n-stack 1
#
# IMPORTANTE: corre desde la raiz del repo, con el working tree LIMPIO (hace git checkout
# entre variantes). El proceso queda en memoria, asi que cambiar de rama no lo afecta.
#
# Uso:
#   python run_all_experiments.py --seeds 0 1 2 3 4 --total-timesteps 1000000 --n-envs 4 --episodes 10
#   python run_all_experiments.py --seeds 0 1 2 3 4 --only geometrica vision_lstm   # subconjunto

import argparse
import subprocess
import sys

# (rama, n_stack, etiqueta)
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
    """True si hay cambios en archivos TRACKEADOS (lo que bloquearia un checkout)."""
    out = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ).decode().strip()
    return out != ""


def parse_args():
    p = argparse.ArgumentParser(
        description="Lanza las 4 variantes del paper, cada una en su rama."
    )
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--total-timesteps", type=int, default=1000000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device", default="auto", help="Dispositivo del train.")
    p.add_argument("--eval-device", default="cpu", help="Dispositivo del eval.")
    p.add_argument(
        "--only", nargs="+", default=None,
        help="Correr solo estas etiquetas: vision_1frame vision_stacked geometrica vision_lstm.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if tracked_changes():
        raise SystemExit(
            "Hay cambios en archivos trackeados -> commitea/limpia antes de lanzar "
            "(el script hace git checkout entre variantes)."
        )

    variants = VARIANTS
    if args.only:
        variants = [v for v in VARIANTS if v[2] in args.only]
        if not variants:
            raise SystemExit(f"--only no matchea ninguna variante. Validas: "
                             f"{[v[2] for v in VARIANTS]}")

    origin = current_branch()
    results = []
    try:
        for branch, n_stack, label in variants:
            print("#" * 72)
            print(f"# VARIANTE {label}  (rama {branch}, n_stack {n_stack})")
            print("#" * 72, flush=True)
            sh(["git", "checkout", branch])
            cmd = [
                sys.executable, "-m", "rl.run_experiment",
                "--seeds", *map(str, args.seeds),
                "--total-timesteps", str(args.total_timesteps),
                "--n-stack", str(n_stack),
                "--n-envs", str(args.n_envs),
                "--device", args.device,
                "--eval-device", args.eval_device,
                "--episodes", str(args.episodes),
            ]
            if args.n_steps is not None:
                cmd += ["--n-steps", str(args.n_steps)]
            code = sh(cmd, check=False)
            results.append((label, branch, "ok" if code == 0 else f"fallo({code})"))
    finally:
        print(f"\nVolviendo a la rama original: {origin}")
        sh(["git", "checkout", origin], check=False)

    print("#" * 72)
    print("RESUMEN DEL EXPERIMENTO COMPLETO")
    for label, branch, status in results:
        print(f"  {label:<16} {branch:<14} {status}")
    print("#" * 72)


if __name__ == "__main__":
    main()
