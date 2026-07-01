"""
Análisis del paper en UN comando: descubre las corridas en models/, aplica el guard
anti-mezcla (500k piloto vs 1M final) UNA sola vez, y corre sobre el MISMO conjunto:
  - aggregate_eval  -> results_summary.{json,md}   (evaluación held-out, tablas del paper)
  - parse_tensorboard -> curves.json, sample_efficiency.json, fig_reward.png, fig_lap_rate.png

Uso:
  python -m analysis.run_analysis                       # auto: elige el grupo de mayor largo
  python -m analysis.run_analysis --timesteps 1000000   # fija la corrida final
  python -m analysis.run_analysis --variants geometrica vision_1frame
  python -m analysis.run_analysis --allow-mixed         # poolea distintos largos (con warning)
"""

import argparse

from analysis import aggregate_eval, parse_tensorboard
from analysis.discover import discover_runs, select_runs, format_report


def parse_args():
    p = argparse.ArgumentParser(description="Corre todo el análisis del paper de una.")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--out-dir", default="analysis")
    p.add_argument("--timesteps", type=int, default=None,
                   help="Fijar el largo exacto de las corridas (guard anti-mezcla).")
    p.add_argument("--variants", nargs="+", default=None, help="Subconjunto de variantes.")
    p.add_argument("--allow-mixed", action="store_true",
                   help="Poolear runs de distinto largo (por defecto NO).")
    return p.parse_args()


def main():
    args = parse_args()
    runs = discover_runs(args.models_dir)
    if not runs:
        print(f"No hay corridas con run_metadata.json en {args.models_dir}/.")
        return
    sel, report = select_runs(runs, timesteps=args.timesteps, variants=args.variants,
                              allow_mixed=args.allow_mixed)
    print("=" * 72)
    print("SELECCION DE CORRIDAS")
    print(format_report(report))
    print("=" * 72)

    # 1) Evaluación (tablas del paper).
    eval_dirs = [r["run_dir"] for r in sel if r["has_eval"]]
    print(f"\n[eval] {len(eval_dirs)} corridas con eval_results")
    if eval_dirs:
        _, md = aggregate_eval.run(eval_dirs, args.out_dir)
        print(md)
    else:
        print("  (ninguna evaluada aún; se saltea results_summary)")

    # 2) Curvas de entrenamiento (TensorBoard).
    tb_dirs = [r["run_dir"] for r in sel if r["has_tb"]]
    print(f"\n[train] {len(tb_dirs)} corridas con event files de TensorBoard")
    if tb_dirs:
        curves, se = parse_tensorboard.run(tb_dirs, args.out_dir)
        if curves is not None:
            parse_tensorboard.print_summary(curves, se)
    else:
        print("  (sin event files; se saltean curvas)")

    print("\n" + "=" * 72)
    print("Escrito en", args.out_dir + "/:")
    for f in ("results_summary.json", "results_summary.md", "curves.json",
              "sample_efficiency.json", "fig_reward.png", "fig_lap_rate.png"):
        print("  -", f)
    print("=" * 72)


if __name__ == "__main__":
    main()
