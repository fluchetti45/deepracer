"""
Agrega las evaluaciones de TODAS las corridas en models/ y produce las estadisticas del
paper (media +/- desvio sobre seeds, por variante). NO entrena ni evalua: solo lee los
artefactos ya generados (run_metadata.json + eval_results_*.json).

Identificacion de la variante (acordado con el setup de los datos):
  - device == "cpu"               -> geometrica  (features de imagen, MlpPolicy)
  - device == "cuda" & n_stack==1 -> vision      (1 frame, CNN)
  - device == "cuda" & n_stack==4 -> vision_stacked (4 frames apilados, CNN)

La seleccion de que runs entran (filtros + guard anti-mezcla 500k/1M) vive en
analysis/discover.py, compartida con parse_tensorboard para analizar el MISMO conjunto.

Salidas (en analysis/):
  - results_summary.json : agregados + por-seed (machine-readable)
  - results_summary.md   : tablas markdown listas para pegar en PAPER.md
Ademas imprime un resumen por consola.

Uso:  python -m analysis.aggregate_eval [--timesteps N] [--variants ...] [--allow-mixed]
"""

import argparse
import glob
import itertools
import json
import os
import statistics as st

from analysis.discover import (
    VARIANT_ORDER, VARIANT_LABEL, classify_variant,
    discover_runs, select_runs, format_report,
)

MODELS_DIR = "models"
OUT_DIR = "analysis"


def load_runs(run_dirs):
    """
    Lee los run dirs DADOS (ya filtrados por discover): metadata + eval_results mas reciente.
    Devuelve lista de dicts. Los que no tienen eval_results se saltean (aun no evaluados).
    """
    runs = []
    for run_dir in run_dirs:
        meta_path = os.path.join(run_dir, "run_metadata.json")
        evals = sorted(glob.glob(os.path.join(run_dir, "eval_results_*.json")))
        if not os.path.exists(meta_path) or not evals:
            continue
        meta = json.load(open(meta_path, encoding="utf-8"))
        hp = meta["hyperparameters"]
        ev = json.load(open(evals[-1], encoding="utf-8"))
        device = meta.get("actual_device", "cpu")
        n_stack = hp.get("n_stack", 1)
        variant = meta.get("variant") or classify_variant(device, n_stack)
        # Regimen de entrenamiento: explicito en metadata, o derivado (los vision_distill*
        # son supervisados; el resto se entreno con el simulador/RL).
        regime = meta.get("training_regime") or (
            "distill" if variant.startswith("vision_distill") else "rl")
        runs.append({
            "run_id": os.path.basename(run_dir.rstrip("/\\")),
            "variant": variant,
            "training_regime": regime,
            "device": device,
            "n_stack": int(n_stack),
            "seed": int(hp.get("seed", -1)),
            "total_timesteps": int(hp.get("total_timesteps", 0)),
            "tracks": ev["tracks"],
        })
    return runs


def per_run_aggregates(run):
    """
    Colapsa los tracks de UNA corrida en escalares (pooled sobre episodios):
      lap_rate_overall, reward_overall, offtrack_rate_overall, lap_time_overall.
    Y guarda el lap_rate por track (para la tabla por-pista).
    """
    tracks = run["tracks"]
    tot_ep = sum(t["episodes"] for t in tracks)
    tot_laps = sum(t["laps"] for t in tracks)
    tot_offtrack = sum(t.get("failures", {}).get("offtrack_grass", 0) for t in tracks)
    rewards = [t["reward_ep_mean"] for t in tracks if t["reward_ep_mean"] is not None]
    lap_times = [t["lap_time_s_mean"] for t in tracks if t.get("lap_time_s_mean") is not None]

    per_track = {t["texture"]: t["lap_rate"] for t in tracks}
    return {
        "lap_rate": (tot_laps / tot_ep) if tot_ep else 0.0,
        "reward": st.mean(rewards) if rewards else 0.0,
        "offtrack_rate": (tot_offtrack / tot_ep) if tot_ep else 0.0,
        "lap_time": st.mean(lap_times) if lap_times else None,
        "per_track_lap_rate": per_track,
    }


def _u_statistic(a, b):
    """U de Mann-Whitney del grupo a vs b (cuenta de a>b, con 0.5 por empate)."""
    u = 0.0
    for x in a:
        for y in b:
            if x > y:
                u += 1.0
            elif x == y:
                u += 0.5
    return u


def mann_whitney_exact(a, b):
    """
    Test de Mann-Whitney EXACTO (two-sided) por permutacion. Maneja empates (usa la
    distribucion nula de permutar las etiquetas sobre los valores reales). Apto para n chico.
    Devuelve (U_observado, p_two_sided).
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return None, None
    pooled = list(a) + list(b)
    u_obs = _u_statistic(a, b)
    mean_u = n1 * n2 / 2.0
    target = abs(u_obs - mean_u)
    total = 0
    extreme = 0
    idx = range(len(pooled))
    for combo in itertools.combinations(idx, n1):
        sel = set(combo)
        ga = [pooled[i] for i in idx if i in sel]
        gb = [pooled[i] for i in idx if i not in sel]
        u = _u_statistic(ga, gb)
        total += 1
        if abs(u - mean_u) >= target - 1e-9:
            extreme += 1
    return u_obs, extreme / total


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def aggregate(run_dirs):
    runs = load_runs(run_dirs)
    by_variant = {v: [] for v in VARIANT_ORDER}
    for run in runs:
        run["agg"] = per_run_aggregates(run)
        by_variant[run["variant"]].append(run)

    # Conjunto de tracks (en orden de aparicion).
    track_names = []
    for run in runs:
        for t in run["tracks"]:
            if t["texture"] not in track_names:
                track_names.append(t["texture"])

    summary = {"variants": {}, "tracks": track_names, "n_runs": len(runs)}
    for v in VARIANT_ORDER:
        group = sorted(by_variant[v], key=lambda r: r["seed"])
        if not group:
            continue
        lap = [r["agg"]["lap_rate"] for r in group]
        rew = [r["agg"]["reward"] for r in group]
        off = [r["agg"]["offtrack_rate"] for r in group]
        lt = [r["agg"]["lap_time"] for r in group]
        per_track = {
            tn: [r["agg"]["per_track_lap_rate"].get(tn) for r in group]
            for tn in track_names
        }
        summary["variants"][v] = {
            "label": VARIANT_LABEL[v],
            "training_regime": group[0].get("training_regime", "rl"),
            "n_seeds": len(group),
            "seeds": [r["seed"] for r in group],
            "lap_rate": dict(zip(("mean", "std"), mean_std(lap))),
            "reward": dict(zip(("mean", "std"), mean_std(rew))),
            "offtrack_rate": dict(zip(("mean", "std"), mean_std(off))),
            "lap_time_s": dict(zip(("mean", "std"), mean_std(lt))),
            "per_track_lap_rate": {
                tn: dict(zip(("mean", "std"), mean_std(vals)))
                for tn, vals in per_track.items()
            },
            "per_seed": [
                {"seed": r["seed"], **{k: r["agg"][k] for k in
                 ("lap_rate", "reward", "offtrack_rate", "lap_time")}}
                for r in group
            ],
        }

    # Significancia: Mann-Whitney exacto sobre lap_rate por seed, entre pares de variantes.
    lap_by_variant = {
        v: [p["lap_rate"] for p in summary["variants"][v]["per_seed"]]
        for v in VARIANT_ORDER if v in summary["variants"]
    }
    significance = []
    for a, b in itertools.combinations(lap_by_variant, 2):
        u, p = mann_whitney_exact(lap_by_variant[a], lap_by_variant[b])
        significance.append({"a": a, "b": b, "metric": "lap_rate", "U": u, "p_two_sided": p})
    summary["significance"] = significance
    return summary


def pct(m, s):
    return f"{100*m:.1f} ± {100*s:.1f}" if m is not None else "—"


def num(m, s, fmt="{:.1f}"):
    if m is None:
        return "—"
    return f"{fmt.format(m)} ± {fmt.format(s)}"


def to_markdown(summary):
    lines = []
    lines.append("# Resultados agregados (eval held-out, media ± desvío sobre seeds)\n")
    lines.append(f"_{summary['n_runs']} corridas · tracks held-out: "
                 f"{', '.join(summary['tracks'])} · 10 episodios/track (modo tasa de éxito)._\n")

    # Tabla global.
    lines.append("## Desempeño global (promedio de los tracks held-out)\n")
    lines.append("| Variante | Régimen | Lap rate (%) | Reward/ep | Off-track (%) | Tiempo vuelta (s) |")
    lines.append("|---|---|---|---|---|---|")
    for v in VARIANT_ORDER:
        d = summary["variants"].get(v)
        if not d:
            continue
        regime = "destilado" if d.get("training_regime") == "distill" else "RL (sim)"
        lines.append(
            f"| {d['label']} | {regime} | {pct(d['lap_rate']['mean'], d['lap_rate']['std'])} "
            f"| {num(d['reward']['mean'], d['reward']['std'])} "
            f"| {pct(d['offtrack_rate']['mean'], d['offtrack_rate']['std'])} "
            f"| {num(d['lap_time_s']['mean'], d['lap_time_s']['std'])} |"
        )
    lines.append("")

    # Tabla por track (lap rate).
    lines.append("## Lap rate por pista (%)\n")
    header = "| Variante | " + " | ".join(summary["tracks"]) + " |"
    lines.append(header)
    lines.append("|---" * (len(summary["tracks"]) + 1) + "|")
    for v in VARIANT_ORDER:
        d = summary["variants"].get(v)
        if not d:
            continue
        cells = [d["label"]]
        for tn in summary["tracks"]:
            pt = d["per_track_lap_rate"][tn]
            cells.append(pct(pt["mean"], pt["std"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Significancia.
    if summary.get("significance"):
        lines.append("## Significancia — Mann-Whitney exacto (lap rate por seed)\n")
        lines.append("| Comparación | U | p (two-sided) |")
        lines.append("|---|---|---|")
        for s in summary["significance"]:
            la = VARIANT_LABEL.get(s["a"], s["a"])
            lb = VARIANT_LABEL.get(s["b"], s["b"])
            lines.append(f"| {la} vs {lb} | {s['U']:.1f} | {s['p_two_sided']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def plot_eval_bars(summary, out_path):
    """
    Barras de lap rate held-out por variante (eval), con ±desvio. Incluye TODAS las variantes
    —incluidas las destiladas, que NO tienen curva de TensorBoard— y las distingue por color +
    hatch/etiqueta segun el regimen (RL de simulador vs destilado). Este es el chart que compara
    geometrico, RL de vision y destilados en el mismo grafico.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [v for v in VARIANT_ORDER if v in summary["variants"]]
    if not variants:
        return
    d = [summary["variants"][v] for v in variants]
    # lap_rate viene en fraccion [0,1]; el chart va en porcentaje.
    means = [x["lap_rate"]["mean"] * 100 for x in d]
    stds = [x["lap_rate"]["std"] * 100 for x in d]
    regimes = [x.get("training_regime", "rl") for x in d]
    xt = [f"{x['label']}\n({'destilado' if r == 'distill' else 'RL sim'}, n={x['n_seeds']})"
          for x, r in zip(d, regimes)]
    colors = [plt.cm.tab10(i % 10) for i in range(len(variants))]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(variants)), 4.6))
    bars = ax.bar(range(len(variants)), means, yerr=stds, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6)
    for b, r in zip(bars, regimes):
        if r == "distill":
            b.set_hatch("//")   # los destilados con hatch, para diferenciarlos de un vistazo
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(xt, fontsize=8)
    ax.set_ylabel("Lap rate held-out (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Lap rate en held-out por variante (eval)  ·  hatch = destilado")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, min(m + s + 2, 104), f"{m:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("Chart de eval escrito en", out_path)


def run(run_dirs, out_dir=OUT_DIR):
    """Agrega los run_dirs dados y escribe results_summary.{json,md} + fig_eval_lap_rate.png."""
    os.makedirs(out_dir, exist_ok=True)
    summary = aggregate(run_dirs)
    json_path = os.path.join(out_dir, "results_summary.json")
    md_path = os.path.join(out_dir, "results_summary.md")
    json.dump(summary, open(json_path, "w", encoding="utf-8"), indent=2, ensure_ascii=True)
    md = to_markdown(summary)
    open(md_path, "w", encoding="utf-8").write(md)
    plot_eval_bars(summary, os.path.join(out_dir, "fig_eval_lap_rate.png"))
    return summary, md


def parse_args():
    p = argparse.ArgumentParser(description="Agrega evaluaciones por variante (paper).")
    p.add_argument("--models-dir", default=MODELS_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--timesteps", type=int, default=None,
                   help="Fijar el largo exacto de las corridas a incluir (guard anti-mezcla).")
    p.add_argument("--variants", nargs="+", default=None, help="Subconjunto de variantes.")
    p.add_argument("--allow-mixed", action="store_true",
                   help="Poolear runs de distinto largo (por defecto NO, ver discover).")
    return p.parse_args()


def main():
    args = parse_args()
    runs = discover_runs(args.models_dir)
    sel, report = select_runs(runs, timesteps=args.timesteps, variants=args.variants,
                              allow_mixed=args.allow_mixed)
    print("Seleccion de corridas (eval):")
    print(format_report(report))
    sel_with_eval = [r["run_dir"] for r in sel if r["has_eval"]]
    if not sel_with_eval:
        print("\nNo hay corridas con eval_results en la seleccion; nada que agregar.")
        return
    _, md = run(sel_with_eval, args.out_dir)
    print("\n" + md)
    print(f"\nEscrito: {os.path.join(args.out_dir, 'results_summary.json')}"
          f"\n         {os.path.join(args.out_dir, 'results_summary.md')}")


if __name__ == "__main__":
    main()
