# -*- coding: utf-8 -*-
"""
Grafico de barras del eval held-out (fig_eval_lap_rate.png) desde results_summary.json:
lap rate por variante con IQM y barras de error = IC del 95% bootstrap (mismos numeros que
la Tabla del paper). Resalta las dos variantes que resuelven la tarea (geometrica + destilada).

Uso:  python -m analysis.fig_eval [--summary analysis/results_summary.json] [--out ...]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.robust_stats import boot_ci

TEAL = "#0E8C79"
MUTED = "#9AA6A2"
BAD = "#A34428"

# Orden del paper: maestro + destilada (resaltadas), luego la vision-RL.
ORDER = ["geometrica", "vision_distill", "vision_1frame",
         "vision_camonly", "vision_stacked", "vision_lstm"]
SHORT = {
    "geometrica": "Geométrica",
    "vision_distill": "Visión\ndestilada",
    "vision_1frame": "Visión\n1 frame",
    "vision_camonly": "Visión pura\n(cámara)",
    "vision_stacked": "Visión\napilada",
    "vision_lstm": "Visión\n+ LSTM",
}
SOLVED = {"geometrica", "vision_distill"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join("analysis", "results_summary.json"))
    ap.add_argument("--out", default=os.path.join("analysis", "fig_eval_lap_rate.png"))
    args = ap.parse_args()

    J = json.load(open(args.summary, encoding="utf-8"))
    V = J["variants"]
    present = [v for v in ORDER if v in V]

    iqm, lo, hi, labels, colors = [], [], [], [], []
    for v in present:
        vals = [p["lap_rate"] * 100 for p in V[v]["per_seed"] if p.get("lap_rate") is not None]
        m, l, h = boot_ci(vals)
        iqm.append(m); lo.append(m - l); hi.append(h - m)
        labels.append(SHORT.get(v, v))
        colors.append(TEAL if v in SOLVED else (BAD if m < 50 else MUTED))

    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(x, iqm, width=0.66, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax.errorbar(x, iqm, yerr=[lo, hi], fmt="none", ecolor="#33403C",
                elinewidth=1.1, capsize=3.5, zorder=4)
    for xi, m in zip(x, iqm):
        ax.text(xi, min(m + max(hi) * 0.06 + 2, 101), f"{m:.0f}", ha="center",
                va="bottom", fontsize=9, color="#22302C")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Lap rate (%)  —  IQM [IC 95%]", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.axhline(0, color="#33403C", linewidth=0.8)
    ax.grid(axis="y", color="#DDE4E1", linewidth=0.7, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("Escrito:", args.out)


if __name__ == "__main__":
    main()
