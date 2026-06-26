"""
Parsea las curvas de entrenamiento de TensorBoard (los event files que SB3 escribe en
`models/<run>/tensorboard/`) y produce las curvas agregadas del paper [TRAIN]:
lap rate y reward vs timesteps, media ± desvío sobre seeds, una curva por variante.
Tambien calcula la eficiencia de muestras (timesteps hasta X% de lap rate).

NO entrena: solo lee los event files ya generados.

Salidas (en analysis/):
  - curves.json          : grilla + mean/std por variante y por tag (machine-readable)
  - fig_lap_rate.png     : lap rate (train) vs timesteps, 3 variantes con banda ±desvío
  - fig_reward.png       : reward medio/episodio (train) vs timesteps
  - sample_efficiency.json : timesteps hasta 25%/50%/80% de lap rate por variante

Uso:  python -m analysis.parse_tensorboard     (desde la raiz del repo)
"""

import glob
import json
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from analysis.aggregate_eval import classify_variant, VARIANT_ORDER, VARIANT_LABEL

MODELS_DIR = "models"
OUT_DIR = "analysis"
# Tags de TensorBoard a extraer (los loguea RLMetricsCallback / SB3).
TAGS = {
    "lap_rate": "custom/lap_rate",
    "reward": "rollout/ep_rew_mean",
    "offtrack_rate": "custom/offtrack_rate",
}
GRID_POINTS = 120          # puntos de la grilla comun de timesteps
# Eficiencia de muestras por REWARD (umbrales absolutos): el lap_rate de TRAIN no sirve
# como metrica (una vuelta ~1080 steps NO entra en el episodio de training de 1000 steps,
# asi que el lap_rate train queda truncado a ~0 para todas las variantes).
SE_REWARD_THRESHOLDS = [25, 50, 75]


def find_event_dir(run_dir):
    """Carpeta que contiene el events.out.tfevents.* de una corrida (o None)."""
    hits = glob.glob(os.path.join(run_dir, "tensorboard", "**", "events.out.tfevents.*"),
                     recursive=True)
    return os.path.dirname(sorted(hits)[0]) if hits else None


def load_scalars(event_dir):
    """Devuelve {tag: (steps_array, values_array)} para los TAGS presentes."""
    ea = EventAccumulator(event_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out = {}
    for key, tag in TAGS.items():
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=float)
        vals = np.array([e.value for e in events], dtype=float)
        out[key] = (steps, vals)
    return out


def collect_runs():
    """Lista de dicts {variant, seed, scalars} por corrida con event files."""
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(MODELS_DIR, "*", ""))):
        meta_path = os.path.join(run_dir, "run_metadata.json")
        if not os.path.exists(meta_path):
            continue
        meta = json.load(open(meta_path, encoding="utf-8"))
        hp = meta["hyperparameters"]
        event_dir = find_event_dir(run_dir)
        if event_dir is None:
            continue
        runs.append({
            "variant": classify_variant(meta.get("actual_device", "cpu"), hp.get("n_stack", 1)),
            "seed": int(hp.get("seed", -1)),
            "total_timesteps": int(hp.get("total_timesteps", 0)),
            "scalars": load_scalars(event_dir),
        })
    return runs


def aggregate_curves(runs):
    """
    Por variante y tag: interpola cada seed sobre una grilla comun de timesteps y devuelve
    mean/std. Grilla: 0..max_total_timesteps (de las corridas de esa variante).
    """
    curves = {}
    for variant in VARIANT_ORDER:
        group = [r for r in runs if r["variant"] == variant]
        if not group:
            continue
        max_ts = max(r["total_timesteps"] for r in group) or 1
        grid = np.linspace(0, max_ts, GRID_POINTS)
        curves[variant] = {"label": VARIANT_LABEL[variant], "n_seeds": len(group),
                           "grid": grid.tolist(), "tags": {}}
        for key in TAGS:
            stacked = []
            for r in group:
                if key not in r["scalars"]:
                    continue
                steps, vals = r["scalars"][key]
                if len(steps) < 2:
                    continue
                # np.interp requiere x creciente; los steps de SB3 ya lo son.
                stacked.append(np.interp(grid, steps, vals, left=vals[0], right=vals[-1]))
            if not stacked:
                continue
            arr = np.vstack(stacked)
            curves[variant]["tags"][key] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0, ddof=1 if arr.shape[0] > 1 else 0).tolist(),
                "n": int(arr.shape[0]),
            }
    return curves


def sample_efficiency(curves):
    """Primer timestep donde el REWARD MEDIO (train) cruza cada umbral absoluto, por variante."""
    out = {}
    for variant, c in curves.items():
        grid = np.array(c["grid"])
        rew = c["tags"].get("reward")
        entry = {}
        for thr in SE_REWARD_THRESHOLDS:
            ts = None
            if rew is not None:
                mean = np.array(rew["mean"])
                idx = np.argmax(mean >= thr) if np.any(mean >= thr) else None
                ts = float(grid[idx]) if idx is not None else None
            entry[f"ts_to_reward_{thr}"] = ts
        out[variant] = {"label": c["label"], **entry}
    return out


def plot_curve(curves, key, ylabel, title, out_path, percent=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for variant in VARIANT_ORDER:
        c = curves.get(variant)
        if not c or key not in c["tags"]:
            continue
        grid = np.array(c["grid"])
        mean = np.array(c["tags"][key]["mean"])
        std = np.array(c["tags"][key]["std"])
        scale = 100.0 if percent else 1.0
        plt.plot(grid, mean * scale, label=c["label"], linewidth=2)
        plt.fill_between(grid, (mean - std) * scale, (mean + std) * scale, alpha=0.18)
    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = collect_runs()
    if not runs:
        print("No se encontraron event files de TensorBoard en models/.")
        return
    curves = aggregate_curves(runs)
    se = sample_efficiency(curves)

    json.dump(curves, open(os.path.join(OUT_DIR, "curves.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=True)
    json.dump(se, open(os.path.join(OUT_DIR, "sample_efficiency.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=True)

    plot_curve(curves, "lap_rate", "Lap rate (%)",
               "Lap rate (train) vs timesteps", os.path.join(OUT_DIR, "fig_lap_rate.png"),
               percent=True)
    plot_curve(curves, "reward", "Reward medio / episodio",
               "Reward (train) vs timesteps", os.path.join(OUT_DIR, "fig_reward.png"))

    # Resumen por consola.
    print("Curvas agregadas por variante (n seeds):")
    for v in VARIANT_ORDER:
        if v in curves:
            tags = ", ".join(curves[v]["tags"].keys())
            print(f"  {VARIANT_LABEL[v]:<20} n={curves[v]['n_seeds']}  tags=[{tags}]")
    print("\nEficiencia de muestras (timesteps hasta reward >= umbral, train):")
    print(f"  {'variante':<20}{'r>=25':>12}{'r>=50':>12}{'r>=75':>12}")
    for v in VARIANT_ORDER:
        if v not in se:
            continue
        e = se[v]
        fmt = lambda x: f"{int(x):>12}" if x is not None else f"{'nunca':>12}"
        print(f"  {VARIANT_LABEL[v]:<20}{fmt(e['ts_to_reward_25'])}{fmt(e['ts_to_reward_50'])}{fmt(e['ts_to_reward_75'])}")
    print(f"\nEscrito: {OUT_DIR}/curves.json, sample_efficiency.json, fig_lap_rate.png, fig_reward.png")


if __name__ == "__main__":
    main()
