# -*- coding: utf-8 -*-
"""
Estadisticas 'paper-grade' desde analysis/results_summary.json.

Reemplaza el media+-desvio de aggregate_eval por metricas ROBUSTAS a seeds que colapsan:
  - IQM (media intercuartil, trim 25% por lado) con IC del 95% por bootstrap de percentil.
  - TOST (two one-sided tests) de EQUIVALENCIA sobre lap rate, con margen pre-especificado.

Auto-incluye cualquier variante presente en el summary (orden de discover.VARIANT_ORDER),
asi que cuando entrenes vision_camonly entra sola, y ademas corre el TOST de la ablacion
clave (vision_1frame vs vision_camonly = cuanto aporta la propiocepcion).

Requiere scipy (pip install scipy).

Uso:
  python -m analysis.robust_stats
  python -m analysis.robust_stats --summary analysis/results_summary.json --delta 5
"""

import argparse
import json
import os

import numpy as np

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    raise SystemExit("Falta scipy: pip install scipy")

from analysis.discover import VARIANT_ORDER, VARIANT_LABEL

RNG = np.random.default_rng(20260713)
NB = 20000  # remuestreos bootstrap

# (clave en per_seed, escala para mostrar, etiqueta, mas-alto-es-mejor)
METRICS = [
    ("lap_rate", 100.0, "Lap rate (%)", True),
    ("offtrack_rate", 100.0, "Off-track (%)", False),
    ("lap_steps", 1.0, "Pasos/vuelta", False),
    ("reward", 1.0, "Reward/ep", True),
]

# Pares de interes para el TOST de equivalencia (se corren si AMBOS estan presentes).
TOST_PAIRS = [
    ("geometrica", "vision_distill"),   # destilada iguala al maestro?
    ("vision_1frame", "vision_camonly"),  # ablacion: aporta la propiocepcion?
]


def iqm(x):
    """Interquartile mean = trim_mean 25% por lado (Rliable). n=5 -> media de los 3 centrales."""
    return stats.trim_mean(np.asarray(x, float), 0.25)


def boot_ci(x, stat=iqm, nb=NB, alpha=0.05):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), size=(nb, len(x)))
    bs = np.array([stat(x[i]) for i in idx])
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return stat(x), lo, hi


def tost_welch(a, b, delta):
    """TOST no pareado (Welch). Devuelve (diff, se, df, p_TOST, IC90 de la diferencia)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a.mean() - b.mean()
    s1, s2, n1, n2 = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    se = np.sqrt(s1 / n1 + s2 / n2) or 1e-9
    df = (s1 / n1 + s2 / n2) ** 2 / (
        (s1 / n1) ** 2 / max(n1 - 1, 1) + (s2 / n2) ** 2 / max(n2 - 1, 1)
    )
    p_lo = stats.t.sf((diff + delta) / se, df)  # H0: diff <= -delta
    p_hi = stats.t.cdf((diff - delta) / se, df)  # H0: diff >=  delta
    tc = stats.t.ppf(0.95, df)  # IC del (1-2alpha)=90%
    return diff, se, df, max(p_lo, p_hi), (diff - tc * se, diff + tc * se)


def variants_present(V):
    """Variantes del summary en orden canonico; las no listadas se agregan al final."""
    return [v for v in VARIANT_ORDER if v in V] + [v for v in V if v not in VARIANT_ORDER]


def main():
    ap = argparse.ArgumentParser(description="IQM [IC95%] + TOST desde results_summary.json.")
    ap.add_argument("--summary", default=os.path.join("analysis", "results_summary.json"))
    ap.add_argument("--delta", type=float, default=5.0,
                    help="Margen de equivalencia del TOST, en puntos de lap rate (default 5).")
    args = ap.parse_args()

    J = json.load(open(args.summary, encoding="utf-8"))
    V = J["variants"]
    present = variants_present(V)

    print("=" * 74)
    print(f"IQM [IC 95% bootstrap, {NB} remuestreos]  ({len(present)} variantes)")
    print("=" * 74)
    for key, sc, label, hi_better in METRICS:
        arrow = "(mas alto mejor)" if hi_better else "(mas bajo mejor)"
        print(f"\n--- {label} {arrow} ---")
        for v in present:
            vals = [d[key] for d in V[v]["per_seed"] if d.get(key) is not None]
            if not vals:
                print(f"  {VARIANT_LABEL.get(v, v)[:26]:26s}  (sin datos)")
                continue
            pt, lo, hi = boot_ci(vals)
            print(f"  {VARIANT_LABEL.get(v, v)[:26]:26s}  IQM={pt*sc:8.2f}  "
                  f"IC95=[{lo*sc:8.2f}, {hi*sc:8.2f}]  (n={len(vals)})")

    d = args.delta / 100.0
    print("\n" + "=" * 74)
    print(f"TOST equivalencia (lap rate, margen pre-especificado +/-{args.delta:.0f} puntos)")
    print("=" * 74)
    for x, y in TOST_PAIRS:
        if x not in V or y not in V:
            print(f"  {x} vs {y}: (falta alguna; salteo)")
            continue
        gx = [p["lap_rate"] for p in V[x]["per_seed"]]
        gy = [p["lap_rate"] for p in V[y]["per_seed"]]
        diff, se, df, p, ci = tost_welch(gx, gy, d)
        verdict = "EQUIVALENTE" if p < 0.05 else "NO concluye equiv."
        print(f"  {VARIANT_LABEL.get(x, x)} vs {VARIANT_LABEL.get(y, y)}:")
        print(f"    diff={diff*100:+.1f} pts  p_TOST={p:.3f} ({verdict})  "
              f"IC90 diff=[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] pts")


if __name__ == "__main__":
    main()
