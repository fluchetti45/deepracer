"""
Descubrimiento + selección de corridas en models/ para el análisis del paper.

Es la ÚNICA fuente de verdad sobre "qué runs entran": la usan tanto aggregate_eval
(evaluación) como parse_tensorboard (curvas), para que ambos analicen el MISMO conjunto.

Incluye el GUARD ANTI-MEZCLA: si en models/ conviven corridas de distinto largo (p.ej. el
piloto de 500k y la final de 1M) o con distintos tracks de eval, pooolearlas daría números
sin sentido. Por defecto se selecciona el grupo de MAYOR total_timesteps y se avisa qué se
excluyó; con --timesteps se fija el largo exacto y con --allow-mixed se poolea todo (con
warning explícito).
"""

import glob
import json
import os

# Orden y etiquetas canónicas de las 4 variantes del ablation.
VARIANT_ORDER = ["geometrica", "vision_1frame", "vision_stacked", "vision_lstm"]
VARIANT_LABEL = {
    "geometrica": "Geometrica",
    "vision_1frame": "Vision (1 frame)",
    "vision_stacked": "Vision apilada (4)",
    "vision_lstm": "Vision + LSTM",
}


def classify_variant(device, n_stack):
    """Fallback para runs SIN el campo 'variant' (metadata vieja): deriva de device+n_stack."""
    if device == "cpu":
        return "geometrica"
    return "vision_1frame" if int(n_stack) == 1 else "vision_stacked"


def _latest_eval_tracks(run_dir):
    """Tupla ordenada de texturas del eval_results mas reciente (o None si no hay eval)."""
    evals = sorted(glob.glob(os.path.join(run_dir, "eval_results_*.json")))
    if not evals:
        return None
    try:
        ev = json.load(open(evals[-1], encoding="utf-8"))
        return tuple(sorted(t["texture"] for t in ev.get("tracks", [])))
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def _has_tb(run_dir):
    return bool(glob.glob(os.path.join(run_dir, "tensorboard", "**", "events.out.tfevents.*"),
                          recursive=True))


def discover_runs(models_dir="models"):
    """
    Escanea models_dir y devuelve una lista de dicts (uno por run con run_metadata.json):
      run_dir, run_id, variant, device, n_stack, seed, total_timesteps,
      eval_tracks (tupla|None), has_eval, has_tb.
    """
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(models_dir, "*", ""))):
        meta_path = os.path.join(run_dir, "run_metadata.json")
        if not os.path.exists(meta_path):
            continue
        try:
            meta = json.load(open(meta_path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        hp = meta.get("hyperparameters", {})
        device = meta.get("actual_device", "cpu")
        n_stack = int(hp.get("n_stack", 1))
        eval_tracks = _latest_eval_tracks(run_dir)
        runs.append({
            "run_dir": run_dir,
            "run_id": os.path.basename(run_dir.rstrip("/\\")),
            "variant": meta.get("variant") or classify_variant(device, n_stack),
            "device": device,
            "n_stack": n_stack,
            "seed": int(hp.get("seed", -1)),
            "total_timesteps": int(hp.get("total_timesteps", 0)),
            "eval_tracks": eval_tracks,
            "has_eval": eval_tracks is not None,
            "has_tb": _has_tb(run_dir),
        })
    return runs


def select_runs(runs, timesteps=None, variants=None, allow_mixed=False):
    """
    Aplica los filtros y el guard anti-mezcla. Devuelve (seleccionados, report) donde
    report = {"selected_timesteps", "excluded", "warnings", "by_variant", "eval_track_sets"}.
    """
    warnings = []
    sel = list(runs)

    if variants:
        sel = [r for r in sel if r["variant"] in set(variants)]

    ts_values = sorted({r["total_timesteps"] for r in sel})
    if timesteps is not None:
        chosen_ts = int(timesteps)
        sel = [r for r in sel if r["total_timesteps"] == chosen_ts]
        if not sel:
            warnings.append(f"Ningun run con total_timesteps == {chosen_ts}.")
    elif len(ts_values) > 1 and not allow_mixed:
        chosen_ts = max(ts_values)
        excluded = [r for r in sel if r["total_timesteps"] != chosen_ts]
        sel = [r for r in sel if r["total_timesteps"] == chosen_ts]
        warnings.append(
            f"MEZCLA DE TIMESTEPS detectada {ts_values}: se selecciona el grupo de "
            f"{chosen_ts} ({len(sel)} runs) y se EXCLUYEN {len(excluded)} de otros largos. "
            f"Usa --timesteps N para fijar otro, o --allow-mixed para poolear todo."
        )
    elif len(ts_values) > 1 and allow_mixed:
        chosen_ts = None
        warnings.append(
            f"--allow-mixed: se poolean runs de DISTINTO largo {ts_values}. "
            f"Los agregados por variante mezclan corridas no comparables."
        )
    else:
        chosen_ts = ts_values[0] if ts_values else None

    # Heterogeneidad de tracks de eval dentro de lo seleccionado (solo runs con eval).
    eval_sets = sorted({r["eval_tracks"] for r in sel if r["eval_tracks"]})
    if len(eval_sets) > 1:
        warnings.append(
            f"MEZCLA DE TRACKS DE EVAL entre los runs seleccionados: {eval_sets}. "
            f"El lap-rate por-pista no es comparable entre sets distintos."
        )

    by_variant = {v: sum(1 for r in sel if r["variant"] == v) for v in VARIANT_ORDER}
    report = {
        "selected_timesteps": chosen_ts,
        "n_selected": len(sel),
        "by_variant": by_variant,
        "eval_track_sets": [list(s) for s in eval_sets],
        "warnings": warnings,
    }
    return sel, report


def format_report(report):
    """Reporte legible por consola de qué entró al análisis (con los warnings arriba)."""
    lines = []
    for w in report["warnings"]:
        lines.append(f"  [!] {w}")
    ts = report["selected_timesteps"]
    lines.append(f"  timesteps seleccionados: {ts if ts is not None else 'MEZCLA'}")
    lines.append(f"  runs por variante: " + ", ".join(
        f"{VARIANT_LABEL[v]}={report['by_variant'][v]}" for v in VARIANT_ORDER
        if report["by_variant"][v]
    ))
    if report["eval_track_sets"]:
        lines.append(f"  tracks de eval: {report['eval_track_sets']}")
    return "\n".join(lines)
