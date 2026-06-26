# analysis/ — análisis de modelos y evaluaciones

Carpeta destinada a **procesar y almacenar** el análisis de las corridas de `models/`
(no entrena ni evalúa; solo lee los artefactos ya generados).

## Contenido

- **`aggregate_eval.py`** — agrega las evaluaciones de todas las corridas en `models/` y
  produce las estadísticas del paper (media ± desvío sobre seeds, por variante) + un test
  de **Mann-Whitney exacto** (permutación, puro Python) sobre el lap rate por seed.
- **`results_summary.json`** — agregados + datos por-seed (machine-readable).
- **`results_summary.md`** — tablas markdown listas para pegar en `PAPER.md`.

## Cómo correrlo

```bash
python -m analysis.aggregate_eval     # desde la raíz del repo
```

## Identificación de la variante

Las corridas no guardan un campo "variante"; se deriva de la metadata:

| Variante | Regla (`run_metadata.json`) |
|---|---|
| Geométrica | `actual_device == "cpu"` |
| Visión (1 frame) | `device == "cuda"` y `n_stack == 1` |
| Visión apilada (4) | `device == "cuda"` y `n_stack == 4` |

(Convención del lote de datos: las geométricas se entrenaron en CPU y las de visión en CUDA;
el `n_stack` distingue 1 vs 4 frames.)

## Notas

- El eval se corrió en **modo `--episodes` (10 episodios/track)** → `lap_rate` = vueltas/10
  sin sesgo. Tracks held-out: `track4`, `track5`.
- `lap_rate` global = pooled sobre los episodios de ambos tracks; `reward/ep` y
  `off-track %` igual. `lap_time` solo promedia tracks con al menos una vuelta.
