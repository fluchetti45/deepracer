# analysis/ — análisis de modelos y evaluaciones

Carpeta destinada a **procesar y almacenar** el análisis de las corridas de `models/`
(no entrena ni evalúa; solo lee los artefactos ya generados).

## Un solo comando (recomendado)

```bash
python -m analysis.run_analysis            # desde la raíz del repo
```

Descubre las corridas de `models/`, aplica el **guard anti-mezcla** una sola vez y corre
todo sobre el MISMO conjunto: tablas de evaluación + curvas de entrenamiento + figuras.

Flags:

| Flag | Efecto |
|---|---|
| `--timesteps N` | incluir solo corridas de largo `N` (p.ej. `1000000` para la final) |
| `--variants a b` | subconjunto de variantes (`geometrica vision_1frame vision_stacked vision_lstm`) |
| `--allow-mixed` | poolear corridas de distinto largo (por defecto NO) |
| `--models-dir` / `--out-dir` | rutas alternativas |

## Guard anti-mezcla (500k piloto vs 1M final)

`models/` suele acumular corridas de distinto largo o con distintos tracks de eval.
Poolearlas daría números sin sentido. Por eso `analysis/discover.py` (fuente única de
"qué runs entran", compartida por ambos scripts):

- si hay **varios `total_timesteps`** y no se pasó `--timesteps`, selecciona el grupo de
  **mayor** largo y **avisa** qué excluyó;
- avisa también si se **mezclan tracks de eval** entre los runs elegidos;
- `--timesteps` fija el largo exacto; `--allow-mixed` poolea todo (con warning explícito).

## Componentes

- **`discover.py`** — descubrimiento + selección + guard. Define `VARIANT_ORDER/LABEL` y
  `classify_variant` (fallback para metadata vieja).
- **`aggregate_eval.py`** — estadísticas de **evaluación** (media ± desvío por variante) +
  **Mann-Whitney exacto** por seed. → `results_summary.{json,md}`.
- **`parse_tensorboard.py`** — curvas de **entrenamiento** (reward/lap_rate vs timesteps) +
  eficiencia de muestras. → `curves.json`, `sample_efficiency.json`, `fig_reward.png`,
  `fig_lap_rate.png`.

Ambos se pueden correr sueltos (`python -m analysis.aggregate_eval [flags]`), con los mismos
flags de selección.

## Identificación de la variante

Se prefiere el campo explícito **`variant`** de `run_metadata.json` (metadata nueva). Si no
está (runs viejos), se deriva:

| Variante | Fallback (`run_metadata.json`) |
|---|---|
| Geométrica | `actual_device == "cpu"` |
| Visión (1 frame) | `device == "cuda"` y `n_stack == 1` |
| Visión apilada (4) | `device == "cuda"` y `n_stack == 4` |

(La variante **`vision_lstm`** siempre trae el campo `variant` explícito.)

## Notas

- El eval se corre en **modo `--episodes` (10 episodios/track)** → `lap_rate` = vueltas/10
  sin sesgo. Tracks held-out actuales: `track9`, `track10`.
- `lap_rate` global = pooled sobre los episodios de todos los tracks; `reward/ep` y
  `off-track %` igual. `lap_time` solo promedia tracks con al menos una vuelta.
