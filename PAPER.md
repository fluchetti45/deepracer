# Conducción autónoma estilo DeepRacer en Webots: el rol de la representación de la observación

> Informe técnico / paper en progreso.
> Las secciones de **Resultados** están como _placeholder_: se completan tras correr
> varias seeds por variante y agregar estadísticas.

---

## Resumen (abstract)

Entrenamos un agente de conducción autónoma sobre pistas estilo AWS DeepRacer
simuladas en Webots, usando **PPO** (Proximal Policy Optimization). El objetivo es
**aislar el efecto de la representación de la observación** sobre el desempeño,
manteniendo _todo lo demás constante_: mismo robot, mismo espacio de acción, mismas
pistas, mismos hiperparámetros y —crucialmente— **la misma función de reward**.

Comparamos tres variantes que solo difieren en _qué ve el agente_:

1. **Geométrica** — la observación es un vector de **métricas derivadas localmente de
   la imagen** (no los píxeles).
2. **Visión** — la observación es la **imagen cruda** de la cámara frontal (un frame).
3. **Visión apilada** — igual que Visión pero con **4 frames apilados** (información
   temporal).

El reward usa **información privilegiada del simulador** (progreso sobre el circuito,
detección de vuelta completa y de conduccion en sentido contrario) que el agente **no** recibe en su observación.
Esto convierte el experimento en un _ablation_ limpio sobre la representación de
entrada.

---

## 1. Introducción y objetivo

El problema es de **seguimiento de pista**: un robot debe recorrer un circuito cerrado
lo más rápido posible sin salirse a la zona de pasto. La pista sigue el modelo
DeepRacer: asfalto negro delimitado por **bordes blancos**, con una **línea amarilla
central decorativa** (pisable), rodeada de **pasto verde** (fuera de pista).

La pregunta de investigación es:

> **¿Cuánto importa la representación de la observación, a igualdad de todo lo demás?**
> ¿Conviene diseñar features a mano sobre la imagen, dárselos crudos a una CNN, o
> agregar memoria temporal apilando frames?

Para que la comparación sea atribuible _solo_ a la observación, fijamos un entorno y un
reward idénticos entre las tres variantes (Sección 3) y variamos únicamente el bloque de
observación + el extractor de features de la política (Sección 4).

---

## 2. Formulación del problema (MDP)

Modelamos la tarea como un **MDP de horizonte finito** resuelto con PPO.

- **Estado del simulador**: pose y velocidad del robot, imagen de la cámara frontal,
  textura/geometría de la pista activa.
- **Observación del agente** `oₜ`: depende de la variante (Sección 4). Es una _vista
  parcial_ del estado.
- **Acción** `aₜ`: comando continuo de las dos ruedas (Sección 4.4).
- **Reward** `rₜ`: función del progreso sobre el circuito y de la seguridad
  (Sección 3.3). **Idéntica entre variantes.**
- **Terminación**: salida de pista, contramano sostenido, carril perdido, o vuelta
  completada (éxito). **Truncado** por límite de pasos.

### 2.1 Robot y simulador

| Ítem                         | Valor                                                              |
| ---------------------------- | ------------------------------------------------------------------ |
| Simulador                    | Webots R2025a                                                      |
| Arena                        | `RectangleArena` 8×8 m, multi-pista por _swap_ de textura del piso |
| Robot                        | e-puck, tracción **diferencial** (2 ruedas)                        |
| Cámara                       | frontal, RGB **84×84**, montada mirando levemente hacia abajo      |
| Frame-skip (`ACTION_REPEAT`) | 5 ticks por acción                                                 |
| Pasos máx. por episodio      | 1000 (`MAX_EPISODE_STEPS`)                                         |
| Domain randomization         | jitter de pose sobre el spawn al inicio del episodio               |

> **Nota de generalización**: el entrenamiento usa varias pistas (`track1`–`track3`) y
> la evaluación pistas _held-out_ (`track4`–`track5`), nunca vistas en entrenamiento.

---

## 3. Entorno y reward compartidos (lo que NO cambia entre variantes)

Esta es la parte central del diseño experimental: **el contrato entre el agente y el
entorno es idéntico en las tres variantes**, salvo la observación.

### 3.1 Arquitectura cliente–servidor

- El **supervisor** de Webots corre la simulación, calcula el reward y arma la
  observación.
- El **trainer** (PPO, Stable-Baselines3) se conecta por socket: manda acciones,
  recibe `(observación, reward, terminated, truncated, info)`.

El supervisor es la **única fuente de verdad** del reward (`_compute_reward_breakdown`).

### 3.2 Información privilegiada (asymmetric information)

**El reward puede mirar información que el agente no observa.**
El supervisor conoce el estado completo del simulador y lo usa para construir un reward
_denso y bien orientado_, pero esa información **no entra en la observación**:

| Señal privilegiada                           | Cómo se obtiene                                                                     | ¿Entra en la obs?       |
| -------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------- |
| **Progreso sobre el circuito** (Δarc-length) | Proyección de la pose `(x,y)` sobre la poligonal de _gates_ del track (`project_s`) | **No**                  |
| **Vuelta completa**                          | Progreso acumulado ≥ perímetro (o cruce de la línea de meta del spawn)              | **No**                  |
| **Contramano**                               | Δarc-length negativo sostenido                                                      | **No**                  |
| **Pose/velocidad ground-truth**              | API del supervisor de Webots                                                        | Solo velocidad propia\* |

\* La velocidad propia `[forward, yaw_rate]` sí entra (es propioceptiva, un sensor real
del robot); la **geometría global del track no**.

Esto es **privileged reward shaping** / _asymmetric actor_: una técnica estándar donde
el reward usa información de oráculo en entrenamiento, pero la política aprende a actuar
solo con su observación parcial. Permite un reward con **sentido de dirección** (premia
avanzar _en el sentido correcto del circuito_, no solo ir rápido) sin filtrar esa
información a la entrada del agente.

### 3.3 Función de reward

Estructura (idéntica en las tres variantes):

```
si vuelta_completa:        r = +LAP_BONUS              (terminal, éxito)
si pasto_en_el_centro:     r = +OFFTRACK_PENALTY       (terminal, salida de pista)
en otro caso:
    r_avance  = REWARD_PROGRESS_W · Δs_norm · clearance     # progreso×centrado
    r_base    = REWARD_BASE · clearance
    r_offset  = -REWARD_OFFSET_W · |offset|                  # (peso 0 por defecto)
    r_white   = -REWARD_WHITE_W · white_center               # (peso 0 por defecto)
    r_lost    = LINE_LOST_PENALTY  si no se ve calzada
    r_step    = -REWARD_STEP_COST
    r = r_base + r_avance + r_offset + r_white + r_lost + r_step

terminación adicional:
    contramano sostenido (wrong_way_steps ≥ N) -> terminal con WRONG_WAY_PENALTY
    carril perdido (lost_line_steps ≥ N)       -> terminal
```

Donde:

- **`clearance`** = fracción de calzada en la banda central de la imagen (∈ [0,1]):
  _gate_ multiplicativo, solo se cobra fuerte yendo **centrado**.
- **`Δs_norm`** = progreso normalizado sobre el circuito (privilegiado). Reemplaza a la
  velocidad cruda como motor del avance → no premia _loopear_ ni ir en contramano.
- **`offset`, `white_center`, `green_center`** = features de color de la imagen
  (centroide de la calzada, blanco/verde en el centro).

| Parámetro                           | Valor       | Significado                                    |
| ----------------------------------- | ----------- | ---------------------------------------------- |
| `LAP_BONUS`                         | +5.0        | bonus terminal por vuelta                      |
| `OFFTRACK_PENALTY`                  | −1.0        | penalización terminal por salir a pasto        |
| `OFFTRACK_GREEN_FRAC`               | 0.4         | umbral de pasto-en-centro para declarar salida |
| `REWARD_PROGRESS_W`                 | 1.0         | peso del progreso                              |
| `WRONG_WAY_PENALTY`                 | −1.0 a −1.5 | penalización terminal por contramano           |
| `WRONG_WAY_MAX_STEPS`               | 12–30       | pasos de retroceso antes de cortar             |
| `REWARD_OFFSET_W`, `REWARD_WHITE_W` | 0.0         | penas laterales (desactivadas)                 |
| `REWARD_STEP_COST`                  | 0.0–0.03    | costo por step                                 |

> **Implicación experimental**: como el reward y la terminación son idénticos, cualquier
> diferencia de desempeño entre variantes es atribuible a **la observación y su
> extractor**, no a la señal de entrenamiento.

---

## 4. Variantes (lo que SÍ cambia)

Las tres variantes comparten robot, acción, pistas, reward e hiperparámetros de PPO.
**Difieren solo en el par (observación, extractor de features de la política).**

### 4.1 Tabla comparativa

|                       | **Geométrica**                                    | **Visión (1 frame)**                         | **Visión apilada (4 frames)**        |
| --------------------- | ------------------------------------------------- | -------------------------------------------- | ------------------------------------ |
| **Rama git**          | `geometrica`                                      | `master`, `--n-stack 1`                      | `master`, `--n-stack 4`              |
| **Observación**       | Vector `Box(9)` de métricas de imagen             | `Dict{image: 3×84×84, velocity: 2}`          | `Dict{image: 12×84×84, velocity: 8}` |
| **Origen de la obs**  | Métricas calculadas **localmente** sobre el frame | Píxeles crudos + velocidad                   | 4 frames + 4 velocidades apilados    |
| **Política SB3**      | `MlpPolicy`                                       | `MultiInputPolicy`                           | `MultiInputPolicy`                   |
| **Extractor**         | MLP                                               | CNN (NatureCNN) + MLP                        | CNN (NatureCNN) + MLP                |
| **Info temporal**     | Solo velocidad propia                             | Solo velocidad propia                        | **Sí** (4 frames)                    |
| **Normalización obs** | VecNormalize sobre todo el vector                 | VecNormalize solo en `velocity`; imagen /255 | ídem                                 |

### 4.2 Observación — Geométrica

Vector de **9 dimensiones** (`Box`, float32), todo derivado **localmente** de la imagen
de la cámara en ese timestep (track-agnóstico, no usa geometría global):

```
[ forward, yaw_rate,          # velocidad propia (propioceptiva, ∈ ~[-1,1])
  road_frac,                  # fracción de calzada visible en la ROI
  green_center,               # pasto en la banda central (proximidad a salirse)
  off_0, off_1, off_2, off_3, off_4 ]   # offset horizontal del centroide de calzada
                                        # en 5 franjas de CERCA -> LEJOS (curvatura)
```

Las features se calculan por clasificación de color (asfalto+amarillo = calzada;
blanco = borde; verde = pasto) sobre la franja inferior del frame. Los `off_i` trazan
_hacia dónde va la pista adelante_ (look-ahead en la imagen), sin waypoints ni geometría
del circuito.

### 4.3 Observación — Visión y Visión apilada

Diccionario con dos claves:

- **`image`**: cámara frontal, RGB, channel-first `uint8` `(3, 84, 84)`. SB3 la
  normaliza dividiendo por 255.
- **`velocity`**: velocidad propia del cuerpo `[forward, yaw_rate]`, normalizada.

En **Visión apilada** se aplica `VecFrameStack(n_stack=4)`: la imagen pasa a `(12,84,84)`
(4 frames en el eje de canales) y la velocidad a `(8,)` (historia temporal). Esto le da
al agente **percepción de movimiento** (no solo una foto estática).

### 4.4 Acción (idéntica en las tres)

Vector continuo de **2 dimensiones** `Box([-1,1]²)`: `[rueda_izq, rueda_der]`.

- El robot mapea cada componente de `[-1, 1]` a velocidad de rueda en
  `[WHEEL_MIN_SPEED, WHEEL_MAX_SPEED] = [1.5, 5.0]` rad/s.
- **Ambas siempre positivas**: el robot **no puede frenar ni ir en reversa**.

### 4.5 Modelo y entrenamiento (idéntico en las tres)

- **Algoritmo**: PPO (Stable-Baselines3).
- **Política**: `MlpPolicy` (geométrica) / `MultiInputPolicy` (visión). El extractor de
  visión es el `CombinedExtractor` de SB3 (NatureCNN sobre la imagen + MLP sobre la
  velocidad, concatenados).
- **VecNormalize**: normaliza la observación (en visión, solo `velocity`; la imagen va
  cruda /255). Opcionalmente normaliza el reward.
- **Hiperparámetros** (defaults del trainer):

| Hiperparámetro  | Valor |
| --------------- | ----- |
| `learning_rate` | 5e-4  |
| `n_steps`       | 1024  |
| `batch_size`    | 128   |
| `n_epochs`      | 5     |
| `gamma`         | 0.995 |
| `target_kl`     | 0.02  |
| `ent_coef`      | 0.02  |
| `vf_coef`       | 0.5   |
| `clip_range`    | 0.2   |
| `max_grad_norm` | 0.5   |

---

## 5. Protocolo experimental

> Esta sección define **cómo** se corren los experimentos; los números van en la
> Sección 6.

### 5.1 Comandos de entrenamiento

```bash
# Visión apilada (4 frames) — rama master
python -m rl.trainer --n-stack 4 --webots-world worlds/track1.wbt --seed <S>

# Visión (1 frame) — rama master
python -m rl.trainer --n-stack 1 --webots-world worlds/track1.wbt --seed <S>

# Geométrica — rama geometrica
git checkout geometrica
python -m rl.trainer --n-stack 1 --webots-world worlds/track1.wbt --seed <S>
```

### 5.2 Seeds y agregación

- **N seeds** por variante (sugerido N ≥ 5): `--seed 0..N-1`.
- Mismo `--total-timesteps` para las tres.
- Se reportan **media ± desvío** (o IQM) sobre las seeds.

### 5.3 Evaluación

Sobre pistas _held-out_ (`track4`, `track5`), con el modelo final de cada seed:

```bash
python -m rl.evaluate --model <run_dir>      # todas las pistas de eval
```

La evaluación **guarda las métricas** en `<run_dir>/eval_results_<timestamp>.json`
(una entrada por track), para poder agregar entre seeds.

### 5.4 Métricas

**Convención**: cada métrica se mide en entrenamiento **[TRAIN]** (sobre las pistas
vistas, durante los rollouts) y/o en evaluación **[EVAL]** (sobre pistas held-out, con
el modelo final). Miden cosas distintas: TRAIN = *cuán rápido y bien aprende*; EVAL =
*cuán bien generaliza*.

**Dónde se guardan**:

- **[TRAIN]** → **TensorBoard** en `<run_dir>/tensorboard/` (namespace `custom/`),
  logueado por `RLMetricsCallback`.
- **[EVAL]** → **JSON** en `<run_dir>/eval_results_<timestamp>.json`, escrito por
  `rl/evaluate.py`.

| Métrica                       | Definición                                             | Fuente                                                          |
| ----------------------------- | ------------------------------------------------------ | -------------------------------------------------------------- |
| **Tasa de vuelta** (lap rate) | % de episodios que completan una vuelta                | **[EVAL]** `lap_rate` · **[TRAIN]** `custom/lap_rate`          |
| **Tiempo a vuelta**           | pasos/segundos hasta completar la vuelta (solo éxitos) | **[EVAL]** `lap_steps_mean`, `lap_time_s_mean`                 |
| **Tasa off-track**            | % de episodios terminados por salir a pasto            | **[EVAL]** `failure_rates` · **[TRAIN]** `custom/offtrack_rate` |
| **Tasa contramano**           | % terminados por contramano                            | **[EVAL]** `failure_rates` · **[TRAIN]** `custom/wrong_way_rate` |
| **Tasa carril perdido**       | % terminados por perder la pista                       | **[EVAL]** `failure_rates` · **[TRAIN]** `custom/line_lost_rate` |
| **Reward medio / episodio**   | retorno promedio                                       | **[EVAL]** `reward_ep_mean` · **[TRAIN]** `rollout/ep_rew_mean` |
| **Eficiencia de muestras**    | timesteps hasta alcanzar X% de lap rate                | **[TRAIN]** (deriva de la curva `custom/lap_rate`)             |

> **Nota metodológica**: el loop de eval corta al juntar `--laps` vueltas, por lo que
> `lap_rate` queda calculado sobre los episodios efectivamente corridos (sesgado si el
> modelo es bueno). Para una **tasa de éxito limpia**, correr con `--laps` alto o agregar
> un modo de **N episodios fijos** (pendiente).

---

## 6. Resultados

> **PLACEHOLDER — pendiente de correr N seeds por variante y agregar estadísticas.**

### 6.1 Curvas de aprendizaje — **[TRAIN]**

> **PLACEHOLDER**: lap rate y reward vs. timesteps, media ± desvío sobre seeds, una
> curva por variante. (Fuente: TensorBoard `custom/lap_rate`, `custom/offtrack_rate`,
> etc., sobre las pistas de entrenamiento.)

```
[figura: lap_rate vs timesteps — geométrica / visión / visión apilada]
[figura: reward medio vs timesteps]
```

### 6.2 Desempeño final (pistas held-out) — **[EVAL]**

> **PLACEHOLDER**: completar con media ± desvío sobre N seeds. (Fuente:
> `eval_results_*.json` de cada corrida.)

| Variante           | Lap rate (%) | Tiempo a vuelta | Off-track (%) | Contramano (%) | Reward/ep |
| ------------------ | ------------ | --------------- | ------------- | -------------- | --------- |
| Geométrica         | —            | —               | —             | —              | —         |
| Visión (1 frame)   | —            | —               | —             | —              | —         |
| Visión apilada (4) | —            | —               | —             | —              | —         |

### 6.3 Eficiencia de muestras — **[TRAIN]**

> **PLACEHOLDER**: timesteps hasta 50% / 80% de lap rate por variante. (Deriva de las
> curvas de §6.1.)

### 6.4 Tests de significancia — **[EVAL]**

> **PLACEHOLDER**: comparación entre variantes (p. ej. Mann-Whitney U sobre las
> métricas de eval por seed), para sostener las afirmaciones del paper.

---

## 7. Discusión

> **PLACEHOLDER** — depende de los resultados. Preguntas a responder:
>
> - ¿Las features a mano (geométrica) **igualan o superan** a la CNN cruda? Si sí,
>   sugiere que la información relevante de la imagen es de baja dimensión y el diseño
>   de features ahorra muestras.
> - ¿El **apilado de frames** (información temporal) mejora el control en curvas
>   cerradas o reduce el contramano?
> - ¿Qué variante es más **robusta** al cambio de pista (gap train→held-out)?

---

## 8. Limitaciones

- **La dirección de marcha no es perceptible desde una observación local.** Ni los
  features geométricos ni un solo frame codifican "para qué lado es adelante en el
  circuito". El reward lo resuelve con información **privilegiada** (progreso/contramano),
  pero la _política_ opera ciega a la dirección global: en pistas que se auto-aproximan,
  la cámara puede ver dos tramos y el agente saltar de carril. El apilado de frames
  mitiga parcialmente (da sentido de movimiento) pero no garantiza orientación global.
- **Escala de las pistas**: diseñadas a mano (Inkscape); curvas muy cerradas pueden
  exceder el radio de giro efectivo del robot diferencial.
- **Reward dependiente de color**: la detección de pista asume el esquema de color
  DeepRacer (blanco/amarillo/verde). Cambios de iluminación/textura podrían degradarla.

---

## 9. Trabajo relacionado / variantes paralelas (fuera del ablation)

- **Rama `ackerman`**: mismo problema con un robot de **dirección Ackermann** (auto
  1/18 estilo DeepRacer), acción `[steering, speed]`. Sirve para estudiar el efecto del
  **modelo cinemático** (el giro acotado del Ackermann limita el cambio brusco de
  rumbo). No forma parte del ablation de observación (cambia el robot y la acción).

---

## 10. Conclusión

> **PLACEHOLDER** — a redactar con los resultados.
>
> Tesis a defender: _a igualdad de reward, robot y acción, la representación de la
> observación es un factor de primer orden en la eficiencia y robustez del agente;
> el diseño de features y la información temporal compensan parte de la dificultad de
> aprender de píxeles crudos._

---

## Apéndice A — Reproducibilidad

- Reward y terminación: `controllers/supervisor_controller/supervisor_controller.py`
  (`_compute_reward_breakdown`, `_handle_step_env_request`).
- Features geométricos: `helpers/geom_obs.py`, `helpers/lane_vision.py`
  (`detect_lane`, `road_band_offsets`).
- Observación de visión: `rl/env.py`, `helpers/image_obs.py`.
- Progreso por gates: `helpers/track_progress.py` (`build_loop`, `project_s`,
  `signed_delta`).
- Trainer / hiperparámetros: `rl/trainer.py`.
- Evaluación: `rl/evaluate.py`.
- Pistas y spawns/gates: `worlds/`, `spawns.json`.
