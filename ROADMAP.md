# ROADMAP — Agente de conducción autónoma estilo DeepRacer (Webots + PPO)

Evolución del proyecto: de un **line follower** a un agente estilo DeepRacer con
**object avoidance**, comparando representaciones de la observación y subiendo la
complejidad del entorno por etapas. Este documento resume **lo hecho** y **lo que viene**.

> Convención: ✅ hecho · 🔧 en progreso · ⬜ pendiente · 💤 diferido / opcional.

---

## Visión general (la "escalera" del proyecto)

```
line follower (CNN)                                   [punto de partida, proyecto previo]
        │
        ▼
DeepRacer time-trial  ── 3 representaciones de obs ──  geométrica / visión / visión apilada   ✅
        │
        ▼
object avoidance ESTÁTICO  (cajas + penalización de choque)                                    ✅ (infra)
        │
        ▼
object avoidance MÓVIL scripteado  (bot cars siguiendo la pista)                               ⬜
        │
        ▼
self-play  (rival = política que aprende)  → liga / fictitious self-play                       💤 lejano
```

Ejes paralelos (fuera de la línea principal): sim-to-real, arquitectura del extractor,
memoria recurrente, imitation / model-based.

---

## 1. Hecho ✅

### 1.1 Plataforma base
- ✅ Simulación en **Webots R2025a**: `RectangleArena` 8×8, multi-pista por *swap* de
  textura del piso, domain randomization de la pose de spawn.
- ✅ Arquitectura **supervisor (servidor) ↔ trainer (PPO/Stable-Baselines3)** por socket.
- ✅ Dos robots: **e-puck diferencial** (acción `[izq, der]`) y **auto Ackermann 1/18**
  (acción `[steering, speed]`, RWD + dirección Ackermann real).

### 1.2 Reward con información privilegiada
- ✅ Reward computado por el supervisor con **info de oráculo que NO entra en la obs**:
  progreso por *gates* (`project_s`), detección de **vuelta**, **contramano**, y
  **off-track** por pasto. (asymmetric / privileged reward shaping).
- ✅ Detección de vuelta por **línea de meta** (spawn) como fallback sin gates.

### 1.3 Tres representaciones de la observación (el ablation del paper)
- ✅ **Geométrica** (rama `geometrica`): obs = vector de métricas derivadas **localmente**
  de la imagen (`road_frac`, `green_center`, band offsets), `MlpPolicy`. Reward **idéntico**
  al de visión (privilegiado) → ablation limpio.
- ✅ **Visión** (rama `master`, `--n-stack 1`): obs = imagen + velocidad, `MultiInputPolicy`
  (NatureCNN + MLP).
- ✅ **Visión apilada** (rama `master`, `--n-stack 4`): + información temporal (VecFrameStack).
- ✅ **Ackermann** (rama `ackerman`): mismas ideas con el auto Ackermann, drive-type modular.

### 1.4 Pipeline de experimentación
- ✅ **`rl/run_experiment.py`**: orquesta **train + eval de N seeds** de una variante.
  Por seed: entrena → detecta `run_dir` → evalúa sobre pistas held-out → escribe manifest.
  Soporta `--n-envs/--n-steps/--base-port` (paralelizar cada agente). Presente en las 3 ramas.
- ✅ **`rl/evaluate.py`**: modo **`--episodes`** (tasa de éxito sin sesgo) + **persistencia**
  de métricas a `<run_dir>/eval_results_<ts>.json` (lap_rate, reward/ep, failure_rates, tiempos).
- ✅ Métricas de **train** en TensorBoard (`custom/*`) vía `RLMetricsCallback`.
- ✅ **`PAPER.md`** (en `master`): informe del ablation de representación, con etiquetas
  TRAIN/EVAL y fuentes de cada métrica.

### 1.5 Object avoidance estático (rama `ackerman_obstacle`)
- ✅ **Cajas físicas** (pool `DEF OBSTACLE_*`, naranjas, 0.1 m) colocables sobre la pista.
- ✅ **Detección de colisión OBB**: el auto se modela como **rectángulo orientado**
  (`helpers/obstacle_geom.py`), choque si la caja queda a ≤ `OBSTACLE_HIT_DIST` del borde
  → capta golpes con esquina/rueda sin falsos positivos al pasar al costado. **Penalización
  terminal fuerte** (`OBSTACLE_PENALTY`).
- ✅ **Banco de prueba manual** para calibrar la detección:
  `worlds/obstacle_test.wbt` + `controllers/keyboard_drive` (manejo por teclado) +
  `controllers/obstacle_monitor` (imprime el clearance y avisa el choque) +
  `launch_manual.py --world ...` (lanza Webots GUI con el venv del proyecto cargado).

---

## 2. Próximo ⬜

### 2.1 Entrenar object avoidance estático
- ⬜ Entrenar la rama de **visión** con las cajas + penalización terminal, y medir si
  esquiva. (La rama **geométrica está ciega a los obstáculos**: sus features solo codifican
  pista/pasto; necesitaría features del obstáculo para esquivar.)

### 2.2 Densificar la penalización — near-miss 💤 (después de entrenar como está)
- ⬜ Penalización **graduada por proximidad** usando el `clearance` ya calculado (empieza a
  doler antes de tocar) para acelerar el aprendizaje vs. la señal terminal *sparse* actual.

### 2.3 Head-to-head: rivales móviles scripteados
- ⬜ **Bot cars**: autos rival (reusar el Ackermann) con un controlador **path-follower
  sobre los gates** a velocidad fija. NO aprenden (son dinámica del entorno).
- ⬜ La detección OBB se **reusa tal cual** (el obstáculo ahora se mueve).
- ⬜ **Frame stacking obligatorio** (percibir velocidad relativa / closing speed).
- ⬜ **Randomizar el color del bot por episodio** (setear `baseColor` del `PBRAppearance`
  en el reset): evita **shortcut learning** (que la CNN aprenda "evitar color X" en vez de
  "evitar un auto") → representación color-invariante. Las libreas de color son cosméticas.

---

## 3. Diferido / lejano 💤

### 3.1 Self-play (rival que aprende)
- 💤 Reemplazar el bot scripteado por una **política** (al principio, copia **congelada** del
  agente → MDP estacionario por ronda; actualizar el pool cada N iteraciones).
- 💤 Luego **fictitious self-play** (pool de versiones pasadas) → **liga** (tipo AlphaStar).
  No estacionario, necesita entrenamiento por población. Ambición lejana.

### 3.2 Memoria recurrente (LSTM) — solo ablation del paper
- 💤 El horizonte temporal útil acá es **corto** (velocidad, closing speed) → el frame
  stacking ya lo captura. Se espera que el LSTM **no mejore** (y sea más finicky). Tratarlo
  como **ablation barato / resultado negativo válido** ("stacking alcanza"), NO como un paso
  de performance. No priorizarlo.

### 3.3 Arquitectura del extractor (CNN)
- 💤 La capacidad **rara vez** es el cuello de botella en RL; agrandar NatureCNN suele empeorar.
  Si sube la **resolución** de cámara o se estanca la **generalización**, el upgrade canónico
  es **IMPALA-CNN (ResNet) + data augmentation (DrQ/RAD)**, no "NatureCNN más ancha".
- 💤 Para el paper: **mantener la CNN fija** entre variantes de visión (comparación limpia);
  la arquitectura es un **eje de ablation aparte**.
- 💤 Diagnóstico antes que tamaño: *underfit* (falla en train → más capacidad/training) vs
  *overfit* (falla en held-out → augmentation/regularización, NO agrandar).

### 3.4 Otros ejes
- 💤 **Sim-to-real**: desplegar en el auto físico 1/18 (domain randomization fuerte, brecha de
  dinámica). El verdadero "fin de juego" de DeepRacer; mete hardware.
- 💤 **Imitation learning** (clonar experto + RL fine-tune) — arranca sin chocar 10000 veces.
- 💤 **Model-based** (Dreamer/MuZero) — eficiencia de muestras; muy alto costo.

---

## 4. Ramas

| Rama | Qué es | Estado |
|---|---|---|
| `master` | Visión (1 frame / apilada), e-puck diferencial + `PAPER.md` | ✅ |
| `geometrica` | Obs = features de percepción local, `MlpPolicy` | ✅ |
| `ackerman` | Auto Ackermann 1/18, acción `[steering, speed]`, drive modular | ✅ |
| `ackerman_obstacle` | Ackermann + object avoidance estático + banco de prueba | 🔧 (sin entrenar) |

---

## 5. Notas para el paper

- Tesis central: **a igualdad de reward, robot y acción, la representación de la observación
  es un factor de primer orden**. El reward privilegiado idéntico hace el ablation limpio.
- Ejes de ablation separados (no mezclar): **observación** (geométrica/visión/stacked) ·
  **arquitectura** (NatureCNN/IMPALA) · **memoria** (stacking/LSTM).
- Un **resultado negativo** controlado (p. ej. "LSTM no mejora sobre stacking") es un hallazgo
  válido y barato.
- Evaluación con **N seeds** + estadísticas (media ± desvío), tasa de éxito con `--episodes`.
