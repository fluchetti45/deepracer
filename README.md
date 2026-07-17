# De la percepción privilegiada a los píxeles

**Destilación de políticas para conducción autónoma** — qué representación de la observación aprende a
conducir en un agente de carreras estilo **AWS DeepRacer**, y por qué **destilar un maestro geométrico**
gana. Entrenado con **PPO (Stable-Baselines3)** dentro del simulador **Webots**.

> 📄 **Página interactiva (paper):** <https://fluchetti45.github.io/deepracer/>
> — versión navegable con las figuras a color, tablas y saliencia. Este README la espeja.
> **Versión LaTeX (IEEE):** [`paper/`](paper/) · [`paper/main.pdf`](paper/main.pdf).

---

## Resumen

Comparamos **cinco representaciones de la observación** para un agente de conducción autónoma, entrenadas
con PPO en Webots sobre el **mismo entorno, recompensa, presupuesto (1M timesteps) y configuración** —
*lo único que cambia es la observación*:

- un **agente geométrico privilegiado** (features de calzada extraídas de la cámara),
- tres **agentes de visión** entrenados con RL (un frame, cuatro frames apilados, y uno recurrente con LSTM),
- un **agente de visión destilado** del geométrico por clonación de comportamiento.

Todas las variantes reciben además una **lectura propioceptiva de velocidad** (avance y *yaw-rate* del
cuerpo, del ground-truth del simulador); lo único que cambia entre ellas es **cómo perciben la pista**
(features explícitas vs imagen cruda). La política de visión es *MultiInput*: NatureCNN sobre la imagen +
un MLP sobre la velocidad.

Sobre pistas *held-out* y con **randomización del fondo**, el agente destilado **se acerca al maestro
privilegiado** —92.5 % vs 97.0 % de vueltas completadas, una brecha pequeña de ~5 puntos— y es la
**única variante de visión estable** (las cinco seeds convergen), completando la vuelta en **menos pasos
de control**; entrenar la visión directamente con RL es **inestable** —algunas seeds colapsan— y
agregar estructura temporal (stacking, recurrencia) **la empeora**. Mapas de saliencia muestran que la
destilación **desplaza la atención de la CNN del fondo hacia la calzada**, lo que explica su invarianza al
entorno.

---

## El pipeline de destilación

Una misma cámara alimenta dos caminos. El **baseline** entrena la visión con RL directo desde la
recompensa. **Nuestro método** primero entrena un maestro geométrico privilegiado, colecta sus decisiones
con ruido DART (para cubrir estados de recuperación) y entrena por clonación un estudiante de visión que
hereda *qué mirar*.

La clonación de comportamiento **cambia la naturaleza del problema**: en vez de optimizar la visión por
refuerzo desde la recompensa —con su *asignación de crédito* (credit assignment) de alta varianza a través
de píxeles— resuelve un **aprendizaje supervisado directo** con targets densos, mucho más fácil de optimizar.

```mermaid
flowchart LR
    CAM["Cámara 84×84×3<br/>(Webots)"]
    CAM --> T["Maestro geométrico<br/>PPO · privilegiado"]
    CAM -.-> RL["Visión-RL<br/>PPO desde reward"]
    T --> D["Colecta con DART<br/>img → acción · ~55k pares · σ=0.15"]
    D --> S["Estudiante de visión<br/>clonación (BC) · CNN 1 frame"]
    S --> OUT["Visión destilada<br/>92.5 % vueltas ✓"]
    RL -.-> BAD["Inestable ✗<br/>10–90 % · sobreajusta el fondo"]

    classDef ours fill:#0E8C79,stroke:#0E8C79,color:#ffffff;
    classDef good fill:#DFF0EA,stroke:#0E8C79,color:#0C6E56;
    classDef bad fill:#F6E1DC,stroke:#A34428,color:#A34428;
    class T,D,S ours;
    class OUT good;
    class RL,BAD bad;
```

### Las representaciones de la observación

| Variante | Régimen | Observación |
|---|---|---|
| **Geométrica** | privilegiado (RL) | Vector de 9 features de percepción (2 de velocidad + 7 de calzada). Techo de referencia. |
| **Visión (1 frame)** | visión-RL | CNN sobre un frame RGB crudo. La variante de visión más simple. |
| **Visión apilada (4)** | visión-RL | Cuatro frames apilados para dar información temporal. |
| **Visión + LSTM** | visión-RL | Política recurrente: la memoria la aporta el LSTM en vez del stacking. |
| **Visión destilada** | destilado (BC) | CNN de 1 frame entrenada por clonación para imitar al maestro geométrico. |
| **Visión pura (cámara)** *(ablación)* | visión-RL | Igual a *Visión 1 frame* pero **sin** la propiocepción de velocidad: obs = solo imagen. Aísla cuánto aporta esa velocidad de 2D. |

---

## Resultados

Evaluación sobre **dos pistas held-out** (`track9`, `track10`) con **fondo aleatorio** y semilla de
evaluación fija (todos los modelos ven exactamente los mismos episodios). Métrica: fracción de vueltas
completadas. **5 seeds por variante** (30 modelos en total).

Valores: **IQM (media intercuartil) con IC del 95 % por _bootstrap_ de percentil** (2×10⁴ remuestreos)
sobre 5 seeds, en formato `IQM [IC 95%]`. La rapidez se mide en **pasos de control por vuelta** (no en
segundos, que dependen de la velocidad de simulación) y es *condicional a completar la vuelta*.

| Variante | Régimen | Lap rate (%) | Off-track (%) | Pasos/vuelta | Reward/ep |
|---|---|--:|--:|--:|--:|
| **Geométrica** | privilegiado (RL) | **97.5** [93, 100] | 1.7 [0, 5.8] | 983 [975, 1036] | 179 [174, 184] |
| **Visión destilada** | destilado (BC) | **92.5** [87, 98] | 7.5 [1.7, 13.3] | **766** [751, 807] | 139 [131, 143] |
| Visión (1 frame) | visión-RL | 90.0 [67, 97] | 7.5 [0.8, 33.3] | 769 [684, 818] | 134 [88, 173] |
| Visión pura (cámara) *(ablación)* | visión-RL | 75.0 [26, 91] | 19.2 [1.7, 70.0] | 851 [689, 940] | 126 [71, 185] |
| Visión apilada (4) | visión-RL | 66.7 [21, 89] | 23.3 [5.8, 52.5] | 740 [561, 963] | 140 [96, 195] |
| Visión + LSTM | visión-RL | 69.2 [33, 89] | 30.8 [9.2, 66.7] | 697 [639, 762] | 82 [45, 106] |

<sub>**Visión pura (cámara)** es la ablación sin propiocepción (obs = solo imagen): baja el lap rate de 90.0
a 75.0 y más que duplica el off-track vs *Visión 1 frame* — incluso la visión más simple se apoya en la
velocidad propioceptiva como muleta.</sub>

> **La visión destilada se acerca al maestro privilegiado** y es la **única variante de visión estable**:
> IQM 92.5 % vs 97.5 % (~5 puntos), pero con IC bootstrap del 95 % **angosto** ([87, 98]) frente a los de la
> visión-RL, que abarcan casi todo el rango (apilada [21, 89], LSTM [33, 89]). El **TOST a ±5 puntos no es
> concluyente con n=5** (p<sub>TOST</sub>=0.43; IC 90 % de la diferencia [−1.1, +10.1]): hay una brecha
> pequeña pero real de ~5 puntos —la destilada hereda el techo del maestro sin igualarlo punto por punto—.
> Aun así es **el conductor confiable más rápido**: completa la vuelta en menos pasos (766 vs 983).

![Lap rate held-out por variante](docs/img/fig_eval_lap_rate.png)

### La visión-RL es una lotería; la destilación, no

El promedio esconde lo esencial: las variantes temporales son **bimodales** — algunas seeds convergen,
otras colapsan. La destilación (y el maestro) rinden parejo en las 5.

| Variante | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|--:|--:|--:|--:|--:|
| Geométrica | 93 | 98 | 100 | 95 | 100 |
| Visión destilada | 90 | 95 | 85 | 100 | 93 |
| Visión (1 frame) | 90 | 95 | 58 | 98 | 85 |
| Visión pura (cámara) *(ablación)* | 58 | 80 | 10 | 88 | 93 |
| Visión apilada (4) | 63 | 98 | 73 | 0 | 65 |
| Visión + LSTM | 45 | 28 | 93 | 80 | 83 |

La enorme varianza de apilada y LSTM (apilada de 0 a 98, LSTM de 28 a 93) es el síntoma: la temporalidad agrega
**dificultad de optimización**, no desempeño. La equivalencia Geométrica–Destilada se evalúa con el **TOST**
de arriba (no con Mann-Whitney: un *p* alto no prueba igualdad).

---

## Por qué funciona: la CNN deja de mirar el fondo

Guided backprop (qué píxeles mueven la decisión) calculado **sobre este mismo frame** —una curva— para el
**valor del crítico** V(s) y para el **steering**, en las cuatro variantes de visión (todas seed 0). La
saliencia corresponde exactamente a la imagen mostrada (no es un promedio).

![Saliencia de las cuatro variantes de visión](docs/img/fig_saliency.png)

**Las tres variantes de visión-RL** (1 frame, stacked, LSTM) encienden la saliencia en una **franja sobre el
horizonte** —el muro y las montañas del fondo—: usan el entorno, constante en entrenamiento, como atajo.
**La destilada** desplaza la atención hacia **abajo, sobre la calzada y el borde curvo del carril**, con el
horizonte apagado: hereda del maestro geométrico *qué mirar*. Por eso se sostiene bajo randomización del
fondo. Dentro de cada modelo, **value** y **steer** coinciden (comparten el extractor CNN).

### Curvas de aprendizaje

Las cuatro variantes RL (5 seeds, banda = ± desvío). La visión-RL aprende más lento y con bandas anchas;
el geométrico converge parejo. La destilada es supervisada (no aparece en estas curvas de RL).

| Reward por episodio | Lap rate held-out durante el entrenamiento |
|---|---|
| ![Reward](docs/img/fig_reward.png) | ![Lap rate](docs/img/fig_lap_rate.png) |

---

## Setup experimental

Las seis variantes comparten todo salvo la observación. Adoptamos **PPO** por su estabilidad y eficiencia
demostradas en control continuo y navegación autónoma. Hiperparámetros reales de los runs finales
(`models/<id>/run_metadata.json`):

**PPO (las cuatro variantes RL)**

| | |
|---|---|
| Algoritmo | PPO (Stable-Baselines3) |
| Política | MultiInput Actor-Critic — NatureCNN (imagen) + MLP (velocidad) |
| Timesteps | 1 000 000 |
| Entornos paralelos | 4 |
| n_steps / rollout | 256 (1 024 transiciones/actualización) |
| Batch size · Épocas | 128 · 5 |
| Learning rate | 5 × 10⁻⁴ |
| γ · Clip · Ent · VF | 0.995 · 0.2 · 0.02 · 0.5 |
| Normalización | VecNormalize (reward) |
| Seeds | 0, 1, 2, 3, 4 |

**Destilación (BC)**

| | |
|---|---|
| Objetivo | Clonación de comportamiento (MSE sobre la acción) |
| Maestro | Geométrico privilegiado, misma seed que el estudiante |
| Colecta | Apareada: limpia (σ=0) + DART (σ=0.15) |
| Pares (estado, acción) | ≈ 55 000 por seed |
| Épocas · LR | 30 · 3 × 10⁻⁴ |
| Backbone | NatureCNN, 1 frame (idéntico a Visión-RL 1 frame) |

**Entorno & evaluación**

| | |
|---|---|
| Simulador | Webots R2025a · Stable-Baselines3 / sb3-contrib |
| Observación (visión) | Imagen RGB 84 × 84 × 3 (× n_stack) **+ velocidad propioceptiva [avance, yaw-rate]** |
| Observación (geométrica) | 9 features de percepción (2 de velocidad + 7 de calzada) |
| Ablación *camera-only* | Obs = **solo la imagen** (sin la velocidad); misma CNN, sin la rama de velocidad |
| Acción | Box continuo: velocidades de las 2 ruedas (tracción diferencial), remapeadas a `[v_min, v_max]` rad/s |
| Reward | Progreso sobre la pista − penalizaciones (compartido) |
| Randomización de dominio | Textura de pared + skybox rotadas por episodio |
| Evaluación | 2 pistas held-out · 20 ep/pista · fondo aleatorio · `--eval-seed 0` |

---

## Reproducir

```powershell
# 1. Instalar
python -m venv env; .\env\Scripts\Activate.ps1
pip install -r requirements.txt
# El .env viene versionado en cada rama (la config de cámara/robot cambia por variante).

# 2. Entrenar una variante suelta (el trainer lanza Webots)
python -m rl.trainer --total-timesteps 1000000 --n-stack 1 --n-envs 4 --n-steps 256 --norm-reward

# 3. Entrenar TODAS las variantes/seeds y evaluarlas (switch de rama automático)
python run_full_pipeline.py            # train (5 variantes × seeds) + eval de esas mismas seeds

# 3b. Ablación camera-only (obs = SOLO imagen, sin propiocepción) — en master
python -m rl.run_experiment --seeds 0 1 2 3 4 --total-timesteps 1000000 --n-stack 1 --n-envs 4 --n-steps 256 --episodes 20 --camera-only

# 4. Destilación apareada geométrico → visión  (checkout vision_distill primero)
git checkout vision_distill
python run_all_distill.py --seeds 0 1 2 3 4        # colecta (limpio+DART) + BC por seed

# 5. Evaluar los 30 modelos con fondo aleatorio y misma secuencia de episodios
python run_all_evals.py --discover --episodes 20 --randomize-background --eval-seed 0
```

> Las variantes viven en **ramas distintas** porque el observation-space cambia: `master` (visión 1 frame
> y stacked), `geometrica`, `vision_lstm`, `vision_distill`. **El pipeline de destilación** (`rl/distill.py`,
> `rl/collect_teacher.py`, `run_all_distill.py`) vive en la rama **`vision_distill`**. Los scripts
> `run_all_*` / `run_full_pipeline` hacen el `git checkout` por vos; corré siempre con el working tree limpio.

Cada corrida genera `models/<timestamp>/` con `final_model.zip`, `vecnormalize.pkl`, `tensorboard/` y
`run_metadata.json`. El análisis se corre desde `analysis/`:

```powershell
python -m analysis.run_analysis --since 20260713   # descubre runs + results_summary.{json,md} + curvas
python -m analysis.fig_eval                          # barras de lap rate (fig_eval_lap_rate.png)
python -m analysis.robust_stats                      # IQM [IC 95% bootstrap] + TOST (incluye camera-only)
```

<sub>`--since 20260713` se queda solo con la tanda final (config de mayor velocidad); sin él, `discover`
agruparía esas corridas con tandas previas del mismo largo (1M pasos), que el guard de timesteps no distingue.</sub>

---

## Conclusiones

- Un agente **privilegiado** con percepción limpia de la calzada resuelve la tarea (97 %).
- **Destilarlo a un agente de visión** recupera casi ese desempeño (92.5 %; una brecha de ~5 puntos) con la
  ventaja decisiva de ser la **única variante de visión estable**, y completando la vuelta en menos pasos de control.
- Entrenar la **misma visión con RL directo** es poco confiable, y **la temporalidad la empeora**: stacking
  y recurrencia desestabilizan la optimización (KL disparado, crítico colapsado) en vez de ayudar.
- **Lectura:** el límite de la visión-RL no es la *capacidad* de la representación, sino la *asignación de
  crédito* a través de píxeles en un problema parcialmente observable. La destilación lo esquiva con targets
  supervisados densos del agente privilegiado.

## Limitaciones

- **Solo simulación** — no evaluamos transferencia a hardware real; el maestro usa features extraídas en el
  simulador, que en el mundo real habría que estimar con percepción. *Dicho esto*, la saliencia sugiere una
  hipótesis optimista para **sim-to-real**: como la destilada aprendió a mirar la calzada e ignorar el fondo
  (justo lo que la randomización vuelve inservible), es mejor candidata a transferencia que la visión-RL
  directa, que colapsaría al desaparecer el fondo específico de Webots.
- **Familia de pistas acotada** — dos pistas held-out de la misma distribución de arena.
- **Randomización de dominio parcial** — varía pared y skybox, no geometría, iluminación ni cámara.
- **El estudiante hereda el techo del maestro** — el BC no puede superar la política del geométrico (la
  mejora en tiempo viene de acciones más suaves).
- **Sensibilidad a la semilla** — la bimodalidad de la visión-RL sugiere alta sensibilidad a seed/HP.

## Related work

- **Chen et al., 2019** — *Learning by Cheating* (CoRL). Destilar un agente privilegiado a uno de visión: la base directa.
- **Ross et al., 2011** — *DAgger* (AISTATS). Corrimiento de covariables del BC ingenuo.
- **Laskey et al., 2017** — *DART* (CoRL). Ruido en la acción del maestro para cubrir estados de recuperación (σ=0.15).
- **Tobin et al., 2017** — *Domain Randomization* (IROS). Randomizar lo irrelevante fuerza invarianza.
- **Schulman et al., 2017** — *PPO* (arXiv). El algoritmo de RL on-policy usado.
- **Mnih et al., 2015** — *NatureCNN* (Nature). El extractor convolucional de la política de visión.
- **Balaji et al., 2020** — *DeepRacer* (ICRA). La plataforma que motiva el escenario.

---

## Arquitectura del sistema

Tres procesos que se comunican por socket:

- **`controllers/agent_controller/`** — corre en el robot (cámara + ruedas). Aplica acciones y devuelve la observación sensorial.
- **`controllers/supervisor_controller/`** — servidor del environment: atiende `reset_env` / `step_env`, avanza la simulación, calcula el reward (desde la cámara) y aplica la randomización de dominio del fondo.
- **`rl/`** — el lado de entrenamiento. `NavEnv` (Gymnasium) habla con el supervisor por TCP; `trainer.py` arma PPO con `VecNormalize` + `VecFrameStack`. `distill.py` hace la clonación; `evaluate.py` corre las evals.

```
deepracer/
├── controllers/
│   ├── agent_controller/        # robot (camara + ruedas)
│   └── supervisor_controller/   # servidor del environment + reward + domain randomization
├── helpers/                     # lane_vision, image_obs, puentes de socket, policy_runner
├── rl/
│   ├── env.py                   # NavEnv (Gymnasium) sobre socket
│   ├── trainer.py               # entrenamiento PPO
│   ├── evaluate.py              # evaluacion (--randomize-background, --eval-seed)
│   ├── distill.py               # clonacion de comportamiento (BC)   [rama vision_distill]
│   └── collect_teacher.py       # colecta del maestro (limpio + DART)  [rama vision_distill]
├── analysis/                    # agregado de evals, saliencia, figuras
├── run_full_pipeline.py         # train + eval de todas las variantes/seeds
├── run_all_experiments.py       # solo train (switch de rama automatico)
├── run_all_evals.py             # solo eval  (--discover)
├── run_all_distill.py           # destilacion apareada multi-seed      [rama vision_distill]
├── docs/img/                    # figuras del README
├── worlds/                      # mundos y texturas de Webots
├── requirements.txt
└── .env.example
```

---

## Cómo citar

```bibtex
@mastersthesis{luchetti2026privilegiada,
  title   = {De la percepci{\'o}n privilegiada a los p{\'i}xeles:
             destilaci{\'o}n de pol{\'i}ticas para conducci{\'o}n aut{\'o}noma},
  author  = {Luchetti, Faustino},
  year    = {2026},
  note    = {Proyecto de tesis. Webots R2025a, Stable-Baselines3 (PPO)},
  url     = {https://github.com/fluchetti45/deepracer}
}
```

---

## Requisitos

- **[Webots](https://cyberbotics.com/) R2025a** (o compatible) · **Python 3.10+**
- Dependencias de [`requirements.txt`](requirements.txt): `stable-baselines3[extra]`, `sb3-contrib`, `gymnasium`, `python-dotenv`, `numpy`.
- El `.env` (versionado por rama) define la resolución de cámara (`CAMERA_WIDTH/HEIGHT`), que **debe coincidir** con el nodo `Camera` del robot en el mundo de Webots, además de los pesos del reward y la config de domain randomization.
