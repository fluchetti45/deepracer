# Ramas del proyecto

Dos enfoques de conducción para el mismo agente de RL (PPO, visión). Comparten el
**pipeline** (entorno gym, supervisor, reward por gates, observación), y se diferencian
en el **robot**, el **espacio de acción** y la **cinemática**.

| | `master` | `ackerman` |
|---|---|---|
| **Robot** | e-puck (diferencial) | Auto Ackermann estilo DeepRacer (1/18) |
| **Tracción** | 2 ruedas, control diferencial | Trasera (RWD) + dirección Ackermann delantera |
| **Acción** | `[rueda_izq, rueda_der]` | `[steering, speed]` |
| **Cinemática** | puede pivotar en el lugar | radio de giro mínimo (no puede girar 180° en el lugar) |
| **World** | `worlds/track1.wbt` | `worlds/ackermann.wbt` |
| **Texturas** | `track1..5.png` (angostas) | `track1..5_ackermann.png` (más anchas) |
| **`drive` en spawns.json** | `"differential"` | `"ackermann"` |
| **Radio de rueda** | 0.02 m | 0.03 m |

> El tipo activo se elige con `--drive {ackermann,differential}` (default `DRIVE_TYPE`
> del `.env`). Esa perilla selecciona el world, filtra los tracks de `spawns.json` a ese
> drive, y queda registrada en `run_metadata.json` (`drive_type`).

---

## Lo que es IGUAL en las dos ramas

### Observación (`rl/env.py`)
Diccionario, idéntico en ambas:

- **`image`**: cámara frontal RGB, `84×84`, layout CHW → shape `(3, 84, 84)` `uint8`.
- **`velocity`**: `(2,)` = `[forward, yaw_rate]` del cuerpo en frame local, normalizado a
  `~[-1, 1]`. Lo computa el supervisor con `getVelocity()`/`getOrientation()` del nodo
  (propriocepción ground-truth).

Con `VecFrameStack(n_stack=4)`: la imagen se apila en canales → `(12, 84, 84)`, y la
velocidad como historia → `(8,)`. `VecNormalize` normaliza solo `velocity`.

### Reward (supervisor, `helpers/track_progress.py` + `helpers/lane_vision.py`)
Economía **progress-only** (solo paga avanzar sobre el circuito):

- **Avance** = Δarc-length sobre los gates (centerline), gateado por `clearance` (calzada
  en el centro de la vista). Adelante suma, atrás resta, girar en el lugar ≈ 0.
- **Off-track** (pasto en el centro) → terminal con penalización.
- **Vuelta completa** → terminal con `LAP_BONUS`. Sin gates, se usa el cruce de la
  **línea de meta única** (sobre el spawn, en el sentido del spawn).
- **Contramano** sostenido → terminal con penalización.
- `REWARD_BASE=0`, `REWARD_STEP_COST>0`: estancarse/loopear es netamente negativo.

### Entrenamiento / evaluación
`rl/trainer.py` (PPO, `MultiInputPolicy`) y `rl/evaluate.py` (cronometra vueltas) son los
mismos; el `--drive` los enruta al world y tracks correctos.

---

## Rama `master` — diferencial (e-puck)

- **Robot**: e-puck, 2 ruedas con `RotationalMotor` en velocidad.
- **Acción** (`controllers/agent_controller/agent_controller.py`):
  `[rueda_izq, rueda_der]`, cada una `[-1, 1]` → `[WHEEL_MIN_SPEED, WHEEL_MAX_SPEED]`
  (ambas **positivas**: nunca reversa ni freno total).
- **Cinemática**: control diferencial → **puede pivotar en el lugar**. Esto le permite
  invertir el sentido sobre la pista y "perderse" (loopear), porque la dirección no es
  observable en una calzada simétrica.

## Rama `ackerman` — auto Ackermann (estilo DeepRacer)

- **Robot** (`worlds/ackermann.wbt`): auto 1/18 con dimensiones tipo DeepRacer.
  - wheelbase **0.16 m**, trocha **0.13 m**, radio de rueda **0.03 m**, masa **~1.5 kg**.
  - Tracción trasera (2 `RotationalMotor` en velocidad, `maxTorque` limitado) + dirección
    delantera (2 `RotationalMotor` en posición).
- **Acción** (`agent_controller.py`):
  - `steering` `[-1, 1]` → ángulo de dirección `±MAX_STEER_ANGLE_DEG` (30°), con
    **Ackermann real**: la rueda interna gira más que la externa (`ackermann_angles`),
    apuntando al mismo centro de giro → sin scrub lateral.
  - `speed` `[-1, 1]` → velocidad de ruedas traseras `[WHEEL_MIN_SPEED, WHEEL_MAX_SPEED]`,
    siempre **positiva**.
- **Cinemática**: el ángulo de dirección acotado da un **radio de giro mínimo**: el auto
  **no puede pivotar ni hacer un 180° en el lugar**. Arrancando bien encarado en el spawn,
  preserva el sentido de marcha de forma intrínseca (igual que el DeepRacer real).

### Simplificaciones actuales del modelo Ackermann (fase 1)
- **RWD** (el DeepRacer real es 4WD).
- Cámara aún `84×84`, FOV `0.84` (el DeepRacer real: 160×120 grises, FOV ancho/fisheye).
- Dimensiones aproximadas (falta la pasada de fidelidad sim2real).

---

## Cómo elegir el modo

```bash
# Ackermann (auto) — default en la rama ackerman:
python -m rl.trainer --total-timesteps 50000 --n-stack 4
python -m rl.evaluate --model models/<run> --laps 3

# Diferencial (e-puck) — sin cambiar de rama:
python -m rl.trainer --drive differential
python -m rl.evaluate --model models/<run> --drive differential --laps 3
```

El `.env` fija el default (`DRIVE_TYPE`), `WHEELBASE`/`TRACK_WIDTH`/`MAX_STEER_ANGLE_DEG`
(geometría Ackermann, deben coincidir con el `.wbt`) y `MAX_LINEAR_SPEED` (divisor de
normalización de velocidad = `WHEEL_MAX_SPEED × radio_rueda`).
