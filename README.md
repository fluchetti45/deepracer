# DeepRacer en Webots — Seguimiento de carril por visión

Agente autónomo estilo **AWS DeepRacer** que aprende a **navegar un circuito sin salirse del carril**, entrenado con **PPO (Stable-Baselines3)** dentro del simulador **Webots**. El robot percibe el mundo de forma **vision-pura**: su observación es un stack de imágenes de la cámara frontal más su velocidad propia, sin coordenadas ni waypoints privilegiados.

La pista se diseñó en Inkscape y se usa como **textura del piso** (`worlds/track1.png`): asfalto gris, línea amarilla punteada al centro, bordes blancos y pasto verde fuera de pista.

---

## Cómo funciona

El sistema son **tres procesos** que se comunican:

```
  rl/trainer.py (PPO)            Webots
  ┌────────────────┐      ┌──────────────────────────────────────┐
  │  NavEnv (gym)  │◄────►│  supervisor_controller (env server)  │
  │  rl/env.py     │socket│   - resetea / stepea la simulacion    │
  └────────────────┘ TCP  │   - computa el REWARD desde la imagen │
                          │   - habla con el robot por emitter/   │
                          │     receiver                          │
                          │            ▲                          │
                          │            │ mensajes                 │
                          │            ▼                          │
                          │  agent_controller (e-puck + camara)  │
                          └──────────────────────────────────────┘
```

- **`controllers/agent_controller/`** — corre en el robot (e-puck con cámara y 2 ruedas). Aplica acciones a las ruedas y devuelve su observación sensorial (frame RGB + estado de ruedas).
- **`controllers/supervisor_controller/`** — corre en el supervisor de Webots. Es el **servidor del environment**: atiende `reset_env` / `step_env` por socket, avanza la simulación, y **calcula el reward** a partir de la cámara del robot. También expone una *Robot Window* con el desglose del reward en vivo.
- **`rl/`** — el lado de entrenamiento. `NavEnv` (Gymnasium) habla con el supervisor por TCP; `trainer.py` arma PPO con `VecNormalize` + `VecFrameStack` y entrena.
- **`helpers/`** — código compartido: puente de sockets, formato de imagen, detector de carril (`lane_vision.py`), runner de inferencia, etc.

### Reward (vision-pura)

El reward se computa **desde la misma imagen de cámara** que ve el agente — sin geometría de la pista ni waypoints. En la franja inferior del frame se clasifican píxeles por color (amarillo / blanco / verde) y se derivan:

| Término | Señal |
|---|---|
| `center_clearance` | banda central de la vista = asfalto limpio → va centrado en el carril |
| `offset` | desplazamiento lateral firmado (da la dirección de corrección) |
| penalización blanco | borde blanco metido en el centro → pisando el borde |
| velocidad × centrado | premia avanzar **solo** si va bien |
| terminal off-track | pasto en el centro → fin de episodio con reward negativo |

Toda la detección vive en [`helpers/lane_vision.py`](helpers/lane_vision.py) (NumPy puro, sin OpenCV) y la composición del reward en `_compute_reward_breakdown` de [`controllers/supervisor_controller/supervisor_controller.py`](controllers/supervisor_controller/supervisor_controller.py). Los pesos y umbrales se ajustan por `.env` (ver `.env.example`).

---

## Requisitos

- **[Webots](https://cyberbotics.com/) R2025a** (o compatible).
- **Python 3.10+**.
- Las dependencias de [`requirements.txt`](requirements.txt): `stable-baselines3[extra]`, `sb3-contrib`, `gymnasium`, `python-dotenv`, `numpy`.

---

## Instalación

```powershell
# 1. Clonar
git clone <tu-repo> deepracer
cd deepracer

# 2. Entorno virtual
python -m venv env
.\env\Scripts\Activate.ps1        # Windows PowerShell
# source env/bin/activate         # Linux / macOS

# 3. Dependencias
pip install -r requirements.txt

# 4. Configuracion
copy .env.example .env            # Windows  (cp en Linux/macOS)
```

> El `.env` define la resolución de cámara (`CAMERA_WIDTH/HEIGHT`), que **debe coincidir** con el nodo `Camera` del robot en `worlds/track1.wbt`.

---

## Entrenar

El trainer lanza Webots automáticamente (con `worlds/track1.wbt`), levanta el environment y entrena PPO:

```powershell
python -m rl.trainer --total-timesteps 100000 --n-stack 4
```

Flags útiles:

| Flag | Default | Qué hace |
|---|---|---|
| `--total-timesteps` | `100000` | timesteps totales de entrenamiento |
| `--n-stack` | `4` | frames apilados (info temporal / velocidad) |
| `--n-envs` | `1` | instancias de Webots en paralelo (cada una en `--base-port + i`) |
| `--learning-rate` | `5e-4` | learning rate de PPO |
| `--n-steps` | `1024` | pasos por rollout (con `--n-envs > 1` conviene bajarlo) |
| `--norm-reward` | off | normaliza el reward con VecNormalize |
| `--device` | `auto` | `cpu` / `cuda` / `auto` |
| `--no-webots-launch` | off | no lanzar Webots (si ya lo abriste a mano) |

Cada corrida genera una carpeta `models/<timestamp>/` con:

- `final_model.zip` — la policy PPO entrenada.
- `vecnormalize.pkl` — estadísticas de normalización.
- `tensorboard/` — logs (incluye métricas `custom/` de carril: `offtrack_rate`, `center_clearance_mean`, `abs_offset_mean`, etc.).
- `run_metadata.json` — hiperparámetros de la corrida.

Para ver las métricas:

```powershell
tensorboard --logdir models
```

---

## Ver una policy entrenada

Abrí `worlds/track1.wbt` en Webots y usá la **Robot Window** del supervisor para cargar el modelo (`models/<timestamp>/final_model.zip`) y activarlo. El supervisor pasa a tiempo real y el robot ejecuta la policy; el panel muestra el desglose del reward en vivo.

---

## Estructura del proyecto

```
deepracer/
├── controllers/
│   ├── agent_controller/        # robot e-puck (camara + ruedas)
│   └── supervisor_controller/   # servidor del environment + reward
├── helpers/
│   ├── lane_vision.py           # detector de carril (reward vision-puro)
│   ├── image_obs.py             # formato/decodificacion de la imagen
│   ├── robot_bridge.py          # puente supervisor <-> robot (emitter/receiver)
│   ├── supervisor_socket_bridge.py  # puente trainer <-> supervisor (TCP)
│   ├── training_server.py       # servidor de requests de entrenamiento
│   ├── policy_runner.py         # inferencia de la policy en el supervisor
│   └── read_env_value.py        # lectura de config desde .env
├── rl/
│   ├── env.py                   # NavEnv (Gymnasium) sobre socket
│   └── trainer.py               # entrenamiento PPO
├── worlds/
│   ├── track1.wbt               # mundo de Webots
│   └── track1.png               # textura del circuito
├── launch_webots.py             # lanzador de instancias de Webots
├── requirements.txt
└── .env.example
```

---

## Notas

- **Acción**: vector de 2 valores (rueda izq./der.) normalizado en `[-1, 1]`; el robot lo escala a rad/s.
- **Observación**: `{ image: cámara RGB (C,H,W), velocity: [forward, yaw_rate] }`, con stacking de `--n-stack` frames.
- El reward es **denso** y está acotado a ~`[-1, 1]` por step, por eso `--norm-reward` viene apagado por default.
- El `.env` **no se versiona** (está en `.gitignore`); usá `.env.example` como plantilla.
