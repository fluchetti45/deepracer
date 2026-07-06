"""
Destilacion geometrica -> vision (behavioral cloning / privileged distillation).

Entrena una CNN de vision (MultiInputPolicy, MISMA arquitectura que vision_1frame) para
IMITAR las acciones del agente GEOMETRICO (maestro privilegiado), a partir del dataset
(imagen, velocity, accion) que genera rl/collect_teacher. Es supervisado:
    loss = MSE( accion_estudiante(imagen, velocity),  accion_maestro )

El estudiante ve pixeles crudos; el maestro veia las features de calzada (ground-truth de
percepcion). Al imitar decisiones basadas en la calzada, el estudiante tiende a IGNORAR el
fondo (el maestro nunca lo uso). Ver analysis/cnn_activations para verificar el sesgo.

Produce, por seed, models/<id>/ con:
  - final_model.zip        (SB3, cargable por rl/evaluate y policy_runner sin cambios)
  - vecnormalize.pkl       (identidad: la velocity ya viene en [-1,1], no se normaliza)
  - run_metadata.json      (variant="vision_distill", para que analysis lo agrupe)

Uso:
  python -m rl.distill --data data/teacher --seeds 0 1 2 3 4
  python -m rl.distill --data data/teacher --seeds 0 --epochs 30 --batch-size 256
"""

import argparse
import glob
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from helpers.image_obs import build_image_space


class _StubEnv(gym.Env):
    """
    Env sin Webots: solo publica los spaces (Dict{image,velocity} + accion Box) para poder
    construir la MultiInputPolicy de SB3 identica a la de vision_1frame. Nunca se stepea.
    """
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "velocity": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "image": build_image_space(),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def load_dataset(data_dir):
    """Concatena todos los *.npz de data_dir en (images, velocities, actions)."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No hay .npz en {data_dir} (corre rl.collect_teacher primero).")
    imgs, vels, acts = [], [], []
    for f in files:
        d = np.load(f)
        imgs.append(d["images"]); vels.append(d["velocities"]); acts.append(d["actions"])
    images = np.concatenate(imgs).astype(np.uint8)          # (N, 3, 84, 84)
    velocities = np.concatenate(vels).astype(np.float32)    # (N, 2)
    actions = np.concatenate(acts).astype(np.float32)       # (N, 2)
    print(f"Dataset: {len(images)} pares de {len(files)} archivo(s).")
    return images, velocities, actions


def build_model(seed, device):
    """PPO/MultiInputPolicy identico a vision_1frame (solo se usa su .policy para el BC)."""
    env = DummyVecEnv([_StubEnv])
    return PPO("MultiInputPolicy", env, seed=seed, device=device, verbose=0)


def student_action_mean(policy, obs):
    """Media (deterministica) de la accion de la policy para un batch de obs (tensores)."""
    features = policy.extract_features(obs)
    latent_pi, _ = policy.mlp_extractor(features)
    return policy.action_net(latent_pi)   # DiagGaussian: action_net -> media


def train_one_seed(images, velocities, actions, seed, args):
    """Entrena un estudiante por BC y devuelve el run_dir con el modelo guardado."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(seed, args.device)
    policy = model.policy
    policy.train()
    device = model.device
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    n = len(images)
    img_t = torch.as_tensor(images, device=device)                    # uint8
    vel_t = torch.as_tensor(velocities, device=device)                # float32
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)

    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            obs = {"image": img_t[idx], "velocity": vel_t[idx]}
            pred = student_action_mean(policy, obs)
            loss = torch.nn.functional.mse_loss(pred, act_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(idx)
        print(f"  seed {seed} | epoch {epoch + 1:3d}/{args.epochs} | MSE {total / n:.5f}")

    return save_outputs(model, seed, args, n)


def save_outputs(model, seed, args, n_pairs):
    """Guarda final_model.zip + vecnormalize.pkl (identidad) + run_metadata.json."""
    stamp = time.strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(args.models_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)

    model.save(os.path.join(run_dir, "final_model"))

    # VecNormalize identidad: la velocity ya esta en [-1,1] y la imagen es uint8 cruda.
    # Se guarda igual para que rl/evaluate.build_vec_env encuentre el .pkl y no avise.
    vn = VecNormalize(DummyVecEnv([_StubEnv]), norm_obs=False, norm_reward=False, training=False)
    vn.save(os.path.join(run_dir, "vecnormalize.pkl"))

    meta = {
        "variant": args.variant,
        "training_regime": "distill",   # supervisado (BC), NO entrenado en el simulador
        "actual_device": str(model.device),
        "teacher_model": args.teacher_id,
        "n_pairs": int(n_pairs),
        "hyperparameters": {
            "n_stack": 1,
            "seed": int(seed),
            # Presupuesto NOMINAL para agrupar en analysis junto al cohorte del maestro
            # (NO son pasos de RL: es destilacion supervisada).
            "total_timesteps": int(args.total_timesteps),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
        },
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  seed {seed} -> {run_dir}")
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(description="Destila el agente geometrico en una CNN de vision (BC).")
    p.add_argument("--data", default="data/teacher", help="Carpeta con los .npz del maestro.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds de estudiante.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--total-timesteps", type=int, default=1000000,
                   help="Presupuesto nominal para agrupar en analysis (no son pasos de RL).")
    p.add_argument("--teacher-id", default="", help="run_id del maestro (para el metadata).")
    p.add_argument("--variant", default="vision_distill",
                   help="Etiqueta de variante para analysis (ej. vision_distill para BC limpio, "
                        "vision_distill_dart para la colecta con ruido/recuperacion).")
    return p.parse_args()


def main():
    args = parse_args()
    images, velocities, actions = load_dataset(args.data)
    runs = []
    for seed in args.seeds:
        print("=" * 56)
        print(f"DISTILL seed {seed}  ({len(images)} pares, {args.epochs} epochs)")
        print("=" * 56)
        runs.append(train_one_seed(images, velocities, actions, seed, args))
    print("\nModelos destilados:")
    for r in runs:
        print("  -", r)


if __name__ == "__main__":
    main()
