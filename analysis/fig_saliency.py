# -*- coding: utf-8 -*-
"""
Saliencia por *occlusion sensitivity* de las variantes de vision (seed 0), sobre un mismo
frame de curva. Para cada modelo mide cuanto cambia el *steering* (diferencial de ruedas,
a_izq - a_der) al tapar cada region de la imagen con un parche gris: donde tapar cambia mas
la decision = region de la que depende la politica.

Usamos occlusion (perturbacion directa, causal) en vez de guided backprop / integrated
gradients: esos son gradientes, mas ruidosos y sensibles a la cabeza de valor ---que en la
destilada NO se entrena (BC solo entrena la accion)---; occlusion mide el efecto real sobre
la salida y es fiel para las cuatro. El *steering* es la unica salida entrenada en todas.

Resultado: la vision-RL se apoya en la franja del HORIZONTE (atajo del fondo); la destilada
mira la CALZADA. Genera analysis/fig_saliency.png.

Uso:  python -m analysis.fig_saliency
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

# (etiqueta, run_id seed 0, clase, n_stack)
MODELS = [
    ("Visión 1 frame",  "20260713120404", PPO,          1),
    ("Visión apilada",  "20260714012211", PPO,          4),
    ("Visión + LSTM",   "20260715041635", RecurrentPPO, 1),
    ("Visión destilada", "20260717110936", PPO,         1),
]
import glob
FRAMES_DIR = os.path.join("analysis", "frames")
FRAME = os.path.join(FRAMES_DIR, "frame_track9_016.png")  # frame representativo para el heatmap
# Frames de curva para el estadistico robusto (promedio de %calzada); un solo frame es ruidoso.
AGG_FRAMES = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_track9_*.png")))[:12]
PATCH, STRIDE, GRAY = 10, 3, 128.0


def load_image_tensor(path, n_stack):
    """Frame -> tensor (1, 3*n_stack, 84, 84) en [0,255]."""
    img = Image.open(path).convert("RGB").resize((84, 84), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)   # (3,84,84)
    arr = np.tile(arr, (n_stack, 1, 1))                          # (3*n_stack,84,84)
    return torch.tensor(arr).unsqueeze(0), img


def steer_of(policy, img, vel_size, recurrent):
    """Steering (a_izq - a_der) de la media de la accion, sin gradiente."""
    obs = {"image": img, "velocity": torch.zeros(1, vel_size)}
    with torch.no_grad():
        if recurrent:
            from sb3_contrib.common.recurrent.type_aliases import RNNStates
            lstm = policy.lstm_actor
            shape = (lstm.num_layers, 1, lstm.hidden_size)
            zero = (torch.zeros(shape), torch.zeros(shape))
            states = RNNStates(zero, (zero[0].clone(), zero[1].clone()))
            dist, _ = policy.get_distribution(obs, states.pi, torch.ones(1))
            mean = dist.distribution.mean
        else:
            feats = policy.extract_features(obs)
            latent_pi, _ = policy.mlp_extractor(feats)
            mean = policy.action_net(latent_pi)
    return float(mean[0, 0] - mean[0, 1])


def occlusion_raw(policy, img, vel_size, recurrent):
    """Mapa 84x84 crudo: |steer(tapado) - steer(base)| promediado por pixel."""
    base = steer_of(policy, img, vel_size, recurrent)
    acc = np.zeros((84, 84)); cnt = np.zeros((84, 84))
    for y in range(0, 84 - PATCH + 1, STRIDE):
        for x in range(0, 84 - PATCH + 1, STRIDE):
            occ = img.clone()
            occ[:, :, y:y + PATCH, x:x + PATCH] = GRAY
            d = abs(steer_of(policy, occ, vel_size, recurrent) - base)
            acc[y:y + PATCH, x:x + PATCH] += d
            cnt[y:y + PATCH, x:x + PATCH] += 1
    return acc / (cnt + 1e-9)


def normalize(m):
    m = gaussian_filter(m, sigma=1.2)
    hi = np.percentile(m, 99.0)
    return np.clip(m / (hi + 1e-9), 0, 1)


def road_fraction(m):
    """% de la sensibilidad en el tercio inferior (calzada)."""
    rm = m.sum(1)
    return 100 * rm[56:].sum() / (rm.sum() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("analysis", "fig_saliency.png"))
    args = ap.parse_args()

    n = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.0))

    for ax, (label, rid, cls, n_stack) in zip(axes, MODELS):
        model = cls.load(f"models/{rid}/final_model.zip", device="cpu")
        policy = model.policy.eval()
        vel_size = model.policy.observation_space.spaces["velocity"].shape[0]
        recurrent = cls is RecurrentPPO

        # Heatmap: un frame representativo. Anotacion: %calzada promediado sobre AGG_FRAMES.
        img, base = load_image_tensor(FRAME, n_stack)
        sal = normalize(occlusion_raw(policy, img, vel_size, recurrent))
        roads = []
        for fr in AGG_FRAMES:
            fimg, _ = load_image_tensor(fr, n_stack)
            roads.append(road_fraction(occlusion_raw(policy, fimg, vel_size, recurrent)))
        road = float(np.mean(roads))
        print(f"  {label:16s} calzada={road:.0f}%  (n={len(roads)} frames)")

        ax.imshow(np.asarray(base, dtype=np.float32) / 255.0, alpha=0.40)
        ax.imshow(sal, cmap="inferno", alpha=0.70)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(f"calzada {road:.0f}%", fontsize=10, fontweight="bold",
                      color=("#0C6E56" if road >= 50 else "#A34428"))

    fig.suptitle("Occlusion sensitivity del steering — mapa: una curva; %: promedio sobre curvas (seed 0)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("Escrito:", args.out)


if __name__ == "__main__":
    main()
