"""
Visualizacion de la CNN de una policy de vision (estilo "Learning to Drive"/WRC6):
para UN frame de entrada muestra en una figura
  1) el input que ve la red,
  2) guided backpropagation (saliencia: que pixeles activan mas las features de la CNN),
  3) los feature maps (activaciones) de cada capa convolucional.

Sirve para el paper: evidencia de QUE mira la red (bordes de pista, linea blanca, pasto).

La CNN es el NatureCNN del extractor de imagen (CnnPolicy -> features_extractor.cnn;
MultiInputPolicy/Dict -> features_extractor.extractors['image'].cnn).

Uso:
  python -m analysis.cnn_activations --model models/<run_id>            # frame sintetico
  python -m analysis.cnn_activations --model models/<run_id> --frame f.png
  python -m analysis.cnn_activations --model models/<run_id> --out analysis/fig_cnn.png

--frame acepta un PNG (se redimensiona a 84x84). Para frames REALES de la camara ver el
dumper de rl/evaluate (--dump-frames) o guardar la obs['image'] de un rollout.
"""

import argparse
import warnings

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from helpers.image_obs import IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402


def get_image_cnn(model):
    """Devuelve el sub-modulo NatureCNN que procesa la imagen (maneja Dict y Box)."""
    fe = model.policy.features_extractor
    if hasattr(fe, "extractors"):  # CombinedExtractor (obs Dict)
        for key, sub in fe.extractors.items():
            if hasattr(sub, "cnn"):
                return sub
    if hasattr(fe, "cnn"):  # NatureCNN directo (CnnPolicy)
        return fe
    raise SystemExit("No encontre una CNN de imagen en el features_extractor de este modelo.")


def synth_frame(h=IMAGE_HEIGHT, w=IMAGE_WIDTH):
    """
    Frame sintetico en PERSPECTIVA que imita la vista de la camara: pasto teal de fondo,
    calzada gris en trapecio que se ensancha hacia abajo, y lineas blancas en los bordes.
    Aproxima la distribucion de entrenamiento (road vs grass vs white) para demostrar la
    visualizacion sin Webots. NO reemplaza un frame real para la figura final.
    """
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (12, 150, 120)  # pasto (teal ~ PMS 3395 C bajo luz de Webots)
    cx = w // 2
    for y in range(h):
        t = y / (h - 1)                       # 0 arriba (lejos) -> 1 abajo (cerca)
        road_half = int((0.05 + 0.43 * t) * w)  # la calzada se ensancha hacia abajo
        x0, x1 = cx - road_half, cx + road_half
        img[y, max(0, x0):min(w, x1)] = (92, 92, 98)   # asfalto gris
        for xe in (x0, x1):                             # lineas blancas de borde
            for dx in (-1, 0, 1):
                xx = xe + dx
                if 0 <= xx < w:
                    img[y, xx] = (235, 235, 235)
    return img


def load_frame(path, h=IMAGE_HEIGHT, w=IMAGE_WIDTH):
    """Carga un PNG y lo lleva a HWC uint8 (h, w, 3)."""
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((w, h))
    return np.asarray(im, dtype=np.uint8)


def model_image_tensor(frame_hwc, model):
    """
    HWC uint8 -> (1, C, H, W) float en [0,255] con C = canales que espera el modelo. Para
    modelos con frame-stacking (C = 3*n_stack) replica el frame para llenar el stack (aprox:
    como si los ultimos n_stack frames fueran iguales). SB3 divide por 255 en su preprocess.
    """
    t = torch.tensor(frame_hwc, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    want = model.observation_space["image"].shape[0] if hasattr(
        model.observation_space, "spaces") else model.observation_space.shape[0]
    if want > t.shape[1] and want % t.shape[1] == 0:
        t = t.repeat(1, want // t.shape[1], 1, 1)
    return t


def collect_activations(cnn, inp):
    """Corre el forward y devuelve las salidas POST-ReLU de cada bloque conv (lista de tensores)."""
    acts = []
    handles = []
    for layer in cnn.cnn:
        if isinstance(layer, torch.nn.ReLU):
            handles.append(layer.register_forward_hook(lambda m, i, o: acts.append(o.detach())))
    with torch.no_grad():
        cnn.cnn(inp)
    for h in handles:
        h.remove()
    return acts


def _register_guided_relu(modules):
    """
    Hooks de guided backprop (Springenberg et al. 2015) sobre TODOS los ReLU de `modules`:
    en el backward corta los gradientes negativos y los que vienen de activaciones <=0.
    Devuelve los handles (para removerlos despues).
    """
    outputs = {}
    handles = []

    def fwd(m, i, o):
        outputs[m] = o

    def bwd(m, grad_in, grad_out):
        act = outputs[m]
        return (grad_out[0].clamp(min=0) * (act > 0).float(),)

    for m in modules:
        if isinstance(m, torch.nn.ReLU):
            handles.append(m.register_forward_hook(fwd))
            handles.append(m.register_full_backward_hook(bwd))
    return handles


def saliency_features(cnn, cnn_input):
    """
    Saliencia respecto a la MAGNITUD de las features conv (que EXCITA a la representacion
    visual, independiente de la cabeza de decision). Este es el modo `features` (default).
    """
    handles = _register_guided_relu(cnn.cnn.modules())
    x = cnn_input.clone().requires_grad_(True)
    score = cnn.cnn(x).pow(2).sum()
    score.backward()
    sal = x.grad[0].abs().sum(0).detach().cpu().numpy()
    for h in handles:
        h.remove()
    return sal


def saliency_decision(model, obs_image, target):
    """
    Saliencia respecto a la DECISION: gradiente del value V(s) o de una componente de la
    accion (steer/speed) respecto a los pixeles. Corre el forward COMPLETO de la policy con
    guided ReLU en toda la red. `obs_image` = (1, C, H, W) float en [0,255] (SB3 lo divide).
    """
    handles = _register_guided_relu(model.policy.modules())
    img = obs_image.clone().requires_grad_(True)
    vel_dim = model.observation_space["velocity"].shape[0]
    obs = {"image": img, "velocity": torch.zeros(1, vel_dim)}

    features = model.policy.extract_features(obs)
    latent_pi, latent_vf = model.policy.mlp_extractor(features)
    if target == "value":
        score = model.policy.value_net(latent_vf).sum()
    else:  # steer / speed => componente 0 / 1 de la media de la accion
        mean_actions = model.policy._get_action_dist_from_latent(latent_pi).distribution.mean
        score = mean_actions[0, 0 if target == "steer" else 1]
    score.backward()
    sal = img.grad[0].abs().sum(0).detach().cpu().numpy()
    for h in handles:
        h.remove()
    return sal


def compute_saliency(model, cnn, model_img, target):
    """Dispatcher: `features` usa la CNN; value/steer/speed usan la policy completa."""
    if target == "features":
        return saliency_features(cnn, model_img / 255.0)
    return saliency_decision(model, model_img, target)


def _norm(a):
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


_TARGET_LABEL = {
    "features": "features (excita a la CNN)",
    "value": "value  V(s)",
    "steer": "accion: steer",
    "speed": "accion: speed",
}


def make_figure(frame, sal, acts, out_path, title, target):
    """Figura: input | guided-backprop | activacion media por capa conv."""
    n_layers = len(acts)
    ncols = 2 + n_layers
    fig, axes = plt.subplots(1, ncols, figsize=(3.1 * ncols, 3.4))

    axes[0].imshow(frame)
    axes[0].set_title("Network input\n(84x84x3)")

    # Guided backprop: saliencia sobre el input en escala de grises.
    axes[1].imshow(frame.mean(2), cmap="gray")
    axes[1].imshow(_norm(sal), cmap="inferno", alpha=0.65)
    axes[1].set_title(f"Guided backprop\n{_TARGET_LABEL.get(target, target)}")

    for k, act in enumerate(acts):
        a = act[0].mean(0).cpu().numpy()   # media sobre canales -> mapa espacial
        ax = axes[2 + k]
        ax.imshow(_norm(a), cmap="jet")
        ax.set_title(f"Activations\nLayer {k + 1}  ({act.shape[1]} ch)")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print("Figura escrita en", out_path)


def analyze_model(model, cnn, paths, target):
    """Saliencia + activaciones PROMEDIADAS sobre `paths` para un modelo. Devuelve
    (frame_display, sal, acts). Es el nucleo reutilizado por el modo simple y el comparativo."""
    sal_sum, acts_sum, first_frame = None, None, None
    for path in paths:
        frame = load_frame(path) if path else synth_frame()
        if first_frame is None:
            first_frame = frame
        model_img = model_image_tensor(frame, model)
        acts = collect_activations(cnn, model_img / 255.0)
        sal = compute_saliency(model, cnn, model_img, target)
        sal_sum = sal if sal_sum is None else sal_sum + sal
        acts_sum = ([a.clone() for a in acts] if acts_sum is None
                    else [s + a for s, a in zip(acts_sum, acts)])
    n = len(paths)
    return first_frame, sal_sum / n, [a / n for a in acts_sum]


def make_compare_figure(rows, out_path, target, title):
    """
    Figura comparativa: una FILA por modelo (input | guided-backprop | activacion por capa).
    `rows` = [(label, frame, sal, acts), ...]. Pensada para el before/after del paper.
    """
    n_layers = len(rows[0][3])
    ncols = 2 + n_layers
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.1 * ncols, 3.4 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]

    for r, (label, frame, sal, acts) in enumerate(rows):
        axes[r, 0].imshow(frame)
        axes[r, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=12)
        if r == 0:
            axes[r, 0].set_title("Network input")
            axes[r, 1].set_title(f"Guided backprop\n{_TARGET_LABEL.get(target, target)}")
        axes[r, 1].imshow(frame.mean(2), cmap="gray")
        axes[r, 1].imshow(_norm(sal), cmap="inferno", alpha=0.65)
        for k, act in enumerate(acts):
            axes[r, 2 + k].imshow(_norm(act[0].mean(0).cpu().numpy()), cmap="jet")
            if r == 0:
                axes[r, 2 + k].set_title(f"Activations\nLayer {k + 1}")

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print("Figura comparativa escrita en", out_path)


def parse_args():
    p = argparse.ArgumentParser(description="Visualiza activaciones + guided backprop de la CNN de vision.")
    p.add_argument("--model", required=True, help="run_dir o .zip del modelo (PPO/RecurrentPPO).")
    p.add_argument("--frame", default=None, help="PNG de entrada. Si falta, usa un frame sintetico.")
    p.add_argument("--out", default="analysis/fig_cnn_activations.png")
    p.add_argument("--target", default="features",
                   choices=["features", "value", "steer", "speed"],
                   help="Sobre QUE se computa la saliencia: 'features' (excitacion de la CNN, "
                        "default; independiente de la decision) o la DECISION -> 'value' V(s), "
                        "'steer' o 'speed' (componentes de la accion).")
    p.add_argument("--frames-dir", default=None,
                   help="Carpeta de PNGs: PROMEDIA saliencia y activaciones sobre TODOS los "
                        "frames (mapa robusto, no anecdotico). Ignora --frame.")
    p.add_argument("--no-montage", action="store_true",
                   help="No generar los montages de canales por capa (*_ch1/2/3.png).")
    # --- Modo comparativo (before/after, p.ej. RL vs destilado) ---
    p.add_argument("--compare-model", default=None,
                   help="Segundo modelo: genera una figura de 2 filas comparando ambos "
                        "(mismos frames/target). Sin montages.")
    p.add_argument("--label", default="Modelo A", help="Etiqueta de fila de --model.")
    p.add_argument("--compare-label", default="Modelo B", help="Etiqueta de fila de --compare-model.")
    return p.parse_args()


def load_model(path):
    """Carga PPO o RecurrentPPO segun corresponda (misma CNN de vision)."""
    import os
    zip_path = path if path.endswith(".zip") else os.path.join(path, "final_model.zip")
    try:
        from stable_baselines3 import PPO
        return PPO.load(zip_path, device="cpu")
    except Exception:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(zip_path, device="cpu")


def channel_montage(act, out_path, layer_idx, cols=8):
    """Montage de TODOS los canales (feature maps) de una capa conv, en grilla."""
    a = act[0].cpu().numpy()
    n = a.shape[0]
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(np.array(axes).ravel()):
        if i < n:
            ax.imshow(_norm(a[i]), cmap="jet")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Layer {layer_idx}: {n} canales", y=1.005, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print("Montage escrito en", out_path)


def _frame_paths(args):
    """Lista de frames a procesar: --frames-dir (todos los PNG) o un unico --frame/sintetico."""
    if args.frames_dir:
        import glob
        import os
        paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
        if not paths:
            raise SystemExit(f"No hay PNGs en {args.frames_dir}")
        return paths
    return [args.frame] if args.frame else [None]


def main():
    args = parse_args()
    paths = _frame_paths(args)
    src = (f"{args.frames_dir} (media de {len(paths)} frames)" if args.frames_dir
           else (args.frame if args.frame else "frame sintetico"))

    model = load_model(args.model)
    cnn = get_image_cnn(model)

    # --- Modo comparativo: 2 modelos, 1 figura (before/after) ---
    if args.compare_model:
        m2 = load_model(args.compare_model)
        cnn2 = get_image_cnn(m2)
        f1, s1, a1 = analyze_model(model, cnn, paths, args.target)
        f2, s2, a2 = analyze_model(m2, cnn2, paths, args.target)
        make_compare_figure(
            [(args.label, f1, s1, a1), (args.compare_label, f2, s2, a2)],
            args.out, target=args.target,
            title=f"target: {args.target}  |  {src}")
        return

    # --- Modo simple: 1 modelo ---
    first_frame, sal, acts = analyze_model(model, cnn, paths, args.target)
    make_figure(first_frame, sal, acts, args.out, target=args.target,
                title=f"{args.model}  |  target: {args.target}  |  {src}")

    # Montage de feature maps por capa conv (todos los canales): *_ch1.png, *_ch2.png, ...
    if not args.no_montage:
        for k, act in enumerate(acts):
            channel_montage(act, args.out.replace(".png", f"_ch{k + 1}.png"), layer_idx=k + 1)


if __name__ == "__main__":
    main()
