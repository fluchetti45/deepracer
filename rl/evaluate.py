# Evalua un modelo entrenado en UN track y mide cuanto tarda en completar vueltas.
#
# Reproduce el MISMO pipeline de obs del entrenamiento (VecNormalize de velocity +
# VecFrameStack) para que la policy vea exactamente lo que vio entrenando. Fuerza el
# track elegido escribiendo un spawns.json temporal con UN solo track (asi el supervisor,
# que elige random, siempre cae en ese) y apaga el domain randomization -> corrida
# determinista. La "vuelta completa" la detecta el supervisor por PROGRESO sobre los
# gates (term_reason == "lap_complete"); por eso el track DEBE tener gates en spawns.json.
#
# Uso tipico:
#   python -m rl.evaluate --model models/20260613145840 --track track4.png --laps 3
#
# Si Webots ya esta abierto con el world cargado, agregar --no-webots-launch.

import argparse
import json
import os
import tempfile
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecNormalize,
)

from helpers.read_env_value import read_env_value

from launch_webots import launch_webots
from rl.env import NavEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalua un modelo PPO en un track y mide el tiempo de vuelta."
    )
    parser.add_argument(
        "--model",
        default="models/20260613145840",
        help=(
            "Carpeta de la corrida (usa final_model.zip + vecnormalize.pkl) o ruta "
            "directa al .zip del modelo."
        ),
    )
    parser.add_argument(
        "--vecnormalize",
        default=None,
        help="Ruta a vecnormalize.pkl. Si falta, se infiere desde --model.",
    )
    parser.add_argument(
        "--track",
        default=None,
        help=(
            "Textura del track a evaluar (debe existir en worlds/ y en spawns.json). "
            "Si se omite, evalua TODOS los tracks marcados \"eval\": true, con la misma "
            "cantidad de vueltas (--laps) en cada uno."
        ),
    )
    parser.add_argument(
        "--spawns",
        default="spawns.json",
        help="spawns.json del que se toma el track (texture, spawn y gates).",
    )
    parser.add_argument(
        "--spawn-index",
        type=int,
        default=0,
        help="Cual spawn del track usar (si hay varios). Determinista.",
    )
    parser.add_argument(
        "--laps",
        type=int,
        default=3,
        help="Cantidad de vueltas COMPLETAS a cronometrar antes de cortar.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help=(
            "Tope de episodios (intentos) por si nunca completa. "
            "Default: laps * 5 + 5. (Solo aplica en modo --laps.)"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Modo TASA DE EXITO: corre EXACTAMENTE N episodios por track, pase lo que "
            "pase, y reporta lap_rate = vueltas/N (sin sesgo). Ignora --laps y "
            "--max-episodes. Recomendado para el paper."
        ),
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=read_env_value("EVAL_MAX_EPISODE_STEPS", 2000, int),
        help=(
            "Time limit por episodio en EVAL (steps antes de truncar por timeout). Da "
            "el tiempo maximo para completar una vuelta. Default: EVAL_MAX_EPISODE_STEPS "
            "del .env (o 2000). Se inyecta al supervisor como MAX_EPISODE_STEPS."
        ),
    )
    parser.add_argument(
        "--n-stack",
        type=int,
        default=None,
        help="Frames apilados. Si falta, se lee de run_metadata.json (o 4).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Muestrear la accion (por defecto se usa la media, determinista).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.16,
        help=(
            "Segundos de simulacion por step de env, para reportar tiempo de vuelta "
            "(basicTimeStep * ACTION_REPEAT / 1000 = 32ms * 5 = 0.16s por defecto)."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host del supervisor.")
    parser.add_argument("--port", type=int, default=10001, help="Puerto del supervisor.")
    parser.add_argument(
        "--webots-world",
        default="worlds/track1.wbt",
        help="World a lanzar (el supervisor le cambia la textura al track elegido).",
    )
    parser.add_argument(
        "--webots-executable",
        default=r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe",
        help="Ruta al ejecutable de Webots.",
    )
    parser.add_argument(
        "--no-webots-launch",
        action="store_true",
        help="No lanzar Webots (ya esta abierto con el world cargado).",
    )
    parser.add_argument(
        "--device", default="cpu", help="Dispositivo de PyTorch para la inferencia."
    )
    # --- Volcado de frames de la camara (para analysis/cnn_activations) ---
    parser.add_argument(
        "--dump-frames", type=int, default=0,
        help="Guardar N frames REALES de la camara por track (0 = off). Para visualizar "
             "activaciones de la CNN con inputs de la distribucion de entrenamiento.",
    )
    parser.add_argument(
        "--dump-every", type=int, default=15,
        help="Steps entre frames guardados (stride): evita volcar frames casi identicos "
             "seguidos y da diversidad (curvas, rectas, off-track).",
    )
    parser.add_argument(
        "--dump-dir", default="analysis/frames",
        help="Carpeta destino de los frames volcados.",
    )
    # --- Robustez: randomizar el fondo TAMBIEN en eval ---
    parser.add_argument(
        "--randomize-background", action="store_true",
        help="Rotar pared+skybox aleatoriamente por episodio TAMBIEN en eval (prob=1.0). "
             "Test de robustez a fondo: mide cuanto depende el modelo del fondo. Default OFF "
             "(fondo fijo, comparable con evals previos).",
    )
    parser.add_argument(
        "--eval-seed", type=int, default=None,
        help="Seed del RNG de reset del supervisor (spawn + fondo). Si se fija, TODOS los "
             "modelos ven la MISMA secuencia de spawns/fondos episodio-a-episodio -> "
             "comparacion justa. Sin fijar = no reproducible (comportamiento previo).",
    )
    parser.add_argument(
        "--no-save-results", action="store_true",
        help="No escribir eval_results_<ts>.json. Para volcar frames (--dump-frames) sin "
             "pisar el eval del modelo (el analysis toma el eval_results mas reciente).",
    )
    return parser.parse_args()


def _extract_latest_rgb(obs):
    """
    Frame RGB HWC uint8 mas RECIENTE desde la obs del VecEnv. Con VecFrameStack la imagen
    viene apilada en el eje de canales (newest al final) -> tomo los ultimos 3 canales.
    La imagen no la normaliza VecNormalize (solo velocity), asi que ya es uint8 [0,255].
    """
    img = obs["image"] if isinstance(obs, dict) else obs
    arr = np.asarray(img)[0]                 # (C, H, W)  (n_envs=1)
    arr = arr[-3:]                           # frame mas nuevo = ultimos 3 canales
    arr = np.transpose(arr, (1, 2, 0))       # HWC
    if arr.dtype != np.uint8:                # defensivo por si viniera float
        top = float(arr.max()) if arr.size else 0.0
        arr = arr * 255.0 if top <= 1.0 else arr
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _save_frame(obs, out_dir, texture, idx):
    """Guarda el frame mas reciente como PNG. Devuelve la ruta (o None si falla)."""
    try:
        from PIL import Image
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(texture)[0]
        path = os.path.join(out_dir, f"frame_{stem}_{idx:03d}.png")
        Image.fromarray(_extract_latest_rgb(obs)).save(path)
        return path
    except Exception as exc:  # nunca romper la eval por el dump
        print(f"  [dump] no pude guardar frame: {exc}")
        return None


def resolve_artifacts(model_arg):
    """Devuelve (model_zip, vecnormalize_pkl, run_metadata_json) desde carpeta o .zip."""
    path = os.path.abspath(model_arg)
    if os.path.isdir(path):
        return (
            os.path.join(path, "final_model.zip"),
            os.path.join(path, "vecnormalize.pkl"),
            os.path.join(path, "run_metadata.json"),
        )
    if not path.endswith(".zip"):
        path += ".zip"
    parent = os.path.dirname(path)
    return (
        path,
        os.path.join(parent, "vecnormalize.pkl"),
        os.path.join(parent, "run_metadata.json"),
    )


def read_n_stack(metadata_path, default):
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return int(data["hyperparameters"]["n_stack"])
    except (OSError, ValueError, KeyError, TypeError):
        return default


def build_eval_spawns(spawns_path, texture, spawn_index):
    """
    Escribe un spawns.json temporal con UN solo track (el elegido), de modo que el
    supervisor siempre lo seleccione. Devuelve (ruta_temporal, tiene_gates).
    """
    with open(spawns_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    tracks = data.get("tracks", []) if isinstance(data, dict) else []
    match = next((t for t in tracks if t.get("texture") == texture), None)
    if match is None:
        raise SystemExit(
            f"No hay un track con texture '{texture}' en {spawns_path}. "
            "Agregalo con su texture, al menos un spawn y los gates."
        )
    spawns = match.get("spawns", [])
    if not spawns:
        raise SystemExit(f"El track '{texture}' no tiene spawns en {spawns_path}.")
    idx = max(0, min(spawn_index, len(spawns) - 1))

    gates = match.get("gates")
    has_gates = isinstance(gates, list) and len(gates) >= 2
    if not has_gates:
        print(
            f"El track '{texture}' no tiene gates -> la vuelta se detecta por la LINEA "
            "DE META unica (cruce sobre el punto del spawn, en el sentido del spawn)."
        )

    one_track = {"texture": texture, "spawns": [spawns[idx]]}
    if has_gates:
        one_track["gates"] = gates

    tmp = tempfile.NamedTemporaryFile(
        "w", suffix="_eval_spawns.json", delete=False, encoding="utf-8"
    )
    json.dump({"tracks": [one_track]}, tmp)
    tmp.close()
    return tmp.name, has_gates


def build_vec_env(host, port, vecnormalize_path, n_stack):
    """DummyVecEnv -> VecNormalize (cargado, sin entrenar) -> VecFrameStack, igual que el train."""
    vec_env = DummyVecEnv([lambda: NavEnv(host=host, port=port)])
    if os.path.exists(vecnormalize_path):
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False  # no actualizar estadisticas en evaluacion
        vec_env.norm_reward = False
    else:
        print(
            f"ADVERTENCIA: no se encontro {vecnormalize_path}; la velocity NO se "
            "normaliza igual que en el train (la policy puede comportarse distinto)."
        )
    if n_stack and n_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)
    return vec_env


def classify_done(info):
    """Devuelve el motivo de cierre del episodio: 'lap_complete'/'timeout'/term_reason."""
    if info.get("TimeLimit.truncated", False):
        return "timeout"
    breakdown = info.get("reward_breakdown")
    if isinstance(breakdown, dict) and breakdown.get("term_reason"):
        return breakdown["term_reason"]
    return "desconocido"


def discover_eval_tracks(spawns_path):
    """Texturas de los tracks marcados "eval": true en spawns.json, en orden."""
    with open(spawns_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    tracks = data.get("tracks", []) if isinstance(data, dict) else []
    return [
        t["texture"]
        for t in tracks
        if isinstance(t, dict) and t.get("eval") and t.get("texture")
    ]


def evaluate_one_track(args, model, n_stack, vecnormalize_path, max_episodes, texture):
    """
    Evalua UN track: lanza su propio Webots forzando ese unico track, corre hasta
    juntar --laps vueltas (o agotar max_episodes) y devuelve un dict de resultados.
    """
    eval_spawns_path, has_gates = build_eval_spawns(
        args.spawns, texture, args.spawn_index
    )
    # SPAWNS_JSON se setea por track ANTES de lanzar Webots (launch copia os.environ).
    os.environ["SPAWNS_JSON"] = eval_spawns_path

    print("=" * 56)
    print(f"TRACK: {texture}  (spawn #{args.spawn_index}, "
          f"deteccion: {'gates' if has_gates else 'linea de meta unica'})")
    print("-" * 56)

    webots_process = None
    vec_env = None
    lap_steps_list = []           # steps de las vueltas COMPLETAS
    failures = {}                 # motivo -> conteo
    episode_rewards = []          # retorno (reward acumulado) de CADA episodio
    deterministic = not args.stochastic
    try:
        if not args.no_webots_launch:
            webots_process = launch_webots(args)

        vec_env = build_vec_env(args.host, args.port, vecnormalize_path, n_stack)

        obs = vec_env.reset()
        episode = 0
        step_in_episode = 0
        ep_reward = 0.0
        wall_start = time.perf_counter()
        laps_done = 0
        fixed_n = args.episodes  # None => modo "hasta --laps vueltas"
        step_global = 0           # steps totales del track (para el stride del dump)
        dumped = 0                # frames volcados en este track

        while True:
            # Corte: N episodios fijos (modo tasa de exito) o hasta juntar --laps vueltas.
            if fixed_n is not None:
                if episode >= fixed_n:
                    break
            elif laps_done >= args.laps or episode >= max_episodes:
                break
            # Volcado de frames con STRIDE: 1 cada --dump-every steps, hasta --dump-frames.
            # Se toma el frame que la policy VE ahora (obs actual) -> input real de la CNN.
            if args.dump_frames and dumped < args.dump_frames \
                    and step_global % max(1, args.dump_every) == 0:
                if _save_frame(obs, args.dump_dir, texture, dumped):
                    dumped += 1
            step_global += 1
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _rewards, dones, infos = vec_env.step(action)
            step_in_episode += 1
            ep_reward += float(_rewards[0])  # reward real (VecNormalize no lo normaliza en eval)

            if not dones[0]:
                continue

            episode += 1
            episode_rewards.append(ep_reward)
            info = infos[0]
            reason = classify_done(info)
            wall = time.perf_counter() - wall_start

            if reason == "lap_complete":
                laps_done += 1
                lap_steps_list.append(step_in_episode)
                sim_s = step_in_episode * args.dt
                print(
                    f"  ep {episode:2d}: VUELTA #{laps_done} en {step_in_episode:4d} steps "
                    f"(~{sim_s:5.1f}s sim, {wall:5.1f}s reloj)"
                )
            else:
                failures[reason] = failures.get(reason, 0) + 1
                print(
                    f"  ep {episode:2d}: SIN vuelta ({reason}) tras {step_in_episode:4d} steps"
                )

            step_in_episode = 0
            ep_reward = 0.0
            wall_start = time.perf_counter()

        if args.dump_frames:
            print(f"  [dump] {dumped} frame(s) guardados en {args.dump_dir}/ "
                  f"(1 cada {args.dump_every} steps)")
    finally:
        if vec_env is not None:
            vec_env.close()
        if webots_process is not None:
            try:
                webots_process.kill()
            except Exception:
                pass
        try:
            os.remove(eval_spawns_path)
        except OSError:
            pass

    return {
        "texture": texture,
        "episodes": episode,
        "lap_steps": lap_steps_list,
        "failures": failures,
        "episode_rewards": episode_rewards,
        "has_gates": has_gates,
    }


def summarize_result(result, laps_requested, dt):
    """
    Deriva las metricas agregadas de UN track desde el dict crudo de
    evaluate_one_track (mismo origen para el JSON guardado y los prints).
    """
    lap_steps = result.get("lap_steps", [])
    episodes = int(result.get("episodes", 0))
    laps = len(lap_steps)
    rewards = result.get("episode_rewards", [])
    failures = dict(result.get("failures", {}))

    summary = {
        "texture": result["texture"],
        "has_gates": result.get("has_gates"),
        "episodes": episodes,
        "laps": laps,
        "laps_requested": int(laps_requested),
        # NOTA: el loop corta al juntar `laps_requested` vueltas, asi que lap_rate es
        # sobre los episodios efectivamente corridos (sesgado si el modelo es bueno).
        # Para una tasa de exito limpia, correr con --laps alto o un nro fijo de episodios.
        "lap_rate": (laps / episodes) if episodes else 0.0,
        "failures": failures,
        "failure_rates": (
            {k: v / episodes for k, v in failures.items()} if episodes else {}
        ),
        "reward_ep_mean": float(np.mean(rewards)) if rewards else None,
        "reward_ep_std": float(np.std(rewards)) if rewards else None,
        "lap_steps": [int(s) for s in lap_steps],
    }
    if lap_steps:
        steps_arr = np.asarray(lap_steps, dtype=float)
        summary["lap_steps_mean"] = float(steps_arr.mean())
        summary["lap_steps_min"] = int(steps_arr.min())
        summary["lap_steps_max"] = int(steps_arr.max())
        summary["lap_time_s_mean"] = float(steps_arr.mean() * dt)
    else:
        summary["lap_steps_mean"] = None
        summary["lap_steps_min"] = None
        summary["lap_steps_max"] = None
        summary["lap_time_s_mean"] = None
    return summary


def save_eval_results(args, model_zip, n_stack, results):
    """
    Persiste las metricas de eval a un JSON dentro de la carpeta de la corrida (o al
    lado del .zip), para poder agregar entre seeds despues. Devuelve la ruta.
    """
    model_path = os.path.abspath(args.model)
    out_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_zip)
    stamp = time.strftime("%Y%m%d%H%M%S")
    out_path = os.path.join(out_dir, f"eval_results_{stamp}.json")
    # Modo real de la corrida: "episodes" (N fijos, tasa de exito sin sesgo) o "laps"
    # (hasta juntar --laps vueltas). requested_per_track = el N efectivo de ese modo.
    mode = "episodes" if args.episodes is not None else "laps"
    requested = int(args.episodes) if args.episodes is not None else int(args.laps)
    payload = {
        "model": os.path.abspath(model_zip),
        "n_stack": int(n_stack),
        "mode": mode,
        "requested_per_track": requested,
        "laps_requested": int(args.laps),  # solo significativo en modo "laps"
        "max_episode_steps": int(args.max_episode_steps),
        "deterministic": not args.stochastic,
        "randomized_background": bool(args.randomize_background),
        "eval_seed": args.eval_seed,
        "dt": float(args.dt),
        "tracks": [summarize_result(r, args.laps, args.dt) for r in results],
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    print(f"\nMetricas de eval guardadas en: {out_path}")
    return out_path


def print_track_summary(result, args):
    """Resumen por track."""
    lap_steps = result["lap_steps"]
    episodes = int(result.get("episodes", 0))
    rewards = result.get("episode_rewards", [])
    print("-" * 56)
    if args.episodes is not None:
        print(f"  vueltas: {len(lap_steps)} en {args.episodes} episodios")
    else:
        print(f"  vueltas: {len(lap_steps)} / {args.laps} pedidas")
    if episodes:
        print(f"  lap rate     : {len(lap_steps)}/{episodes} episodios "
              f"({100.0 * len(lap_steps) / episodes:.0f}%)")
    if rewards:
        print(f"  reward/ep    : prom {np.mean(rewards):6.2f}  desv {np.std(rewards):.2f}  "
              f"(n={len(rewards)})")
    if lap_steps:
        steps_arr = np.asarray(lap_steps, dtype=float)
        print(
            f"  steps/vuelta : min {int(steps_arr.min())}  "
            f"prom {steps_arr.mean():.0f}  max {int(steps_arr.max())}"
        )
        print(
            f"  tiempo/vuelta: min {steps_arr.min() * args.dt:.1f}s  "
            f"prom {steps_arr.mean() * args.dt:.1f}s  "
            f"max {steps_arr.max() * args.dt:.1f}s (sim, dt={args.dt})"
        )
    if result["failures"]:
        detalle = ", ".join(f"{k}={v}" for k, v in sorted(result["failures"].items()))
        print(f"  fallos       : {detalle}")


def print_overall_summary(results, args):
    """Tabla final comparando todos los tracks evaluados."""
    # Denominador y etiqueta segun el MODO real (episodios fijos vs hasta --laps
    # vueltas), no el default de --laps: en modo --episodes la fraccion es vueltas/ep.
    if args.episodes is not None:
        n_req, unidad = args.episodes, "episodios"
    else:
        n_req, unidad = args.laps, "vueltas pedidas"
    print("=" * 56)
    print(f"RESUMEN GLOBAL ({len(results)} tracks, {n_req} {unidad} c/u)")
    print("-" * 56)
    header = "vueltas/ep" if args.episodes is not None else "vueltas"
    print(f"  {'track':<16}{header:>11}{'prom steps':>12}{'prom seg':>10}")
    for result in results:
        lap_steps = result["lap_steps"]
        if lap_steps:
            mean_steps = float(np.mean(lap_steps))
            steps_txt = f"{mean_steps:.0f}"
            secs_txt = f"{mean_steps * args.dt:.1f}"
        else:
            steps_txt = "-"
            secs_txt = "-"
        vueltas = f"{len(lap_steps)}/{n_req}"
        print(f"  {result['texture']:<16}{vueltas:>11}{steps_txt:>12}{secs_txt:>10}")
    print("=" * 56)


def run_evaluation(args):
    model_zip, inferred_vecnorm, metadata_path = resolve_artifacts(args.model)
    if not os.path.exists(model_zip):
        raise SystemExit(f"No se encontro el modelo: {model_zip}")
    vecnormalize_path = os.path.abspath(args.vecnormalize or inferred_vecnorm)
    n_stack = args.n_stack if args.n_stack is not None else read_n_stack(metadata_path, 4)
    max_episodes = args.max_episodes if args.max_episodes is not None else args.laps * 5 + 5

    # Lista de tracks a evaluar: el indicado, o TODOS los marcados "eval": true.
    if args.track:
        tracks = [args.track]
    else:
        tracks = discover_eval_tracks(args.spawns)
        if not tracks:
            raise SystemExit(
                f"No hay tracks con \"eval\": true en {args.spawns}. "
                "Marca los de evaluacion o pasa --track <textura>."
            )
    if args.no_webots_launch and len(tracks) > 1:
        raise SystemExit(
            "--no-webots-launch solo sirve para un track (Webots ya cargado). "
            "Para varios tracks de eval, deja que el script lance Webots por track."
        )

    # Config comun para todos los tracks (heredada por cada Webots que se lanza).
    os.environ["DOMAIN_RANDOMIZATION_ENABLED"] = "0"
    if args.randomize_background:
        # Test de robustez: fondo aleatorio por episodio (siempre, prob=1.0).
        os.environ["BACKGROUND_RANDOMIZATION_ENABLED"] = "1"
        os.environ["BACKGROUND_RANDOMIZATION_PROBABILITY"] = "1.0"
    else:
        os.environ["BACKGROUND_RANDOMIZATION_ENABLED"] = "0"  # eval determinista: fondo fijo
    if args.eval_seed is not None:
        # Fija spawn + fondo para que todos los modelos vean la misma secuencia de episodios.
        os.environ["RESET_RNG_SEED"] = str(int(args.eval_seed))
    os.environ["MAX_EPISODE_STEPS"] = str(int(args.max_episode_steps))

    if args.episodes is not None:
        modo_txt = f"{args.episodes} episodios fijos/track (tasa de exito)"
    else:
        modo_txt = f"hasta {args.laps} vueltas/track"
    print("=" * 56)
    print(f"Modelo:        {model_zip}")
    print(f"VecNormalize:  {vecnormalize_path}")
    print(f"Tracks eval:   {', '.join(tracks)}")
    print(f"Modo:          {modo_txt}   Time limit: {int(args.max_episode_steps)} steps")
    print(f"Accion:        {'estocastica' if args.stochastic else 'determinista'}  "
          f"(n_stack={n_stack})")

    model = PPO.load(model_zip, device=args.device)

    results = []
    for texture in tracks:
        result = evaluate_one_track(
            args, model, n_stack, vecnormalize_path, max_episodes, texture
        )
        print_track_summary(result, args)
        results.append(result)

    if len(results) > 1:
        print_overall_summary(results, args)

    if getattr(args, "no_save_results", False):
        print("[dump] --no-save-results: NO se escribe eval_results (solo volcado de frames).")
    else:
        save_eval_results(args, model_zip, n_stack, results)


def main():
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
