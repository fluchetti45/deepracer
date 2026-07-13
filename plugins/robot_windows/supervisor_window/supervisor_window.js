import RobotWindow from "https://cyberbotics.com/wwi/R2025a/RobotWindow.js";

window.robotWindow = new RobotWindow();

const controls = [
  { id: "forward",  label: "Avanzar",          action: [0.6,  0.6]  },
  { id: "left",     label: "Doblar Izquierda",  action: [-0.2, 0.6]  },
  { id: "right",    label: "Doblar Derecha",    action: [0.6, -0.2]  },
  { id: "backward", label: "Retroceder",        action: [-0.4, -0.4] },
  { id: "stop",     label: "Frenar",            action: [0.0,  0.0]  },
];

function sendMessage(payload) {
  console.log("[SupervisorWindow] sendMessage", payload);
  window.robotWindow.send(JSON.stringify(payload));
}

function isExplicitModelPath(modelPath) {
  if (!modelPath) return false;
  return (
    modelPath.includes("\\") ||
    modelPath.includes("/") ||
    /^[a-zA-Z]:/.test(modelPath)
  );
}

function getPolicyModelPathInput() {
  return document.getElementById("policy-model-path");
}

function getNormalizedPolicyModelPath() {
  return String(getPolicyModelPathInput()?.value ?? "").trim();
}

function isDeterministicSelected() {
  return document.getElementById("policy-deterministic")?.checked ?? true;
}

function requestPolicyLoad(modelPath, source = "unknown") {
  const normalizedPath = String(modelPath ?? "").trim();
  console.log("[SupervisorWindow] requestPolicyLoad", { source, modelPath, normalizedPath });
  if (!normalizedPath) {
    console.warn("[SupervisorWindow] requestPolicyLoad sin ruta valida.");
    return;
  }
  sendMessage({
    type: "configure_policy",
    model_path: normalizedPath,
    deterministic: isDeterministicSelected(),
  });
}

function updateText(id, value) {
  document.getElementById(id).textContent = value;
}

function formatVec2(v) {
  if (!Array.isArray(v) || v.length < 2) return "N/A";
  return `[${v[0].toFixed(4)}, ${v[1].toFixed(4)}]`;
}

function formatPolicyStatus(policyStatus) {
  if (!policyStatus) return "Todavia no hay informacion de politica.";

  const lines = [
    `Cargada: ${policyStatus.loaded ? "si" : "no"}`,
    `Activa: ${policyStatus.enabled ? "si" : "no"}`,
    `Deterministica: ${policyStatus.deterministic ? "si" : "no"}`,
    `Pasos ejecutados: ${policyStatus.step_count ?? 0}`,
    `Modelo: ${policyStatus.model_path ?? "N/A"}`,
  ];
  if (Array.isArray(policyStatus.last_action)) {
    lines.push(`Ultima accion: ${JSON.stringify(policyStatus.last_action)}`);
  }
  if (policyStatus.last_error) {
    lines.push(`Ultimo error: ${policyStatus.last_error}`);
  }
  return lines.join("\n");
}

function policyBannerState(policyStatus) {
  if (!policyStatus) {
    return {
      className: "idle",
      title: "Sin informacion de politica",
      description: "Todavia no se consulto el estado del robot.",
    };
  }
  if (policyStatus.last_error) {
    return { className: "error", title: "Politica con error", description: policyStatus.last_error };
  }
  if (policyStatus.enabled) {
    return {
      className: "running",
      title: "Politica ejecutandose",
      description: policyStatus.model_path ?? "Modelo activo en el robot.",
    };
  }
  if (policyStatus.loaded) {
    return { className: "loaded", title: "Politica cargada", description: policyStatus.model_path ?? "Lista para activarse." };
  }
  return { className: "idle", title: "Sin politica cargada", description: "Selecciona un modelo para poder activarlo." };
}

function formatPolicyDebugSnapshot(snapshot) {
  if (!snapshot) return "Todavia no hay snapshot de policy.";

  const lines = [
    `Policy cargada: ${snapshot.policy_loaded ? "si" : "no"}`,
    `Policy activa: ${snapshot.policy_enabled ? "si" : "no"}`,
    `Deterministica: ${snapshot.policy_deterministic ? "si" : "no"}`,
    `Modelo: ${snapshot.model_path ?? "N/A"}`,
    `Velocity [forward, yaw]: ${JSON.stringify(snapshot.velocity ?? [])}`,
    `Goal direction: ${formatVec2(snapshot.goal_dir)}`,
    `Distancia al objetivo: ${snapshot.distance_to_goal != null ? snapshot.distance_to_goal.toFixed(4) : "N/A"}`,
    `Ultima accion aplicada: ${JSON.stringify(snapshot.last_action ?? null)}`,
    `Accion predicha ahora: ${JSON.stringify(snapshot.predicted_action ?? null)}`,
  ];
  if (snapshot.prediction_error) {
    lines.push(`Error al predecir: ${snapshot.prediction_error}`);
  }
  return lines.join("\n");
}

function formatDebugValue(value) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(6);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "null";
  if (Array.isArray(value) || typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function renderKeyValueRows(containerId, entries, emptyText) {
  const container = document.getElementById(containerId);
  container.replaceChildren();

  if (!entries || entries.length === 0) {
    const empty = document.createElement("code");
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }

  entries.forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "kv-row";

    const keyNode = document.createElement("span");
    keyNode.className = "kv-key";
    keyNode.textContent = key;

    const valueNode = document.createElement("span");
    valueNode.className = "kv-value";
    valueNode.textContent = formatDebugValue(value);

    row.appendChild(keyNode);
    row.appendChild(valueNode);
    container.appendChild(row);
  });
}

function renderNavMetrics(snapshot) {
  const entries = [];
  if (snapshot?.goal_dir != null) entries.push(["goal_dir", snapshot.goal_dir]);
  if (snapshot?.distance_to_goal != null) entries.push(["distance_to_goal", snapshot.distance_to_goal]);
  if (snapshot?.robot_pos != null) entries.push(["robot_pos", snapshot.robot_pos]);
  if (snapshot?.target_pos != null) entries.push(["target_pos", snapshot.target_pos]);
  if (snapshot?.velocity != null) entries.push(["velocity", snapshot.velocity]);
  renderKeyValueRows("policy-debug-nav-metrics", entries, "Todavia no hay metricas de navegacion.");
}

function renderRewardMetrics(snapshot) {
  const reward = snapshot?.reward_inputs;
  if (!reward || typeof reward !== "object") {
    renderKeyValueRows("policy-debug-reward-metrics", [], "Todavia no hay metricas de reward.");
    return;
  }
  // El breakdown ya viene curado por el supervisor (reward de vision actual): mostrar
  // todas sus claves en vez de filtrar por una lista fija (que quedaba vieja).
  renderKeyValueRows("policy-debug-reward-metrics", Object.entries(reward), "Todavia no hay metricas de reward.");
}

function renderLane(lane) {
  const swatch = document.getElementById("lane-rgb-swatch");
  const rgb = lane?.center_rgb;
  if (Array.isArray(rgb) && rgb.length === 3) {
    updateText("lane-rgb", `[${rgb[0]}, ${rgb[1]}, ${rgb[2]}]`);
    swatch.style.background = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
  } else {
    updateText("lane-rgb", "N/A");
    swatch.style.background = "#000";
  }
  const fix = (v) => (typeof v === "number" ? v.toFixed(4) : "N/A");
  updateText("lane-green-center", fix(lane?.green_center));
  updateText("lane-white-center", fix(lane?.white_center));
  updateText("lane-clearance", fix(lane?.center_clearance));
  updateText("lane-offset", fix(lane?.offset));
  updateText("lane-line-visible", lane?.line_visible ? "si" : "no");
}

let trackSelectPopulated = false;

function populateTrackSelector(tracks, current) {
  const sel = document.getElementById("track-select");
  if (!sel) return;
  if (Array.isArray(tracks) && tracks.length && !trackSelectPopulated) {
    sel.replaceChildren();
    tracks.forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      sel.appendChild(opt);
    });
    trackSelectPopulated = true;
  }
  if (current) sel.value = current;
  updateText("current-track", current ?? "N/A");
}

let wallSelectPopulated = false;
let skyboxSelectPopulated = false;

// Popula un <select> UNA vez con las opciones dadas y refleja la seleccion actual.
function populateSelect(selectId, currentLabelId, options, current, alreadyPopulated) {
  const sel = document.getElementById(selectId);
  if (!sel) return alreadyPopulated;
  let populated = alreadyPopulated;
  if (Array.isArray(options) && options.length && !alreadyPopulated) {
    sel.replaceChildren();
    options.forEach((o) => {
      const opt = document.createElement("option");
      opt.value = o;
      opt.textContent = o;
      sel.appendChild(opt);
    });
    populated = true;
  }
  if (current) sel.value = current;
  updateText(currentLabelId, current ?? "N/A");
  return populated;
}

function normalizeSelectedPath(fileInput) {
  const selectedFile = fileInput?.files?.[0];
  if (!selectedFile) return "";
  if (selectedFile.path) return selectedFile.path;
  const rawValue = fileInput.value || "";
  return rawValue.replace(/^C:\\fakepath\\/i, "") || selectedFile.name || "";
}

function renderCameraFrame(canvasId, placeholderId, frame) {
  const canvas = document.getElementById(canvasId);
  const placeholder = document.getElementById(placeholderId);

  if (!frame || !Array.isArray(frame.data) || !frame.width || !frame.height) {
    placeholder.textContent = "Todavia no hay imagen del robot.";
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const { width, height, data: rgbData } = frame;
  const rgbaData = new Uint8ClampedArray(width * height * 4);

  for (let i = 0, j = 0; i < rgbData.length; i += 3, j += 4) {
    rgbaData[j]     = rgbData[i];
    rgbaData[j + 1] = rgbData[i + 1];
    rgbaData[j + 2] = rgbData[i + 2];
    rgbaData[j + 3] = 255;
  }

  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").putImageData(new ImageData(rgbaData, width, height), 0, 0);
  placeholder.textContent = `Camara completa (${width} x ${height})`;
}

function renderPolicyBanner(policyStatus) {
  const banner = document.getElementById("policy-banner");
  const title = document.getElementById("policy-banner-title");
  const description = document.getElementById("policy-banner-description");
  const state = policyBannerState(policyStatus);
  banner.className = `policy-banner ${state.className}`;
  title.textContent = state.title;
  description.textContent = state.description;
}

function wireButtons() {
  controls.forEach((control) => {
    document.getElementById(control.id).addEventListener("click", () => {
      sendMessage({ type: "action", label: control.id, action: control.action });
    });
  });

  document.getElementById("reset-pose").addEventListener("click", () => {
    sendMessage({ type: "reset_pose" });
  });

  document.getElementById("refresh-state").addEventListener("click", () => {
    sendMessage({ type: "request_state" });
  });

  document.getElementById("capture-frame").addEventListener("click", () => {
    sendMessage({ type: "capture_frame", mode: "full" });
  });

  // --- Test de velocidad constante (calibracion de min/max) ---
  const speedInput = document.getElementById("test-speed-input");
  const applyTestSpeed = () => {
    const v = parseFloat(speedInput?.value);
    if (!Number.isNaN(v)) sendMessage({ type: "test_speed", rad_s: v });
  };
  document.getElementById("test-speed-min").addEventListener("click", () => {
    sendMessage({ type: "test_speed", preset: "min" });
  });
  document.getElementById("test-speed-max").addEventListener("click", () => {
    sendMessage({ type: "test_speed", preset: "max" });
  });
  document.getElementById("test-speed-apply").addEventListener("click", applyTestSpeed);
  speedInput?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") applyTestSpeed();
  });
  document.getElementById("test-speed-stop").addEventListener("click", () => {
    sendMessage({ type: "test_speed", rad_s: 0 });
  });

  document.getElementById("track-select").addEventListener("change", (event) => {
    const texture = event.target.value;
    if (texture) sendMessage({ type: "set_track", texture });
  });

  document.getElementById("wall-select").addEventListener("change", (event) => {
    const texture = event.target.value;
    if (texture) sendMessage({ type: "set_wall_texture", texture });
  });

  document.getElementById("skybox-select").addEventListener("change", (event) => {
    const skybox = event.target.value;
    if (skybox) sendMessage({ type: "set_skybox", skybox });
  });

  document.getElementById("request-policy-debug").addEventListener("click", () => {
    sendMessage({ type: "request_policy_debug" });
  });

  document.getElementById("select-policy-model").addEventListener("click", () => {
    document.getElementById("policy-model-file").click();
  });

  document.getElementById("policy-model-file").addEventListener("change", (event) => {
    const modelPath = normalizeSelectedPath(event.target);
    if (!modelPath) {
      console.warn("[SupervisorWindow] no se pudo resolver modelPath desde el file input.");
      return;
    }
    getPolicyModelPathInput().value = modelPath;
    updateText("policy-model-selected", modelPath);
    if (!isExplicitModelPath(modelPath)) {
      updateText(
        "policy-model-selected",
        `${modelPath}\nADVERTENCIA: solo llego el nombre. Pega la ruta completa en el campo superior y usa 'Cargar Ruta Escrita'.`
      );
      return;
    }
    requestPolicyLoad(modelPath, "file_input");
  });

  document.getElementById("load-policy-from-path").addEventListener("click", () => {
    requestPolicyLoad(getPolicyModelPathInput().value, "manual_input");
  });

  getPolicyModelPathInput().addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    requestPolicyLoad(event.target.value, "manual_input_enter");
  });

  document.getElementById("enable-policy").addEventListener("click", () => {
    sendMessage({
      type: "configure_policy",
      model_path: getNormalizedPolicyModelPath() || undefined,
      enabled: true,
      deterministic: isDeterministicSelected(),
    });
  });

  // Cambiar determinismo en vivo (util para A/B sin recargar el modelo).
  document.getElementById("policy-deterministic").addEventListener("change", () => {
    sendMessage({ type: "configure_policy", deterministic: isDeterministicSelected() });
  });

  document.getElementById("disable-policy").addEventListener("click", () => {
    sendMessage({ type: "configure_policy", enabled: false });
  });

  document.getElementById("unload-policy").addEventListener("click", () => {
    sendMessage({ type: "configure_policy", unload: true });
  });
}

window.onload = () => {
  wireButtons();

  window.robotWindow.receive = (message) => {
    console.log("[SupervisorWindow] mensaje recibido crudo", message);
    const data = JSON.parse(message);
    console.log("[SupervisorWindow] mensaje parseado", data);
    if (data.type !== "status") {
      console.log("[SupervisorWindow] ignorando mensaje no-status", data.type);
      return;
    }

    updateText("status-text",   data.status ?? "Sin estado");
    updateText("last-action",   data.last_action ?? "N/A");
    updateText("step-count",    String(data.step_count ?? 0));
    updateText("episode-id",    String(data.episode_id ?? 0));
    updateText("episode-step",  String(data.episode_step ?? 0));
    updateText("training-client", data.training_client_connected ? "Conectado" : "Sin cliente");
    // Rango de velocidad actual (.env) para el modo test de calibracion.
    updateText("wheel-min",
      data.wheel_min_speed != null ? data.wheel_min_speed.toFixed(2) + " rad/s" : "N/A");
    updateText("wheel-max",
      data.wheel_max_speed != null ? data.wheel_max_speed.toFixed(2) + " rad/s" : "N/A");
    updateText("last-robot-message",
      data.last_robot_message
        ? JSON.stringify(data.last_robot_message, null, 2)
        : "Todavia no hay respuestas del robot."
    );

    // Navigation state
    updateText("distance-to-goal",
      data.distance_to_goal != null ? data.distance_to_goal.toFixed(4) + " m" : "N/A"
    );
    updateText("goal-dir",   formatVec2(data.goal_dir ?? null));
    updateText("robot-pos",  formatVec2(data.robot_pos ?? null));
    updateText("target-pos", formatVec2(data.target_pos ?? null));

    renderPolicyBanner(data.policy_status ?? null);
    updateText("policy-status",        formatPolicyStatus(data.policy_status ?? null));
    updateText("policy-model-current", data.policy_status?.model_path ?? "Ningun modelo cargado.");

    // Reflejar el determinismo real del backend en el checkbox.
    if (data.policy_status?.deterministic != null) {
      document.getElementById("policy-deterministic").checked =
        !!data.policy_status.deterministic;
    }

    // Estado de carril (vision-pura) + selector de pista.
    renderLane(data.lane ?? null);
    populateTrackSelector(data.available_tracks ?? null, data.current_track ?? null);
    // Selectores de fondo (pared / skybox).
    wallSelectPopulated = populateSelect(
      "wall-select", "current-wall",
      data.available_wall_textures ?? null, data.current_wall_texture ?? null,
      wallSelectPopulated);
    skyboxSelectPopulated = populateSelect(
      "skybox-select", "current-skybox",
      data.available_skyboxes ?? null, data.current_skybox ?? null,
      skyboxSelectPopulated);

    const snapshot = data.policy_debug_snapshot ?? null;
    updateText("policy-debug-summary", formatPolicyDebugSnapshot(snapshot));
    renderNavMetrics(snapshot);
    renderRewardMetrics(snapshot);

    renderCameraFrame("camera-frame",       "camera-placeholder",    data.camera_frame ?? null);
    renderCameraFrame("policy-full-frame",  "policy-full-placeholder", snapshot?.full_frame ?? null);
  };

  sendMessage({ type: "request_state" });
};
