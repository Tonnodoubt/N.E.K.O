const pluginId = "mahjong_companion";
let autoRefreshTimer = null;

const terminalRunStatuses = new Set(["succeeded", "failed", "canceled", "timeout"]);
const failedRunStatuses = new Set(["failed", "canceled", "timeout"]);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function describeError(payload, fallback) {
  if (payload && typeof payload === "object") {
    const detail = payload.detail;
    if (typeof detail === "string" && detail) return detail;
    if (detail && typeof detail === "object") {
      return detail.message || detail.code || JSON.stringify(detail);
    }

    const error = payload.error;
    if (typeof error === "string" && error) return error;
    if (error && typeof error === "object") {
      return error.message || error.code || JSON.stringify(error);
    }

    if (typeof payload.message === "string" && payload.message) {
      return payload.message;
    }
  }
  return fallback;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: text };
    }
  }
  if (!response.ok) {
    throw new Error(describeError(data, `HTTP ${response.status}`));
  }
  return data;
}

async function waitForRun(runId) {
  const deadline = Date.now() + 320000;
  while (true) {
    const run = await fetchJson(`/runs/${encodeURIComponent(runId)}`);
    if (terminalRunStatuses.has(String(run.status || ""))) {
      return run;
    }
    if (Date.now() > deadline) {
      throw new Error(`Run ${runId} did not finish in time`);
    }
    await sleep(300);
  }
}

function extractRunExportPayload(exportPayload) {
  const items = Array.isArray(exportPayload?.items) ? exportPayload.items : [];
  const item = items.find((entry) => entry?.metadata?.kind === "trigger_response") || items[0];
  if (!item || typeof item !== "object") return null;
  if (item.json && typeof item.json === "object") return item.json;
  if (item.json_data && typeof item.json_data === "object") return item.json_data;
  return null;
}

async function fetchRunPayload(run) {
  const runId = run?.run_id;
  if (!runId) return run;

  const exportPayload = await fetchJson(`/runs/${encodeURIComponent(runId)}/export?limit=20`);
  const pluginPayload = extractRunExportPayload(exportPayload);
  if (pluginPayload) return pluginPayload;

  if (failedRunStatuses.has(String(run.status || ""))) {
    throw new Error(describeError(run, `Run ${runId} ${run.status}`));
  }
  return run;
}

async function callEntry(entryId, args = {}) {
  const created = await fetchJson("/runs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      plugin_id: pluginId,
      entry_id: entryId,
      args,
    }),
  });

  const runId = created?.run_id;
  if (!runId) {
    return created;
  }

  const run = await waitForRun(runId);
  const payload = await fetchRunPayload(run);
  if (payload?.success === false || failedRunStatuses.has(String(run.status || ""))) {
    throw new Error(describeError(payload, describeError(run, `Run ${runId} ${run.status}`)));
  }
  return payload;
}

function renderJson(elementId, payload) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.textContent = JSON.stringify(payload, null, 2);
}

function setText(elementId, value) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.textContent = String(value);
}

function unwrapPayload(payload) {
  if (payload && typeof payload === "object" && payload.data && typeof payload.data === "object") {
    return payload.data;
  }
  return payload;
}

function windowOptionLabel(candidate) {
  const title = String(candidate?.title || "").trim();
  if (!title) return "";
  const size = candidate?.width && candidate?.height ? ` (${candidate.width}x${candidate.height})` : "";
  const hint = candidate?.matches_keywords ? " *" : "";
  return `${title}${size}${hint}`;
}

function renderWindowCandidates(payload) {
  const data = unwrapPayload(payload) || {};
  const select = document.getElementById("window-select");
  if (!select) return;
  const previous = select.value;
  const selected = String(data.selected_window_title || "").trim();
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  select.textContent = "";

  const autoOption = document.createElement("option");
  autoOption.value = "";
  autoOption.textContent = "自动匹配雀魂窗口";
  select.appendChild(autoOption);

  for (const candidate of candidates) {
    const title = String(candidate?.title || "").trim();
    if (!title) continue;
    const option = document.createElement("option");
    option.value = title;
    option.textContent = windowOptionLabel(candidate);
    select.appendChild(option);
  }

  const values = Array.from(select.options).map((option) => option.value);
  if (selected && values.includes(selected)) {
    select.value = selected;
  } else if (previous && values.includes(previous)) {
    select.value = previous;
  }
}

async function refreshWindowCandidates() {
  const data = await callEntry("list_window_candidates");
  renderWindowCandidates(data);
  return data;
}

function tileLabel(value) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    return String(value.tile || value.tile_id || value.label || "?");
  }
  return "?";
}

function summarizeTileList(items, limit = 18) {
  if (!Array.isArray(items) || !items.length) return "-";
  const labels = items.map(tileLabel);
  const visible = labels.slice(0, limit).join(" ");
  return labels.length > limit ? `${visible} +${labels.length - limit}` : visible;
}

function summarizeDiscardPiles(piles) {
  if (!piles || typeof piles !== "object") return "-";
  const labels = {
    self: "自家",
    left_opponent: "左家",
    top_opponent: "上家",
    right_opponent: "右家",
  };
  const orderedPlayers = ["self", "left_opponent", "top_opponent", "right_opponent"];
  const extraPlayers = Object.keys(piles).filter((player) => !orderedPlayers.includes(player)).sort();
  const parts = [...orderedPlayers, ...extraPlayers].map((player) => {
    const pile = Array.isArray(piles[player]) ? piles[player] : [];
    const latest = pile.slice(-6).map(tileLabel).join(" ");
    return latest ? `${labels[player] || player}: ${pile.length} (${latest})` : `${labels[player] || player}: ${pile.length}`;
  });
  return parts.length ? parts.join(" / ") : "-";
}

function countDiscardTiles(piles) {
  if (!piles || typeof piles !== "object") return 0;
  return Object.values(piles).reduce((total, pile) => total + (Array.isArray(pile) ? pile.length : 0), 0);
}

function sceneLabel(scene) {
  const labels = {
    in_match: "牌局中",
    dialog: "弹窗",
    replay: "回放",
    lobby: "大厅",
    menu: "菜单",
    matching: "匹配中",
    result: "结算",
    unknown: "未识别",
  };
  return labels[scene] || scene || "-";
}

function sessionLabel(status, runtimeStatus) {
  const statusLabels = {
    standby: "待机",
    scanning: "识别中",
    running: "运行中",
    stopped: "已停止",
  };
  const runtimeLabels = {
    active: "主动",
    standby: "待机",
    off: "关闭",
  };
  const primary = statusLabels[status] || status || "-";
  const runtime = runtimeLabels[runtimeStatus] || runtimeStatus || "-";
  return `${primary} / ${runtime}`;
}

function getAnalysisHints(data) {
  const perception = data.last_perception && typeof data.last_perception === "object" ? data.last_perception : {};
  return perception.analysis_hints && typeof perception.analysis_hints === "object" ? perception.analysis_hints : {};
}

function topDiscardCandidate(data) {
  const single = data.last_decision?.engine_meta?.single_recommendation;
  if (single && (single.kind === "discard" || single.kind === "preturn_discard")) {
    if (single.candidate && typeof single.candidate === "object") return single.candidate;
    if (single.tile) return { tile: single.tile };
  }
  const analysis = data.last_decision?.mahjong_analysis;
  const candidates = analysis && Array.isArray(analysis.candidate_discards) ? analysis.candidate_discards : [];
  return candidates.find((item) => item && typeof item === "object") || null;
}

function formatTileCode(tile) {
  const raw = String(tile || "").trim();
  const match = raw.match(/^([1-9])([mpsz])$/i);
  if (!match) return raw || "-";
  const numberLabels = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
  };
  const suitLabels = {
    m: "万",
    p: "筒",
    s: "索",
    z: "字",
  };
  return `${numberLabels[match[1]] || match[1]}${suitLabels[match[2].toLowerCase()] || match[2]}`;
}

function replaceTileCodes(text) {
  return String(text || "").replace(/\b([1-9])([mpsz])\b/gi, (_match, number, suit) => {
    return formatTileCode(`${number}${suit}`);
  });
}

function candidateSuggestionText(data) {
  const candidate = topDiscardCandidate(data);
  if (!candidate) return "";
  const tile = String(candidate.tile || candidate.tile_id || "").trim();
  if (!tile) return "";
  const parts = [`推荐 ${formatTileCode(tile)}`];
  const reason = String(candidate.reason || "").trim();
  return reason ? `${parts.join(" · ")}\n${replaceTileCodes(reason)}` : parts.join(" · ");
}

function suggestionText(data) {
  return candidateSuggestionText(data)
    || data.last_decision?.suggestion
    || data.last_narration_text
    || (data.window_bound ? "已绑定窗口，点“刷新屏幕并给建议”获取当前出牌建议。" : "先绑定雀魂窗口。");
}

function recognizedHandCount(data) {
  const hints = getAnalysisHints(data);
  const count = Number(hints.recognized_hand_tile_count);
  return Number.isFinite(count) && count > 0 ? count : null;
}

function renderSummary(payload) {
  const data = unwrapPayload(payload) || {};
  const perception = data.last_perception && typeof data.last_perception === "object" ? data.last_perception : {};
  const discardPiles = perception.discard_piles && typeof perception.discard_piles === "object"
    ? perception.discard_piles
    : {};
  const windowBound = data.window_bound === true ? "已绑定" : "未绑定";
  const windowTitle = data.window_title ? ` · ${data.window_title}` : "";
  const captureState = data.last_capture_ok === true ? "成功" : data.last_capture_ok === false ? "未成功" : "-";
  const suggestion = suggestionText(data);
  const handCount = recognizedHandCount(data);
  setText("header-status", sessionLabel(data.status, data.runtime_status || data.runtime_mode));
  setText("quick-session", sessionLabel(data.status, data.runtime_status || data.runtime_mode));
  setText("quick-window", `${windowBound}${windowTitle}`);
  setText("quick-scene", `${sceneLabel(data.last_scene || data.scene)} · ${data.last_is_user_turn ? "轮到你" : "等待"}`);
  setText("quick-hand", handCount ? `手牌 ${handCount} 张` : "手牌未稳定识别");
  setText("quick-capture", `${captureState}${data.last_capture_source ? ` · ${data.last_capture_source}` : ""}`);
  setText("quick-suggestion", suggestion);
  setText("quick-voice", data.voice_mode || "-");
  setText("quick-error", data.last_error || "无");
  setText("host-status", data.status || "-");
  setText("runtime-status", data.runtime_status || "-");
  setText("current-mode", data.mode || "-");
  setText("runtime-mode", data.runtime_mode || "-");
  setText("game-runtime-status", data.game_runtime_status || "-");
  setText("runtime-inbound-pending", data.runtime_inbound_pending ?? "-");
  setText("runtime-outbound-pending", data.runtime_outbound_pending ?? "-");
  setText("runtime-deduped-outbound", data.runtime_deduped_outbound ?? "-");
  setText("runtime-interrupt-seq", data.runtime_interrupt_seq ?? "-");
  setText("runtime-last-action", data.last_runtime_command_action || "-");
  setText("runtime-last-interrupt-reason", data.last_runtime_interrupt_reason || "-");
  const modeSelect = document.getElementById("mode-select");
  if (modeSelect && typeof data.mode === "string" && data.mode) {
    modeSelect.value = data.mode;
  }
  const runtimeModeSelect = document.getElementById("runtime-mode-select");
  if (runtimeModeSelect && typeof data.runtime_mode === "string" && data.runtime_mode) {
    runtimeModeSelect.value = data.runtime_mode;
  }
  syncUnifiedModeSelect(data);
  syncVoiceModeSelect(data);
  setText("window-bound", data.window_bound ?? "-");
  setText("window-title", data.window_title || "-");
  setText("capture-ok", data.last_capture_ok ?? "-");
  setText("capture-source", data.last_capture_source || "-");
  setText("frame-path", data.last_frame_path || "-");
  setText("scene", data.last_scene || data.scene || "-");
  setText("scene-confidence", data.last_scene_confidence ?? "-");
  setText("user-turn", data.last_is_user_turn ?? "-");
  setText("buttons", Array.isArray(data.last_buttons) && data.last_buttons.length
    ? data.last_buttons.join(", ")
    : "-");
  setText("discard-count", countDiscardTiles(discardPiles) || "-");
  setText("discard-pile-summary", summarizeDiscardPiles(discardPiles));
  setText("visible-tiles", summarizeTileList(perception.visible_tiles));
  setText("known-genbutsu", summarizeTileList(perception.known_genbutsu_tiles));
  setText("perception-ok", data.last_perception_ok ?? "-");
  setText("perception-at", data.last_perception_at || "-");
  setText("decision-type", data.last_decision_type || "-");
  setText("decision-risk", data.last_decision_risk_level || "-");
  setText("decision-focus", data.last_decision?.recommended_focus || "-");
  setText("decision-tags", Array.isArray(data.last_decision?.review_tags) && data.last_decision.review_tags.length
    ? data.last_decision.review_tags.join(", ")
    : "-");
  setText("decision-at", data.last_decision_at || "-");
  setText("tile-analysis-available", data.last_tile_analysis_available ?? "-");
  setText("shanten-estimate", data.last_shanten_estimate ?? "-");
  setText("ukeire-estimate", data.last_ukeire_estimate ?? "-");
  setText("narration-type", data.last_narration_type || "-");
  setText("narration-channel", data.last_narration_channel || "-");
  setText("narration-delivery", data.last_narration_delivery || "-");
  setText("companion-mood", data.last_companion_mood || "-");
  setText("suggestion-level", data.last_companion_view?.suggestion_level || "-");
  setText("decision-suggestion", data.last_decision?.suggestion || "-");
  setText("narration-text", data.last_narration_text || "-");
  setText("voice-mode", data.voice_mode || "-");
  setText("notification-at", data.last_notification_at || "-");
  setText("spoken-at", data.last_spoken_at || "-");
  setText("last-error", data.last_error || "-");
}

async function refreshStatus(options = {}) {
  const preserveOutput = Boolean(options.preserveOutput);
  const data = await callEntry("get_session_status");
  renderSummary(data);
  renderJson("status", data);
  if (!preserveOutput) {
    renderJson("output", data);
  }
}

function syncAutoRefresh(enabled) {
  if (autoRefreshTimer) {
    clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }

  if (!enabled) return;

  autoRefreshTimer = setInterval(() => {
    refreshStatus().catch((error) => renderJson("output", { error: String(error) }));
  }, 1000);
}

async function runAction(entryId) {
  try {
    const data = await callEntry(entryId);
    renderJson("output", data);
    await refreshStatus({ preserveOutput: true });
  } catch (error) {
    renderJson("output", { error: String(error) });
  }
}

document.getElementById("refresh-btn")?.addEventListener("click", () => {
  refreshStatus().catch((error) => renderJson("output", { error: String(error) }));
});

document.getElementById("start-btn")?.addEventListener("click", () => {
  runAction("start_session");
});

document.getElementById("stop-btn")?.addEventListener("click", () => {
  runAction("stop_session");
});

document.getElementById("set-mode-btn")?.addEventListener("click", async () => {
  const mode = document.getElementById("mode-select")?.value || "teaching";
  try {
    const data = await callEntry("set_mode", { mode });
    renderJson("output", data);
    await refreshStatus({ preserveOutput: true });
  } catch (error) {
    renderJson("output", { error: String(error) });
  }
});

document.getElementById("set-runtime-mode-btn")?.addEventListener("click", async () => {
  const mode = document.getElementById("runtime-mode-select")?.value || "active";
  try {
    const data = await callEntry("set_runtime_mode", { mode });
    renderJson("output", data);
    await refreshStatus({ preserveOutput: true });
  } catch (error) {
    renderJson("output", { error: String(error) });
  }
});

document.getElementById("refresh-windows-btn")?.addEventListener("click", () => {
  refreshWindowCandidates()
    .then((data) => renderJson("output", data))
    .catch((error) => renderJson("output", { error: String(error) }));
});

document.getElementById("bind-btn")?.addEventListener("click", async () => {
  const windowTitle = String(document.getElementById("window-select")?.value || "").trim();
  try {
    const data = await callEntry("bind_window", { window_title: windowTitle });
    renderJson("output", data);
    await refreshStatus({ preserveOutput: true });
  } catch (error) {
    renderJson("output", { error: String(error) });
  }
});

document.getElementById("unbind-btn")?.addEventListener("click", () => {
  runAction("unbind_window");
});

document.getElementById("capture-btn")?.addEventListener("click", () => {
  runAction("capture_debug_frame");
});

document.getElementById("analyze-btn")?.addEventListener("click", () => {
  runAction("analyze_debug_frame");
});

document.getElementById("decision-btn")?.addEventListener("click", () => {
  runAction("generate_decision");
});

document.getElementById("narration-btn")?.addEventListener("click", () => {
  runAction("generate_narration");
});

document.getElementById("pipeline-btn")?.addEventListener("click", async () => {
  try {
    const data = await callEntry("run_companion_pipeline", {
      capture: true,
      dispatch: true,
      force_reply: true,
    });
    renderJson("output", data);
    await refreshStatus({ preserveOutput: true });
  } catch (error) {
    renderJson("output", { error: String(error) });
  }
});

document.getElementById("preview-btn")?.addEventListener("click", () => {
  runAction("preview_companion_view");
});

function unifiedModeFromState(data) {
  const runtime = String(data?.runtime_mode || "").toLowerCase();
  const mode = String(data?.mode || "").toLowerCase();
  if (runtime === "off") return "off";
  if (runtime === "standby") return "standby";
  if (mode === "silent") return "silent";
  return "teaching";
}

function syncUnifiedModeSelect(data) {
  const select = document.getElementById("unified-mode-select");
  if (!select) return;
  const target = unifiedModeFromState(data);
  if (Array.from(select.options).some((opt) => opt.value === target)) {
    select.value = target;
  }
}

function syncVoiceModeSelect(data) {
  const select = document.getElementById("voice-mode-select");
  if (!select) return;
  const target = String(data?.voice_mode || "").trim();
  if (target && Array.from(select.options).some((opt) => opt.value === target)) {
    select.value = target;
  }
}

document.getElementById("apply-unified-mode-btn")?.addEventListener("click", async () => {
  const mode = document.getElementById("unified-mode-select")?.value || "teaching";
  try {
    const data = await callEntry("set_unified_mode", { mode });
    renderJson("output", data);
    await refreshStatus({ preserveOutput: true });
  } catch (error) {
    renderJson("output", { error: String(error) });
  }
});

document.getElementById("apply-voice-mode-btn")?.addEventListener("click", async () => {
  const mode = document.getElementById("voice-mode-select")?.value || "off";
  try {
    const data = await callEntry("set_voice_mode", { mode });
    renderJson("output", data);
    await refreshStatus({ preserveOutput: true });
  } catch (error) {
    renderJson("output", { error: String(error) });
  }
});

document.getElementById("auto-refresh-toggle")?.addEventListener("change", (event) => {
  syncAutoRefresh(Boolean(event.target?.checked));
});

function applyDevMode() {
  let enabled = false;
  try {
    const params = new URLSearchParams(window.location.search);
    if (params.get("dev") === "1") enabled = true;
    if (window.localStorage?.getItem("mahjong_companion_dev") === "1") enabled = true;
  } catch {
    enabled = false;
  }
  document.body.classList.toggle("dev-active", enabled);
}

applyDevMode();
refreshWindowCandidates().catch((error) => {
  console.error("[mahjong_companion] initial refreshWindowCandidates failed:", error);
});
refreshStatus().catch((error) => renderJson("output", { error: String(error) }));
