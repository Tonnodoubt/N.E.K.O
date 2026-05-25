const PLUGIN_ID = 'mahjong_coach';
const RUNS_URL = '/runs';
const RUN_TIMEOUT_MS = 30000;

const statusLine = document.getElementById('statusLine');
const refreshBtn = document.getElementById('refreshBtn');
const resetBtn = document.getElementById('resetBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const startLiveBtn = document.getElementById('startLiveBtn');
const stopLiveBtn = document.getElementById('stopLiveBtn');
const imagePathInput = document.getElementById('imagePathInput');
const turnInput = document.getElementById('turnInput');
const buttonsInput = document.getElementById('buttonsInput');
const forceCheckpointInput = document.getElementById('forceCheckpointInput');
const keywordsInput = document.getElementById('keywordsInput');
const intervalInput = document.getElementById('intervalInput');
const overlayInput = document.getElementById('overlayInput');
const mainPlan = document.getElementById('mainPlan');
const planDetail = document.getElementById('planDetail');
const biasValue = document.getElementById('biasValue');
const lastReason = document.getElementById('lastReason');
const confidenceValue = document.getElementById('confidenceValue');
const updateCount = document.getElementById('updateCount');
const targetList = document.getElementById('targetList');
const cautionList = document.getElementById('cautionList');
const handTiles = document.getElementById('handTiles');
const handCount = document.getElementById('handCount');
const decisionType = document.getElementById('decisionType');
const decisionOutput = document.getElementById('decisionOutput');
const liveState = document.getElementById('liveState');
const liveFrame = document.getElementById('liveFrame');
const liveWindow = document.getElementById('liveWindow');
const liveError = document.getElementById('liveError');
let autoRefreshTimer = 0;

function setStatus(text) {
  statusLine.textContent = text || '';
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function compact(value, fallback = '-') {
  const text = String(value || '').trim();
  return text || fallback;
}

function percent(value) {
  const number = Number(value || 0);
  return `${Math.round(Math.max(0, Math.min(1, number)) * 100)}%`;
}

async function fetchJson(url, init = {}, timeoutMs = RUN_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } finally {
    window.clearTimeout(timeout);
  }
}

async function callPlugin(entryId, args = {}, timeoutMs = RUN_TIMEOUT_MS) {
  const created = await fetchJson(RUNS_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plugin_id: PLUGIN_ID, entry_id: entryId, args }),
  }, timeoutMs);
  const runId = created.run_id || created.id;
  if (!runId) {
    throw new Error('run_id_missing');
  }

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await sleep(250);
    const record = await fetchJson(`${RUNS_URL}/${runId}`, {}, Math.max(1000, deadline - Date.now()));
    if (record.status === 'succeeded') {
      const exported = await fetchJson(`${RUNS_URL}/${runId}/export`, {}, Math.max(1000, deadline - Date.now()));
      const item = (exported.items || []).find((candidate) => candidate.type === 'json' && candidate.json);
      const payload = item ? item.json : {};
      if (payload.success === false || payload.error) {
        throw new Error(payload.error?.message || payload.message || 'plugin_call_failed');
      }
      return payload.data || {};
    }
    if (['failed', 'canceled', 'timeout'].includes(record.status)) {
      throw new Error(record.error?.message || record.message || record.status);
    }
  }
  throw new Error('plugin_call_timeout');
}

function renderList(node, values, emptyText, className) {
  node.replaceChildren();
  const items = Array.isArray(values) ? values.filter((value) => String(value || '').trim()) : [];
  if (!items.length) {
    const empty = document.createElement('span');
    empty.className = 'empty-text';
    empty.textContent = emptyText;
    node.appendChild(empty);
    return;
  }
  items.forEach((value) => {
    const item = document.createElement('span');
    item.className = className;
    item.textContent = String(value);
    node.appendChild(item);
  });
}

function renderHand(tiles) {
  handTiles.replaceChildren();
  const values = Array.isArray(tiles) ? tiles.filter(Boolean) : [];
  handCount.textContent = `${values.length} 张`;
  if (!values.length) {
    const empty = document.createElement('span');
    empty.className = 'empty-text';
    empty.textContent = '暂无手牌';
    handTiles.appendChild(empty);
    return;
  }
  values.forEach((tile) => {
    const node = document.createElement('span');
    node.className = `tile tile-${String(tile).slice(-1)}`;
    node.textContent = String(tile);
    handTiles.appendChild(node);
  });
}

function renderDashboard(data = {}) {
  const state = data.round_state || data.coach_state || data || {};
  const decision = data.last_decision || data;
  const detail = decision.detail || state.opening_plan || '';
  const live = data.live || {};

  mainPlan.textContent = compact(state.current_plan || state.opening_plan || decision.suggestion, '等待手牌');
  planDetail.textContent = compact(detail, '还没有稳定手牌输入');
  biasValue.textContent = compact(state.attack_defense_bias, 'neutral');
  lastReason.textContent = compact(state.last_update_reason || decision.decision_type, '-');
  confidenceValue.textContent = percent(state.last_hand_confidence);
  updateCount.textContent = `${Number(state.update_count || 0)} updates`;
  decisionType.textContent = compact(decision.decision_type, 'observe');
  renderList(targetList, state.target_shapes, '暂无目标形状', 'tag');
  renderList(cautionList, state.caution_points, '暂无风险点', 'note');
  renderHand(state.last_hand_tiles || decision.hand_tiles || []);
  decisionOutput.textContent = JSON.stringify(decision && Object.keys(decision).length ? decision : state, null, 2);
  renderLive(live);
}

async function refreshStatus() {
  setStatus('刷新中');
  const data = await callPlugin('mahjong_coach_status', {}, 15000);
  renderDashboard(data);
  setStatus('ready');
}

function renderLive(live = {}) {
  const running = Boolean(live.running);
  liveState.textContent = compact(live.status, 'stopped');
  liveState.classList.toggle('is-running', running);
  liveFrame.textContent = `${Number(live.frame_index || 0)} frames`;
  liveWindow.textContent = compact(live.last_window_title || live.last_binding?.window_title, '未绑定窗口');
  liveError.textContent = compact(live.last_error || live.last_capture_source || live.last_frame_path, '-');
  startLiveBtn.disabled = running;
  stopLiveBtn.disabled = !running;
  scheduleAutoRefresh(running);
}

function scheduleAutoRefresh(running) {
  if (!running) {
    if (autoRefreshTimer) {
      window.clearInterval(autoRefreshTimer);
      autoRefreshTimer = 0;
    }
    return;
  }
  if (autoRefreshTimer) {
    return;
  }
  autoRefreshTimer = window.setInterval(() => {
    refreshStatus().catch((error) => {
      setStatus(error instanceof Error ? error.message : String(error));
    });
  }, 1200);
}

function keywordValues() {
  return String(keywordsInput.value || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

async function resetRound() {
  setStatus('重置中');
  const data = await callPlugin('mahjong_coach_reset_round', { round_id: `round-${Date.now()}` }, 15000);
  renderDashboard(data.round_state || data);
  setStatus('ready');
}

async function analyzeFrame() {
  setStatus('分析中');
  const observedButtons = String(buttonsInput.value || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  const data = await callPlugin('mahjong_coach_analyze_frame', {
    image_path: imagePathInput.value.trim(),
    observed_buttons: observedButtons,
    self_turn_index: Number(turnInput.value || 0),
    force_checkpoint: Boolean(forceCheckpointInput.checked),
  }, 30000);
  renderDashboard(data);
  setStatus(data.summary || 'ready');
}

async function startLive() {
  setStatus('启动实战观察');
  const data = await callPlugin('mahjong_coach_start_live', {
    keywords: keywordValues(),
    interval_ms: Number(intervalInput.value || 1200),
    overlay: Boolean(overlayInput.checked),
  }, 15000);
  renderLive(data.live || {});
  await refreshStatus();
}

async function stopLive() {
  setStatus('停止实战观察');
  const data = await callPlugin('mahjong_coach_stop_live', {}, 15000);
  renderLive(data.live || {});
  await refreshStatus();
}

function bind(button, handler) {
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      await handler();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      button.disabled = false;
    }
  });
}

bind(refreshBtn, refreshStatus);
bind(resetBtn, resetRound);
bind(analyzeBtn, analyzeFrame);
bind(startLiveBtn, startLive);
bind(stopLiveBtn, stopLive);

refreshStatus().catch((error) => {
  setStatus(error instanceof Error ? error.message : String(error));
});
