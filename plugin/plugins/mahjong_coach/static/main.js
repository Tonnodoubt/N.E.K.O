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
const roundWindInput = document.getElementById('roundWindInput');
const seatWindInput = document.getElementById('seatWindInput');
const doraTilesInput = document.getElementById('doraTilesInput');
const analysisRoundWindInput = document.getElementById('analysisRoundWindInput');
const analysisSeatWindInput = document.getElementById('analysisSeatWindInput');
const analysisDoraTilesInput = document.getElementById('analysisDoraTilesInput');
const analysisSource = document.getElementById('analysisSource');
const mainPlan = document.getElementById('mainPlan');
const planDetail = document.getElementById('planDetail');
const aiSource = document.getElementById('aiSource');
const aiPlan = document.getElementById('aiPlan');
const aiPlanDetail = document.getElementById('aiPlanDetail');
const biasValue = document.getElementById('biasValue');
const lastReason = document.getElementById('lastReason');
const confidenceValue = document.getElementById('confidenceValue');
const updateCount = document.getElementById('updateCount');
const targetList = document.getElementById('targetList');
const cautionList = document.getElementById('cautionList');
const handTiles = document.getElementById('handTiles');
const handCount = document.getElementById('handCount');
const riverTiles = document.getElementById('riverTiles');
const riverCount = document.getElementById('riverCount');
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

function cleanStrategyText(value) {
  let text = String(value || '').replace(/\s+/g, ' ').trim();
  [
    ['主线：', ''],
    ['保留：', ''],
    ['对子：', ''],
    ['路线选择：', ''],
    ['筒子占比很高', '筒子多'],
    ['万子占比很高', '万子多'],
    ['索子占比很高', '索子多'],
    ['保留同色块', '保留同色'],
    ['同色块', '同色'],
    ['做搭子', '找顺子'],
    ['先清', '先打'],
    ['不硬染', '别强做清一色'],
    ['吃碰杠', '鸣牌'],
    ['进听', '听牌'],
  ].forEach(([from, to]) => {
    text = text.replaceAll(from, to);
  });
  return text.trim();
}

function firstSentence(value) {
  const text = cleanStrategyText(value);
  const cutAt = ['，', '；', ';', '。'].map((mark) => text.indexOf(mark)).filter((index) => index >= 0);
  return cutAt.length ? text.slice(0, Math.min(...cutAt)).trim() : text;
}

function firstPrefixedValue(values, prefix) {
  const items = Array.isArray(values) ? values : [];
  const match = items.find((item) => String(item || '').startsWith(prefix));
  return match ? String(match).slice(prefix.length).trim() : '';
}

function listValues(value) {
  return Array.isArray(value) ? value.filter((item) => String(item || '').trim()) : [];
}

function extractAfter(value, keywords, stopMarkers) {
  const text = String(value || '');
  let start = -1;
  let keywordLength = 0;
  keywords.forEach((keyword) => {
    const index = text.indexOf(keyword);
    if (index >= 0 && (start < 0 || index < start)) {
      start = index;
      keywordLength = keyword.length;
    }
  });
  if (start < 0) {
    return '';
  }
  let tail = text.slice(start + keywordLength).replace(/^[\s：:]+/, '');
  const stopAt = stopMarkers.map((mark) => tail.indexOf(mark)).filter((index) => index >= 0);
  if (stopAt.length) {
    tail = tail.slice(0, Math.min(...stopAt));
  }
  return tail.trim();
}

function briefItems(value, limit = 4) {
  const text = cleanStrategyText(value).replace(/^[，、\s]+|[，、\s]+$/g, '');
  if (!text) {
    return '';
  }
  const items = text.split(/[、，,\s]+/).map((item) => item.trim()).filter(Boolean);
  if (items.length <= limit) {
    return items.length ? items.join('、') : text;
  }
  return `${items.slice(0, limit).join('、')}等${items.length}张`;
}

function strategyHeadline(plan, targets, fallback) {
  const targetItems = Array.isArray(targets) ? targets : [];
  const mainTarget = targetItems.find((item) => String(item || '').startsWith('主线：'));
  return firstSentence(mainTarget || plan) || fallback;
}

function strategyBrief(plan, detail, targets, cautions, fallback) {
  const targetItems = Array.isArray(targets) ? targets : [];
  const cautionItems = Array.isArray(cautions) ? cautions : [];
  const keep = briefItems(firstPrefixedValue(targetItems, '保留：') || extractAfter(plan, ['保留'], ['，先', '；', '。']), 4);
  const discard = briefItems(
    firstPrefixedValue(cautionItems, '优先清理：') || extractAfter(`${plan}。${detail}`, ['先打', '先清', '打：'], ['，', '；', '。']),
    4,
  );
  const lines = [];
  if (keep) {
    lines.push(`留：${keep}`);
  }
  if (discard) {
    lines.push(`打：${discard}`);
  }
  return lines.join('\n') || firstSentence(detail) || fallback;
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

function renderRiver(piles = {}) {
  riverTiles.replaceChildren();
  const entries = Object.entries(piles || {});
  const total = entries.reduce((count, [, items]) => count + (Array.isArray(items) ? items.length : 0), 0);
  riverCount.textContent = `${total} 张`;
  if (!total) {
    const empty = document.createElement('span');
    empty.className = 'empty-text';
    empty.textContent = '暂无牌河';
    riverTiles.appendChild(empty);
    return;
  }
  entries.forEach(([player, items]) => {
    const row = document.createElement('div');
    row.className = 'river-row';
    const label = document.createElement('span');
    label.className = 'river-player';
    label.textContent = player;
    row.appendChild(label);
    (Array.isArray(items) ? items : []).forEach((item) => {
      const tile = String(item?.tile || '').trim();
      if (!tile) {
        return;
      }
      const node = document.createElement('span');
      node.className = `tile tile-small tile-${tile.slice(-1)}`;
      node.textContent = tile;
      row.appendChild(node);
    });
    riverTiles.appendChild(row);
  });
}

function renderDashboard(data = {}) {
  const state = data.round_state || data.coach_state || data || {};
  const decision = data.last_decision || data;
  const detail = decision.detail || state.opening_plan || '';
  const live = data.live || {};
  const source = decision.analysis_source || decision.engine_meta?.analysis_source || state.plan_source || 'heuristic';
  const localPlan = state.local_direction || state.local_plan || (source === 'llm' ? state.opening_plan : state.current_plan) || decision.suggestion;
  const localDetail = state.local_detail || (source === 'llm' ? '' : detail);
  const localTargets = listValues(state.local_targets).length ? listValues(state.local_targets) : listValues(state.target_shapes);
  const localCautions = listValues(state.local_cautions).length ? listValues(state.local_cautions) : listValues(state.caution_points);
  const aiPlanText = state.ai_direction || state.ai_plan || (source === 'llm' ? state.current_plan || decision.suggestion : '');
  const aiDetailText = state.ai_detail || (source === 'llm' ? detail : '');
  const aiTargets = listValues(state.ai_targets);
  const aiCautions = listValues(state.ai_cautions);
  const llmStatus = String(state.llm_status || '').toLowerCase();
  const llmError = String(state.llm_error || '').trim();
  const aiPending = Number(data.llm_pending || 0) > 0 || llmStatus === 'pending';
  const aiFailed = ['timeout', 'error', 'empty'].includes(llmStatus);

  mainPlan.textContent = strategyHeadline(localPlan, localTargets, '等待手牌');
  planDetail.textContent = strategyBrief(localPlan, localDetail, localTargets, localCautions, '还没有稳定手牌输入');
  analysisSource.textContent = 'Heuristic';
  analysisSource.classList.remove('is-ai');
  aiPlan.textContent = aiPlanText
    ? strategyHeadline(aiPlanText, aiTargets, llmStatus === 'ready_previous_hand' ? 'AI参考' : 'AI策略')
    : compact('', aiPending ? 'AI 思考中' : (aiFailed ? 'AI 未返回' : '等待 AI'));
  aiPlanDetail.textContent = compact(
    aiPlanText ? strategyBrief(aiPlanText, aiDetailText, aiTargets, aiCautions, 'AI 已更新策略') : '',
    aiPending ? '模型请求已发出，返回后会更新这里' : (aiFailed ? llmError || '模型没有返回可用策略' : 'AI 只在开局/阶段更新后异步生成'),
  );
  aiSource.textContent = llmStatus === 'ready_previous_hand' ? 'AI参考' : 'AI';
  aiSource.classList.toggle('is-ai', Boolean(aiPlanText));
  const style = state.play_style || 'riichi';
  const styleLabel = style === 'fast' ? '快攻' : '立直';
  biasValue.textContent = `${styleLabel} / ${compact(state.attack_defense_bias, 'neutral')}`;
  lastReason.textContent = compact(state.last_update_reason || decision.decision_type, '-');
  confidenceValue.textContent = percent(state.last_hand_confidence);
  updateCount.textContent = `${Number(state.update_count || 0)} updates`;
  decisionType.textContent = compact(decision.decision_type, 'observe');
  renderList(targetList, state.target_shapes, '暂无目标形状', 'tag');
  renderList(cautionList, state.caution_points, '暂无风险点', 'note');
  renderHand(state.last_hand_tiles || decision.hand_tiles || []);
  renderRiver(state.last_discard_piles || decision.perception?.river?.discard_piles || {});
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

function tileValues(text) {
  return String(text || '')
    .split(/[,，、\s]+/)
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
    round_wind: analysisRoundWindInput.value.trim(),
    seat_wind: analysisSeatWindInput.value.trim(),
    dora_tiles: tileValues(analysisDoraTilesInput.value),
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
    round_wind: roundWindInput.value.trim(),
    seat_wind: seatWindInput.value.trim(),
    dora_tiles: tileValues(doraTilesInput.value),
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
