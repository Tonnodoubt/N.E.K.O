# Wave 5: v1.2 Orchestrator Split Blueprint

> Single Source of Truth. 本文件是 Wave 5 的拆分蓝图；执行代码迁移前，先以这里的语义边界和迁移顺序为准。
>
> 当前基线（2026-05-03）：`plugin/plugins/mahjong_companion/orchestrator.py` 为 3378 行。`CODE_REVIEW_v1.2.md` 中的 3211 行是早一轮快照。

---

## 1. 目标与边界

### 1.1 目标

Wave 5 解决剩余三个大件：

- M4：把 `orchestrator.py` 中明显可独立的 4 块迁出，降低单文件体积和认知负担。
- M5：明确并修正悬浮窗手动刷新建议时的 review artifact 持久化语义。
- M7：明确悬浮窗刷新与正常 live cycle / fast-path 的调度边界，避免“跳过一轮 live cycle”继续停留在隐式行为。

### 1.2 非目标

- 不重写麻将决策、感知、复盘生成算法。
- 不改变插件对外 API 名称和前端按钮语义。
- 不把所有 orchestrator 私有方法一次性搬空；本轮只迁出已具备清晰边界的 4 块。
- 不在 fast-path 中直接落 review artifact，除非下面的语义边界被重新审定。

---

## 2. M5 / M7 语义边界

### 2.1 定义

**Reviewable decision**：来自真实截图、能代表一个可复盘局面、且调用方明确声明应参与复盘的决策。它可以写入：

- `session_cache/review_candidates.json`
- `session_cache/game_private_memory.json`
- `session_cache/memory_bridge_queue.json`（满足优先级与配置时）

**Ephemeral decision**：用于即时 UI、播报或 speculative fast-path 的临时建议。它可以更新状态和悬浮窗，但不直接写 review artifact。

### 2.2 各入口的持久化语义

| 入口 | 当前意图 | review artifact | 调度语义 |
| --- | --- | --- | --- |
| 正常 live cycle `_run_live_cycle_locked` | 自动观察真实对局 | 是；当前已传 `persist_review_artifacts=True` | 每个 active tick 最多一次完整 capture/perception/decision/narration |
| 悬浮窗 `refresh_advice` | 用户主动要求刷新当前建议 | 是；应显式传 `persist_review_artifacts=True` | 视为一轮手动完整 pipeline，并消费当前 runtime tick |
| fast-path `_maybe_emit_fast_preturn_advice_locked` | 摸牌前计划命中后的低延迟临时建议 | 否；保持 `persist_review_artifacts=False` | 只更新状态与可播报事件；后续正常 live cycle 再写可复盘数据 |
| `run_companion_pipeline(frame_path=...)` 且 session 未运行 | 离线调试/手动分析 | 否；保留现有测试语义 | 不参与实时复盘队列 |
| runtime `explain_current_hand` | 当前手牌解释命令 | 否；当前已传 `persist_review_artifacts=False` | 不污染复盘数据 |

### 2.3 M7 处理方向

悬浮窗刷新后“跳过同一个 tick 的正常 live cycle”可以成立，但必须从隐式 early return 变成显式结果：

- `refresh_advice` 是一轮手动完整 pipeline，不需要同 tick 再跑一次自动 live cycle。
- runtime cycle 应返回或记录一个类似 `RuntimeCycleOutcome(consumed_live_tick=True, reason="overlay_refresh")` 的结果。
- 测试应从“不会调用 live cycle 的偶然断言”改名为“manual overlay refresh consumes current live tick by design”。

### 2.4 M5 代码落点

需要先扩展 `_run_companion_pipeline_locked` 的参数：

```python
def _run_companion_pipeline_locked(
    self,
    frame_path: str,
    *,
    capture: bool,
    dispatch: bool,
    force_reply: bool,
    persist_review_artifacts: bool | None = None,
) -> dict[str, Any]:
    ...
```

然后：

- 悬浮窗 `refresh_advice` 传 `persist_review_artifacts=True`。
- 公开 `run_companion_pipeline(...)` 保持默认 `None`，继续走 `_should_persist_review_artifacts` 的现有 guard。
- fast-path 和 `explain_current_hand` 继续显式传 `False`。

---

## 3. 拆分模块

### 3.1 Status Snapshot Builder

建议新文件：

- `plugin/plugins/mahjong_companion/status_snapshot.py`

迁出内容：

- `_derive_report_status`
- `_build_status_snapshot`
- `_current_screen_overlays` 可一起迁出为纯 helper

建议接口：

```python
@dataclass
class StatusSnapshotContext:
    state: SessionState
    selected_window_title: str
    overlay_visible: bool
    preturn_discard_plan: Any | None
    last_preturn_plan_meta: dict[str, Any]
    last_fast_advice_frame_path: str

def build_status_snapshot(ctx: StatusSnapshotContext) -> dict[str, Any]:
    ...
```

迁出原则：

- 先做纯函数迁移，`_emit_status` 仍留在 orchestrator。
- 迁出后 `orchestrator.get_status()` 只调用 builder。
- 不在 builder 里调用 plugin、overlay 或文件 I/O。

### 3.2 Runtime Mailbox Scheduler

建议新文件：

- `plugin/plugins/mahjong_companion/runtime/scheduler.py`

迁出内容：

- `_run_runtime_cycle_locked`
- `_process_runtime_command_locked`
- `_handle_runtime_command_locked`
- `_build_runtime_command_registry`
- `_queue_runtime_outbound_event_locked`
- `_flush_runtime_outbox_locked`
- `_sync_runtime_mailbox_state_locked`（可最后迁，避免第一步牵太多 state）

建议接口：

```python
@dataclass
class RuntimeCycleOutcome:
    mode: str
    processed_command: bool = False
    processed_overlay_command: bool = False
    consumed_live_tick: bool = False
    flushed_outbound: int = 0

class RuntimeScheduler:
    def run_cycle_locked(self) -> RuntimeCycleOutcome:
        ...
```

迁出原则：

- scheduler 可以接收一组 callbacks：`run_live_cycle`, `emit_status`, `handle_runtime_command`, `handle_overlay_commands`。
- 不让 scheduler 直接 import `SessionOrchestrator`，避免循环依赖。
- 第一步允许命令 handler 仍留在 orchestrator；第二步再把 registry 初始化和 dispatch 搬进去。

### 3.3 Overlay Controller

建议新文件：

- `plugin/plugins/mahjong_companion/overlay_controller.py`

迁出内容：

- `_overlay_enabled`
- `_overlay_auto_show_on_bind`
- `_show_overlay_locked`
- `_hide_overlay_locked`
- `_process_overlay_commands_locked`
- `_handle_overlay_command_locked`
- `_dispatch_overlay_narration_locked`
- `_get_active_overlay_poll_interval_ms`
- `_get_overlay_max_age_ms`
- `_get_overlay_region_change_threshold`
- `_maybe_clear_expired_screen_overlays_locked`
- `_clear_screen_overlays_locked`
- `_screen_overlay_region_changed`

迁出原则：

- `overlay.py` 继续只做 Tk UI；`overlay_controller.py` 管 orchestrator 侧生命周期和命令。
- controller 不应直接知道完整 pipeline 细节，通过 callbacks 执行：
  - `refresh_advice`
  - `dispatch_current_narration`
  - `emit_status`
  - `current_screen_overlays`
- M5 的 `persist_review_artifacts=True` 在 `refresh_advice` callback 边界显式表达。

### 3.4 Fast-Path Service

建议新文件：

- `plugin/plugins/mahjong_companion/fast_path.py`

迁出内容：

- `_maybe_emit_fast_preturn_advice_locked`
- `_fast_path_base_state`
- `_snapshot_fast_path_state`
- `_restore_fast_path_state`
- `_PERCEPTION_FIELDS_FOR_FAST_PATH`
- `_DECISION_FIELDS_FOR_FAST_PATH`
- `_NARRATION_FIELDS_FOR_FAST_PATH`

建议接口：

```python
@dataclass
class FastPathStateSnapshot:
    fields: dict[str, Any]
    fast_poll_until: float

class PreturnFastPathService:
    def maybe_emit_locked(self, frame_path: Path) -> bool:
        ...
```

迁出原则：

- fast-path 仍在 orchestrator lock 内更新状态，但慢 dispatch 继续走 `_pending_fast_dispatch_events` 锁外派发。
- fast-path 继续显式 `persist_review_artifacts=False`。
- state snapshot/restore 使用 dataclass 或 typed helper，避免新增裸 dict 契约。
- 优先把 `_fast_path_base_state` 抽成纯函数并单测，再迁 orchestration。

---

## 4. 迁移顺序

### Step 0：文档冻结

交付：

- 本蓝图。
- 若需要更细，可追加 `plans/design-v1.2-review-artifact-boundary.md`，专门记录 M5/M7 的最终语义决定。

验收：

- 不改 runtime 代码。

### Step 1：迁出 status snapshot

原因：纯度最高，最容易做“搬家不改行为”。

交付：

- 新增 `status_snapshot.py`。
- `orchestrator.py` 删除 `_derive_report_status` / `_build_status_snapshot` 主体。
- 新增 focused unit tests，比较关键字段：`status`、`runtime_status`、`screen_overlays`、runtime mailbox counters、review/coaching/action fields。

建议测试：

```bash
python -m pytest plugin/tests/unit/sdk/plugin/test_mahjong_companion_layering.py plugin/tests/unit/sdk/plugin/test_mahjong_companion_v9_memory_and_coaching.py
```

### Step 2：迁出 runtime mailbox scheduler

原因：现有 `runtime/` package 已存在，调度层自然落在同目录。

交付：

- 新增 `runtime/scheduler.py`。
- 保持 `send_runtime_message` / `get_runtime_mailbox` 外观不变。
- runtime cycle 返回显式 outcome，为 Step 3 的 M7 语义铺路。

建议测试：

```bash
python -m pytest plugin/tests/unit/sdk/plugin/test_mahjong_companion_runtime_mailbox.py plugin/tests/unit/sdk/plugin/test_mahjong_companion_standby_mode.py
```

### Step 3：迁出 overlay controller，并落 M5/M7 语义

交付：

- 新增 `overlay_controller.py`。
- `refresh_advice` 调用 pipeline 时传 `persist_review_artifacts=True`。
- runtime cycle outcome 中标记 `consumed_live_tick=True`。
- 更新/新增测试覆盖：
  - 手动 overlay refresh 会写 review artifact。
  - 手动 overlay refresh 消费当前 tick 是显式设计。
  - 非运行态手动 pipeline 仍不写 review artifact。

建议测试：

```bash
python -m pytest plugin/tests/unit/sdk/plugin/test_mahjong_companion_v4_flow.py plugin/tests/unit/sdk/plugin/test_mahjong_companion_static_ui.py
```

### Step 4：迁出 fast-path service

交付：

- 新增 `fast_path.py`。
- fast-path snapshot/restore 从裸 dict 过渡到 typed helper。
- 保持 fast-path 不写 review artifacts。
- 保持慢 dispatch 锁外执行。

建议测试：

```bash
python -m pytest plugin/tests/unit/sdk/plugin/test_mahjong_companion_orchestrator_with_fakes.py plugin/tests/unit/sdk/plugin/test_mahjong_companion_v11_tile_efficiency_v03.py
```

### Step 5：收口与行数验收

交付：

- `orchestrator.py` 只保留 public API、session lifecycle、pipeline glue 和少量 state transitions。
- 删除搬迁后失效 imports。
- 更新 `CODE_REVIEW_v1.2.md` 或新增 closeout note，标记 M4/M5/M7 的最终状态。

验收建议：

```bash
wc -l plugin/plugins/mahjong_companion/orchestrator.py
python -m pytest plugin/tests/unit/sdk/plugin/test_mahjong_companion_*.py
```

---

## 5. 回归测试矩阵

最小分阶段矩阵：

- Status：`test_mahjong_companion_layering.py`、`test_mahjong_companion_v9_memory_and_coaching.py`
- Runtime：`test_mahjong_companion_runtime_mailbox.py`、`test_mahjong_companion_standby_mode.py`
- Overlay：`test_mahjong_companion_v4_flow.py`、`test_mahjong_companion_overlay.py`、`test_mahjong_companion_static_ui.py`
- Fast-path：`test_mahjong_companion_orchestrator_with_fakes.py`、`test_mahjong_companion_v11_tile_efficiency_v03.py`
- Review persistence：新增/更新 `test_mahjong_companion_v4_flow.py` 中 overlay refresh 与 manual pipeline 边界测试

合并前矩阵：

```bash
python -m pytest plugin/tests/unit/sdk/plugin/test_mahjong_companion_*.py
```

---

## 6. 风险与护栏

- **锁语义**：所有被命名为 `_locked` 的迁出方法仍必须只在 orchestrator lock 内调用；新模块不要自己创建第二把 lock。
- **循环 import**：新模块可以 import `SessionState` / contracts / runtime types，但不要 import `SessionOrchestrator`。
- **review artifact 去重**：overlay refresh 改为持久化后，依赖 `append_review_candidate` 的 dedupe window 防止重复写爆；新增测试应覆盖同一帧不会重复膨胀。
- **fast-path speculative state**：fast-path 不落 review artifact，避免将“基于上轮 hand + 新摸牌识别”的临时状态写成复盘事实。
- **测试 monkeypatch**：现有测试直接 monkeypatch orchestrator 私有方法；迁出时要同步测试锚点，优先 monkeypatch callback 或新 service 方法。
- **前端状态字段**：status snapshot 的 key 不应删改；如果要改名，必须同步 `static/main.js` 和 static UI tests。

---

## 7. LOW 项清理建议

这 3 个 LOW 可以在 Wave 5 前后独立清理，不应和大拆分混在同一个 commit：

- N-L2：`data_lifecycle._safe_package_name("..zip", timestamp)` 不应产出 `..zip`。建议剥离全点号 stem，空值回退默认包名。测试落在 `test_data_lifecycle.py`。
- N-L4：`overlay.py` 的 `user_moved` 裸 dict 改为 `dataclass` 或 `SimpleNamespace`。测试落在 overlay focused tests；这是可读性改动。
- N-L5：`diagnostics._check_recent_status()["ok"]` 与 `_health_from_issues` 对齐，`error` 和 `warning` 都应让 ok 为 false。测试落在 diagnostics focused test 或新增小单测。

建议顺序：先 LOW 小清理，再执行 M4 大拆分；这样大拆分 diff 更干净。

---

## 8. 执行前待审决定

执行代码前建议确认两条语义：

1. 悬浮窗 `refresh_advice` 是否接受“写 review artifact，并消费当前 runtime tick”的设计。
2. fast-path 是否继续保持“只更新即时建议，不写 review artifact，等正常 live cycle 落复盘”的设计。

若这两条确认，Wave 5 就可以按 Step 1 -> Step 5 顺序拆。

---

## 9. 执行 Closeout（2026-05-03）

Wave 5 已执行，并在原 4 块之外继续抽出低风险边界：

- `status_snapshot.py`
- `fast_path.py`
- `runtime/scheduler.py`
- `overlay_controller.py`
- `config_accessors.py`
- `pipeline_execution.py`
- `action_executor.py`
- `frame_resources.py`
- `review_workflow.py`
- `state_transitions.py`

最终 `orchestrator.py` 从 Wave 5 前 **3378 行**降至 **885 行**。剩余内容主要是 public API、session lifecycle、窗口绑定、status/cache 写出和少量语义胶水。

验证结果：

```bash
uv run pytest plugin/tests/unit/sdk/plugin/test_mahjong_companion_*.py plugin/tests/unit/sdk/plugin/test_data_lifecycle.py plugin/tests/unit/sdk/plugin/test_runtime_diagnostics.py
# 167 passed

git diff --check
# pass
```
