# 雀魂陪伴插件 v1.1 执行计划

日期：2026-05-05
前置版本：`v1.0.1`
目标发布版本：`v1.1.0`
关联文档：`blueprint-v1.0-closeout.md`、`plans/artifacts/test-status-v1.0.1.md`、`plans/artifacts/eval-report-v1.0.1-skip-smoke.json`

## 1. 目标与范围

v1.1 的核心目标：把 v1.0.1 在 release gate 上暴露出来的回归全部修掉，把 hand_recognition 从 v1.0.1 的"刚刚开门"提升到能撑住 0.50+ 全手准确率，并把 v1.0.1 推迟的 smoke runner 清理掉。

v1.1 对外承诺：

1. `evaluate_v10_release` 不带 `--skip-smoke` 直接 `ok=True`。
2. `hand_recognition.full_hand_accuracy` 阈值回到 ≥0.50，tile_accuracy 阈值回到 ≥0.92。
3. `pytest plugin/tests/unit/sdk/plugin/test_mahjong_companion*.py` 在 Python 3.11 + 全套依赖下零 fail / 零 collect error，stale skip 数 ≤ 5（仅保留语义未定的 ViT classifier_config 类）。
4. `check_v10_release --pretty` 返回 `ok=True`，数据生命周期和运行诊断入口恢复。
5. macOS / Linux 也能拿到 overlay 红框（子进程化 Tk）。
6. 4 家 toy 模拟器有 1000+ 局 × 5 seeds 的稳定基线，包含 oracle 头对头。

v1.1 不做（推到 v1.2 或之后）：

- 真正的麻将 AI（多巡 mcts、点棒/役/赤五综合估值）。
- 雀魂段位赛实战胜率统计（需要长期试用数据）。
- 多皮肤完整承诺。
- 自动代打或自动点击对局按钮（永久不做，advice-only 是产品边界）。

## 2. 工作分轨与验收

按 P0 → P1 → P2 → P3 排列；每条都给出文件清单、验收命令、预估改动量。

### Step 0（P0，已完成）：恢复 release hygiene 入口

**现状**：`check_v10_release --pretty` 仍返回 `ok=False`。`data_lifecycle.py` / `diagnostics.py` 的底层实现还在，但插件入口和 orchestrator wrapper 已经丢失：

- `get_data_lifecycle`
- `export_local_data`
- `clear_local_runtime_data`
- `clear_calibration_raw_data`
- `get_runtime_diagnostics`

**任务**：

1. 在 `orchestrator.py` 增加上述 5 个 async wrapper，调用 `data_lifecycle.py` 与 `diagnostics.py` 现有实现。
2. 在 `__init__.py` 恢复 5 个 `@plugin_entry`，统一返回 `Ok(...)` / `Err(...)`。
3. 取消或修复 `test_data_lifecycle.py` / `test_runtime_diagnostics.py` 中因此失败的用例。

**验收**：

- `python -m plugin.plugins.mahjong_companion.scripts.check_v10_release --pretty` 返回 `ok=True`。
- `pytest plugin/tests/unit/sdk/plugin/test_data_lifecycle.py plugin/tests/unit/sdk/plugin/test_runtime_diagnostics.py` 通过。

**改动量**：约 80-140 行代码 + 少量测试修正。

**完成记录（2026-05-05）**：已恢复 5 个 plugin entry / orchestrator wrapper；`check_v10_release --pretty` 返回 `ok=True`，data lifecycle 与 runtime diagnostics 单测通过。

### Step 1（P0，已完成）：修 v05.genbutsu_hint.recall 回归

**现状**：`evaluate_v10_release --skip-smoke` 在 v1.0.1 头上唯一硬失败。1.0 是 1/1 → v1.0.1 是 0/1。门禁阈值 ≥0.95。

**怀疑点**：180e101a 的策略重构改了 `decision/risk_estimator.py::estimate_defense_alerts` 或 `decision/generator.py` 里的 reason_codes 拼接，导致 `confirmed_genbutsu` 信号没进 alert 链。

**任务**：

1. `git log -p 180e101a -- plugin/plugins/mahjong_companion/decision/risk_estimator.py` 圈出策略重构对 risk_estimator 的具体改动。
2. 跑 `evaluate_v05 --max-details` 把 1 个 mismatch case 的 expected vs actual reason_codes / defense_alerts 对出来。
3. 修复点二选一：
   - 若回归路径成立 → 在 risk_estimator 里恢复 confirmed_genbutsu 信号写入 reason_codes
   - 若新策略主动拿掉了这个信号 → 升级 fixture 的 expected reason_codes，并在 `decision/risk_estimator.py` 里加注释说明语义变化
4. 加/扩一条单测覆盖"立直家 + 现物已知 → defense_alerts 第一项标 genbutsu"（现有 v11 tile efficiency 测试可复用，防止下次再回归）。

**验收**：

- `python -m plugin.plugins.mahjong_companion.scripts.evaluate_v05 --strict` 返回 `genbutsu_hint.recall = 1.0`。
- 新增或扩展 genbutsu alert 单测，至少 2 个 case（立直前 / 立直后）。

**改动量**：约 30-80 行代码 + 1 个新测试文件。

**完成记录（2026-05-05）**：实际回归点是评测脚本只查原始牌码（如 `9m`），而 alert 文本使用中文牌名（如 `九万`）；已改为同时匹配原始牌码和本地化牌名。`evaluate_v05 --strict --pretty` 返回 `genbutsu_hint.recall = 1.0`。

### Step 2（P0，已完成）：清理 smoke_test.py 与 v1.0.1 接口对齐

**现状**：`smoke_test.py` 仍有 5 处调用已删 orchestrator 方法（`execute_assist_action` / `generate_review_summary` / `sync_memory_bridge` / `list_assist_actions` / `get_action_log`）。导致：
- `evaluate_v10_release` 不带 `--skip-smoke` 直接 `AttributeError` 中断
- `test_mahjong_companion_smoke.py` 整体被 skip

**任务**：

1. 删除 `smoke_test.py` 中以下三段：
   - v5：`generate_review_summary` / `sync_memory_bridge` 的 review/memory 段
   - v6：`generate_review_summary` 的二次调用段
   - v7：`list_assist_actions` / `execute_assist_action` / `get_action_log` 的辅助动作段
   - v9：coaching topics 段
2. 保留并加固：
   - v1：runtime contract 规则（仍有效）
   - v4：感知 → 决策 → narration 主链
   - v8：tile efficiency hint
3. 把 smoke 报告 schema 从"5 段"改成"3 段"。同步更新 `plans/artifacts/eval-report-v1.0-release-gate.json` 期望（或重新生成）。
4. 取消 `test_mahjong_companion_smoke.py` 的 skip 标记，确保单测覆盖新 schema。

**验收**：

- `python -m plugin.plugins.mahjong_companion.scripts.evaluate_v10_release --pretty` 不带 skip-smoke 返回 `ok=True`。
- `pytest plugin/tests/unit/sdk/plugin/test_mahjong_companion_smoke.py` 通过。

**改动量**：smoke_test.py 大约 -150/+30 行；schema 期望同步更新 1-2 处。

**完成记录（2026-05-05）**：smoke runner 已收敛为 3 段有效检查（runtime mode/dispatch、真实样本主链、tile efficiency hint），skip 已取消，`eval-report-v1.0-release-gate.json` 已按新 schema 重新生成。

### Step 3（P1，已完成）：hand_recognition 准确率拉回 v1.1 标准

**现状**（v1.0.1 baseline，21 case）：

- tile_accuracy = 88.24%（255/289）
- full_hand_accuracy = 23.81%（5/21）
- 阈值放宽到 0.85 / 0.20

**目标**（v1.1）：

- tile_accuracy ≥ 0.92（≥266/289）
- full_hand_accuracy ≥ 0.50（≥11/21）

**已知混淆分布**（按出现频次排序）：

| 混淆方向 | 次数 | 推断原因 |
|---|---|---|
| `5p → 6p` | 12 | 模板像素差小，主导错误 |
| `6p → 7p` | 11 | 同上，连锁错位 |
| `7s → 9s` | 5 | 索子上方圈数模板边界模糊 |
| `9s → 6s` | 3 | 9s 旋转/对称错认 |
| `4p → 5p` | 2 | 红五归一前的边界 |
| `R5p → 5p` | 1 | 红五未识别为 R 标 |

**任务（任选 / 组合）**：

1. **样本扩充路径**：再加 30 张含 5p/6p/7p/7s/9s 的 1080p 真实 hand 标注，跑 `scripts.label_calibration` 重训练 1080p profile，更新 `data/calibration/profiles/majsoul-pc-manual-2026.05-1920x1080.json`。
2. **ViT fallback 路径**：把 `perception/tile_parser.py::_from_template_profile` 改成"模板余弦相似度 < 0.92 → 调 ViT 分类器二次确认"，让 `vit_tile_classifier.py` 真正参与 hand 识别（目前只在 discard 用）。
3. **模板分离单测**：`tests/unit/sdk/plugin/test_tile_template_separation.py`，对 `(5p, 6p)`、`(6p, 7p)`、`(7s, 9s)` 三对模板断言"自匹配 - 跨匹配 ≥ 0.10"。

**验收**：

- `THRESHOLDS` 在 `evaluate_v03.py` 改回 `tile_accuracy: 0.92`、`full_hand_accuracy: 0.50`。
- `evaluate_v10_release` 仍 `ok=True`。
- 在 closeout 里报告每一类混淆从 v1.0.1 → v1.1 的变化。

**改动量**：取决于路径选择。样本路径 ~2-4 小时人工标注；ViT 路径 ~150 行代码 + 接入测试。

**完成记录（2026-05-05）**：实际根因是 `vit-discard` profile 以更高 confidence 抢过了 manual hand profile；已改为同分辨率 profile 运行时按领域合成，hand 取 manual 模板、discard 取 vit-discard 模板。`hand_recognition.tile_accuracy/full_hand_accuracy = 1.0/1.0`，`evaluate_v03.py` hand 阈值已恢复到 `0.92 / 0.50`。

### Step 4（P1，已完成）：升级 v0.3 决策 fixture 到新策略

**现状**：`decision_top1.match_rate` 1.0 → 0.667（4/6）。两个不通过的 fixture 都还在断言 180e101a 之前的 strategy 排序。具体哪两个 case，要跑：

```bash
python -m plugin.plugins.mahjong_companion.scripts.evaluate_v03 --details --max-details 20
```

**任务**：

1. 列出 2 个 mismatch fixture 的 expected_top1 vs actual_top1。
2. 评估 actual 是不是合理（dead-honor 5z 排第一通常合理）：
   - 合理 → 更新 fixture 的 `expected_top1` 字段，加注释说明 v1.0.1 strategy 把 dead-honor 提前
   - 不合理 → 反推 strategy 哪段过于激进，回到 Step 1 的修法
3. 同步把 `test_mahjong_companion_v11_tile_efficiency_v03.py::test_complete_deck_state_enables_real_ukeire_for_candidate_ordering` 的 skip 标记拿掉，更新 expected `1m → 5z`，附 comment 说明语义。

**验收**：

- `decision_top1.match_rate ≥ 0.95`（5-6/6）。
- v11 那条单测在 v1.1 重新启用。

**改动量**：~20 行 fixture 改动 + 1 行测试改动。

**完成记录（2026-05-05）**：2 个 mismatch 已确认是新策略合理变化：`complete-deck-ukeire` 改为死字牌 `5z` 优先，`riichi-genbutsu-present` 改为确认现物 `9m` 优先。`decision_top1.match_rate = 1.0`，门禁阈值已恢复到 `0.95`；v11 tile efficiency 单测 skip 已移除。

### Step 5（P2，已完成）：Toy 模拟器扩到 1000 局 + oracle 头对头

**现状**（v1.0.1，sandbox 受限）：

- companion vs random：120 局 × 2 seeds，平均 50.4%
- companion vs shanten：120 局 × 2 seeds，平均 28.3%
- companion vs oracle：未跑（沙箱超时）
- shanten vs oracle / oracle vs oracle：未跑

**任务**：

1. 本地 Python 3.11 环境跑：
   ```bash
   python scripts/mahjong_simulator_baseline.py --games 500 --seeds 5 --pretty \
       --output plugin/plugins/mahjong_companion/plans/artifacts/simulator-baseline-v1.1.json
   ```
2. 在 `mahjong_simulator_baseline.py` 里补：
   - `companion vs oracle` matchup（看离最优策略多远）
   - 新指标 `risk_adjusted_score = win_rate - deal_in_rate * 1.5`
   - 95% 置信区间（基于 5 seeds 的 stdev）
3. 把 win_rate 95% CI 的下界写进 `closeout` 作为 v1.1 → v1.2 回归基线。

**验收**：

- 新基线 JSON 至少 12 个 matchup（4 hero × 3 opponent，opp ∈ {random, shanten, oracle}）。
- `companion vs shanten` 95% CI 下界 ≥ 26%。
- `companion vs oracle` 至少 ≥ 18%（不被绝对吊打）。

**改动量**：simulator harness +50 行，加 stdev/CI 计算；产物文件由本地跑。

**完成记录（2026-05-05）**：`scripts/mahjong_simulator_baseline.py` 已切到 v1.1 baseline 默认参数（500 games × 5 seeds），补 `risk_adjusted_score`、基于 5 seeds sample stdev 的 95% CI、进程池并行和单 matchup 过滤参数。`scripts/mahjong_simulate_four_player_strategy.py` 增加 tuple-count shanten/ukeire 快路径；`tile_efficiency.py` 在无防守压力、同向听候选内改为真实 ukeire 优先，防守/guarded push 仍保持 strategy-first。产物 `plans/artifacts/simulator-baseline-v1.1.json` 共 12 个 matchup；`companion vs shanten` 平均胜率 28.92%，95% CI 下界 26.51%；`companion vs oracle` 平均胜率 26.48%，通过 ≥18% 验收。

### Step 6（P2，已完成）：跨平台 overlay（子进程化 Tk）

**现状**：`overlay.py::_overlay_platform_supported()` 只在 Windows 返回 True，macOS / Linux 直接 `logger.warning` 跳过。

**任务**：

1. 拆 `overlay.py` 为：
   - `overlay/process.py`：纯子进程 worker，跑 Tk mainloop
   - `overlay/ipc.py`：主进程到子进程的命令/事件管道
   - `overlay/__init__.py`：保持现有 `CompanionOverlay` 公开接口
2. 主进程通过 `multiprocessing.Process` + `Queue` 推命令；子进程独立 GIL，不再有 daemon Thread + Tkinter mainloop 的死锁风险。
3. macOS 测试：在 macOS 终端跑 `python -m plugin.plugins.mahjong_companion.overlay.smoke`，能看到红框窗口出现 3 秒后关闭。
4. Linux X11 / Wayland：先验证 X11，Wayland 留 follow-up。

**验收**：

- macOS / Linux X11 上 `_overlay_platform_supported()` 返回 True，红框能正常显示与关闭。
- Windows 不回归（重跑 v1.0.1 的 overlay 测试集）。

**改动量**：~300-500 行重构（含测试），是 v1.1 工作量最重的一项。

**完成记录（2026-05-05）**：`overlay.py` 已重构为 `overlay/` package：`overlay/__init__.py` 保持 `CompanionOverlay` 公开接口，主进程通过 `multiprocessing.get_context("spawn")` 启动 Tk 子进程；`overlay/process.py` 跑 Tk mainloop 和红框 marker；`overlay/ipc.py` 负责 latest-status queue 与命令 drain；`overlay/view.py` 保留文案与 marker 渲染 helper。`_overlay_platform_supported()` 现在支持 Windows、macOS、Linux X11（`DISPLAY`），Wayland-only 继续留 follow-up。macOS 已跑 `python -m plugin.plugins.mahjong_companion.overlay.smoke`，窗口 3 秒后正常关闭；overlay/v4 flow 单测通过。

### Step 7（P3，验收项已完成）：Cleanup

整理工作，单独 PR：

1. **stale 测试分类处置**：
   - 删除或改名归档：v6（review summary）、v9（memory and coaching）、v7（assisted actions）、runtime_mailbox 中已经确认不会回来的旧行为测试
   - 升级保留：v8 tile_parser 的 2 个用例补 `**kwargs` 支持新 ViT classifier_config kwarg
   - 重写：static_ui 的 `quick-marker` 测试改为断言新的 quick-status-grid 结构
2. **orchestrator.py 体积守住**：当前 612 行，已低于旧目标 800。v1.1 只要求新增功能不要把 orchestrator 推回 650+，必要时把 lifecycle/diagnostics wrapper 放到小 mixin。
3. **版本号统一（follow-up）**：`plugin.toml` 是插件 manifest 版本，`pyproject.toml` 当前是主应用动态版本源；本轮不硬绑，避免把插件发布节奏和主应用版本误合并。
4. **`scripts/check_v10_release.py` 更名（follow-up）**：现有命令仍作为兼容入口保留；后续可新增 `check_v1_release` 别名后再迁移文档。

**验收**：

- `pytest plugin/tests/unit/sdk/plugin/test_mahjong_companion*.py` 总用例数下降至少 30，全部通过（无效 skip 清零）。
- `wc -l plugin/plugins/mahjong_companion/orchestrator.py` ≤ 650。

**改动量**：~600 行删除 + ~200 行迁移，主要是体力活。

**完成记录（2026-05-05）**：已删除 v6/v7/v9/runtime_mailbox/standby_mode 旧行为测试，v8 tile parser 低置信与 6s/9s 歧义用例重新启用并兼容新 classifier_config 传参，static UI 断言已更新到 `quick-status-grid` 当前结构。`lifecycle_controller.py` 接走 data lifecycle 与 runtime diagnostics async wrapper，`orchestrator.py` 当前 612 行，仍低于 650。`pytest plugin/tests/unit/sdk/plugin/test_mahjong_companion*.py` 结果为 `123 passed`、无 skip；`rg "pytest.mark.skip|pytestmark = pytest.mark.skip|skip\\(" plugin/tests/unit/sdk/plugin/test_mahjong_companion*.py` 无命中。`check_v10_release --pretty`、`evaluate_v10_release --pretty` 与 `git diff --check` 均通过。

## 3. 时间线建议

按"先解硬门禁、再补能力、最后做体力活"的节奏：

| 周 | 工作 | 出口 |
|---|---|---|
| W1 | Step 0 + Step 1 + Step 2 | release hygiene 通过，release gate 不带 skip-smoke 就 ok=True |
| W2 | Step 3 路径选择 + 实施 | hand_recognition 拉回 0.50 / 0.92 |
| W3 | Step 4 + Step 5 | decision/sim 基线锁定 |
| W4 | Step 6 主体 | macOS overlay 可用 |
| W5 | Step 6 收尾 + Step 7 cleanup | 切 v1.1.0 |

## 4. 风险与回退

- **Step 1 风险**：如果 genbutsu_hint 回归是 fast-path narration dispatcher 的副作用（N-H2 那波改动让 alert 入队后被 dedupe 掉），修起来会比预想更深，影响 fast-path 全链。回退方案：在 v1.1 里只把 fixture 升级到新预期 + 阈值降到 0.50，`coverage_warnings` 标 partial，留个 follow-up。
- **Step 3 风险**：ViT fallback 路径会让 hand 识别延迟从 ~470ms 涨到 ~700ms+（ViT 推理本身 ~150-200ms），可能撞 v0.5 `decision_latency_p95_ms ≤ 900` 门禁。回退方案：只对置信度 < 0.85 的 tile 走 ViT 二次确认（约 10-15% 的 tile），p95 涨幅可控。
- **Step 6 风险**：子进程 IPC 在 Windows 上有时启动慢（spawn 模式 200-500ms），可能让 overlay 首次出现的延迟超过用户感知阈值。回退方案：进程预热（startup 时就拉起子进程，不等到第一次 `show_overlay`）。

## 5. 不变项（advice-only 边界）

v1.1 不打破 v1.0 的 advice-only 承诺：

- 对局内 chi/pon/kan/riichi/ron/tsumo/skip 按钮**仍只用于识别、讲解、建议，不注册为可执行自动点击动作**
- 复盘摘要 / 长期记忆同步 v1.0.1 已删，v1.1 不重新引入；如果要回来，单独走 v1.2 设计评审

## 6. 参考产物

- `plans/blueprint-v1.0-closeout.md`：v1.0.1 已知遗留段
- `plans/artifacts/test-status-v1.0.1.md`：30 条 stale 测试分类
- `plans/artifacts/eval-report-v1.0.1-skip-smoke.json`：当前门禁全量数据
- `plans/artifacts/simulator-baseline-v1.0.1.json`：v1.0.1 模拟器对比基线
- `plans/artifacts/simulator-baseline-v1.1.json`：v1.1 模拟器回归基线（12 matchup，含 risk-adjusted 与 95% CI）
- `CODE_REVIEW_v1.2.md`：v1.0.1 已修 / 未修问题清单

## 7. 合并修复事件（2026-05-05）

`cleanup/mahjong-docs` 分支合并回 `bird-dev` 时，冲突解决吃掉了三处 v1.1 cleanup 工作。Release gate 表面绿色因为旧 `overlay.py` 给了 fallback，但 `pytest` 一跑就暴露 13 个 collection error。

**修复**（已合并到当前工作树）：

| 丢失项 | 后果 | 恢复方式 |
|---|---|---|
| `overlay/view.py`（331 行）| `overlay/__init__.py` 的 `from .view import _DragState, ...` 全部 ImportError，13 个测试 collect 失败 | 从 stash `9005e6dd` 恢复 |
| `perception/calibration.py::_merge_specialized_profile_templates` 等 7 个 helper（91 行）| vit-discard / manual-hand 不再合成；`test_resolve_calibration_profile_merges_specialized_template_profiles` 失败 | 重新落到 line 318 起，命名为 `_coerce_template_int` 避开和 `decision/utils.py::coerce_int` 重名 |
| `orchestrator.py` 缺 `LifecycleControllerMixin` import / 基类 / `forced_refresh` 短路（7 行） | overlay refresh 不再消费 runtime tick；`test_overlay_refresh_button_runs_advice_pipeline` 失败 | 加回 import + Mixin + `if forced_refresh: emit; return` |

修后验证：

- `pytest test_mahjong_companion*.py` → **123 passed, 0 skipped, 0 failed**
- `evaluate_v10_release` → `ok=true`，hand_recognition tile=1.0 / full=1.0，v05 latency p95=739ms（沙箱）
- `check_v10_release` → `ok=true`（10/10）

旧的 monolithic `plugin/plugins/mahjong_companion/overlay.py` 已被新 `overlay/` 包完全替代；本轮已置零字节，**最终 commit 时需要 `git rm`**。
