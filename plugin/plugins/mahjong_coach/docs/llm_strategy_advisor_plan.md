# Mahjong Coach Target Plan

## 1. 背景

`mahjong_coach` 的目标不是做每巡代打，而是做一个安静的局内教练：

- 本地感知负责截图、按钮、手牌和牌河识别。
- LLM 负责低频战略判断，不参与短时间窗口的实时反应。
- UI/overlay 给中文建议，避免每一手都打断用户。

本阶段的核心调整是把插件从“每帧分析后直接分流”改成明确的局面状态机。开局、普通跟踪、阶段重评估、防守、吃碰杠窗口、和牌窗口应该有不同优先级。

## 2. 目标决策链

### 2.1 新局开局

新局开始时不应该优先判断 `ron`、`tsumo`、`riichi`，因为这些按钮在开局阶段理论上不会出现。开局阶段只做稳定手牌识别和初始策略生成。

目标流程：

```text
round_idle
  -> opening_hand_scan
  -> opening_strategy
  -> normal_tracking
```

开局行为：

1. 连续识别到稳定手牌后，进入 `opening_strategy`。
2. 不跑牌河 ONNX，因为此时没有有效牌河信息。
3. 本地生成保底开局策略。
4. 将结构化手牌发给 LLM，生成中文初始策略。
5. LLM 超时或失败时保留本地策略。

开局 LLM 可以允许更长一点的等待，因为前几巡通常不需要极快反应，尤其是字牌、孤张和役牌判断更依赖全局策略。

### 2.2 中央协调器

需要一个插件内部的中央协调器，但不需要全系统级总司令。

建议结构：

```text
Screenshot capture
  -> Local perception
  -> Round state machine
  -> Heuristic fallback
  -> Optional LLM strategy advisor
  -> DecisionCoordinator
  -> Overlay / Dashboard
```

`DecisionCoordinator` 的职责：

- 汇总手牌、按钮、牌河、局面阶段和已有策略。
- 决定哪些事件需要 LLM，哪些事件只能走本地快判。
- 防止旧 LLM 结果覆盖新手牌、新局或新阶段。
- 输出一个最终展示用的 `CoachDecision`。

### 2.3 LLM 的职责

LLM 是战略顾问，不是实时按钮处理器。

接入范围：

- 开局初始策略。
- 每约三次自身手牌稳定变化后的策略重评估。
- 局势显著变化后的策略重评估，例如进入防守、中后盘或副露明显改变。

不接入范围：

- `ron` / `tsumo`。
- `riichi` 按钮窗口。
- `chi` / `pon` / `kan` 的即时窗口。
- 普通无变化截图。

LLM 输入仍然是结构化文本，不发送截图：

- 当前手牌。
- 当前本地策略。
- 已知牌河摘要。
- 已知按钮窗口。
- 当前阶段。

Prompt 的系统指令和推理约束使用英文，返回给用户的字段使用中文。

### 2.4 Ron / Tsumo / Riichi

`ron` 和 `tsumo` 不需要策略建议，出现时直接最高优先级提示点击。

`riichi` 不应该等 LLM，但也不应该无条件自动建议。立直存在默听、打点、防守和分差判断，目标行为是：

1. 用当前策略和本地规则快速判断。
2. UI 显示“建议立直”或“可立直但不强制”。
3. 不调用 LLM，不等待牌河扫描。

### 2.5 Chi / Pon / Kan

吃碰杠窗口偶尔需要建议，但必须走快路径。

目标行为：

1. 使用当前开局/阶段策略作为背景。
2. 结合本地规则判断是否推进役、速度或防守。
3. 不等待 LLM。
4. 不临时触发牌河 ONNX，除非已有最近一次牌河结果可用。

默认倾向仍然保守，避免无意义鸣牌破坏手牌价值。

### 2.6 牌河识别插入点

牌河 ONNX 是本地感知，不是 VLM。它不应该每帧都跑，也不应该在开局立刻跑。

建议触发点：

- 进入 `checkpoint_strategy` 前，如果已经过了前几巡，跑一次牌河识别作为 LLM 输入。
- 检测到对手立直或进入防守阶段时，立即跑牌河识别，用立直家现物生成防守建议。
- 中后盘策略重评估时，低频刷新牌河。
- 普通前几巡只做手牌跟踪，不刷新牌河。

建议频率：

```text
opening_hand_scan: no river scan
opening_strategy: no river scan
normal_tracking: low frequency, only if stale and useful
checkpoint_strategy: scan once before strategy update when round is no longer opening
defense_mode: scan immediately and refresh more often
action_window: use cached river only
```

牌河识别结果只作为决策输入之一。低置信度、空牌或无法映射座位时，不能强行生成确定防守结论。

### 2.7 截图频率和模型频率

本地截图轮询和 LLM 调用频率必须分离。

目标策略：

- 本地截图保持低成本轮询，用于捕捉手牌变化和按钮窗口。
- 普通状态可使用 `1200-2000ms` 间隔。
- 动作窗口出现时短暂使用 `fast_interval_ms`。
- LLM 只在开局和阶段重评估时触发。
- 牌河 ONNX 按状态和事件触发，不跟随每张截图。

## 3. 状态机

目标状态：

```text
round_idle
  -> opening_hand_scan
  -> opening_strategy
  -> normal_tracking
  -> checkpoint_strategy
  -> defense_mode
  -> action_window
  -> round_end
```

状态说明：

- `round_idle`: 等待新局手牌出现。
- `opening_hand_scan`: 只识别手牌，等待稳定。
- `opening_strategy`: 生成本地保底策略，并触发 LLM 初始策略。
- `normal_tracking`: 跟踪自己手牌变化，不频繁打扰。
- `checkpoint_strategy`: 每约三次自身手牌变化后重评估，可带牌河摘要。
- `defense_mode`: 对手立直或明显危险时启用，优先使用牌河现物。
- `action_window`: 处理吃碰杠、立直、荣和、自摸等按钮窗口。
- `round_end`: 手牌消失或换局后清理临时状态，等待下一局。

优先级：

1. `ron` / `tsumo` 直接提示。
2. `chi` / `pon` / `kan` / `riichi` 走本地快判。
3. 防守阶段使用牌河和本地规则。
4. 开局和 checkpoint 使用 LLM 战略。
5. 普通截图不产出新建议。

## 4. 模型和 Prompt

模型 tier 固定复用项目现有 `summary` 配置，不新增 tier，不要求用户额外配置 key。

实现方式：

```python
api_config = get_config_manager().get_model_api_config("summary")
llm = create_chat_llm(
    model=api_config["model"],
    base_url=api_config["base_url"],
    api_key=api_config["api_key"],
    timeout=timeout_seconds,
)
```

要求：

- 不传 `temperature=`。
- 用严格 JSON 输出。
- 失败、超时、JSON 解析失败都回退本地策略。
- LLM 结果必须带 round/state/hand token，防止过期覆盖。

建议 JSON：

```json
{
  "summary": "",
  "detail": "",
  "bias": "attack|neutral|defense",
  "targets": [],
  "cautions": [],
  "discard_priority": []
}
```

中文字段：

- `summary`
- `detail`
- `targets`
- `cautions`
- `discard_priority`

## 5. 代码落点

目标修改范围：

```text
config/prompts/prompts_mahjong.py
plugin/plugins/mahjong_coach/llm_coach.py
plugin/plugins/mahjong_coach/decision_coordinator.py
plugin/plugins/mahjong_coach/coach.py
plugin/plugins/mahjong_coach/models.py
plugin/plugins/mahjong_coach/plugin.toml
plugin/plugins/mahjong_coach/static/index.html
plugin/plugins/mahjong_coach/static/main.js
plugin/plugins/mahjong_coach/static/style.css
```

核心改动：

- `RoundCoachEngine` 明确区分开局、普通跟踪、checkpoint、防守和动作窗口。
- 开局阶段先稳定识别手牌，不跑牌河。
- 开局策略和 checkpoint 策略允许 LLM 参与。
- `ron` / `tsumo` 直接提示。
- `riichi` 使用本地快判，不等待 LLM。
- `chi` / `pon` / `kan` 使用当前策略和本地规则快判。
- 牌河 ONNX 只在 checkpoint、防守和中后盘事件中低频触发。

## 6. 验证计划

自动验证：

```bash
uv run pytest plugin/plugins/mahjong_coach/tests/ -q
uv run ruff check plugin/plugins/mahjong_coach config/prompts/prompts_mahjong.py
uv run python scripts/check_async_blocking.py
uv run python scripts/check_prompt_hygiene.py
uv run python scripts/check_no_temperature.py
```

需要覆盖的场景：

- 新局开始只识别手牌，不触发牌河识别。
- 开局稳定手牌后触发 LLM 初始策略。
- LLM 超时后保留本地保底策略。
- 三次自身手牌变化后触发 checkpoint。
- checkpoint 可在非开局阶段带入牌河摘要。
- `ron` / `tsumo` 直接最高优先级提示。
- `riichi` 不等待 LLM。
- `chi` / `pon` / `kan` 不等待 LLM，也不临时等待牌河。
- 对手立直后触发防守模式和牌河 ONNX。
- 旧 LLM 返回不会覆盖新局或新手牌。

手动验证：

1. 无 summary 模型配置时，插件仍然能用本地策略运行。
2. 有 summary 模型配置时，开局策略能被中文 AI 策略增强。
3. 前几巡没有无意义牌河刷新。
4. 中后盘和防守时能看到牌河输入影响建议。
5. dashboard 显示当前建议来源：`Heuristic` 或 `AI`。

## 7. 非目标

本阶段明确不做：

- 把截图发给 VLM。
- 每巡弃牌都问 LLM。
- 自动点击按钮。
- 赛后复盘。
- LLM 结果持久化。
- 新增模型 tier。
- 引入新依赖。

## 8. 推荐实施顺序

1. 先调整文档和测试目标，锁定状态机行为。
2. 修改 `RoundCoachEngine` 的阶段判断和优先级。
3. 调整 LLM 触发点：开局和 checkpoint。
4. 调整牌河 ONNX 触发点：非开局、checkpoint、防守。
5. 调整动作窗口：Ron/Tsumo 直出，Riichi/Chi/Pon/Kan 本地快判。
6. 更新 UI 展示策略来源和牌河状态。
7. 运行插件测试、ruff 和 prompt hygiene。
