# Mahjong Companion

雀魂陪伴插件是一个基于截图感知的实时打牌辅助插件。它观察雀魂窗口，识别牌局场景、手牌、牌河、立直棒和常见操作按钮，然后给出轻量的弃牌、吃碰杠、立直、和牌确认与防守提示。

当前版本：`v1.0.1`

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 看屏幕 | 绑定雀魂窗口，自动截图，识别对局、弹窗、回放、菜单等状态 |
| 识别牌局 | 尝试识别手牌、牌河、可见牌、立直玩家、现物和操作按钮 |
| 给建议 | 基于轻量牌理、向听/进张估算、按钮窗口和防守信号生成建议 |
| 显示提示 | 在插件 UI、悬浮 overlay 和可选语音通道中展示当前建议 |
| 保持安全 | 只给建议和红框提示，不自动点击吃、碰、杠、立直、和牌或跳过 |

## 当前入口

- 会话：`start_session` / `stop_session` / `get_session_status`
- 模式：`set_mode` / `set_runtime_mode` / `set_unified_mode`
- 窗口：`list_window_candidates` / `bind_window` / `unbind_window`
- 调试截图：`capture_debug_frame` / `analyze_debug_frame` / `analyze_frame_path`
- 建议链路：`generate_decision` / `generate_narration` / `run_companion_pipeline`
- 最近结果：`get_last_perception` / `get_last_decision` / `get_last_narration` / `preview_companion_view`
- 输出：`show_overlay` / `hide_overlay` / `set_voice_mode` / `cycle_voice_mode` / `speak_last_narration`

## 运行模式

- `active`：持续绑定窗口、截图、识别并生成建议。
- `standby`：暂停实时识别循环，保留状态和 UI。
- `off`：关闭运行循环。

统一模式：

- `teaching`：生成正常讲解和关键提醒。
- `silent`：保留分析结果，减少主动讲解。
- `standby`：暂停实时辅助。
- `off`：关闭运行时。

## 设计边界

本轮 MVP 已移除复盘摘要、长期记忆同步、动作执行、runtime mailbox、批量标注脚本和 release 评测工具链。那些内容适合独立开发工具或后续 PR，不再混入实时打牌辅助主链路。

插件当前目标很明确：帮助玩家看清当前牌局并做判断，而不是代打。

## 快速验证

```bash
python -m py_compile plugin/plugins/mahjong_companion/*.py
python -m pytest -q plugin/tests/unit/sdk/plugin/test_button_region_detector.py plugin/tests/unit/sdk/plugin/test_discard_layout.py plugin/tests/unit/sdk/plugin/test_discard_parser.py plugin/tests/unit/sdk/plugin/test_discard_quad_finder.py plugin/tests/unit/sdk/plugin/test_window_binding.py
```
