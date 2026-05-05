# Mahjong Companion Data Lifecycle

雀魂陪伴插件默认按本地优先处理用户数据。插件不会自动上传原始截图、牌河标注、复盘缓存或校准素材；如果接入宿主记忆，同步内容也应只包含摘要、标签和训练建议，而不是完整对局流水。

## 本地数据位置

- `data/session_cache/`：运行时状态、动作日志、复盘候选、复盘摘要、训练趋势和记忆桥队列。
- `data/debug_samples/`：用户主动抓取或调试时保存的截图和感知调试文件。
- `data/calibration/profiles/`：本地校准 profile，包含从标注样本训练出的模板签名。
- `data/calibration/raw/`：本地原始校准截图和 sidecar 标注。这个目录可能包含完整牌局画面，默认由 `.gitignore` 排除，不应进入发布包。
- `plugin/tests/data/mahjong_companion/eval/`：少量纳入仓库的评测 fixture，用于自动化回归。

## 导出

需要留存或迁移时，可以调用插件入口 `export_local_data` 生成 zip 包。默认导出：

- `data/session_cache/`
- `data/debug_samples/`
- `data/calibration/profiles/`

导出包会写到 `data/exports/`，并带有 `manifest.json`。`data/calibration/raw/` 默认不会导出；只有显式传入 `include_raw_calibration=true` 时才会纳入，因为它可能包含完整牌局画面。

牌河评测数据也可以用导出脚本生成 JSONL：

```bash
.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.export_discard_recognition_dataset \
  --label-root plugin/tests/data/mahjong_companion/eval/discard_recognition \
  --output-dir plugin/plugins/mahjong_companion/plans/artifacts/discard_recognition_dataset/v0.5-eval \
  --refine-quads \
  --pretty
```

## 删除

删除本地运行数据时，可以先调用 `clear_local_runtime_data`。该入口默认 `dry_run=true`，会先返回计划删除的内容；确认后再传 `dry_run=false`。实际清理时会保留 `.gitkeep`，并且只允许清理运行态目录：

- `data/session_cache/`
- `data/debug_samples/`
- `data/exports/`（需要显式传入 `include_exports=true`）

原始校准截图属于训练中间素材，不是插件运行时必需数据。v0.5/v1.0 收口后，如果对应的 calibration profile、reviewed eval fixture 和 release gate 报告已经保存，可以清掉 `data/calibration/raw/` 来降低仓库目录体积和隐私风险。插件提供单独入口 `clear_calibration_raw_data`，默认同样只做 dry-run；真正删除时必须传入 `confirm_token="DELETE_CALIBRATION_RAW"`。该入口只清 raw 原图和 sidecar，不会删除：

- `data/calibration/profiles/`
- `plugin/tests/data/mahjong_companion/eval/`
- `plans/artifacts/`

也可以关闭插件后手动删除以下目录内容：

```bash
rm -rf plugin/plugins/mahjong_companion/data/session_cache/*
rm -rf plugin/plugins/mahjong_companion/data/debug_samples/*
```

如果也要删除本地校准原始素材，可以删除：

```bash
rm -rf plugin/plugins/mahjong_companion/data/calibration/raw/*
```

校准 profile 位于 `data/calibration/profiles/`，删除 profile 后插件会退回默认降级识别；如果只是清理私人原始截图，通常不需要删除 profile。

查看本地数据状态可以调用 `get_data_lifecycle`，它会列出各目录路径、大小、文件数、默认导出策略和运行态清理边界。

## 上传边界

- 不会自动上传原始截图。
- 不会自动上传完整对局流水。
- 不会自动上传 `data/calibration/raw/`。
- 宿主记忆同步不可用时，摘要会停留在本地队列。
- 外部牌河识别器只在用户配置了 `MAHJONG_COMPANION_DISCARD_RECOGNIZER_CMD` 或 `MAHJONG_COMPANION_DISCARD_RECOGNIZER_URL` 时启用。

## 发布检查

发版前运行：

```bash
.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.check_v10_release --pretty
```

该检查会确认 raw calibration 目录仍被 `.gitignore` 排除，并确认 release gate 报告、版本文档和 advice-only 动作边界仍然有效。
