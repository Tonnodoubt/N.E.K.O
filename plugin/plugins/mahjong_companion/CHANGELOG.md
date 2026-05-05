# Changelog

## v1.0.1 - 2026-05-05

### UI / API 精简
- 调试面板从 31 个按钮收敛到约 9 个常用控件，把分步调试按钮、运行时邮箱、底层 mode/runtime_mode 双 select 全部移到 `?dev=1` 折叠区。
- 新增 `set_voice_mode(mode)` plugin entry，替代命令式的 `cycle_voice_mode`；UI 改为 select + 应用按钮。
- 新增 `set_unified_mode(mode)` plugin entry，把"教学/静默/暂停/关闭"四档单一概念映射到底层 `(mode, runtime_mode)`，UI 不再要求用户理解两个独立概念。
- 辅助操作 select 改为通过 `list_assist_actions` 动态填充，不再硬编码。

### Bug fixes（CODE_REVIEW_v1.2 Wave 1）
- **H4**: `_dedupe` 抽到 `tile_labels.py`，统一 7 个文件（review/decision/perception 各模块）的重复定义。
- **H5**: `_normalize_tile` / `_normalize_tile_list` / `_normalize_tile_set` 抽到 `tile_labels.py`，统一 risk_estimator 与 tile_efficiency 的重复定义。
- **N-L1**: `normalize_tile` 加入 `0m` / `0p` / `0s` 赤五映射（雀魂/天凤导出格式），之前会被丢成无效牌。
- **M1**: `_resolve_draw_slot` 把守卫挪到索引访问之前，避免 `len(hand_slots) == 13` + `draw_slot_index == 14` 时整条 fast path 崩溃。
- **M6**: `main.js` 的 `refreshWindowCandidates().catch(() => {})` 改为 `console.error`，至少保留一行可观测性。
- **N-M3**: `diagnostics.py` 接 `utils.logger_config`，校准 profile 加载失败现在会 `logger.warning` 而不是静默吞掉。
- **N-L3**: 删除 `orchestrator._maybe_emit_fast_preturn_advice_locked` 里对 `plan.draw_slot_index` 的双重 clamp。

### Performance（CODE_REVIEW_v1.2 Wave 2）
- **N-H1**: `_estimate_taatsu` / `_estimate_standard_shanten_with_open_melds` 内层 `search` 闭包提到模块级 `_taatsu_search` / `_standard_shanten_search`，挂 `@lru_cache(maxsize=131072)` 真正生效。之前每次外层调用都会重建闭包丢弃缓存，fast-poll 路径每帧 ~476 次冷启动递归；现在跨调用复用，本地基准 34 个不同状态冷 6.43ms / 暖 0.01ms（~919× speedup）。
- **N-M4**: `window_binding._activate_window_best_effort` 加 `isActive` 短路 + 1 秒节流（`ACTIVATION_THROTTLE_SECONDS`，按 hwnd / id(window) 记账），fast tick 不再每次盲目阻塞 80ms。
- **N-M1**: `data_lifecycle._iter_path_members` 改为 `Iterator[Path]` 流式遍历，不再 `sorted(rglob("*"))` 一次物化整树；`_collect_export_files` 的去重排序挪到 post-filter 切片上做，长跑下 `debug_samples` 上万 PNG 不再压住状态查询和导出流程。

### Threading / 锁正确性（CODE_REVIEW_v1.2 Wave 3）
- **H2**: `overlay.py` 加 `_overlay_platform_supported()` 平台守卫，仅在 Windows 启动 Tk worker 线程；macOS / Linux 直接 `logger.warning` 跳过悬浮窗（Tkinter mainloop 不支持子线程，跨平台会崩溃挂死）。
- **H3 + N-M2**: overlay `_run` 内引入闭包标志 `closed = {"value": False}` + `safe_close()` 互斥三条 destroy 路径（`__control__: close`、`_running` 失效、`WM_DELETE_WINDOW`），并在 `poll()` 顶端检查标志位 bail，避免重复 `root.destroy()` 抛 `TclError` 和 `root.after` 在 destroy 后再触发。
- **N-H3**: fast-path 引入 `_snapshot_fast_path_state` / `_restore_fast_path_state` 整体快照 perception / decision / narration 字段（含 `_fast_poll_until`）；决策失败时整组回滚而不是只回 `last_perception`，避免 scene / last_buttons / 已清的 decision/narration 留在脏半应用状态。
- **N-H2**: fast-path 不再在锁内调 `_dispatch_narration_locked`（其内部 flush 会触发 `plugin.push_message` 的 TTS / IPC，几百 ms 级阻塞）。事件改为入队 `_pending_fast_dispatch_events`，`_run_loop` 释放锁后由 `_drain_pending_fast_dispatch_events` 用 `asyncio.to_thread` 跑 `_dispatch_fast_event_unlocked`，UI 入口（`bind_window` / `set_runtime_mode` 等）不再被卡。dispatcher 自带 `dedupe_key` + `require_running` 守卫，去重和会话状态变化都安全。

### 数据正确性 / 维护性（CODE_REVIEW_v1.2 Wave 4）
- **N-H4**: `_find_drawn_tile` 把"previous tile 数量等于 current tile 数量"的强相等检查放宽为"current[t] >= previous[t]"。原本"摸到一张已有牌"的合法场景（例如 5p:1 → 5p:2）会被误判为 "" 触发 fallback；现在正确返回 5p。drift 场景仍由 deltas 长度 + sum 校验拦下。
- **H1**: `orchestrator._fast_path_base_state` 不再用 `[[""] for _ in range(plan.meld_count)]` 伪造 melds。该占位会让 `_meld_group_count`（看 list 长度）与 `_normalize_group_list`（过滤空字符串）口径前后矛盾。`analysis_hints["recognized_meld_group_count"]` 已经携带数量，`_meld_group_count` 在 state.melds 为空时会回退到 hints。
- **N-M5**: `decision/preturn_planner` 的 `apply_preturn_discard_plan` / `apply_preturn_draw_tile` 改用 `dataclasses.replace`，不再 `PerceivedGameState(**state.to_dict())` 反射式重建。`to_dict` 后续加派生字段不会再在锁内 fast path 抛 `TypeError`。
- **N-M6**: `_with_drawn_tile_slot_hint` 中 `hints.get("hand_tile_slots")` 改为单次取值后做类型判断，去掉重复 `.get`。
- **M3**: `test_button_region_detector.test_analyze_image_path_promotes_action_button_frame_to_in_match` 从依赖未入库的 `20260502-153123-253840-frame.png` 改为用 `chi.png` 模板贴在 1920×1080 mahjong-table 蓝色背景上 + `tmp_path` 写盘，跨机器跑稳定。

### Tests
- 新增 `test_mahjong_companion_tile_labels.py`，覆盖 `dedupe` / `normalize_tile`（含 r5/0/字牌别名/garbage）/ `format_tile_label` 红五。


## v1.0.0 - 2026-05-02

- Added plugin-level local data lifecycle entries: `get_data_lifecycle`, `export_local_data`, and `clear_local_runtime_data`.
- Added a safe local data export package flow that defaults to session cache, debug samples, and calibration profiles while excluding raw calibration screenshots.
- Added a dry-run-first runtime data cleanup flow that can clear session/debug/export artifacts without deleting calibration profiles or raw screenshots.
- Added `clear_calibration_raw_data` and debug UI controls for dry-run-first cleanup of raw calibration screenshots after profiles and eval fixtures have been saved.
- Added debug UI controls for local data export/cleanup and a first-glance discard river summary.
- Added `get_runtime_diagnostics` for read-only checks of calibration profiles, button templates, external discard recognizer config, recent status, and advice-only action boundaries.
- Added same-aspect calibration profile scaling so the 1920x1080 profile can act as a 2560x1440 fallback baseline.
- Added visual riichi-stick detection from the central score panel, populating `riichi_players` without OCR when white riichi sticks with red dots are visible.
- Added `scripts.prepare_riichi_stick_review` to batch-generate riichi-stick detection JSON, Markdown, and contact-sheet review artifacts.

## v0.5.0 - 2026-05-02

- Added discard tile-surface quad refinement so side/top river crops can use fitted four-point polygons instead of only layout rectangles.
- Added duplicate suppression for overlapping discard detections that come from the same visible tile surface.
- Added refined-quad slot ownership checks so a side-river tile detected through a neighboring slot is assigned to the slot it overlaps best.
- Added `scripts.export_discard_recognition_dataset` to export labeled river screenshots as model-friendly JSONL with `bbox`, `quad`, player, turn index, tile, orientation, and optional refined quads.
- Added `scripts.prepare_discard_quad_review` to generate per-screenshot quad overlays, refined slot sheets, parser acceptance markers, and JSON review payloads for side/top river alignment work.
- Added `scripts.prepare_discard_crop_debug` to generate per-screenshot accepted-quad overlays, side/top crop sheets, and a markdown index for visually checking how opponent rivers are cut.
- Added `scripts.prepare_discard_gap_review` to turn v0.5 coverage gaps, such as unscored top-opponent predictions, into focused candidate contact sheets, stable candidate IDs, per-candidate crops, and confirmation commands.
- Added `scripts.apply_discard_candidate_confirmations` to turn reviewed candidate IDs into grouped partial fixtures without hand-copying `player:turn:tile` labels.
- Added an external discard recognizer contract hook for command/HTTP recognizers that can return full `discard_piles` or per-tile detections.
- Added partial-label semantics to the v0.5 discard evaluator so sampled white-dragon labels do not count every unlabelled real discard as a false positive.
- Added v0.5 discard metrics by player and by orientation, plus coverage warnings, to make side/top river coverage gaps visible instead of hiding them inside aggregate recall.
- Defaulted discard fixture/labeling tools to partial labels and exported label-scope counts in the discard-recognition dataset manifest.
- Combined discard and hand tile template payloads for river recognition, so a sparse discard profile can still use broader hand samples for non-white-dragon tiles.
- Vectorized tile-template matching with a cached NumPy signature matrix, keeping v0.5 river recognition latency under the strict gate after template expansion.
- Added `--train-extra-root` to `scripts.label_calibration`, allowing reviewed discard fixtures to reinforce the local discard template profile.
- Retrained the 1920x1080 calibration profile with reviewed opponent-river samples and 2560x1440 right-opponent holdout confirmations: 132 discard source samples across 14 tile kinds, with v0.5 strict scoring at 111/111 labeled discards and right opponent at 53/53.
- Added `scripts.evaluate_v10_release` as the first v1.0 release gate wrapper around v0.3, v0.4, v0.5, and smoke checks.
- Added `scripts.check_v10_release` and `DATA_LIFECYCLE.md` to start v1.0 release hygiene checks around version sync, local-data handling, raw screenshot exclusion, release reports, and advice-only action boundaries.

## v0.4.0 - 2026-05-01

- Completed v0.4 button localization seed coverage with `chi` / `pon` / `kan` / `riichi` / `ron` / `tsumo` / `skip` templates and `button_regions` in perception output.
- Added `ButtonCandidateLocator` implementation for consuming button regions while keeping in-match game-button clicks out of the built-in action registry.
- Added action risk levels, richer action audit fields, confirmation-chain logging, and window-focus human override guard support.
- Scoped v0.4 template coverage to advice-relevant in-match buttons; dialog `confirm` / `cancel` are not required for the advice-only release gate.
- Added `scripts/evaluate_v04.py` with button localization, assist guard false-abort, audit-chain completeness, and template-inventory gates.
- Added `scripts.prepare_button_template` to crop new button templates, update template metadata, and create seed localization fixtures from fresh screenshots.

## v0.3.0 - 2026-05-01

- Added 1920x1080 Mahjong Soul hand calibration with local raw labels, trained profile output, and template-based hand tile recognition.
- Added v0.3 evaluation tooling for decision, risk, review, hand holdout, coverage-adjusted hand metrics, and red-five-normalized diagnostics.
- Added real ukeire calculation when deck state is complete, while preserving the lightweight fallback for incomplete visual state.
- Added confirmed-genbutsu risk hints and kept suji/wall claims out of v0.3 until river recognition lands.
- Added structured review summaries and repeated training pattern aggregation across sessions.
- Marked red-five exactness as an enhancement metric; basic hand recognition is not blocked on rare `R5p` samples.

## v0.2.0 - 2026-05-01

- Added runtime command registry to keep catgirl-to-game commands explicit and testable.
- Added perception and narration adapter boundaries while preserving existing debug payload and view model behavior.
- Added host memory writer boundary with explicit unavailable and SDK-backed writer behavior.
- Added action locator skeleton around the existing fixed-offset action coordinates.
- Kept v0.2 scoped to contract boundaries and release hygiene; deeper visual button localization and mahjong analysis upgrades remain planned for later versions.
