from __future__ import annotations

import asyncio
import contextlib
from dataclasses import replace
from pathlib import Path
import threading
import time
from typing import Any

from plugin.sdk.plugin import Err, NekoPluginBase, Ok, SdkError, lifecycle, neko_plugin, plugin_entry, tr

from .capture import DefaultCaptureProvider, prune_frames
from .coach import RoundCoachEngine
from .models import LiveSessionState, MahjongCoachConfig, _clean_string_list, _valid_play_style
from .overlay import CoachOverlayController, overlay_text_from_payload
from .window_binding import list_window_candidates


@neko_plugin
class MahjongCoachPlugin(NekoPluginBase):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.file_logger = self.enable_file_logging(log_level="INFO")
        self.logger = self.file_logger
        self._cfg = MahjongCoachConfig()
        self._engine: RoundCoachEngine | None = None
        self._last_decision: dict[str, Any] = {}
        self._live_state = LiveSessionState()
        self._live_task: asyncio.Task | None = None
        self._live_stop_event: asyncio.Event | None = None
        self._live_last_hand_signature = ""
        self._live_last_checkpoint_at = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tk_loop: asyncio.AbstractEventLoop | None = None
        self._tk_loop_thread: threading.Thread | None = None
        self._overlay = CoachOverlayController()

    @lifecycle(id="startup")
    async def startup(self, **_):
        try:
            raw = await self.config.dump(timeout=5.0)
            self._cfg = MahjongCoachConfig.from_payload(raw if isinstance(raw, dict) else {})
            self._engine = RoundCoachEngine(
                self._cfg,
                calibration_dir=Path(__file__).resolve().parent / "data" / "calibration" / "profiles",
            )
            self._loop = asyncio.get_running_loop()
            self._overlay = CoachOverlayController(
                on_start=self._on_overlay_start_sync,
                on_stop=self._on_overlay_stop_sync,
            )
            self._show_overlay(strategy=False)
            self.register_static_ui("static")
            self.set_list_actions(
                [
                    {
                        "id": "open_ui",
                        "kind": "ui",
                        "target": f"/plugin/{self.plugin_id}/ui/",
                        "open_in": "new_tab",
                    },
                    {
                        "id": "status",
                        "kind": "entry",
                        "target": "mahjong_coach_status",
                    },
                    {
                        "id": "show_overlay",
                        "kind": "entry",
                        "target": "mahjong_coach_show_overlay",
                    },
                    {
                        "id": "analyze_frame",
                        "kind": "entry",
                        "target": "mahjong_coach_analyze_frame",
                    },
                    {
                        "id": "start_live",
                        "kind": "entry",
                        "target": "mahjong_coach_start_live",
                    },
                    {
                        "id": "stop_live",
                        "kind": "entry",
                        "target": "mahjong_coach_stop_live",
                    },
                ]
            )
            return Ok({"status": "ready", "config": self._cfg.to_dict()})
        except Exception as exc:
            self.logger.warning("mahjong coach startup failed: {}", exc)
            return Err(SdkError("failed to start mahjong_coach"))

    def _show_overlay(self, *, strategy: bool = False) -> bool:
        if not self._overlay.start():
            return False
        if strategy:
            self._overlay.show_strategy()
        else:
            self._overlay.show_config()
        return True

    def _ensure_tk_loop(self) -> asyncio.AbstractEventLoop | None:
        if self._tk_loop is not None and self._tk_loop.is_running():
            return self._tk_loop
        if self._tk_loop_thread is not None and self._tk_loop_thread.is_alive():
            return self._tk_loop
        try:
            self._tk_loop = asyncio.new_event_loop()
            ready = threading.Event()

            def _run():
                asyncio.set_event_loop(self._tk_loop)
                ready.set()
                self._tk_loop.run_forever()

            self._tk_loop_thread = threading.Thread(target=_run, daemon=True)
            self._tk_loop_thread.start()
            if not ready.wait(timeout=2.0) or not self._tk_loop.is_running():
                self.logger.warning("tk event loop failed to start")
                return None
            return self._tk_loop
        except Exception as exc:
            self.logger.warning("failed to start tk event loop: {}", exc)
            return None

    def _on_overlay_start_sync(self, style: str) -> None:
        self.logger.info("_on_overlay_start_sync called style={}", style)
        self._schedule_on_loop(self._overlay_start_live(play_style=style), "overlay_start")

    def _on_overlay_stop_sync(self) -> None:
        self._schedule_on_loop(self._overlay_stop_live(hide_overlay=True), "overlay_stop")

    def _schedule_on_loop(self, coro, label: str) -> None:
        loop = None
        if self._loop is not None and self._loop.is_running():
            loop = self._loop
        else:
            loop = self._ensure_tk_loop()
        if loop is None:
            self.logger.warning("{} requested but no running event loop available", label)
            return
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as exc:
            self.logger.warning("asyncio.run_coroutine_threadsafe failed for {}: {}", label, exc)
            return

        def _on_done(f):
            exc = f.exception()
            if exc:
                self.logger.warning("{} failed: {}", label, exc)

        future.add_done_callback(_on_done)

    @lifecycle(id="shutdown")
    async def shutdown(self, **_):
        await self._stop_live_task()
        self.clear_list_actions()
        return Ok({"status": "stopped"})

    @plugin_entry(
        id="mahjong_coach_status",
        name=tr("entries.status.name", default="Mahjong Coach Status"),
        description=tr("entries.status.description", default="Inspect current Mahjong Coach round state."),
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["status", "current_plan", "last_update_reason"],
    )
    async def mahjong_coach_status(self, **_):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        state = self._engine.state.to_dict()
        payload = {"last_decision": dict(self._last_decision), "round_state": state}
        return Ok(
            {
                "status": "ready",
                "config": self._cfg.to_dict(),
                "round_state": state,
                "last_decision": dict(self._last_decision),
                "overlay_text": overlay_text_from_payload(payload),
                "live": self._live_state.to_dict(),
                "ui_path": f"/plugin/{self.plugin_id}/ui/",
                **state,
            }
        )

    @plugin_entry(
        id="mahjong_coach_reset_round",
        name=tr("entries.reset_round.name", default="Reset Mahjong Coach Round"),
        description=tr("entries.reset_round.description", default="Reset opening and checkpoint memory for a new hand."),
        input_schema={"type": "object", "properties": {"round_id": {"type": "string", "default": "default"}}},
        llm_result_fields=["round_id"],
    )
    async def mahjong_coach_reset_round(self, round_id: str = "default", **_):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        state = self._engine.reset_round(round_id)
        self._last_decision = {}
        self._live_last_hand_signature = ""
        self._live_state.observed_hand_changes = 0
        self._live_state.missing_hand_frames = 0
        self._live_last_checkpoint_at = 0.0
        return Ok({"status": "reset", "round_state": state.to_dict(), "round_id": state.round_id})

    @plugin_entry(
        id="mahjong_coach_analyze_frame",
        name=tr("entries.analyze_frame.name", default="Analyze Mahjong Frame"),
        description=tr(
            "entries.analyze_frame.description",
            default="Analyze one Mahjong Soul screenshot and update the quiet round coach state.",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "default": ""},
                "observed_buttons": {"type": "array", "items": {"type": "string"}, "default": []},
                "self_turn_index": {"type": "integer", "default": 0},
                "force_checkpoint": {"type": "boolean", "default": False},
                "riichi_players": {"type": "array", "items": {"type": "string"}, "default": []},
                "round_wind": {"type": "string", "default": ""},
                "seat_wind": {"type": "string", "default": ""},
                "dora_tiles": {"type": "array", "items": {"type": "string"}, "default": []},
                "play_style": {"type": "string", "default": ""},
            },
        },
        timeout=20.0,
        llm_result_fields=["decision_type", "summary", "suggestion"],
    )
    async def mahjong_coach_analyze_frame(
        self,
        image_path: str = "",
        observed_buttons: list[str] | None = None,
        self_turn_index: int | None = None,
        force_checkpoint: bool = False,
        riichi_players: list[str] | None = None,
        round_wind: str = "",
        seat_wind: str = "",
        dora_tiles: list[str] | None = None,
        play_style: str = "",
        **_,
    ):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        self._apply_runtime_round_context(round_wind=round_wind, seat_wind=seat_wind, dora_tiles=dora_tiles, play_style=play_style)
        try:
            decision = await asyncio.to_thread(
                self._engine.analyze_frame,
                image_path or None,
                observed_buttons=observed_buttons or [],
                self_turn_index=self_turn_index if self_turn_index and self_turn_index > 0 else None,
                force_checkpoint=bool(force_checkpoint),
                riichi_players=riichi_players or [],
            )
        except Exception as exc:
            self.logger.warning("mahjong coach frame analysis failed: {}", exc)
            return Err(SdkError(str(exc)))
        self._last_decision = decision.to_dict()
        return Ok(dict(self._last_decision))

    @plugin_entry(
        id="mahjong_coach_start_live",
        name=tr("entries.start_live.name", default="Start Mahjong Coach Live"),
        description=tr(
            "entries.start_live.description",
            default="Start screenshot-only live observation for Mahjong Soul and update the strategy board.",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}, "default": []},
                "interval_ms": {"type": "integer", "default": 0},
                "overlay": {"type": "boolean", "default": True},
                "round_wind": {"type": "string", "default": ""},
                "seat_wind": {"type": "string", "default": ""},
                "dora_tiles": {"type": "array", "items": {"type": "string"}, "default": []},
                "play_style": {"type": "string", "default": ""},
            },
        },
        llm_result_fields=["status", "running"],
    )
    async def mahjong_coach_start_live(
        self,
        keywords: list[str] | None = None,
        interval_ms: int | None = None,
        overlay: bool = True,
        round_wind: str = "",
        seat_wind: str = "",
        dora_tiles: list[str] | None = None,
        play_style: str = "",
        **_,
    ):
        return await self._overlay_start_live(
            keywords=keywords,
            interval_ms=interval_ms,
            overlay=overlay,
            round_wind=round_wind,
            seat_wind=seat_wind,
            dora_tiles=dora_tiles,
            play_style=play_style,
        )

    async def _overlay_start_live(
        self,
        keywords: list[str] | None = None,
        interval_ms: int | None = None,
        overlay: bool = True,
        round_wind: str = "",
        seat_wind: str = "",
        dora_tiles: list[str] | None = None,
        play_style: str = "",
    ):
        self.logger.info("_overlay_start_live called play_style={}", play_style)
        if self._engine is None:
            self.logger.warning("_overlay_start_live early return: engine is None")
            return Err(SdkError("mahjong coach is not initialized"))
        if self._live_task is not None and not self._live_task.done():
            self.logger.warning("_overlay_start_live early return: live_task already running")
            if overlay:
                self._show_overlay(strategy=True)
            return Ok({"status": "already_running", "running": True, "live": self._live_state.to_dict()})
        selected_keywords = _clean_string_list(keywords) or list(self._cfg.live_window_keywords)
        selected_interval = max(200, int(interval_ms or self._cfg.live_interval_ms))
        overlay_enabled = bool(overlay and self._cfg.live_overlay_enabled)
        self._apply_runtime_round_context(round_wind=round_wind, seat_wind=seat_wind, dora_tiles=dora_tiles, play_style=play_style)
        self._live_stop_event = asyncio.Event()
        self._live_state = LiveSessionState(
            running=True,
            status="starting",
            started_at=time.time(),
            updated_at=time.time(),
            overlay_enabled=overlay_enabled,
        )
        if overlay_enabled:
            self.logger.info("_overlay_start_live calling show_strategy")
            self._show_overlay(strategy=True)
        self._live_task = asyncio.create_task(
            self._run_live_loop(
                keywords=selected_keywords,
                interval_ms=selected_interval,
                overlay_enabled=overlay_enabled,
            )
        )
        return Ok({"status": "starting", "running": True, "live": self._live_state.to_dict()})

    def _apply_runtime_round_context(
        self,
        *,
        round_wind: str = "",
        seat_wind: str = "",
        dora_tiles: list[str] | None = None,
        play_style: str = "",
    ) -> None:
        dora_list = None if dora_tiles is None else _clean_string_list(dora_tiles)
        style = _valid_play_style(play_style) if play_style else ""
        updated = replace(
            self._cfg,
            round_wind=str(round_wind or self._cfg.round_wind or "").strip(),
            seat_wind=str(seat_wind or self._cfg.seat_wind or "").strip(),
            dora_tiles=dora_list if dora_list is not None else list(self._cfg.dora_tiles),
            play_style=style or self._cfg.play_style,
        )
        self._cfg = updated
        if self._engine is not None:
            self._engine.config = updated
            self._engine.state.play_style = updated.play_style

    @plugin_entry(
        id="mahjong_coach_stop_live",
        name=tr("entries.stop_live.name", default="Stop Mahjong Coach Live"),
        description=tr("entries.stop_live.description", default="Stop live screenshot observation and overlay updates."),
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["status", "running"],
    )
    async def mahjong_coach_stop_live(self, **_):
        return await self._overlay_stop_live(hide_overlay=True)

    @plugin_entry(
        id="mahjong_coach_show_overlay",
        name=tr("entries.show_overlay.name", default="Show Mahjong Coach Overlay"),
        description=tr("entries.show_overlay.description", default="Reopen the Mahjong Coach overlay after it has been closed."),
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["status", "running"],
    )
    async def mahjong_coach_show_overlay(self, **_):
        running = self._live_task is not None and not self._live_task.done()
        if not self._show_overlay(strategy=running):
            return Err(SdkError("failed to show mahjong coach overlay"))
        return Ok({"status": "overlay_open", "running": running, "live": self._live_state.to_dict()})

    async def _overlay_stop_live(self, hide_overlay: bool = False):
        await self._stop_live_task()
        if hide_overlay:
            self._overlay.stop()
        return Ok({"status": self._live_state.status, "running": self._live_state.running, "live": self._live_state.to_dict()})

    @plugin_entry(
        id="mahjong_coach_window_candidates",
        name=tr("entries.window_candidates.name", default="List Mahjong Window Candidates"),
        description=tr("entries.window_candidates.description", default="List visible windows considered by Mahjong Coach live capture."),
        input_schema={"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}, "default": []}}},
        llm_result_fields=["candidates"],
    )
    async def mahjong_coach_window_candidates(self, keywords: list[str] | None = None, **_):
        selected_keywords = _clean_string_list(keywords) or list(self._cfg.live_window_keywords)
        candidates = await asyncio.to_thread(list_window_candidates, selected_keywords)
        return Ok({"keywords": selected_keywords, "candidates": candidates})

    @plugin_entry(
        id="mahjong_coach_extract_hand_crops",
        name=tr("entries.extract_crops.name", default="Extract Hand Crops for Training"),
        description=tr(
            "entries.extract_crops.description",
            default="Extract hand tile crops from saved live frames for classifier training.",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "output_dir": {"type": "string", "default": ""},
                "classify": {"type": "boolean", "default": True},
            },
        },
        timeout=60.0,
        llm_result_fields=["total", "per_label"],
    )
    async def mahjong_coach_extract_hand_crops(
        self,
        output_dir: str = "",
        classify: bool = True,
        **_,
    ):
        from .perception.calibration import resolve_calibration_profile
        from .perception.hand_layout import build_hand_layout
        from .perception.roi import collect_region_metrics
        from .perception.tile_classifier_dispatch import classify_hand_tile
        from .perception.tile_templates import is_probably_occupied_hand_slot
        from .tile_labels import normalize_tile

        frames_dir = self.data_path("live_frames")
        out = Path(output_dir) if output_dir else self.data_path("hand_crops")
        calibration_dir = Path(__file__).parent / "data" / "calibration" / "profiles"

        images = sorted(p for p in frames_dir.glob("*-frame.*") if p.is_file())
        if not images:
            return Err(SdkError(f"No frames found in {frames_dir}"))

        counts: dict[str, int] = {}
        min_confidence = 0.10
        IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

        for image_path in images:
            from PIL import Image
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            w, h = image.size
            calibration = resolve_calibration_profile(w, h, calibration_dir=calibration_dir)
            template_payload = calibration.hand_tile_templates
            layout = build_hand_layout(w, h, calibration=calibration)

            for slot in layout["hand"][:14]:
                metrics = collect_region_metrics(image, slot.box, sample_step=6)
                occupied = is_probably_occupied_hand_slot({
                    "slot_mean_luma": metrics.get("mean_luma"),
                    "slot_bright_ratio": metrics.get("bright_ratio"),
                    "slot_dark_ratio": metrics.get("dark_ratio"),
                    "slot_stddev": metrics.get("stddev"),
                })
                if not occupied:
                    continue
                crop = image.crop((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom))

                if classify and template_payload:
                    match = classify_hand_tile(crop, template_payload)
                    if match and match.confidence >= min_confidence:
                        label = match.tile
                    else:
                        label = "unclassified"
                else:
                    label = "unclassified"

                label_dir = out / label
                label_dir.mkdir(parents=True, exist_ok=True)
                crop.save(label_dir / f"{image_path.stem}_{slot.slot_id}.png")
                counts[label] = counts.get(label, 0) + 1

        total = sum(counts.values())
        self.logger.info("Extracted {} hand crops: {}", total, counts)
        return Ok({"total": total, "per_label": counts, "output_dir": str(out)})

    async def _run_live_loop(self, *, keywords: list[str], interval_ms: int, overlay_enabled: bool) -> None:
        assert self._engine is not None
        provider = DefaultCaptureProvider()
        frames_dir = self.data_path("live_frames")
        if overlay_enabled:
            self._overlay.start()
            self._overlay.update("Mahjong Coach\n等待雀魂窗口")
        try:
            while self._live_stop_event is not None and not self._live_stop_event.is_set():
                loop_started = time.monotonic()
                sleep_ms = interval_ms
                try:
                    binding = await asyncio.to_thread(provider.locate_window, keywords)
                    self._live_state.last_binding = binding.to_dict()
                    if not binding.bound:
                        self._live_state.status = "waiting_for_window"
                        self._live_state.last_error = binding.error or "window_not_found"
                        self._live_state.updated_at = time.time()
                        self._update_overlay({"last_decision": {"summary": "等待雀魂窗口", "suggestion": self._live_state.last_error}, "round_state": self._engine.state.to_dict()})
                        await self._sleep_live(loop_started, sleep_ms)
                        continue

                    packet = await asyncio.to_thread(
                        provider.capture_frame,
                        samples_dir=frames_dir,
                        binding_result=binding,
                        save_format=self._cfg.live_save_format,
                    )
                    force_checkpoint = self._checkpoint_due_by_time()
                    decision = await asyncio.to_thread(
                        self._engine.analyze_frame,
                        packet.image_path,
                        self_turn_index=self._live_state.observed_hand_changes or None,
                        force_checkpoint=force_checkpoint,
                    )
                    self._last_decision = decision.to_dict()
                    self._observe_live_hand_change()
                    round_idle = self._maybe_reset_live_round_idle(decision)
                    if not round_idle and decision.decision_type in {"opening_plan", "coach_checkpoint", "defense_alert"}:
                        self._live_last_checkpoint_at = time.time()
                    self._live_state.running = True
                    self._live_state.status = "waiting_for_next_round" if round_idle else "observing"
                    self._live_state.frame_index += 1
                    self._live_state.updated_at = time.time()
                    self._live_state.last_error = ""
                    self._live_state.last_frame_path = packet.image_path
                    self._live_state.last_capture_source = packet.source
                    self._live_state.last_window_title = packet.window_title
                    if not getattr(decision, "quiet", False):
                        payload = {"last_decision": dict(self._last_decision), "round_state": self._engine.state.to_dict()}
                        self._update_overlay(payload)
                    await asyncio.to_thread(prune_frames, frames_dir, keep=self._cfg.live_keep_frames)
                    if decision.action_required and not round_idle:
                        sleep_ms = self._cfg.live_fast_interval_ms
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._live_state.status = "error"
                    self._live_state.last_error = repr(exc)
                    self._live_state.updated_at = time.time()
                    self._update_overlay({"last_decision": {"summary": "实战观察错误", "suggestion": repr(exc)}, "round_state": self._engine.state.to_dict()})
                await self._sleep_live(loop_started, sleep_ms)
        finally:
            self._live_state.running = False
            self._live_state.status = "stopped"
            self._live_state.updated_at = time.time()

    async def _stop_live_task(self) -> None:
        if self._live_stop_event is not None:
            self._live_stop_event.set()
        task = self._live_task
        self._live_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                current_loop = asyncio.get_running_loop()
                if task.get_loop() is current_loop:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            except RuntimeError:
                pass
        self._live_state.running = False
        self._live_state.status = "stopped"
        self._live_state.updated_at = time.time()

    async def _sleep_live(self, loop_started: float, sleep_ms: int) -> None:
        elapsed_ms = (time.monotonic() - loop_started) * 1000.0
        remaining = max(0.05, (float(sleep_ms) - elapsed_ms) / 1000.0)
        if self._live_stop_event is None:
            await asyncio.sleep(remaining)
            return
        try:
            await asyncio.wait_for(self._live_stop_event.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            pass

    def _observe_live_hand_change(self) -> None:
        if self._engine is None:
            return
        signature = str(self._engine.state.last_hand_signature or "")
        if not signature or signature == self._live_last_hand_signature:
            return
        self._live_last_hand_signature = signature
        self._live_state.observed_hand_changes += 1

    def _maybe_reset_live_round_idle(self, decision: Any) -> bool:
        if self._engine is None:
            return False
        if getattr(decision, "action_required", False) or getattr(decision, "hand_tiles", []):
            self._live_state.missing_hand_frames = 0
            return False
        if not self._engine.state.opening_emitted:
            self._live_state.missing_hand_frames = 0
            return False
        reason_codes = [str(item) for item in (getattr(decision, "reason_codes", []) or [])]
        if not any(code.startswith("hand_") for code in reason_codes):
            self._live_state.missing_hand_frames = 0
            return False
        self._live_state.missing_hand_frames += 1
        if self._live_state.missing_hand_frames < 4:
            return False

        state = self._engine.reset_round("auto_waiting_next_round")
        self._live_last_hand_signature = ""
        self._live_last_checkpoint_at = 0.0
        self._live_state.observed_hand_changes = 0
        self._last_decision = {
            "decision_type": "round_idle",
            "priority": 1,
            "action_required": False,
            "summary": "等待下一局",
            "detail": "连续几帧没有稳定手牌，上一局大概率已经结束。",
            "suggestion": "新手牌出现后会自动重新给开局主线。",
            "buttons": [],
            "hand_tiles": [],
            "reason_codes": ["live_round_idle", "hand_missing_streak"],
            "coach_state": state.to_dict(),
            "perception": getattr(decision, "perception", {}),
            "engine_meta": {"source": "live_round_idle", "missing_hand_frames": self._live_state.missing_hand_frames},
        }
        return True

    def _checkpoint_due_by_time(self) -> bool:
        if self._engine is None or not self._engine.state.last_hand_tiles:
            return False
        if self._live_last_checkpoint_at <= 0:
            return False
        return (time.time() - self._live_last_checkpoint_at) >= self._cfg.live_checkpoint_interval_seconds

    def _update_overlay(self, payload: dict[str, Any]) -> None:
        if not self._live_state.overlay_enabled:
            return
        overlay_payload = dict(payload)
        self._overlay.update(overlay_text_from_payload(overlay_payload))
