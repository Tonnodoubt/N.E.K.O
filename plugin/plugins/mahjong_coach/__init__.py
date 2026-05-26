from __future__ import annotations

import asyncio
import contextlib
from dataclasses import replace
from pathlib import Path
import time
from typing import Any

from plugin.sdk.plugin import Err, NekoPluginBase, Ok, SdkError, lifecycle, neko_plugin, plugin_entry, tr

from .capture import DefaultCaptureProvider, prune_frames
from .coach import RoundCoachEngine
from .decision_coordinator import DecisionCoordinator
from .llm_coach import build_round_plan_llm
from .models import CoachDecision, LiveSessionState, MahjongCoachConfig
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
        self._live_missing_hand_frames = 0
        self._overlay = CoachOverlayController()
        self._decision_coordinator = DecisionCoordinator()
        self._llm_tasks: set[asyncio.Task] = set()

    @lifecycle(id="startup")
    async def startup(self, **_):
        try:
            raw = await self.config.dump(timeout=5.0)
            self._cfg = MahjongCoachConfig.from_payload(raw if isinstance(raw, dict) else {})
            self._engine = RoundCoachEngine(
                self._cfg,
                calibration_dir=Path(__file__).resolve().parent / "data" / "calibration" / "profiles",
            )
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

    @lifecycle(id="shutdown")
    async def shutdown(self, **_):
        await self._stop_live_task()
        await self._cancel_llm_tasks()
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
        return Ok(
            {
                "status": "ready",
                "config": self._cfg.to_dict(),
                "round_state": state,
                "last_decision": dict(self._last_decision),
                "live": self._live_state.to_dict(),
                "llm_pending": len(self._llm_tasks),
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
        await self._cancel_llm_tasks()
        state = self._engine.reset_round(round_id)
        self._last_decision = {}
        self._live_last_hand_signature = ""
        self._live_state.observed_hand_changes = 0
        self._live_state.missing_hand_frames = 0
        self._live_missing_hand_frames = 0
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
        **_,
    ):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        self._apply_runtime_round_context(round_wind=round_wind, seat_wind=seat_wind, dora_tiles=dora_tiles)
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
        decision = await self._enhance_decision_with_llm(decision)
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
        **_,
    ):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        if self._live_task is not None and not self._live_task.done():
            return Ok({"status": "already_running", "running": True, "live": self._live_state.to_dict()})
        selected_keywords = [str(item).strip() for item in (keywords or []) if str(item).strip()] or list(self._cfg.live_window_keywords)
        selected_interval = max(200, int(interval_ms or self._cfg.live_interval_ms))
        overlay_enabled = bool(overlay and self._cfg.live_overlay_enabled)
        self._apply_runtime_round_context(round_wind=round_wind, seat_wind=seat_wind, dora_tiles=dora_tiles)
        self._live_stop_event = asyncio.Event()
        self._live_missing_hand_frames = 0
        self._live_state = LiveSessionState(
            running=True,
            status="starting",
            started_at=time.time(),
            updated_at=time.time(),
            overlay_enabled=overlay_enabled,
        )
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
    ) -> None:
        dora_list = None if dora_tiles is None else [str(item).strip() for item in dora_tiles if str(item).strip()]
        updated = replace(
            self._cfg,
            round_wind=str(round_wind or self._cfg.round_wind or "").strip(),
            seat_wind=str(seat_wind or self._cfg.seat_wind or "").strip(),
            dora_tiles=dora_list if dora_list is not None else list(self._cfg.dora_tiles),
        )
        self._cfg = updated
        if self._engine is not None:
            self._engine.config = updated

    @plugin_entry(
        id="mahjong_coach_stop_live",
        name=tr("entries.stop_live.name", default="Stop Mahjong Coach Live"),
        description=tr("entries.stop_live.description", default="Stop live screenshot observation and overlay updates."),
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["status", "running"],
    )
    async def mahjong_coach_stop_live(self, **_):
        await self._stop_live_task()
        return Ok({"status": self._live_state.status, "running": self._live_state.running, "live": self._live_state.to_dict()})

    @plugin_entry(
        id="mahjong_coach_window_candidates",
        name=tr("entries.window_candidates.name", default="List Mahjong Window Candidates"),
        description=tr("entries.window_candidates.description", default="List visible windows considered by Mahjong Coach live capture."),
        input_schema={"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}, "default": []}}},
        llm_result_fields=["candidates"],
    )
    async def mahjong_coach_window_candidates(self, keywords: list[str] | None = None, **_):
        selected_keywords = [str(item).strip() for item in (keywords or []) if str(item).strip()] or list(self._cfg.live_window_keywords)
        candidates = await asyncio.to_thread(list_window_candidates, selected_keywords)
        return Ok({"keywords": selected_keywords, "candidates": candidates})

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
                    payload = {"last_decision": dict(self._last_decision), "round_state": self._engine.state.to_dict()}
                    self._update_overlay(payload)
                    if not round_idle:
                        self._schedule_llm_enhancement(decision)
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
            self._overlay.stop()

    async def _stop_live_task(self) -> None:
        if self._live_stop_event is not None:
            self._live_stop_event.set()
        task = self._live_task
        self._live_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._cancel_llm_tasks()
        self._overlay.stop()
        self._live_state.running = False
        self._live_state.status = "stopped"
        self._live_state.updated_at = time.time()

    def _schedule_llm_enhancement(self, decision: CoachDecision) -> None:
        if self._engine is None or not self._decision_coordinator.should_enhance_with_llm(decision, self._cfg):
            return
        for task in list(self._llm_tasks):
            if not task.done():
                task.cancel()
            self._llm_tasks.discard(task)
        self._engine.state.llm_status = "pending"
        self._engine.state.llm_error = ""
        token = self._decision_coordinator.build_enhancement_token(
            decision,
            self._engine.state,
            self._engine.state.last_hand_signature,
        )
        heuristic_plan = self._heuristic_plan_from_decision(decision)
        task = asyncio.create_task(self._run_llm_enhancement(decision, token, heuristic_plan))
        self._llm_tasks.add(task)
        task.add_done_callback(self._llm_tasks.discard)
        self._update_overlay({"last_decision": dict(self._last_decision), "round_state": self._engine.state.to_dict()})

    async def _run_llm_enhancement(
        self,
        decision: CoachDecision,
        token: str,
        heuristic_plan: dict[str, Any],
    ) -> None:
        diagnostics: dict[str, str] = {}
        plan = await build_round_plan_llm(
            decision.hand_tiles,
            previous_plan=self._llm_previous_plan(decision),
            turn_number=self._llm_turn_number(decision),
            heuristic_plan=heuristic_plan,
            timeout=self._cfg.llm_timeout,
            diagnostics=diagnostics,
        )
        if plan is None:
            if self._engine is not None:
                self._engine.state.llm_status = diagnostics.get("status") or "empty"
                self._engine.state.llm_error = diagnostics.get("error") or "AI 没有返回可用策略"
                self._update_overlay({"last_decision": dict(self._last_decision), "round_state": self._engine.state.to_dict()})
            return
        enhanced = self._apply_llm_enhancement(decision, token, heuristic_plan, plan)
        if enhanced is None:
            return
        self._update_overlay({"last_decision": dict(self._last_decision), "round_state": self._engine.state.to_dict()})

    async def _enhance_decision_with_llm(self, decision: CoachDecision) -> CoachDecision:
        if self._engine is None or not self._decision_coordinator.should_enhance_with_llm(decision, self._cfg):
            return decision
        self._engine.state.llm_status = "pending"
        self._engine.state.llm_error = ""
        token = self._decision_coordinator.build_enhancement_token(
            decision,
            self._engine.state,
            self._engine.state.last_hand_signature,
        )
        heuristic_plan = self._heuristic_plan_from_decision(decision)
        diagnostics: dict[str, str] = {}
        plan = await build_round_plan_llm(
            decision.hand_tiles,
            previous_plan=self._llm_previous_plan(decision),
            turn_number=self._llm_turn_number(decision),
            heuristic_plan=heuristic_plan,
            timeout=self._cfg.llm_timeout,
            diagnostics=diagnostics,
        )
        if plan is None:
            self._engine.state.llm_status = diagnostics.get("status") or "empty"
            self._engine.state.llm_error = diagnostics.get("error") or "AI 没有返回可用策略"
            return decision
        return self._apply_llm_enhancement(decision, token, heuristic_plan, plan) or decision

    def _apply_llm_enhancement(
        self,
        decision: CoachDecision,
        token: str,
        heuristic_plan: dict[str, Any],
        llm_plan: dict[str, Any],
    ) -> CoachDecision | None:
        if self._engine is None:
            return None
        state = self._engine.state
        if self._decision_coordinator.is_stale(
            token,
            current_hand_signature=state.last_hand_signature,
            round_id=state.round_id,
            update_count=state.update_count,
        ):
            return None
        enhanced = self._decision_coordinator.apply_llm_plan(decision, heuristic_plan, llm_plan)
        token_update_count = self._decision_coordinator.token_update_count(token)
        token_hand_signature = self._decision_coordinator.token_hand_signature(token)
        same_hand = not token_hand_signature or token_hand_signature == state.last_hand_signature
        can_promote_plan = token_update_count == state.update_count and same_hand
        state.llm_status = "ready" if same_hand else "ready_previous_hand"
        state.llm_error = ""
        summary = str(enhanced.suggestion or "").strip()
        if summary:
            if can_promote_plan and decision.decision_type == "opening_plan":
                state.opening_plan = summary
            state.ai_plan = summary
            if can_promote_plan:
                state.current_plan = summary
                state.plan_source = "llm"
            if not state.local_plan:
                state.local_plan = str(heuristic_plan.get("summary") or decision.suggestion or decision.summary or "").strip()
                state.local_detail = str(heuristic_plan.get("detail") or decision.detail or "").strip()
            if not state.local_targets:
                heuristic_targets = heuristic_plan.get("targets")
                if isinstance(heuristic_targets, list):
                    state.local_targets = [str(item) for item in heuristic_targets if str(item).strip()]
            if not state.local_cautions:
                heuristic_cautions = heuristic_plan.get("cautions")
                if isinstance(heuristic_cautions, list):
                    state.local_cautions = [str(item) for item in heuristic_cautions if str(item).strip()]
        bias = str(llm_plan.get("bias") or "").strip()
        if can_promote_plan and bias in {"attack", "neutral", "defense"}:
            state.attack_defense_bias = bias
        targets = llm_plan.get("targets")
        if isinstance(targets, list):
            ai_targets = [str(item) for item in targets if str(item).strip()]
            state.ai_targets = ai_targets
            if can_promote_plan:
                state.target_shapes = list(ai_targets)
        cautions = llm_plan.get("cautions")
        if isinstance(cautions, list):
            ai_cautions = [str(item) for item in cautions if str(item).strip()]
            state.ai_cautions = ai_cautions
            if can_promote_plan:
                state.caution_points = list(ai_cautions)
        detail = str(llm_plan.get("detail") or enhanced.detail or "").strip()
        if detail:
            state.ai_detail = detail
        state.update_count += 1
        payload = enhanced.to_dict()
        payload["coach_state"] = state.to_dict()
        self._last_decision = payload
        return enhanced

    def _llm_previous_plan(self, decision: CoachDecision) -> str:
        if self._engine is None or decision.decision_type == "opening_plan":
            return ""
        return self._engine.state.opening_plan

    def _llm_turn_number(self, decision: CoachDecision) -> int | None:
        if decision.decision_type == "opening_plan":
            return None
        return self._live_state.observed_hand_changes or None

    def _heuristic_plan_from_decision(self, decision: CoachDecision) -> dict[str, Any]:
        state = self._engine.state if self._engine is not None else None
        return {
            "summary": decision.suggestion or decision.summary,
            "detail": decision.detail,
            "bias": state.attack_defense_bias if state is not None else "neutral",
            "targets": list(state.target_shapes) if state is not None else [],
            "cautions": list(state.caution_points) if state is not None else [],
            "discard_priority": [],
        }

    async def _cancel_llm_tasks(self) -> None:
        tasks = [task for task in self._llm_tasks if not task.done()]
        self._llm_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
            self._live_missing_hand_frames = 0
            self._live_state.missing_hand_frames = 0
            return False
        if not self._engine.state.opening_emitted:
            self._live_missing_hand_frames = 0
            self._live_state.missing_hand_frames = 0
            return False
        reason_codes = [str(item) for item in (getattr(decision, "reason_codes", []) or [])]
        if not any(code.startswith("hand_") for code in reason_codes):
            self._live_missing_hand_frames = 0
            self._live_state.missing_hand_frames = 0
            return False
        self._live_missing_hand_frames += 1
        self._live_state.missing_hand_frames = self._live_missing_hand_frames
        if self._live_missing_hand_frames < 4:
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
            "engine_meta": {"source": "live_round_idle", "missing_hand_frames": self._live_missing_hand_frames},
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
        overlay_payload["llm_pending"] = len([task for task in self._llm_tasks if not task.done()])
        self._overlay.update(overlay_text_from_payload(overlay_payload))


MahjongCoachBridgePlugin = MahjongCoachPlugin
