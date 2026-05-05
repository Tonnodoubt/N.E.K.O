from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Optional

from plugin.sdk.plugin import Ok

from .adapters import DefaultNarrationAdapter, DefaultPerceptionAdapter, NarrationAdapter, PerceptionAdapter
from .capture import DefaultCaptureProvider
from .config_accessors import ConfigAccessorMixin
from .decision.adapter import DefaultDecisionAdapter
from .decision.preturn_planner import PreturnDiscardPlan
from .fast_path import PreturnFastPathMixin
from .frame_resources import FrameResourceMixin, _prune_debug_samples
from .lifecycle_controller import LifecycleControllerMixin
from .meld_selection import MeldSelectionMixin
from .gates import DefaultFrameChangeGate
from .narration import NarrationEvent
from .narration.dispatcher import NarrationDispatcher
from .overlay import CompanionOverlay
from .overlay_controller import OverlayControllerMixin
from .pipeline_execution import PipelineExecutionMixin
from .session_state import SessionState, now_iso
from .state_transitions import (
    StateTransitionMixin,
    _image_region_signature,
    _image_region_signature_distance,
)
from .storage import locked_json_path, write_json_atomic
from .status_snapshot import (
    StatusSnapshotContext,
    build_status_snapshot,
    current_screen_overlays,
    derive_report_status,
)
from .window_binding import WindowBindingResult, bind_window_from_title, list_window_candidates


class SessionOrchestrator(
    PreturnFastPathMixin,
    MeldSelectionMixin,
    LifecycleControllerMixin,
    OverlayControllerMixin,
    ConfigAccessorMixin,
    PipelineExecutionMixin,
    FrameResourceMixin,
    StateTransitionMixin,
):
    def __init__(
        self,
        plugin: Any,
        *,
        perception_adapter: PerceptionAdapter | None = None,
        narration_adapter: NarrationAdapter | None = None,
    ):
        self.plugin = plugin
        self.logger = plugin.logger
        self.state = SessionState.create()
        self._task: Optional[asyncio.Task] = None
        self._config: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._state_lock = threading.RLock()
        self._consecutive_capture_failures = 0
        self._capture_provider = DefaultCaptureProvider()
        self._perception_adapter = perception_adapter or DefaultPerceptionAdapter(
            calibration_dir=self.plugin.data_path("calibration"),
        )
        self._decision_adapter = DefaultDecisionAdapter()
        self._narration_adapter = narration_adapter or DefaultNarrationAdapter()
        self._frame_change_gate = DefaultFrameChangeGate()
        self._narration_dispatcher = NarrationDispatcher(plugin)
        self._selected_window_title = ""
        self._overlay = CompanionOverlay(self.logger)
        self._overlay_visible = False
        self._preturn_discard_plan: PreturnDiscardPlan | None = None
        self._last_preturn_plan_meta: dict[str, Any] = {}
        # Latency-sensitive narration events are dispatched OUTSIDE the
        # orchestrator lock by `_drain_pending_fast_dispatch_events` so that
        # plugin.push_message (TTS / IPC, hundreds of ms) does not block UI
        # entry points or the next screenshot/recognition tick.
        self._pending_fast_dispatch_events: list[NarrationEvent] = []
        self._last_fast_advice_frame_path = ""
        self._last_debug_sample_prune_at = 0.0
        self._fast_poll_until = 0.0
        self._last_frame_gate_decision: Any | None = None
        self._last_screen_overlay_update_at = 0.0
        self._last_fast_button_scan_at = 0.0
        self._last_fast_button_scan_meta: dict[str, Any] = {}
        self._meld_selection_pending = False
        self._pending_meld_type = ""
        self._pending_meld_call_tile = ""
        self._last_meld_selection_frame_path = ""
        self._meld_selection_snapshot: dict[str, Any] = {}

    def apply_config(self, config: dict[str, Any]) -> None:
        self._config = config
        companion_cfg = config.get("mahjong_companion", {})
        default_mode = companion_cfg.get("default_mode")
        speech_cfg = companion_cfg.get("speech_policy", {})
        if isinstance(default_mode, str) and not self.state.running:
            self.state.mode = default_mode
        if isinstance(speech_cfg, dict):
            self.state.voice_enabled = bool(speech_cfg.get("voice_enabled", True))
            if not self.state.running:
                voice_mode = str(speech_cfg.get("voice_mode", "key_events_only")).strip()
                self.state.voice_mode = voice_mode or "key_events_only"
        perception_cfg = companion_cfg.get("perception", {})
        if isinstance(self._perception_adapter, DefaultPerceptionAdapter) and isinstance(perception_cfg, dict):
            self._perception_adapter.apply_perception_config(perception_cfg)
        runtime_cfg = companion_cfg.get("game_agent_runtime", {})
        if isinstance(runtime_cfg, dict):
            runtime_mode = str(runtime_cfg.get("mode", self.state.runtime_mode)).strip().lower() or "active"
            if runtime_mode in {"active", "standby", "off"}:
                self.state.runtime_mode = runtime_mode
                self.state.runtime_status = runtime_mode

    async def start(self):
        async with self._lock:
            if self.state.running:
                return Ok({"already_running": True, **self.get_status()})

            self._start_session_locked()
            return Ok(self.get_status())

    async def stop(self):
        async with self._lock:
            self.state.running = False
            self.state.status = "stopping"
            self.state.runtime_status = "stopping"
            self._emit_status()

            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    self.logger.exception("mahjong companion loop stopped with an error")
                self._task = None

            self.state.status = "idle"
            self.state.runtime_status = "idle"
            self._consecutive_capture_failures = 0
            self._frame_change_gate.reset()
            self._fast_poll_until = 0.0
            self._last_frame_gate_decision = None
            self._last_screen_overlay_update_at = 0.0
            self._clear_preturn_discard_plan_locked()
            self._emit_status()
            return Ok(self.get_status())

    async def shutdown(self):
        await self.stop()
        async with self._lock:
            self._hide_overlay_locked()
            self._emit_status()
            return Ok(self.get_status())

    async def show_overlay(self):
        async with self._lock:
            payload = self._show_overlay_locked()
            self._emit_status()
            return Ok(payload)

    async def hide_overlay(self):
        async with self._lock:
            self._hide_overlay_locked()
            self._emit_status()
            return Ok({
                "ok": True,
                "overlay_visible": False,
                "status": self.state.status,
            })

    def _start_session_locked(self) -> None:
        self.state.running = True
        self.state.status = "starting"
        self.state.runtime_status = "starting"
        self.state.last_error = ""
        self._consecutive_capture_failures = 0
        self.state.last_notification_at = ""
        self.state.last_notification_text = ""
        self.state.last_notification_key = ""
        self.state.last_notification_channel = ""
        self.state.last_notification_delivery = ""
        self.state.last_notification_ok = False
        self.state.last_spoken_at = ""
        self.state.last_spoken_text = ""
        self.state.last_speak_ok = False
        self.state.last_runtime_interrupt_reason = ""
        self.state.started_at = self.state.started_at or now_iso()
        self._frame_change_gate.reset()
        self._fast_poll_until = 0.0
        self._last_frame_gate_decision = None
        self._last_screen_overlay_update_at = 0.0
        self._clear_preturn_discard_plan_locked()
        self._emit_status()
        self._task = asyncio.create_task(self._run_loop(), name="mahjong-companion-loop")

    async def set_mode(self, mode: str) -> None:
        async with self._lock:
            self.state.mode = mode
            self._emit_status()

    async def set_runtime_mode(self, mode: str):
        async with self._lock:
            normalized = str(mode).strip().lower()
            if normalized not in {"active", "standby", "off"}:
                return Ok({
                    "ok": False,
                    "error": f"invalid runtime mode: {mode}",
                    "allowed": ["active", "standby", "off"],
                })
            self.state.runtime_mode = normalized
            if normalized == "off":
                self.state.status = "idle"
            elif normalized == "standby" and self.state.running:
                self.state.status = "standby"
            elif normalized == "active" and self.state.running and self.state.status == "standby":
                self.state.status = "scanning"
            self.state.runtime_status = self.state.runtime_mode
            self._emit_status()
            return Ok({
                "ok": True,
                "runtime_mode": self.state.runtime_mode,
                "runtime_status": self.state.runtime_status,
            })

    def get_status(self) -> dict[str, Any]:
        return self._build_status_snapshot()

    async def list_window_candidates(self):
        async with self._lock:
            candidates = await asyncio.to_thread(list_window_candidates, self._get_keywords())
            return Ok({
                "ok": True,
                "selected_window_title": self._selected_window_title,
                "candidates": candidates,
            })

    async def bind_window(self, window_title: str = ""):
        async with self._lock:
            requested_title = str(window_title or "").strip()
            if requested_title:
                self._selected_window_title = requested_title
            result = await asyncio.to_thread(self._bind_window)
            auto_started = False
            runtime_activated = False
            if result.bound and self.state.runtime_mode != "active":
                self.state.runtime_mode = "active"
                self.state.runtime_status = self.state.runtime_mode
                if self.state.status == "standby":
                    self.state.status = "scanning"
                runtime_activated = True
            if result.bound and not self.state.running:
                self._start_session_locked()
                auto_started = True
            overlay_payload: dict[str, Any] = {"ok": True, "overlay_visible": self._overlay_visible}
            if result.bound and self._overlay_auto_show_on_bind():
                overlay_payload = self._show_overlay_locked()
            self._emit_status()
            payload = result.to_dict()
            payload["auto_started"] = auto_started
            payload["selected_window_title"] = self._selected_window_title
            payload["runtime_activated"] = runtime_activated
            payload["runtime_mode"] = self.state.runtime_mode
            payload["overlay"] = overlay_payload
            payload["status"] = self.state.status
            return Ok(payload)

    async def unbind_window(self):
        async with self._lock:
            self._clear_binding()
            self._selected_window_title = ""
            self._emit_status()
            return Ok({
                "bound": False,
                "window_title": "",
                "match_keyword": "",
                "status": self.state.status,
            })

    async def capture_debug_frame(self):
        async with self._lock:
            binding_result = await asyncio.to_thread(self._bind_window)
            try:
                return Ok(await asyncio.to_thread(self._capture_debug_frame_locked, binding_result))
            except Exception as exc:
                self.logger.exception("capture_debug_frame failed")
                should_cancel = self._handle_capture_failure_locked(exc)
                self._emit_status()
                if should_cancel:
                    await self._cancel_background_loop_locked()
                return Ok({
                    "saved": False,
                    "error": str(exc),
                    "window_bound": self.state.window_bound,
                    "window_title": self.state.window_title,
                    "match_keyword": self.state.window_match_keyword,
                    "binding_error": binding_result.error,
                    "capture_error": str(exc),
                })

    async def analyze_debug_frame(self):
        async with self._lock:
            frame_path = await asyncio.to_thread(self._resolve_latest_frame_path)
            if frame_path is None:
                self._mark_perception_failure("no captured frame available")
                self._emit_status()
                return Ok({
                    "ok": False,
                    "error": "no captured frame available",
                })

            return Ok(await asyncio.to_thread(self._analyze_frame_locked, frame_path))

    async def analyze_frame_path(self, frame_path: str):
        async with self._lock:
            candidate = self._resolve_user_frame_path(frame_path)
            if isinstance(candidate, dict):
                return Ok(candidate)
            return Ok(await asyncio.to_thread(self._analyze_frame_locked, candidate))

    async def get_last_perception(self):
        async with self._lock:
            return Ok({
                "ok": self.state.last_perception_ok,
                "data": self.state.last_perception,
                "last_perception_at": self.state.last_perception_at,
            })

    async def generate_decision(self):
        async with self._lock:
            ready, payload = await asyncio.to_thread(self._ensure_perception_locked)
            if not ready:
                return Ok(payload)
            return Ok(await asyncio.to_thread(self._generate_decision_locked))

    async def get_last_decision(self):
        async with self._lock:
            return Ok({
                "ok": self.state.last_decision_ok,
                "data": self.state.last_decision,
                "last_decision_at": self.state.last_decision_at,
            })

    async def generate_narration(self):
        async with self._lock:
            ready, payload = await asyncio.to_thread(self._ensure_decision_locked)
            if not ready:
                return Ok(payload)
            return Ok(await asyncio.to_thread(self._generate_narration_locked))

    async def get_last_narration(self):
        async with self._lock:
            return Ok({
                "ok": self.state.last_narration_ok,
                "data": self.state.last_narration,
                "last_narration_at": self.state.last_narration_at,
            })

    async def preview_companion_view(self):
        async with self._lock:
            ready, payload = await asyncio.to_thread(self._ensure_narration_locked)
            if not ready:
                return Ok(payload)
            return Ok({
                "ok": True,
                "data": self.state.last_companion_view,
                "last_narration_at": self.state.last_narration_at,
            })

    async def run_companion_pipeline(
        self,
        frame_path: str = "",
        *,
        capture: bool = False,
        dispatch: bool = True,
        force_reply: bool = True,
    ):
        async with self._lock:
            return Ok(await asyncio.to_thread(
                self._run_companion_pipeline_locked,
                frame_path,
                capture=capture,
                dispatch=dispatch,
                force_reply=force_reply,
            ))

    async def speak_last_narration(self):
        async with self._lock:
            return Ok(await asyncio.to_thread(self._speak_last_narration_locked))

    async def cycle_voice_mode(self):
        async with self._lock:
            modes = ["off", "key_events_only", "companion"]
            try:
                current_index = modes.index(self.state.voice_mode)
            except ValueError:
                current_index = 0
            self.state.voice_mode = modes[(current_index + 1) % len(modes)]
            self._emit_status()
            return Ok({
                "ok": True,
                "voice_mode": self.state.voice_mode,
                "voice_enabled": self.state.voice_enabled,
            })

    async def set_voice_mode(self, mode: str):
        valid = {"off", "key_events_only", "companion"}
        normalized = str(mode).strip().lower()
        async with self._lock:
            if normalized not in valid:
                return Ok({
                    "ok": False,
                    "error": f"invalid voice mode: {mode}",
                    "allowed": sorted(valid),
                })
            self.state.voice_mode = normalized
            self._emit_status()
            return Ok({
                "ok": True,
                "voice_mode": self.state.voice_mode,
                "voice_enabled": self.state.voice_enabled,
            })

    async def set_unified_mode(self, mode: str):
        # 单一概念 → 底层 (mode, runtime_mode) 映射。
        # teaching/silent 影响讲解口径，并把 runtime 拉回 active；
        # standby/off 只切运行时，保留当前 mode，方便切回。
        mapping: dict[str, tuple[str | None, str]] = {
            "teaching": ("teaching", "active"),
            "silent": ("silent", "active"),
            "standby": (None, "standby"),
            "off": (None, "off"),
        }
        normalized = str(mode).strip().lower()
        if normalized not in mapping:
            return Ok({
                "ok": False,
                "error": f"invalid unified mode: {mode}",
                "allowed": list(mapping.keys()),
            })
        target_mode, target_runtime = mapping[normalized]
        async with self._lock:
            if target_mode is not None:
                self.state.mode = target_mode
            self.state.runtime_mode = target_runtime
            if target_runtime == "off":
                self.state.status = "idle"
            elif target_runtime == "standby" and self.state.running:
                self.state.status = "standby"
            elif target_runtime == "active" and self.state.running and self.state.status == "standby":
                self.state.status = "scanning"
            self.state.runtime_status = self.state.runtime_mode
            self._emit_status()
            return Ok({
                "ok": True,
                "unified_mode": normalized,
                "mode": self.state.mode,
                "runtime_mode": self.state.runtime_mode,
                "runtime_status": self.state.runtime_status,
            })

    async def _run_loop(self) -> None:
        self.state.status = "scanning"
        self.state.runtime_status = self.state.runtime_mode
        self._emit_status()
        try:
            while self.state.running:
                async with self._lock:
                    await asyncio.to_thread(self._run_runtime_cycle_locked)
                # Dispatch fast-path narration events outside the orchestrator
                # lock so plugin.push_message (TTS / IPC) does not block UI
                # entry points (CODE_REVIEW_v1.2 N-H2).
                await self._drain_pending_fast_dispatch_events()
                if not self.state.running:
                    break
                await asyncio.sleep(self._get_current_sample_interval_ms() / 1000.0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.exception("mahjong companion loop failed")
            self.state.running = False
            self.state.last_error = str(exc)
            self.state.status = "error"
            self.state.runtime_status = "error"
            self._emit_status()

    def _run_runtime_cycle_locked(self) -> None:
        forced_refresh = self._process_overlay_commands_locked()
        if forced_refresh:
            self._emit_status()
            return
        if self.state.runtime_mode == "off":
            self.state.runtime_status = "off"
            self.state.status = "idle"
            self._emit_status()
            return
        if self.state.runtime_mode == "standby":
            self.state.runtime_status = "standby"
            if self.state.running and self.state.status not in {"warning", "error"}:
                self.state.status = "standby"
            self._emit_status()
            return

        self.state.runtime_mode = "active"
        self.state.runtime_status = "active"
        self._run_live_cycle_locked()

    async def _drain_pending_fast_dispatch_events(self) -> None:
        # Snapshot under the lock to keep the queue read race-free, then
        # release the lock for the slow push_message I/O.
        async with self._lock:
            events = list(self._pending_fast_dispatch_events)
            self._pending_fast_dispatch_events.clear()
        for event in events:
            try:
                await asyncio.to_thread(self._dispatch_fast_event_unlocked, event)
            except Exception:
                self.logger.exception("fast-path narration dispatch failed")

    def _dispatch_fast_event_unlocked(self, event: NarrationEvent) -> None:
        # Runs in a worker thread WITHOUT the asyncio lock held.
        # _state_lock (threading.RLock) serializes state mutations with
        # _emit_status snapshot reads, preventing torn reads across threads.
        with self._state_lock:
            try:
                self._narration_dispatcher.dispatch(
                    event,
                    state=self.state,
                    emit_status=self._emit_status,
                    target_lanlan=self._get_voice_target_lanlan(),
                    require_running=True,
                    require_window_bound=True,
                )
            except Exception:
                self.logger.exception("fast narration dispatcher raised")

    def _bind_window(self) -> WindowBindingResult:
        if self._selected_window_title:
            result = bind_window_from_title(self._selected_window_title, self._get_keywords())
        else:
            result = self._capture_provider.locate_window(self._get_keywords())
        self.state.window_bound = result.bound
        self.state.window_title = result.window_title or result.app_name
        self.state.window_match_keyword = result.match_keyword
        self.state.window_left = result.left
        self.state.window_top = result.top
        self.state.window_width = result.width
        self.state.window_height = result.height
        if result.bound:
            self.state.last_error = ""
        elif result.error:
            self.state.last_error = result.error
        return result

    async def _cancel_background_loop_locked(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("mahjong companion loop cancellation surfaced an error")
        finally:
            self._task = None

    def _derive_report_status(self) -> str:
        return derive_report_status(self.state)

    def _build_status_snapshot(self) -> dict[str, Any]:
        return build_status_snapshot(self._status_snapshot_context())

    def _status_snapshot_context(self) -> StatusSnapshotContext:
        return StatusSnapshotContext(
            state=self.state,
            selected_window_title=self._selected_window_title,
            overlay_visible=self._overlay_visible,
            preturn_discard_plan=self._preturn_discard_plan,
            last_preturn_plan_meta=self._last_preturn_plan_meta,
            last_fast_advice_frame_path=self._last_fast_advice_frame_path,
            last_fast_button_scan_meta=self._last_fast_button_scan_meta,
        )

    def _current_screen_overlays(self) -> list[dict[str, Any]]:
        return current_screen_overlays(self.state.last_decision)

    def _emit_status(self) -> None:
        with self._state_lock:
            if self._overlay_visible and not self._overlay.is_running():
                self._overlay_visible = False
            snapshot = self._build_status_snapshot()
        self.plugin.report_status(snapshot)
        self._write_session_cache(snapshot)
        if self._overlay_visible:
            try:
                self._overlay.update_status(snapshot)
            except Exception:
                self.logger.exception("failed to update mahjong companion overlay")
                with self._state_lock:
                    self._overlay_visible = False

    def _write_session_cache(self, snapshot: dict[str, Any]) -> None:
        cache_dir = self.plugin.data_path("session_cache")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "latest_session.json"
            with locked_json_path(cache_path):
                write_json_atomic(cache_path, snapshot)
        except Exception:
            self.logger.exception("failed to write mahjong companion session cache")
