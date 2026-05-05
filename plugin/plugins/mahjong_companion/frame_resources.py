from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from .config_accessors import _coerce_int
from .session_state import now_iso
from .window_binding import WindowBindingResult


class FrameResourceMixin:
    def _capture_debug_frame_locked(self, binding_result: WindowBindingResult) -> dict[str, Any]:
        packet = self._capture_provider.capture_frame(
            samples_dir=self._runtime_debug_samples_dir(),
            binding_result=binding_result,
            save_format=self._get_capture_format(),
        )
        self.state.last_frame_path = packet.image_path
        self.state.last_frame_at = now_iso()
        self.state.last_capture_source = packet.source
        self.state.last_capture_ok = True
        self.state.last_error = ""
        self._consecutive_capture_failures = 0
        self._maybe_prune_debug_samples_locked()
        self._maybe_clear_expired_screen_overlays_locked()
        if self.state.running and self.state.status == "warning":
            self.state.status = "scanning"
        self._emit_status()
        return {
            "saved": True,
            "path": packet.image_path,
            "source": packet.source,
            "window_bound": binding_result.bound,
            "window_title": self.state.window_title,
            "match_keyword": self.state.window_match_keyword,
            "binding_error": binding_result.error,
        }

    def _apply_failure_degrade(self) -> bool:
        if not self.state.running:
            return False
        if self._consecutive_capture_failures >= 3:
            self.state.status = "warning"
        if self._consecutive_capture_failures >= 6:
            self.state.running = False
            self.state.status = "idle"
            return True
        return False

    def _resolve_latest_frame_path(self) -> Optional[Path]:
        if self.state.last_frame_path:
            candidate = Path(self.state.last_frame_path)
            if candidate.exists():
                return candidate

        samples_dir = self._runtime_debug_samples_dir()
        if not samples_dir.exists():
            return None

        candidates = [
            path for path in samples_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _runtime_debug_samples_dir(self) -> Path:
        return self.plugin.data_path("debug_samples", "live")

    def _maybe_prune_debug_samples_locked(self) -> None:
        cfg = self._get_debug_samples_cfg()
        if not bool(cfg.get("auto_prune_enabled", True)):
            return

        now = time.time()
        interval_sec = _coerce_int(cfg.get("prune_interval_sec", 30), default=30, minimum=1)
        if now - self._last_debug_sample_prune_at < interval_sec:
            return
        self._last_debug_sample_prune_at = now

        samples_dir = self._runtime_debug_samples_dir()
        try:
            _prune_debug_samples(
                samples_dir,
                max_frames=_coerce_int(cfg.get("max_frames", 360), default=360, minimum=1),
                max_age_sec=_coerce_int(cfg.get("max_age_sec", 1800), default=1800, minimum=0),
                now=now,
            )
        except Exception:
            self.logger.exception("failed to prune mahjong companion debug samples")

    def _resolve_pipeline_frame_path(self, frame_path: str, capture: bool) -> Path | dict[str, Any]:
        if frame_path.strip():
            return self._resolve_user_frame_path(frame_path)

        if capture:
            binding_result = self._bind_window()
            try:
                payload = self._capture_debug_frame_locked(binding_result)
                return Path(str(payload["path"]))
            except Exception as exc:
                self.logger.exception("run_companion_pipeline capture failed")
                self._handle_capture_failure_locked(exc)
                self._emit_status()
                return {
                    "ok": False,
                    "stage": "capture",
                    "error": str(exc),
                    "window_bound": self.state.window_bound,
                    "window_title": self.state.window_title,
                    "binding_error": binding_result.error,
                }

        candidate = self._resolve_latest_frame_path()
        if candidate is None:
            return {
                "ok": False,
                "stage": "frame_selection",
                "error": "no frame available; provide frame_path or enable capture",
            }
        return candidate

    def _resolve_user_frame_path(self, frame_path: str) -> Path | dict[str, Any]:
        candidate = self._resolve_user_path(frame_path)
        if candidate is None:
            return {
                "ok": False,
                "error": "frame_path is required",
            }
        if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            return {
                "ok": False,
                "error": "frame_path must be a png/jpg/jpeg file",
                "frame_path": str(candidate),
            }
        allowed_roots = [
            self.plugin.data_path("debug_samples"),
            self._bundled_debug_samples_dir(),
        ]
        if not self._is_path_under_any(candidate, allowed_roots):
            return {
                "ok": False,
                "error": "frame_path must be under plugin debug_samples",
                "frame_path": str(candidate),
                "allowed_roots": [str(root.resolve()) for root in allowed_roots],
            }
        return candidate

    def _resolve_user_path(self, raw_path: str) -> Path | None:
        value = str(raw_path).strip()
        if not value:
            return None
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve()

    def _bundled_debug_samples_dir(self) -> Path:
        return Path(__file__).resolve().parent / "data" / "debug_samples"

    def _is_path_under_any(self, candidate: Path, roots: list[Path]) -> bool:
        resolved = candidate.resolve()
        for root in roots:
            try:
                resolved.relative_to(root.resolve())
                return True
            except ValueError:
                continue
        return False

    def _debug_artifacts_allowed(self, frame_path: Path) -> bool:
        return self._is_path_under_any(frame_path, [self.plugin.data_path("debug_samples")])

    def _should_process_frame_locked(self, frame_path: Path) -> bool:
        frame_gate_cfg = self._get_frame_gate_cfg()
        decision = self._frame_change_gate.evaluate(
            frame_path,
            enabled=bool(frame_gate_cfg.get("enabled", True)),
            min_change_distance=int(frame_gate_cfg.get("min_change_distance", 3)),
            stable_skip_limit=int(frame_gate_cfg.get("stable_skip_limit", 300)),
        )
        self._last_frame_gate_decision = decision
        return bool(decision.should_process or self._fast_poll_force_process_enabled())


def _prune_debug_samples(
    samples_dir: Path,
    *,
    max_frames: int,
    max_age_sec: int,
    now: float | None = None,
) -> int:
    if not samples_dir.exists() or not samples_dir.is_dir():
        return 0

    now_ts = time.time() if now is None else float(now)
    frame_paths = sorted(
        (
            path for path in samples_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and path.name.endswith(("-frame.png", "-frame.jpg", "-frame.jpeg"))
        ),
        key=lambda path: _path_mtime(path),
        reverse=True,
    )
    removed = 0
    for index, frame_path in enumerate(frame_paths):
        within_count_limit = index < max(1, max_frames)
        within_age_limit = max_age_sec <= 0 or now_ts - _path_mtime(frame_path) <= max_age_sec
        if within_count_limit and within_age_limit:
            continue
        removed += _remove_debug_sample_group(frame_path)
    return removed


def _remove_debug_sample_group(frame_path: Path) -> int:
    removed = 0
    for path in _debug_sample_group_paths(frame_path):
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def _debug_sample_group_paths(frame_path: Path) -> list[Path]:
    paths = [frame_path]
    base_path = frame_path.with_suffix("")
    for suffix in ("-perception.json", "-overlay.json", "-decision.json", "-narration.json"):
        paths.append(base_path.with_name(base_path.name + suffix))
    return paths


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
