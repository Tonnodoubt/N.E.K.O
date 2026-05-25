from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalibrationOffsets:
    x_px: int = 0
    y_px: int = 0
    width_px: int = 0
    height_px: int = 0
    gap_px: int = 0
    draw_gap_px: int = 0


@dataclass(frozen=True)
class CalibrationProfile:
    profile_id: str = "default"
    enabled: bool = False
    screen_width: int = 0
    screen_height: int = 0
    confidence: float = 0.0
    hand_offsets: CalibrationOffsets = field(default_factory=CalibrationOffsets)
    hand_tile_templates: dict[str, Any] = field(default_factory=dict)


def resolve_calibration_profile(
    width: int,
    height: int,
    *,
    calibration_dir: Path | None = None,
) -> CalibrationProfile:
    if calibration_dir is not None and calibration_dir.exists():
        exact = _find_profile(calibration_dir, width, height)
        if exact is not None:
            return exact
    return CalibrationProfile(
        profile_id=f"default-{width}x{height}",
        screen_width=max(0, int(width)),
        screen_height=max(0, int(height)),
    )


def _find_profile(calibration_dir: Path, width: int, height: int) -> CalibrationProfile | None:
    candidates: list[CalibrationProfile] = []
    for path in sorted(calibration_dir.glob("*.json")):
        try:
            profile = load_calibration_profile(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not profile.enabled or not profile.hand_tile_templates:
            continue
        if profile.screen_width == int(width) and profile.screen_height == int(height):
            candidates.append(profile)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.confidence)


def load_calibration_profile(path: Path) -> CalibrationProfile:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("calibration profile must be a JSON object")
    return CalibrationProfile(
        profile_id=str(payload.get("profile_id") or path.stem),
        enabled=bool(payload.get("enabled", True)),
        screen_width=int(payload.get("screen_width") or 0),
        screen_height=int(payload.get("screen_height") or 0),
        confidence=float(payload.get("confidence") or 0.0),
        hand_offsets=_load_offsets(payload.get("hand_offsets")),
        hand_tile_templates=_load_template_payload(payload.get("hand_tile_templates")),
    )


def _load_offsets(value: Any) -> CalibrationOffsets:
    if not isinstance(value, dict):
        return CalibrationOffsets()
    return CalibrationOffsets(
        x_px=int(value.get("x_px") or 0),
        y_px=int(value.get("y_px") or 0),
        width_px=int(value.get("width_px") or 0),
        height_px=int(value.get("height_px") or 0),
        gap_px=int(value.get("gap_px") or 0),
        draw_gap_px=int(value.get("draw_gap_px") or 0),
    )


def _load_template_payload(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}

