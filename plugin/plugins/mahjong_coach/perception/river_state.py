from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .calibration import resolve_calibration_profile
from .discard_parser import parse_discards_from_image


@dataclass(frozen=True)
class RiverStateResult:
    ok: bool = False
    discard_piles: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    visible_tiles: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    elapsed_ms: float = 0.0
    raw_detections: list[dict[str, Any]] = field(default_factory=list)
    analysis_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["elapsed_ms"] = round(float(self.elapsed_ms), 1)
        payload["confidence"] = round(float(self.confidence), 4)
        return payload


def detect_river_state_path(
    image_path: Path,
    *,
    calibration_dir: Path | None = None,
    min_confidence: float = 0.90,
) -> RiverStateResult:
    started = time.perf_counter()
    if not image_path.exists():
        return RiverStateResult(reason="image_missing")

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        calibration = resolve_calibration_profile(*image.size, calibration_dir=calibration_dir)
        parsed = parse_discards_from_image(
            image,
            calibration.hand_tile_templates,
            min_confidence=min_confidence,
        )

    hints = dict(parsed.analysis_hints)
    available = bool(hints.get("discard_parser_available"))
    visible_count = len(parsed.visible_tiles)
    reason = "recognized_discards" if visible_count else "no_visible_discards"
    if not available:
        reason = str(hints.get("discard_parser_reason") or "river_parser_unavailable")
    return RiverStateResult(
        ok=available,
        discard_piles=parsed.discard_piles,
        visible_tiles=parsed.visible_tiles,
        confidence=float(hints.get("discard_analysis_confidence") or 0.0),
        reason=reason,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        raw_detections=parsed.raw_detections,
        analysis_hints=hints,
    )
