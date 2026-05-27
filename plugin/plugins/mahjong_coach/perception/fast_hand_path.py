from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .calibration import resolve_calibration_profile
from .hand_layout import build_hand_layout
from .roi import collect_region_metrics
from .tile_classifier_dispatch import classify_hand_tile
from .tile_templates import is_probably_occupied_hand_slot


MIN_FAST_HAND_CONFIDENCE = 0.12


@dataclass(frozen=True)
class FastHandResult:
    ok: bool = False
    hand_tiles: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    elapsed_ms: float = 0.0
    raw_detections: list[dict[str, Any]] = field(default_factory=list)
    draw_slot_index: int = 14

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["elapsed_ms"] = round(float(self.elapsed_ms), 1)
        payload["confidence"] = round(float(self.confidence), 4)
        return payload


def detect_fast_hand_path(
    image_path: Path,
    *,
    calibration_dir: Path | None = None,
    min_hand_tiles: int = 12,
    max_hand_tiles: int = 14,
) -> FastHandResult:
    started = time.perf_counter()
    if not image_path.exists():
        return FastHandResult(reason="image_missing")
    min_tiles = max(1, min(14, int(min_hand_tiles or 12)))
    max_tiles = max(min_tiles, min(14, int(max_hand_tiles or 14)))

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        calibration = resolve_calibration_profile(*image.size, calibration_dir=calibration_dir)
        template_payload = calibration.hand_tile_templates
        if not calibration.enabled or not template_payload:
            return FastHandResult(
                reason="missing_hand_tile_templates",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        layout = build_hand_layout(*image.size, calibration=calibration)
        hand_tiles: list[str] = []
        confidences: list[float] = []
        raw_detections: list[dict[str, Any]] = []
        for slot in layout["hand"][:14]:
            metrics = collect_region_metrics(image, slot.box, sample_step=6)
            occupied = is_probably_occupied_hand_slot(
                {
                    "slot_mean_luma": metrics.get("mean_luma"),
                    "slot_bright_ratio": metrics.get("bright_ratio"),
                    "slot_dark_ratio": metrics.get("dark_ratio"),
                    "slot_stddev": metrics.get("stddev"),
                }
            )
            detection = {
                "slot_id": slot.slot_id,
                "candidate_tile": "",
                "confidence": 0.0,
                "box": slot.box.to_dict(),
                "occupied": occupied,
                "source": "legacy_fast_hand_path",
            }
            if not occupied:
                raw_detections.append(detection)
                continue
            crop = image.crop((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom))
            match = classify_hand_tile(crop, template_payload)
            if match is None:
                raw_detections.append(detection)
                continue
            confidence = float(match.confidence)
            detection.update(
                {
                    "candidate_tile": match.tile,
                    "confidence": confidence,
                    "template_distance": match.distance,
                    "runner_up_tile": match.runner_up_tile,
                    "runner_up_distance": match.runner_up_distance,
                }
            )
            if confidence < MIN_FAST_HAND_CONFIDENCE:
                detection["accepted"] = False
                detection["rejection_reason"] = "low_confidence"
                raw_detections.append(detection)
                continue
            detection["accepted"] = True
            hand_tiles.append(match.tile)
            confidences.append(confidence)
            raw_detections.append(detection)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    mean_confidence = sum(confidences) / max(1, len(confidences))
    hand_count = len(hand_tiles)
    if not (min_tiles <= hand_count <= max_tiles):
        return FastHandResult(
            hand_tiles=hand_tiles,
            confidence=mean_confidence,
            reason="unstable_hand_count",
            elapsed_ms=elapsed_ms,
            raw_detections=raw_detections,
        )
    confidence = mean_confidence
    if hand_count < 13:
        confidence = round(mean_confidence * (0.92 if hand_count >= 12 else 0.84), 4)
    reason_prefix = "matched_open" if hand_count < 12 else "matched"
    return FastHandResult(
        ok=True,
        hand_tiles=hand_tiles,
        confidence=confidence,
        reason=f"{reason_prefix}_{hand_count}_hand_tiles",
        elapsed_ms=elapsed_ms,
        raw_detections=raw_detections,
    )
