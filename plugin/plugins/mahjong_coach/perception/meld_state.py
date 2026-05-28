from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .roi import RoiBox, collect_region_metrics
from .tile_classifier_dispatch import classify_discard_tiles_batch, onnx_discard_available


DEFAULT_MIN_MELD_CONFIDENCE = 0.72
MAX_SELF_MELDS = 4


@dataclass(frozen=True)
class MeldSlot:
    slot_id: str
    meld_index: int
    tile_index: int
    box: RoiBox

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "meld_index": self.meld_index,
            "tile_index": self.tile_index,
            "box": self.box.to_dict(),
        }


@dataclass(frozen=True)
class MeldStateResult:
    ok: bool = False
    open_meld_count: int = 0
    melds: list[dict[str, Any]] = field(default_factory=list)
    tiles: list[str] = field(default_factory=list)
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


def detect_meld_state_path(
    image_path: Path,
    *,
    min_confidence: float = DEFAULT_MIN_MELD_CONFIDENCE,
    closed_hand_count: int | None = None,
) -> MeldStateResult:
    started = time.perf_counter()
    if not image_path.exists():
        return MeldStateResult(reason="image_missing")
    if not onnx_discard_available():
        return MeldStateResult(reason="onnx_tile_classifier_unavailable")
    if closed_hand_count is not None and int(closed_hand_count) >= 12:
        return MeldStateResult(reason="closed_hand_count_no_melds", elapsed_ms=(time.perf_counter() - started) * 1000.0)

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        parsed = parse_self_melds_from_image(
            image,
            min_confidence=min_confidence,
            closed_hand_count=closed_hand_count,
        )

    return MeldStateResult(
        ok=bool(parsed.open_meld_count),
        open_meld_count=parsed.open_meld_count,
        melds=parsed.melds,
        tiles=parsed.tiles,
        confidence=parsed.confidence,
        reason="recognized_self_melds" if parsed.open_meld_count else "no_self_melds",
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        raw_detections=parsed.raw_detections,
        analysis_hints=parsed.analysis_hints,
    )


def parse_self_melds_from_image(
    image: Image.Image,
    *,
    min_confidence: float = DEFAULT_MIN_MELD_CONFIDENCE,
    include_empty_detections: bool = False,
    closed_hand_count: int | None = None,
) -> MeldStateResult:
    slots = build_self_meld_layout(*image.size, closed_hand_count=closed_hand_count)
    plans: list[tuple[MeldSlot, dict[str, Any], Image.Image]] = []
    raw_detections: list[dict[str, Any]] = []
    occupied_count = 0

    for slot in slots:
        metrics = _collect_slot_metrics(image, slot)
        occupied = _is_probably_occupied_meld_slot(metrics)
        detection = _base_detection(slot, slot_metrics=metrics, occupied=occupied)
        if not occupied:
            if include_empty_detections:
                raw_detections.append(detection)
            continue
        occupied_count += 1
        plans.append((slot, detection, image.crop(_crop_box(slot.box, image.size))))

    matches = classify_discard_tiles_batch([crop for _, _, crop in plans])
    accepted: list[dict[str, Any]] = []
    for (slot, detection, _crop), match in zip(plans, matches, strict=True):
        if match is None:
            detection["accepted"] = False
            detection["rejection_reason"] = "classification_failed"
            raw_detections.append(detection)
            continue
        detection.update(
            {
                "candidate_tile": match.tile,
                "confidence": match.confidence,
                "template_distance": match.distance,
                "runner_up_tile": match.runner_up_tile,
                "runner_up_distance": match.runner_up_distance,
            }
        )
        rejection_reason = _meld_match_rejection_reason(match.tile, match.confidence, min_confidence=min_confidence)
        if rejection_reason:
            detection["accepted"] = False
            detection["rejection_reason"] = rejection_reason
            raw_detections.append(detection)
            continue
        detection["accepted"] = True
        raw_detections.append(detection)
        accepted.append(
            {
                "tile": match.tile,
                "confidence": match.confidence,
                "slot_id": slot.slot_id,
                "meld_index": slot.meld_index,
                "tile_index": slot.tile_index,
                "box": slot.box.to_dict(),
                "center_x": slot.box.left + (slot.box.width / 2.0),
            }
        )

    melds = _cluster_meld_tiles(accepted, image.width)
    tiles = [str(item.get("tile") or "") for meld in melds for item in meld.get("tiles", []) if item.get("tile")]
    confidences = [
        float(item.get("confidence") or 0.0)
        for meld in melds
        for item in meld.get("tiles", [])
        if item.get("confidence") is not None
    ]
    confidence = round(sum(confidences) / max(1, len(confidences)), 4) if confidences else 0.0
    return MeldStateResult(
        ok=bool(melds),
        open_meld_count=len(melds),
        melds=melds,
        tiles=tiles,
        confidence=confidence,
        raw_detections=raw_detections,
        analysis_hints={
            "meld_parser_available": True,
            "meld_parser_source": "onnx_tile_classifier",
            "tile_identity_reliable": False,
            "meld_slot_count": len(slots),
            "occupied_meld_slot_count": occupied_count,
            "recognized_meld_tile_count": len(tiles),
        },
    )


def build_self_meld_layout(width: int, height: int, *, closed_hand_count: int | None = None) -> list[MeldSlot]:
    screen_width = max(1, int(width))
    screen_height = max(1, int(height))
    aspect_ratio = screen_width / max(1, screen_height)
    tile_width = max(42, int(screen_width * 0.048))
    if aspect_ratio >= 2.35:
        tile_height = max(60, int(screen_height * 0.18))
        top_ratio = 0.61
        left_ratio = 0.62
    elif screen_height <= 720:
        tile_height = max(60, int(screen_height * 0.18))
        top_ratio = 0.68
        left_ratio = 0.64
    else:
        tile_height = max(60, int(screen_height * 0.146))
        top_ratio = 0.852
        left_ratio = _meld_left_ratio_for_hand_count(closed_hand_count)
    top = _clamp_int(round(screen_height * top_ratio), 0, screen_height - tile_height)
    left = _clamp_int(round(screen_width * left_ratio), 0, screen_width - tile_width)
    step = max(18, int(tile_width * 0.92))
    slots: list[MeldSlot] = []
    max_slots = 16
    for index in range(max_slots):
        slot_left = left + index * step
        if slot_left + tile_width > int(screen_width * 0.995):
            break
        meld_index = min(MAX_SELF_MELDS, 1 + index // 3)
        tile_index = 1 + index % 3
        box = RoiBox(
            name=f"meld_self_{index + 1:02d}",
            left=_clamp_int(slot_left, 0, screen_width - tile_width),
            top=top,
            width=tile_width,
            height=tile_height,
        )
        slots.append(MeldSlot(slot_id=box.name, meld_index=meld_index, tile_index=tile_index, box=box))
    return slots


def _cluster_meld_tiles(accepted: list[dict[str, Any]], _screen_width: int) -> list[dict[str, Any]]:
    if not accepted:
        return []
    ordered = sorted(accepted, key=lambda value: float(value.get("center_x") or 0.0))
    group_count = _estimate_open_meld_count(len(ordered))
    if group_count <= 0:
        return []
    melds: list[dict[str, Any]] = []
    start = 0
    base_size, extra = divmod(len(ordered), group_count)
    for index in range(1, group_count + 1):
        size = base_size + (1 if index <= extra else 0)
        cluster = ordered[start:start + size]
        start += size
        deduped = _dedupe_cluster_tiles(cluster)
        if len(deduped) < 2:
            continue
        melds.append(
            {
                "player": "self",
                "meld_index": index,
                "tiles": deduped,
                "tile_count": len(deduped),
                "confidence": round(
                    sum(float(item.get("confidence") or 0.0) for item in deduped) / max(1, len(deduped)),
                    4,
                ),
                "source": "onnx_tile_classifier",
            }
        )
    return melds


def _meld_left_ratio_for_hand_count(closed_hand_count: int | None) -> float:
    if closed_hand_count is None:
        return 0.70
    count = max(0, int(closed_hand_count or 0))
    if count <= 6:
        return 0.58
    if count <= 8:
        return 0.64
    return 0.70


def _estimate_open_meld_count(recognized_tile_count: int) -> int:
    count = max(0, int(recognized_tile_count or 0))
    if count < 2:
        return 0
    if count <= 4:
        return 1
    if count <= 7:
        return 2
    if count <= 10:
        return 3
    return MAX_SELF_MELDS


def _dedupe_cluster_tiles(cluster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_slot: dict[str, dict[str, Any]] = {}
    for item in cluster:
        tile = str(item.get("tile") or "")
        if not tile:
            continue
        slot_id = str(item.get("slot_id") or "")
        current = by_slot.get(slot_id)
        if current is None or float(item.get("confidence") or 0.0) > float(current.get("confidence") or 0.0):
            by_slot[slot_id] = {
                "tile": tile,
                "confidence": float(item.get("confidence") or 0.0),
                "slot_id": slot_id,
                "box": dict(item.get("box") or {}),
            }
    return sorted(by_slot.values(), key=lambda item: str(item.get("slot_id") or ""))


def _collect_slot_metrics(image: Image.Image, slot: MeldSlot) -> dict[str, Any]:
    metrics = collect_region_metrics(image, slot.box, sample_step=4)
    return {
        "slot_mean_luma": metrics["mean_luma"],
        "slot_bright_ratio": metrics["bright_ratio"],
        "slot_dark_ratio": metrics["dark_ratio"],
        "slot_colorful_ratio": metrics["colorful_ratio"],
        "slot_stddev": metrics["stddev"],
    }


def _is_probably_occupied_meld_slot(slot_metrics: dict[str, Any]) -> bool:
    mean_luma = _float_metric(slot_metrics, "slot_mean_luma")
    bright_ratio = _float_metric(slot_metrics, "slot_bright_ratio")
    dark_ratio = _float_metric(slot_metrics, "slot_dark_ratio")
    stddev = _float_metric(slot_metrics, "slot_stddev")
    return mean_luma >= 90.0 and bright_ratio >= 0.10 and dark_ratio <= 0.72 and stddev >= 24.0


def _base_detection(
    slot: MeldSlot,
    *,
    slot_metrics: dict[str, Any],
    occupied: bool,
) -> dict[str, Any]:
    return {
        "slot_id": slot.slot_id,
        "group": "meld",
        "player": "self",
        "meld_index": slot.meld_index,
        "tile_index": slot.tile_index,
        "candidate_tile": "",
        "confidence": 0.0,
        "box": slot.box.to_dict(),
        "occupied": occupied,
        "source": "meld_state",
        **slot_metrics,
    }


def _meld_match_rejection_reason(tile: str, confidence: float, *, min_confidence: float) -> str:
    if tile == "empty":
        return "empty_tile_class"
    if confidence < min_confidence:
        return "low_confidence"
    return ""


def _crop_box(box: RoiBox, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left = _clamp_int(box.left, 0, max(0, width - 1))
    top = _clamp_int(box.top, 0, max(0, height - 1))
    right = _clamp_int(box.left + box.width, left + 1, width)
    bottom = _clamp_int(box.top + box.height, top + 1, height)
    return (left, top, right, bottom)


def _float_metric(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _clamp_int(value: float | int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
