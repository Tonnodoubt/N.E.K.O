from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from .hand_baseline import HandBaselineAnchor, detect_hand_baseline
from .tile_classifier_dispatch import classify_tiles_batch
from .tile_detector import DetectionParams, TileBox, detect_tiles


@dataclass(frozen=True)
class BottomHandSlot:
    slot_id: str
    tile: str
    confidence: float
    accepted: bool
    bbox: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "tile": self.tile,
            "confidence": round(self.confidence, 4),
            "accepted": self.accepted,
            "bbox": list(self.bbox),
        }


@dataclass(frozen=True)
class BottomHandDetection:
    hand_tiles: list[str] = field(default_factory=list)
    slots: list[BottomHandSlot] = field(default_factory=list)
    anchor: HandBaselineAnchor | None = None
    confidence: float = 0.0
    source: str = "bottom_hand_detector"
    error: str = ""

    def analysis_hints(self) -> dict[str, object]:
        hints: dict[str, object] = {
            "tile_parser_source": self.source,
            "tile_level_available": bool(self.hand_tiles),
            "recognized_hand_tile_count": len(self.hand_tiles),
            "analysis_confidence": round(self.confidence, 4),
            "bottom_hand_raw_slots": [slot.to_dict() for slot in self.slots],
        }
        if self.anchor is not None:
            hints["bottom_hand_anchor"] = self.anchor.to_dict()
        if self.error:
            hints["bottom_hand_error"] = self.error
        return hints


def detect_bottom_hand_tiles(image: Image.Image) -> BottomHandDetection:
    rgb = image.convert("RGB")
    anchor = detect_hand_baseline(rgb)
    if anchor is None:
        return BottomHandDetection(source="bottom_hand_detector_no_anchor")

    boxes = _bottom_hand_boxes(rgb, anchor)
    if not boxes:
        return BottomHandDetection(anchor=anchor, source="bottom_hand_detector_no_boxes")

    matches = classify_tiles_batch(
        [rgb.crop((box.left, box.top, box.right, box.bottom)) for box in boxes],
        {},
    )
    slots: list[BottomHandSlot] = []
    hand_tiles: list[str] = []
    confidences: list[float] = []
    for index, (box, match) in enumerate(zip(boxes, matches, strict=True), start=1):
        tile = str(match.tile) if match is not None else ""
        confidence = float(match.confidence) if match is not None else 0.0
        accepted = bool(match is not None and match.confidence >= 0.55)
        slots.append(
            BottomHandSlot(
                slot_id=f"bottom_hand_{index}",
                tile=tile,
                confidence=confidence,
                accepted=accepted,
                bbox=[box.left, box.top, box.right, box.bottom],
            )
        )
        if accepted:
            hand_tiles.append(tile)
            confidences.append(confidence)

    if not hand_tiles:
        return BottomHandDetection(anchor=anchor, slots=slots, source="bottom_hand_detector_no_match")

    mean_confidence = sum(confidences) / max(1, len(confidences))
    return BottomHandDetection(
        hand_tiles=hand_tiles,
        slots=slots,
        anchor=anchor,
        confidence=round(mean_confidence, 4),
    )


def _bottom_hand_boxes(image: Image.Image, anchor: HandBaselineAnchor) -> list[TileBox]:
    params = DetectionParams(
        adaptive_C=-5,
        morph_close_kernel=3,
        min_area_frac=0.0008,
        max_area_frac=0.008,
        min_fill=0.55,
        min_solidity=0.75,
        nms_iou=0.20,
    )
    boxes = [
        box
        for box in detect_tiles(image, params=params)
        if box.top >= anchor.top_y - 20
        and box.bottom <= image.height
        and box.left >= anchor.left_x - 80
        and box.right <= anchor.right_x + 80
        and box.width >= 36
        and box.height >= 54
    ]
    boxes.sort(key=lambda item: item.left)
    return boxes
