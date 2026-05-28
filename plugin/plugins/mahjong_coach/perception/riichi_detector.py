from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .roi import RoiBox


@dataclass
class RiichiDetectResult:
    riichi_players: list[str] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    stick_count: int | None = None
    counter_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "riichi_players": list(self.riichi_players),
            "detections": list(self.detections),
            "stick_count": self.stick_count,
            "counter_confidence": round(float(self.counter_confidence), 4),
        }


def detect_riichi_sticks(image_path: Path) -> RiichiDetectResult:
    if not image_path.exists():
        return RiichiDetectResult()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    counter = _detect_riichi_stick_counter(image)
    if counter.get("active"):
        return RiichiDetectResult(
            riichi_players=["unknown"],
            detections=[counter],
            stick_count=_int_or_none(counter.get("count")),
            counter_confidence=float(counter.get("confidence") or 0.0),
        )
    return RiichiDetectResult(
        detections=[counter],
        stick_count=_int_or_none(counter.get("count")),
        counter_confidence=float(counter.get("confidence") or 0.0),
    )


def _detect_riichi_stick_counter(image: Image.Image) -> dict[str, Any]:
    width, height = image.size
    roi = RoiBox(
        "riichi_stick_counter",
        int(width * 0.045),
        int(height * 0.118),
        int(width * 0.055),
        int(height * 0.055),
    ).clipped(width, height)
    crop = image.crop((roi.left, roi.top, roi.right, roi.bottom)).convert("RGB")
    components = _bright_components(crop)
    digit = _select_counter_digit_component(components, crop.size)
    detection: dict[str, Any] = {
        "method": "riichi_stick_counter",
        "box": roi.to_dict(),
        "active": False,
        "count": 0,
        "confidence": 0.0,
        "reason": "digit_not_found",
        "components": components[:8],
    }
    if digit is None:
        return detection

    digit_box = _component_box_to_roi(digit, roi)
    zero_score = _zero_digit_score(crop, digit)
    count = _classify_counter_digit(zero_score, digit)
    confidence = abs(zero_score - 0.5) * 2.0
    active = count != 0
    detection.update(
        {
            "digit_box": digit_box.to_dict(),
            "active": active,
            "count": count,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "zero_score": round(zero_score, 4),
            "reason": "nonzero_stick_counter" if active else "zero_stick_counter",
        }
    )
    return detection


def _bright_components(crop: Image.Image) -> list[dict[str, Any]]:
    arr = np.asarray(crop.convert("RGB"), dtype=np.int16)
    mask = (arr[..., 0] >= 165) & (arr[..., 1] >= 165) & (arr[..., 2] >= 165)
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or not mask[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                cx, cy = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if seen[ny, nx] or not mask[ny, nx]:
                            continue
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            if len(xs) < 20:
                continue
            left = min(xs)
            top = min(ys)
            right = max(xs) + 1
            bottom = max(ys) + 1
            components.append(
                {
                    "area": len(xs),
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "width": right - left,
                    "height": bottom - top,
                }
            )
    return sorted(components, key=lambda item: (int(item["left"]), -int(item["area"])))


def _select_counter_digit_component(components: list[dict[str, Any]], crop_size: tuple[int, int]) -> dict[str, Any] | None:
    crop_width, crop_height = crop_size
    candidates = [
        item
        for item in components
        if int(item.get("height") or 0) >= crop_height * 0.34
        and int(item.get("width") or 0) >= crop_width * 0.08
        and int(item.get("left") or 0) >= crop_width * 0.25
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (int(item.get("left") or 0), int(item.get("area") or 0)))


def _zero_digit_score(crop: Image.Image, component: dict[str, Any]) -> float:
    arr = np.asarray(crop.convert("RGB"), dtype=np.int16)
    left = max(0, int(component.get("left") or 0))
    top = max(0, int(component.get("top") or 0))
    right = min(arr.shape[1], int(component.get("right") or 0))
    bottom = min(arr.shape[0], int(component.get("bottom") or 0))
    if right <= left or bottom <= top:
        return 0.0
    region = arr[top:bottom, left:right]
    mask = (region[..., 0] >= 165) & (region[..., 1] >= 165) & (region[..., 2] >= 165)
    h, w = mask.shape
    if h < 4 or w < 4:
        return 0.0
    mid = mask[int(h * 0.35) : max(int(h * 0.65), int(h * 0.35) + 1)]
    third = max(1, w // 3)
    side_score = (float(mid[:, :third].mean()) + float(mid[:, -third:].mean())) / 2.0
    center_score = float(mid[:, third : max(third + 1, 2 * third)].mean())
    aspect_score = 1.0 - min(1.0, abs((w / max(h, 1)) - 0.62) / 0.62)
    return max(0.0, min(1.0, side_score * 0.45 + (1.0 - center_score) * 0.4 + aspect_score * 0.15))


def _classify_counter_digit(zero_score: float, component: dict[str, Any]) -> int | None:
    if zero_score >= 0.5:
        return 0
    width = max(1, int(component.get("width") or 1))
    height = max(1, int(component.get("height") or 1))
    if width / height <= 0.58:
        return 1
    return None


def _component_box_to_roi(component: dict[str, Any], parent: RoiBox) -> RoiBox:
    left = parent.left + int(component.get("left") or 0)
    top = parent.top + int(component.get("top") or 0)
    width = max(1, int(component.get("width") or 1))
    height = max(1, int(component.get("height") or 1))
    return RoiBox("riichi_stick_counter_digit", left, top, width, height)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
