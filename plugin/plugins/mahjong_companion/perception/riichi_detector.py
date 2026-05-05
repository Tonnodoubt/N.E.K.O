from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ..tile_labels import dedupe as _dedupe
from .roi import RoiBox


BASE_WIDTH = 1920
BASE_HEIGHT = 1080


@dataclass(frozen=True)
class RiichiStickDetection:
    player: str
    bbox: list[int]
    confidence: float
    orientation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": "riichi_stick",
            "player": self.player,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "orientation": self.orientation,
            "source": "riichi_stick_detector",
        }


@dataclass(frozen=True)
class _StickSpec:
    player: str
    box: tuple[int, int, int, int]
    orientation: str


_STICK_SPECS = (
    _StickSpec("top_opponent", (875, 312, 1030, 350), "horizontal"),
    _StickSpec("self", (860, 506, 1042, 540), "horizontal"),
    _StickSpec("left_opponent", (805, 330, 850, 485), "vertical"),
    _StickSpec("right_opponent", (1072, 330, 1122, 485), "vertical"),
)


def detect_riichi_sticks(image: Image.Image) -> list[RiichiStickDetection]:
    detections: list[RiichiStickDetection] = []
    for spec in _STICK_SPECS:
        box = _scaled_box(spec, width=image.width, height=image.height)
        crop = image.crop((box.left, box.top, box.right, box.bottom)).convert("RGB")
        detection = _detect_stick_in_crop(crop, spec=spec, origin=(box.left, box.top))
        if detection is not None:
            detections.append(detection)
    return detections


def detect_riichi_players(image: Image.Image) -> tuple[list[str], list[dict[str, Any]]]:
    detections = detect_riichi_sticks(image)
    players = _dedupe([item.player for item in detections])
    return players, [item.to_dict() for item in detections]


def _detect_stick_in_crop(
    crop: Image.Image,
    *,
    spec: _StickSpec,
    origin: tuple[int, int],
) -> RiichiStickDetection | None:
    pixels = np.asarray(crop, dtype=np.int16)
    if pixels.size == 0:
        return None

    white_mask = _white_mask(pixels)
    red_mask = _red_mask(pixels)
    best: tuple[float, list[int]] | None = None
    for component in _connected_components(white_mask):
        bbox = _component_bbox(component)
        if bbox is None:
            continue
        score = _stick_shape_score(
            bbox,
            area=int(len(component[0])),
            crop_size=crop.size,
            orientation=spec.orientation,
        )
        if score <= 0.0:
            continue
        red_score = _red_dot_score(red_mask, bbox)
        if red_score <= 0.0:
            continue
        confidence = min(0.98, 0.52 + score * 0.26 + red_score * 0.20)
        if best is None or confidence > best[0]:
            best = (confidence, bbox)

    if best is None:
        return None

    confidence, local_bbox = best
    bbox = [
        origin[0] + local_bbox[0],
        origin[1] + local_bbox[1],
        origin[0] + local_bbox[2],
        origin[1] + local_bbox[3],
    ]
    return RiichiStickDetection(
        player=spec.player,
        bbox=bbox,
        confidence=round(confidence, 3),
        orientation=spec.orientation,
    )


def _white_mask(pixels: np.ndarray) -> np.ndarray:
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    max_channel = np.maximum.reduce([red, green, blue])
    min_channel = np.minimum.reduce([red, green, blue])
    luma = (red * 30 + green * 59 + blue * 11) / 100.0
    saturation = max_channel - min_channel
    return (luma >= 176) & (saturation <= 74)


def _red_mask(pixels: np.ndarray) -> np.ndarray:
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    return (red >= 160) & (green <= 95) & (blue <= 95) & (red >= green + 70) & (red >= blue + 70)


def _connected_components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[np.ndarray, np.ndarray]] = []
    height, width = mask.shape
    for start_y, start_x in zip(*np.where(mask & ~visited), strict=True):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_x), int(start_y))]
        xs: list[int] = []
        ys: list[int] = []
        visited[start_y, start_x] = True
        while stack:
            x, y = stack.pop()
            xs.append(x)
            ys.append(y)
            for next_x, next_y in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if next_x < 0 or next_y < 0 or next_x >= width or next_y >= height:
                    continue
                if visited[next_y, next_x] or not mask[next_y, next_x]:
                    continue
                visited[next_y, next_x] = True
                stack.append((next_x, next_y))
        if len(xs) >= 16:
            components.append((np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)))
    return components


def _component_bbox(component: tuple[np.ndarray, np.ndarray]) -> list[int] | None:
    xs, ys = component
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _stick_shape_score(
    bbox: list[int],
    *,
    area: int,
    crop_size: tuple[int, int],
    orientation: str,
) -> float:
    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    crop_width, crop_height = crop_size
    fill = min(1.0, area / max(1, width * height))
    if orientation == "horizontal":
        long_ratio = width / max(1, crop_width)
        thin_ratio = height / max(1, crop_height)
        aspect = width / height
        if long_ratio < 0.42 or thin_ratio > 0.70 or aspect < 3.2:
            return 0.0
        return min(1.0, long_ratio * 0.45 + (1.0 - thin_ratio) * 0.25 + min(1.0, aspect / 6.0) * 0.20 + fill * 0.10)

    long_ratio = height / max(1, crop_height)
    thin_ratio = width / max(1, crop_width)
    aspect = height / width
    if long_ratio < 0.42 or thin_ratio > 0.75 or aspect < 2.8:
        return 0.0
    return min(1.0, long_ratio * 0.45 + (1.0 - thin_ratio) * 0.25 + min(1.0, aspect / 6.0) * 0.20 + fill * 0.10)


def _red_dot_score(red_mask: np.ndarray, bbox: list[int]) -> float:
    left, top, right, bottom = bbox
    height, width = red_mask.shape
    pad_x = max(3, (right - left) // 10)
    pad_y = max(3, (bottom - top) // 2)
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)
    red_count = int(red_mask[top:bottom, left:right].sum())
    if red_count < 8:
        return 0.0
    expected = max(18, ((right - left) * (bottom - top)) // 120)
    return min(1.0, red_count / expected)


def _scaled_box(spec: _StickSpec, *, width: int, height: int) -> RoiBox:
    scale_x = width / BASE_WIDTH
    scale_y = height / BASE_HEIGHT
    left, top, right, bottom = spec.box
    scaled_left = _clamp_int(round(left * scale_x), 0, width - 1)
    scaled_top = _clamp_int(round(top * scale_y), 0, height - 1)
    scaled_right = _clamp_int(round(right * scale_x), scaled_left + 1, width)
    scaled_bottom = _clamp_int(round(bottom * scale_y), scaled_top + 1, height)
    return RoiBox(
        name=f"riichi_stick_{spec.player}",
        left=scaled_left,
        top=scaled_top,
        width=scaled_right - scaled_left,
        height=scaled_bottom - scaled_top,
    )


def _clamp_int(value: int | float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
