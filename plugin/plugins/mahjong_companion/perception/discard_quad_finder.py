from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from PIL import Image

from .discard_layout import DiscardSlot
from .roi import RoiBox


Quad = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class DiscardQuadRefinement:
    quad: Quad
    bbox: list[int]
    search_box: RoiBox
    confidence: float
    component_area: int

    @property
    def output_size(self) -> tuple[int, int]:
        left, top, right, bottom = self.bbox
        return (max(1, right - left), max(1, bottom - top))

    def to_dict(self) -> dict[str, Any]:
        return {
            "quad": [[x, y] for x, y in self.quad],
            "bbox": list(self.bbox),
            "search_box": self.search_box.to_dict(),
            "confidence": self.confidence,
            "component_area": self.component_area,
        }


def refine_discard_slot_quad(
    image: Image.Image,
    slot: DiscardSlot,
    *,
    min_confidence: float = 0.34,
) -> DiscardQuadRefinement | None:
    search_box = _expanded_search_box(image, slot)
    crop = image.crop((search_box.left, search_box.top, search_box.right, search_box.bottom)).convert("RGB")
    mask = _tile_face_mask(crop)
    component = _best_component(mask, slot=slot, search_box=search_box)
    if component is None:
        return None

    ys, xs = component
    quad = _quad_from_component(xs=xs, ys=ys, search_box=search_box, image_size=image.size)
    bbox = _bbox_from_quad(quad)
    slot_bbox = (slot.box.left, slot.box.top, slot.box.right, slot.box.bottom)
    slot_area = max(1, slot.box.width * slot.box.height)
    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    overlap = _intersection_area((bbox[0], bbox[1], bbox[2], bbox[3]), slot_bbox)
    if overlap / min(slot_area, bbox_area) < 0.08:
        return None
    confidence = _component_confidence(
        xs=xs,
        ys=ys,
        bbox=bbox,
        slot=slot,
        search_box=search_box,
    )
    if confidence < min_confidence:
        return None

    return DiscardQuadRefinement(
        quad=quad,
        bbox=bbox,
        search_box=search_box,
        confidence=round(confidence, 3),
        component_area=int(len(xs)),
    )


def _expanded_search_box(image: Image.Image, slot: DiscardSlot) -> RoiBox:
    width, height = image.size
    box = slot.box
    scale_x = width / 1920.0
    scale_y = height / 1080.0
    if slot.orientation in {"left", "right"}:
        pad_x = max(18, int(round(36 * scale_x)))
        pad_y = max(12, int(round(24 * scale_y)))
    elif slot.orientation == "top":
        pad_x = max(14, int(round(24 * scale_x)))
        pad_y = max(14, int(round(28 * scale_y)))
    else:
        pad_x = max(10, int(round(18 * scale_x)))
        pad_y = max(10, int(round(18 * scale_y)))

    left = max(0, box.left - pad_x)
    top = max(0, box.top - pad_y)
    right = min(width, box.right + pad_x)
    bottom = min(height, box.bottom + pad_y)
    return RoiBox(
        name=f"{box.name}_quad_search",
        left=left,
        top=top,
        width=max(1, right - left),
        height=max(1, bottom - top),
    )


def _tile_face_mask(crop: Image.Image) -> np.ndarray:
    pixels = np.asarray(crop.convert("RGB"), dtype=np.int16)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    max_channel = np.maximum.reduce([red, green, blue])
    min_channel = np.minimum.reduce([red, green, blue])
    saturation = max_channel - min_channel
    luma = (red * 30 + green * 59 + blue * 11) / 100.0

    neutral_face = (luma >= 132) & (saturation <= 78)
    bright_face = (luma >= 176) & (saturation <= 118)
    orange_back = (red >= 170) & (green >= 95) & (green <= 190) & (blue <= 125) & (red >= blue + 45)
    table_blue = (blue >= red + 28) & (blue >= green + 8) & (luma <= 125)
    return (neutral_face | bright_face) & ~orange_back & ~table_blue


def _best_component(
    mask: np.ndarray,
    *,
    slot: DiscardSlot,
    search_box: RoiBox,
) -> tuple[np.ndarray, np.ndarray] | None:
    if mask.size == 0:
        return None

    visited = np.zeros(mask.shape, dtype=bool)
    minimum_area = max(42, int(slot.box.width * slot.box.height * 0.055))
    slot_local = (
        slot.box.left - search_box.left,
        slot.box.top - search_box.top,
        slot.box.right - search_box.left,
        slot.box.bottom - search_box.top,
    )
    slot_center = (
        (slot_local[0] + slot_local[2]) / 2.0,
        (slot_local[1] + slot_local[3]) / 2.0,
    )

    # Single O(W*H) scan for all True pixels instead of one np.where per component.
    all_true_ys, all_true_xs = np.where(mask)
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for i in range(len(all_true_ys)):
        sy, sx = int(all_true_ys[i]), int(all_true_xs[i])
        if visited[sy, sx]:
            continue
        xs, ys = _collect_component(mask, visited, sx, sy)
        area = len(xs)
        if area < minimum_area:
            continue
        left = int(xs.min())
        right = int(xs.max()) + 1
        top = int(ys.min())
        bottom = int(ys.max()) + 1
        if right - left < max(8, int(slot.box.width * 0.22)):
            continue
        if bottom - top < max(8, int(slot.box.height * 0.22)):
            continue

        overlap = _intersection_area((left, top, right, bottom), slot_local)
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        distance = math.dist((center_x, center_y), slot_center)
        score = area + overlap * 1.8 - distance * 5.0
        if best is None or score > best[0]:
            best = (score, xs, ys)

    if best is None:
        return None
    return (best[2], best[1])


def _collect_component(
    mask: np.ndarray,
    visited: np.ndarray,
    start_x: int,
    start_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    stack = [(start_x, start_y)]
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

    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)


def _quad_from_component(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    search_box: RoiBox,
    image_size: tuple[int, int],
) -> Quad:
    top = float(np.percentile(ys, 4))
    bottom = float(np.percentile(ys, 96))
    height = max(1.0, bottom - top)
    top_band = ys <= top + height * 0.28
    bottom_band = ys >= bottom - height * 0.28
    if int(top_band.sum()) < 6:
        top_band = ys <= float(np.percentile(ys, 22))
    if int(bottom_band.sum()) < 6:
        bottom_band = ys >= float(np.percentile(ys, 78))

    margin = max(2, int(round(min(image_size) * 0.0025)))
    top_left_x = float(np.percentile(xs[top_band], 3)) - margin
    top_right_x = float(np.percentile(xs[top_band], 97)) + margin
    bottom_left_x = float(np.percentile(xs[bottom_band], 3)) - margin
    bottom_right_x = float(np.percentile(xs[bottom_band], 97)) + margin
    top_y = top - margin
    bottom_y = bottom + margin

    image_width, image_height = image_size
    return (
        (_clamp_int(round(search_box.left + top_left_x), 0, image_width - 1), _clamp_int(round(search_box.top + top_y), 0, image_height - 1)),
        (_clamp_int(round(search_box.left + bottom_left_x), 0, image_width - 1), _clamp_int(round(search_box.top + bottom_y), 0, image_height - 1)),
        (_clamp_int(round(search_box.left + bottom_right_x), 0, image_width - 1), _clamp_int(round(search_box.top + bottom_y), 0, image_height - 1)),
        (_clamp_int(round(search_box.left + top_right_x), 0, image_width - 1), _clamp_int(round(search_box.top + top_y), 0, image_height - 1)),
    )


def _component_confidence(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    bbox: list[int],
    slot: DiscardSlot,
    search_box: RoiBox,
) -> float:
    left, top, right, bottom = bbox
    bbox_area = max(1, (right - left) * (bottom - top))
    slot_area = max(1, slot.box.width * slot.box.height)
    fill_ratio = min(1.0, len(xs) / bbox_area)
    size_ratio = min(1.0, len(xs) / (slot_area * 0.38))
    component_center = (
        search_box.left + (float(xs.min()) + float(xs.max())) / 2.0,
        search_box.top + (float(ys.min()) + float(ys.max())) / 2.0,
    )
    slot_center = ((slot.box.left + slot.box.right) / 2.0, (slot.box.top + slot.box.bottom) / 2.0)
    max_distance = max(slot.box.width, slot.box.height, 1)
    distance_score = max(0.0, 1.0 - math.dist(component_center, slot_center) / max_distance)
    overlap = _intersection_area((left, top, right, bottom), (slot.box.left, slot.box.top, slot.box.right, slot.box.bottom))
    overlap_score = min(1.0, overlap / min(slot_area, bbox_area))
    return min(1.0, 0.18 + fill_ratio * 0.22 + size_ratio * 0.26 + distance_score * 0.18 + overlap_score * 0.16)


def _bbox_from_quad(quad: Quad) -> list[int]:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def _intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0, right - left) * max(0, bottom - top)


def _clamp_int(value: int | float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
