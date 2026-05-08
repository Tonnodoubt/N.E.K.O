"""Theme-agnostic tile bounding-box detector.

Replaces the ``hardcoded grid + brightness gate`` discovery step with a
classical CV pipeline that locates every tile-shaped bright rectangle in
a screenshot. Output bboxes can then be fed into the existing ONNX
classifier without any per-theme calibration.

Pipeline::

    grayscale → adaptive threshold → morph close → contours
        → filter by (area, aspect, solidity, fill) → NMS → boxes

The detector is **orientation-aware**: it returns each box together with
a coarse orientation hint (``"upright"`` for self / top opponent,
``"sideways"`` for left / right opponents) inferred from the box aspect
ratio. Downstream code can use this to pick the right rotation before
classification.

This module has no dependency on the existing layout / calibration
system so it can be developed and evaluated standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Public types


@dataclass(frozen=True)
class TileBox:
    """One detected tile bounding box in image pixel coordinates."""

    left: int
    top: int
    right: int
    bottom: int
    orientation: str  # "upright" (taller than wide) or "sideways"
    score: float  # 0..1 — how tile-like the contour was

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    def as_bbox(self) -> list[int]:
        return [self.left, self.top, self.right, self.bottom]


@dataclass
class DetectionParams:
    """Tunable knobs. Defaults work on 1920×1080 雀魂 screenshots and most themes."""

    # Tile faces are typically very bright (cream/white) versus the table.
    # We binarize with adaptive threshold to handle uneven lighting.
    adaptive_block_size: int = 41
    adaptive_C: int = -10  # negative C → bright pixels become foreground

    # Morphology to merge tile interior characters with the tile body.
    morph_close_kernel: int = 5

    # Tile area as a fraction of total image area. 1920×1080 frame has
    # ~2.07M px; a self-hand tile is ~58×80=4640 px ≈ 0.22% of frame.
    # Discard tiles for the top opponent are smaller (~30×40=1200 ≈ 0.06%).
    min_area_frac: float = 0.0003  # >= ~620 px on 1920×1080
    max_area_frac: float = 0.015  # <= ~31000 px on 1920×1080 (self hand draw)

    # Tile aspect ratio (height / width).
    # Upright: ~1.4. Sideways (rotated 90°): ~0.7. Allow a wide tolerance
    # because perspective and crops shift this.
    upright_aspect_min: float = 1.10
    upright_aspect_max: float = 1.85
    sideways_aspect_min: float = 0.55
    sideways_aspect_max: float = 0.92

    # Solidity = contour area / convex hull area. Tiles are nearly
    # rectangular so this should be very high.
    min_solidity: float = 0.85

    # Fill ratio = contour area / bbox area. Same reasoning.
    min_fill: float = 0.70

    # NMS IoU threshold for de-duplicating nested detections.
    nms_iou: float = 0.30


# ---------------------------------------------------------------------------
# Public API


def detect_tiles(
    image: Image.Image | np.ndarray,
    *,
    params: DetectionParams | None = None,
) -> list[TileBox]:
    """Detect every tile-shaped bounding box in ``image``.

    Accepts either a PIL ``Image`` (RGB) or a HxWx3 / HxWx4 numpy array
    (which will be converted to BGR for OpenCV). Returns a list of
    :class:`TileBox` sorted top-to-bottom then left-to-right.
    """
    p = params or DetectionParams()
    bgr = _to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        _ensure_odd(p.adaptive_block_size),
        p.adaptive_C,
    )
    if p.morph_close_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (p.morph_close_kernel, p.morph_close_kernel),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_h, image_w = gray.shape
    image_area = float(image_h * image_w)
    min_area = p.min_area_frac * image_area
    max_area = p.max_area_frac * image_area

    candidates: list[TileBox] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        bbox_area = float(w * h)
        fill = area / bbox_area
        if fill < p.min_fill:
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        solidity = area / hull_area
        if solidity < p.min_solidity:
            continue
        aspect = h / float(w)
        orientation = _classify_orientation(aspect, p)
        if orientation is None:
            continue
        score = _tile_score(fill=fill, solidity=solidity, aspect=aspect, orientation=orientation, p=p)
        candidates.append(
            TileBox(
                left=int(x),
                top=int(y),
                right=int(x + w),
                bottom=int(y + h),
                orientation=orientation,
                score=score,
            )
        )

    deduped = _non_max_suppress(candidates, iou_threshold=p.nms_iou)
    deduped.sort(key=lambda b: (b.top, b.left))
    return deduped


# ---------------------------------------------------------------------------
# Internal helpers


def _to_bgr(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if image.shape[2] == 3:
            # Assume incoming numpy is RGB (PIL convention), convert to BGR.
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        raise ValueError(f"unsupported numpy image shape: {image.shape}")
    pil = image.convert("RGB")
    arr = np.asarray(pil, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _ensure_odd(value: int) -> int:
    if value < 3:
        return 3
    return value if value % 2 == 1 else value + 1


def _classify_orientation(aspect: float, p: DetectionParams) -> str | None:
    if p.upright_aspect_min <= aspect <= p.upright_aspect_max:
        return "upright"
    if p.sideways_aspect_min <= aspect <= p.sideways_aspect_max:
        return "sideways"
    return None


def _tile_score(*, fill: float, solidity: float, aspect: float, orientation: str, p: DetectionParams) -> float:
    """Combined 0..1 score expressing how tile-like a contour is."""
    # Fill and solidity each map [min, 1.0] → [0, 1].
    fill_score = max(0.0, (fill - p.min_fill) / (1.0 - p.min_fill))
    solidity_score = max(0.0, (solidity - p.min_solidity) / (1.0 - p.min_solidity))
    if orientation == "upright":
        target = (p.upright_aspect_min + p.upright_aspect_max) / 2.0
        half_range = (p.upright_aspect_max - p.upright_aspect_min) / 2.0
    else:
        target = (p.sideways_aspect_min + p.sideways_aspect_max) / 2.0
        half_range = (p.sideways_aspect_max - p.sideways_aspect_min) / 2.0
    aspect_score = max(0.0, 1.0 - abs(aspect - target) / max(half_range, 1e-6))
    return round(0.45 * fill_score + 0.35 * solidity_score + 0.20 * aspect_score, 4)


def _non_max_suppress(boxes: list[TileBox], *, iou_threshold: float) -> list[TileBox]:
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: list[TileBox] = []
    for cand in sorted_boxes:
        if any(_iou(cand, k) >= iou_threshold for k in kept):
            continue
        kept.append(cand)
    return kept


def _iou(a: TileBox, b: TileBox) -> float:
    x1 = max(a.left, b.left)
    y1 = max(a.top, b.top)
    x2 = min(a.right, b.right)
    y2 = min(a.bottom, b.bottom)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return float(inter) / float(union) if union else 0.0


# ---------------------------------------------------------------------------
# Optional: detection summary for debugging


def detection_summary(boxes: list[TileBox]) -> dict[str, Any]:
    upright = [b for b in boxes if b.orientation == "upright"]
    sideways = [b for b in boxes if b.orientation == "sideways"]
    return {
        "total": len(boxes),
        "upright": len(upright),
        "sideways": len(sideways),
        "avg_score": round(float(np.mean([b.score for b in boxes])) if boxes else 0.0, 3),
        "size_stats": {
            "median_w": int(np.median([b.width for b in boxes])) if boxes else 0,
            "median_h": int(np.median([b.height for b in boxes])) if boxes else 0,
        },
    }


# ---------------------------------------------------------------------------
# Band-based tile segmentation (P0-5)
#
# Instead of hardcoded grid steps, scan a known discard row / column with an
# ivory projection profile and locate each tile by its peak.  This eliminates
# the systematic offset that affects opponent discard piles when the fixed
# grid parameters are wrong for the current theme / resolution.


@dataclass(frozen=True)
class TileBandBox:
    """One tile located inside a band via ivory-profile scanning."""

    left: int
    top: int
    right: int
    bottom: int
    peak_position: float  # band-local centre of the ivory peak
    confidence: float     # 0..1 — peak strength relative to expected peak

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def find_tiles_in_band(
    image: Image.Image,
    band_left: int,
    band_top: int,
    band_width: int,
    band_height: int,
    *,
    orientation: str = "horizontal",
    expected_count: int | None = None,
    tile_length: int | None = None,
    min_peak_ratio: float = 0.15,
) -> list[TileBandBox]:
    """Locate individual tiles inside a known discard row or column.

    ``orientation`` is ``"horizontal"`` for self / top opponent rows
    (ivory summed per column) and ``"vertical"`` for left / right opponent
    columns (ivory summed per row).

    When ``expected_count`` is given, only the strongest *N* peaks are
    returned (ties broken by prominence).  ``tile_length``, when provided,
    is used as the half-peak-separation distance; otherwise it is estimated
    as ``band_width / max(expected_count, 1)``.

    Returns tile boxes in **full-image pixel coordinates**, sorted
    left-to-right (horizontal) or top-to-bottom (vertical).
    """
    crop = image.crop((band_left, band_top, band_left + band_width, band_top + band_height))
    arr = np.asarray(crop.convert("RGB"), dtype=np.int16)

    # Ivory predicate – same colour space as hand_baseline detection.
    ivory = (
        (arr[..., 0] >= 165)
        & (arr[..., 1] >= 150)
        & (arr[..., 2] >= 110)
        & (arr[..., 0] >= arr[..., 2] - 5)
    )

    if orientation == "horizontal":
        signal = ivory.mean(axis=0).astype(np.float64)  # shape (band_width,)
    else:
        signal = ivory.mean(axis=1).astype(np.float64)  # shape (band_height,)

    signal = _smooth_1d(signal, sigma=1.5)

    band_len = band_width if orientation == "horizontal" else band_height
    half_sep = tile_length // 2 if tile_length else max(3, band_len // 24)

    peaks = _find_peaks(signal, min_height=min_peak_ratio, min_distance=half_sep)
    if not peaks:
        return []

    boxes: list[TileBandBox] = []
    for peak_idx, peak_val in peaks:
        left_edge = _find_boundary(signal, peak_idx, direction=-1, threshold_ratio=0.4)
        right_edge = _find_boundary(signal, peak_idx, direction=+1, threshold_ratio=0.4)
        confidence = round(float(min(peak_val / max(min_peak_ratio, 1e-6), 1.0)), 4)

        if orientation == "horizontal":
            box = TileBandBox(
                left=band_left + left_edge,
                top=band_top,
                right=band_left + right_edge,
                bottom=band_top + band_height,
                peak_position=float(peak_idx),
                confidence=confidence,
            )
        else:
            box = TileBandBox(
                left=band_left,
                top=band_top + left_edge,
                right=band_left + band_width,
                bottom=band_top + right_edge,
                peak_position=float(peak_idx),
                confidence=confidence,
            )
        boxes.append(box)

    boxes = _deduplicate_overlapping(boxes)
    if expected_count and len(boxes) > expected_count:
        boxes.sort(key=lambda b: b.confidence, reverse=True)
        boxes = boxes[:expected_count]
        boxes.sort(key=lambda b: (b.top, b.left))
    return boxes


def _smooth_1d(signal: np.ndarray, *, sigma: float) -> np.ndarray:
    """Gaussian-smoothed copy of a 1-D signal.  Length is preserved."""
    from math import exp

    radius = max(1, int(round(sigma * 3)))
    kernel = np.array([
        exp(-0.5 * (i / sigma) ** 2) for i in range(-radius, radius + 1)
    ], dtype=np.float64)
    kernel /= kernel.sum()
    padded = np.pad(signal, (radius, radius), mode="edge")
    convolved = np.convolve(padded, kernel, mode="same")
    return convolved[radius:-radius]


def _find_peaks(
    signal: np.ndarray,
    *,
    min_height: float,
    min_distance: int,
) -> list[tuple[int, float]]:
    """Return ``(index, height)`` for each local maximum exceeding
    ``min_height`` and separated by at least ``min_distance``."""
    n = int(signal.shape[0])
    peaks: list[tuple[int, float]] = []
    for i in range(1, n - 1):
        if signal[i] <= min_height:
            continue
        if signal[i] <= signal[i - 1] or signal[i] < signal[i + 1]:
            continue
        if peaks and i - peaks[-1][0] < min_distance:
            if signal[i] > peaks[-1][1]:
                peaks[-1] = (i, float(signal[i]))
            continue
        peaks.append((i, float(signal[i])))
    return peaks


def _find_boundary(
    signal: np.ndarray,
    peak_idx: int,
    *,
    direction: int,
    threshold_ratio: float,
) -> int:
    """Walk from ``peak_idx`` in ``direction`` (-1 left, +1 right) until
    the signal drops below ``threshold_ratio * peak_value``, then return
    the boundary index (clamped to the array edge)."""
    peak_val = signal[peak_idx]
    threshold = peak_val * threshold_ratio
    n = int(signal.shape[0])
    step = 1 if direction > 0 else -1
    i = peak_idx
    while 0 < i < n - 1:
        if signal[i] < threshold:
            break
        i += step
    return max(0, min(n, i))


def _deduplicate_overlapping(boxes: list[TileBandBox]) -> list[TileBandBox]:
    """Remove boxes whose centres fall within the bounds of a
    higher-confidence box.  Keeps the strongest per cluster."""
    if len(boxes) <= 1:
        return boxes
    sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: list[TileBandBox] = []
    for cand in sorted_boxes:
        if any(_band_box_contains(k, cand) for k in kept):
            continue
        kept.append(cand)
    kept.sort(key=lambda b: (b.top, b.left))
    return kept


def _band_box_contains(outer: TileBandBox, inner: TileBandBox) -> bool:
    """True if ``inner``'s centre lies inside ``outer``."""
    cx = (inner.left + inner.right) // 2
    cy = (inner.top + inner.bottom) // 2
    return outer.left <= cx <= outer.right and outer.top <= cy <= outer.bottom


# ---------------------------------------------------------------------------
# Discard layout refinement (P0-5 integration)
# ---------------------------------------------------------------------------


def refine_discard_layout_with_bands(
    image: Image.Image,
    layout: dict[str, list[Any]],
    *,
    max_shift_px: int = 20,
) -> dict[str, list[Any]]:
    """Refine discard-slot quads using ivory-profile band scanning.

    For each player's discard rows (self / top) or columns (left / right),
    scans the ivory profile with :func:`find_tiles_in_band`, then snaps
    each layout slot to the nearest detected tile centre.

    Only slots whose nearest detected tile is within *max_shift_px* of the
    expected centre get their quad updated.  Others keep their original
    geometry.

    Returns a shallow copy of *layout* with updated ``DiscardSlot``
    objects for refined slots.
    """
    from .discard_layout import DISCARD_PLAYERS

    refined: dict[str, list[Any]] = {}
    for player in DISCARD_PLAYERS:
        if player not in layout:
            refined[player] = layout[player]
            continue
        slots = list(layout[player])
        if not slots:
            refined[player] = slots
            continue
        first = slots[0]
        orient = "horizontal" if first.orientation in ("bottom", "top") else "vertical"
        # Reconstruct the grid from anchor-derived origin + step.
        spec = _discard_grid_spec(slots, orient)
        if spec is None:
            refined[player] = slots
            continue

        # Scan each row (horizontal) or column (vertical).
        all_boxes: list[TileBandBox] = []
        if orient == "horizontal":
            for row in range(spec["rows"]):
                row_top = spec["origin_top"] + row * spec["step_y"]
                boxes = find_tiles_in_band(
                    image,
                    spec["origin_left"],
                    row_top - spec["tile_height"] // 4,
                    spec["band_width"],
                    spec["tile_height"] * 3 // 2,
                    orientation="horizontal",
                    expected_count=spec["columns"],
                    tile_length=spec["tile_width"],
                )
                # Tag each box so we know which row it belongs to.
                for b in boxes:
                    all_boxes.append(b)
        else:
            for col in range(spec["columns"]):
                col_left = spec["origin_left"] + col * spec["step_x"]
                boxes = find_tiles_in_band(
                    image,
                    col_left - spec["tile_width"] // 4,
                    spec["origin_top"],
                    spec["tile_width"] * 3 // 2,
                    spec["band_height"],
                    orientation="vertical",
                    expected_count=spec["rows"],
                    tile_length=spec["tile_height"],
                )
                for b in boxes:
                    all_boxes.append(b)

        if not all_boxes:
            refined[player] = slots
            continue

        # Match each detected box to the nearest slot.
        refined_slots = []
        for slot in slots:
            expected_cx = (slot.box.left + slot.box.right) // 2
            expected_cy = (slot.box.top + slot.box.bottom) // 2
            best_dist = float("inf")
            best_box: TileBandBox | None = None
            for b in all_boxes:
                bcx = (b.left + b.right) // 2
                bcy = (b.top + b.bottom) // 2
                dist = abs(bcx - expected_cx) + abs(bcy - expected_cy)
                if dist < best_dist:
                    best_dist = dist
                    best_box = b
            if best_box is not None and best_dist <= max_shift_px:
                b = best_box
                new_quad = (
                    (b.left, b.top),
                    (b.left, b.bottom),
                    (b.right, b.bottom),
                    (b.right, b.top),
                )
                refined_slots.append(_discard_slot_with_quad(slot, new_quad))
            else:
                refined_slots.append(slot)
        refined[player] = refined_slots

    # Carry over players not in DISCARD_PLAYERS.
    for player in layout:
        if player not in refined:
            refined[player] = layout[player]
    return refined


def _discard_grid_spec(
    slots: list[Any],
    orient: str,
) -> dict[str, int] | None:
    """Infer the grid parameters from a list of DiscardSlot objects."""
    if not slots:
        return None
    first = slots[0]
    last = slots[-1]
    if orient == "horizontal":
        columns = len({s.box.top for s in slots})  # unique row tops
        if columns == 0:
            return None
        rows = len(slots) // columns if columns else 0
        return {
            "origin_left": first.box.left,
            "origin_top": first.box.top,
            "tile_width": first.box.width,
            "tile_height": first.box.height,
            "step_x": (last.box.left - first.box.left) // max((len(slots) // max(rows, 1)) - 1, 1) if len(slots) > rows else 64,
            "step_y": (slots[rows].box.top - first.box.top) if rows > 0 and len(slots) > rows else 70,
            "columns": columns,
            "rows": rows,
            "band_width": last.box.right - first.box.left + first.box.width,
        }
    else:
        rows = len({s.box.left for s in slots})
        if rows == 0:
            return None
        columns = len(slots) // rows if rows else 0
        return {
            "origin_left": first.box.left,
            "origin_top": first.box.top,
            "tile_width": first.box.width,
            "tile_height": first.box.height,
            "step_x": (slots[rows].box.left - first.box.left) if rows > 0 and len(slots) > rows else 82,
            "step_y": (last.box.top - first.box.top) // max((len(slots) // max(columns, 1)) - 1, 1) if len(slots) > columns else 62,
            "columns": columns,
            "rows": rows,
            "band_height": last.box.bottom - first.box.top + first.box.height,
        }


def _discard_slot_with_quad(slot: Any, quad: tuple) -> Any:
    """Return a new DiscardSlot with *quad* replacing the original."""
    from .discard_layout import DiscardSlot
    from .roi import RoiBox

    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    new_box = RoiBox(
        name=slot.box.name,
        left=min(xs),
        top=min(ys),
        width=max(xs) - min(xs),
        height=max(ys) - min(ys),
    )
    return DiscardSlot(
        slot_id=slot.slot_id,
        player=slot.player,
        turn_index=slot.turn_index,
        orientation=slot.orientation,
        box=new_box,
        quad=quad,
    )
