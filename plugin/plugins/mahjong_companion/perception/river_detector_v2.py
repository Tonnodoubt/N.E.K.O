from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np
from PIL import Image


RIVER_PLAYERS = ("self", "left_opponent", "top_opponent", "right_opponent")
RIVER_ORIENTATIONS = {
    "self": "bottom",
    "left_opponent": "left",
    "top_opponent": "top",
    "right_opponent": "right",
}
MIN_SIDE_CLASSIFICATION_CANDIDATE_CONFIDENCE = 0.45
MIN_RIVER_TILE_CLASSIFICATION_CONFIDENCE = 0.50
Quad = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
SurfaceBox = tuple[int, int, int, int, float, Quad | None]


@dataclass(frozen=True)
class RiverRoi:
    player: str
    left: int
    top: int
    right: int
    bottom: int
    order: str

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "bbox": [self.left, self.top, self.right, self.bottom],
            "order": self.order,
        }


@dataclass(frozen=True)
class RiverTileCandidate:
    player: str
    order_index: int
    bbox: tuple[int, int, int, int]
    quad: Quad
    center: tuple[int, int]
    confidence: float
    source: str = "river_detector_v2"

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "order_index": self.order_index,
            "bbox": list(self.bbox),
            "quad": [[x, y] for x, y in self.quad],
            "quad_order": "upper_left,lower_left,lower_right,upper_right",
            "center": list(self.center),
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class RiverDetectionResult:
    candidates: list[RiverTileCandidate] = field(default_factory=list)
    rois: list[RiverRoi] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)

    @property
    def by_player(self) -> dict[str, list[RiverTileCandidate]]:
        grouped: dict[str, list[RiverTileCandidate]] = {player: [] for player in RIVER_PLAYERS}
        for candidate in self.candidates:
            grouped.setdefault(candidate.player, []).append(candidate)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        grouped = {
            player: [candidate.to_dict() for candidate in candidates]
            for player, candidates in self.by_player.items()
        }
        return {
            "source": "river_detector_v2",
            "image_size": list(self.image_size),
            "candidate_count": len(self.candidates),
            "rois": [roi.to_dict() for roi in self.rois],
            "discard_piles": grouped,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class RiverDetectorParams:
    min_area_ratio: float = 0.00045
    max_area_ratio: float = 0.0032
    min_fill_ratio: float = 0.33
    min_aspect_ratio: float = 0.42
    max_aspect_ratio: float = 1.45
    max_component_width_ratio: float = 0.055
    max_component_height_ratio: float = 0.085
    nms_iou_threshold: float = 0.18
    cluster_axis_tolerance_ratio: float = 0.034
    pad_ratio: float = 0.004


def detect_river_tiles_v2(
    image: Image.Image,
    *,
    params: RiverDetectorParams | None = None,
) -> RiverDetectionResult:
    """Detect visible discard-river tile boxes without using the old fixed grid.

    The detector is intentionally class-agnostic. It first defines broad river
    regions around the table center, then uses an ivory/white face mask inside
    each region to find tile-sized connected components. Classification is a
    separate downstream step.
    """
    p = params or RiverDetectorParams()
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    arr = np.asarray(rgb_image, dtype=np.uint8)
    rois = build_river_rois(width, height)
    all_candidates: list[RiverTileCandidate] = []
    image_area = float(width * height)

    for roi in rois:
        raw_boxes = _detect_roi_boxes(arr, roi, image_area=image_area, params=p)
        ordered_boxes = _order_boxes(raw_boxes, order=roi.order, axis_tolerance=max(8, int(height * p.cluster_axis_tolerance_ratio)))
        for index, box in enumerate(ordered_boxes, start=1):
            all_candidates.append(_candidate_from_box(roi.player, index, box))

    candidates = _dedupe_candidates(all_candidates, iou_threshold=p.nms_iou_threshold)
    candidates = _suppress_self_action_panel_candidates(candidates, arr)
    candidates = _complete_side_river_candidates(candidates, rois, arr, image_area=image_area, params=p)
    candidates = _renumber_by_player(candidates, rois)
    candidates = _stabilize_lower_side_visible_quads(candidates, rois)
    return RiverDetectionResult(candidates=candidates, rois=rois, image_size=(width, height))


def crop_river_candidate(
    image: Image.Image,
    candidate: RiverTileCandidate,
    *,
    output_size: tuple[int, int] | None = None,
    padding_px: int | None = None,
) -> Image.Image:
    """Perspective-crop one v2 river candidate for tile classification."""
    from .discard_parser import crop_discard_quad

    padded_quad = expand_candidate_quad_for_classification(candidate, padding_px=padding_px)
    padded_bbox = _bbox_from_quad(padded_quad)
    width, height = output_size or (
        max(candidate.width, padded_bbox[2] - padded_bbox[0]),
        max(candidate.height, padded_bbox[3] - padded_bbox[1]),
    )
    return crop_discard_quad(
        image,
        padded_quad,
        output_size=(max(1, width), max(1, height)),
        orientation=RIVER_ORIENTATIONS.get(candidate.player, "bottom"),
    )


def river_candidate_looks_blank(image: Image.Image, candidate: RiverTileCandidate) -> bool:
    crop = image.crop(candidate.bbox).convert("RGB")
    if crop.size[0] <= 0 or crop.size[1] <= 0:
        return False
    arr = np.asarray(crop, dtype=np.int16)
    height, width = arr.shape[:2]
    inner = arr[
        int(round(height * 0.18)): int(round(height * 0.82)),
        int(round(width * 0.12)): int(round(width * 0.88)),
    ]
    if inner.size == 0:
        return False
    r = inner[..., 0]
    g = inner[..., 1]
    b = inner[..., 2]
    blue_face = (r <= 170) & (g >= 120) & (b >= 135) & (b >= r + 8) & (g >= r + 4)
    dark_ink = (r <= 85) & (g <= 85) & (b <= 85)
    red_ink = (r >= 135) & (g <= 90) & (b <= 90) & (r >= g + 35)
    blue_ratio = float(blue_face.sum()) / max(1, blue_face.size)
    ink_ratio = float((dark_ink | red_ink).sum()) / max(1, blue_face.size)
    return blue_ratio >= 0.45 and ink_ratio <= 0.035


def river_candidate_classification_rejection_reason(image: Image.Image, candidate: RiverTileCandidate) -> str:
    if river_candidate_looks_blank(image, candidate):
        return "blank_river_candidate"
    if candidate.player in {"left_opponent", "right_opponent"} and candidate.confidence < MIN_SIDE_CLASSIFICATION_CANDIDATE_CONFIDENCE:
        return "low_side_candidate_confidence"
    return ""


def expand_candidate_quad_for_classification(
    candidate: RiverTileCandidate,
    *,
    padding_px: int | None = None,
) -> Quad:
    """Return a slightly larger quad for robust classification crops.

    Debug overlays stay tight to the detected tile face, but classifiers need
    a little context around side-river and geometry-completed candidates.
    """
    pad = padding_px if padding_px is not None else _classification_padding_px(candidate)
    base_quad = candidate.quad
    if candidate.player in {"left_opponent", "right_opponent"} and candidate.source == "river_detector_v2_completion":
        base_quad = _perspective_quad(candidate.player, candidate.bbox)
    return _expand_ordered_quad(base_quad, pad=max(0, int(pad)))


def build_river_rois(width: int, height: int) -> list[RiverRoi]:
    """Return broad table-center ROIs for the four discard rivers.

    These are deliberately wider than the old layout slots. They describe where
    Mahjong Soul renders discard rivers on the table, while excluding hands,
    avatars, side walls, action buttons, and most chrome.
    """
    return [
        _roi("top_opponent", width, height, 0.39, 0.14, 0.59, 0.295, order="row_major"),
        _roi("left_opponent", width, height, 0.275, 0.265, 0.41, 0.525, order="row_major"),
        _roi("right_opponent", width, height, 0.59, 0.265, 0.73, 0.525, order="row_major"),
        _roi("self", width, height, 0.385, 0.50, 0.605, 0.735, order="row_major"),
    ]


def _detect_roi_boxes(
    arr: np.ndarray,
    roi: RiverRoi,
    *,
    image_area: float,
    params: RiverDetectorParams,
) -> list[SurfaceBox]:
    crop = arr[roi.top:roi.bottom, roi.left:roi.right]
    if crop.size == 0:
        return []
    mask = _tile_face_mask(crop)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)

    min_area = image_area * params.min_area_ratio
    max_area = image_area * params.max_area_ratio
    max_width = max(1, int(np.sqrt(image_area) * params.max_component_width_ratio))
    max_height = max(1, int(np.sqrt(image_area) * params.max_component_height_ratio))
    pad = max(1, int(round(np.sqrt(image_area) * params.pad_ratio)))
    boxes: list[SurfaceBox] = []
    for component_id in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[component_id]]
        if area < min_area or area > max_area:
            continue
        if w > max_width or h > max_height:
            continue
        aspect = h / max(1.0, float(w))
        if aspect < params.min_aspect_ratio or aspect > params.max_aspect_ratio:
            continue
        fill = area / max(1.0, float(w * h))
        if fill < params.min_fill_ratio:
            continue
        left = max(roi.left, roi.left + x - pad)
        top = max(roi.top, roi.top + y - pad)
        right = min(roi.right, roi.left + x + w + pad)
        bottom = min(roi.bottom, roi.top + y + h + pad)
        score = _score_component(area=area, fill=fill, aspect=aspect, image_area=image_area)
        component_mask = np.where(labels == component_id, 255, 0).astype(np.uint8)
        quad = _quad_from_component(component_mask, roi=roi, fallback_bbox=(left, top, right, bottom), pad=pad)
        boxes.append((left, top, right, bottom, score, quad))
    return _dedupe_boxes(boxes, iou_threshold=params.nms_iou_threshold)


def _tile_face_mask(crop: np.ndarray) -> np.ndarray:
    rgb = crop.astype(np.int16)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    warm_ivory = (
        (r >= 165)
        & (g >= 150)
        & (b >= 125)
        & (r >= b - 8)
        & (r <= b + 105)
        & (g >= b - 25)
    )
    pale_blank = (
        (r >= 175)
        & (g >= 175)
        & (b >= 170)
        & (np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b]) <= 38)
    )
    return np.where(warm_ivory | pale_blank, 255, 0).astype(np.uint8)


def _score_component(*, area: int, fill: float, aspect: float, image_area: float) -> float:
    area_ratio = area / max(1.0, image_area)
    area_score = min(1.0, area_ratio / 0.0012)
    fill_score = min(1.0, max(0.0, (fill - 0.30) / 0.55))
    aspect_score = max(0.0, 1.0 - abs(aspect - 0.82) / 0.70)
    return round(0.40 * area_score + 0.40 * fill_score + 0.20 * aspect_score, 4)


def _order_boxes(
    boxes: list[SurfaceBox],
    *,
    order: str,
    axis_tolerance: int,
) -> list[SurfaceBox]:
    if order != "row_major":
        return sorted(boxes, key=lambda box: (box[0], box[1]))
    rows = _cluster_by_axis(boxes, axis="y", tolerance=axis_tolerance)
    ordered: list[SurfaceBox] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda box: (box[0], box[1])))
    return ordered


def _cluster_by_axis(
    boxes: list[SurfaceBox],
    *,
    axis: str,
    tolerance: int,
) -> list[list[SurfaceBox]]:
    if not boxes:
        return []
    index = 1 if axis == "y" else 0
    sorted_boxes = sorted(boxes, key=lambda box: ((box[index] + box[index + 2]) // 2, box[0], box[1]))
    clusters: list[list[SurfaceBox]] = []
    centers: list[float] = []
    for box in sorted_boxes:
        center = (box[index] + box[index + 2]) / 2.0
        if centers and abs(center - centers[-1]) <= tolerance:
            clusters[-1].append(box)
            centers[-1] = float(np.mean([(b[index] + b[index + 2]) / 2.0 for b in clusters[-1]]))
        else:
            clusters.append([box])
            centers.append(center)
    return clusters


def _dedupe_boxes(
    boxes: list[SurfaceBox],
    *,
    iou_threshold: float,
) -> list[SurfaceBox]:
    kept: list[SurfaceBox] = []
    for box in sorted(boxes, key=lambda item: item[4], reverse=True):
        if any(_box_iou(box[:4], kept_box[:4]) >= iou_threshold for kept_box in kept):
            continue
        kept.append(box)
    return sorted(kept, key=lambda item: (item[1], item[0]))


def _dedupe_candidates(
    candidates: list[RiverTileCandidate],
    *,
    iou_threshold: float,
) -> list[RiverTileCandidate]:
    kept: list[RiverTileCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if any(_candidate_duplicate(candidate, kept_candidate, iou_threshold=iou_threshold) for kept_candidate in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: (item.player, item.order_index, item.bbox[1], item.bbox[0]))


def _candidate_duplicate(
    candidate: RiverTileCandidate,
    other: RiverTileCandidate,
    *,
    iou_threshold: float,
) -> bool:
    if candidate.player != other.player:
        return _box_iou(candidate.bbox, other.bbox) >= iou_threshold
    if candidate.player in {"left_opponent", "right_opponent"}:
        tile_width = max(1, min(candidate.width, other.width))
        tile_height = max(1, min(candidate.height, other.height))
        return _same_grid_cell(candidate.center, other.center, tile_width=tile_width, tile_height=tile_height)
    return _box_iou(candidate.bbox, other.bbox) >= iou_threshold


def _suppress_self_action_panel_candidates(
    candidates: list[RiverTileCandidate],
    arr: np.ndarray,
) -> list[RiverTileCandidate]:
    panel_top = _self_action_panel_top(arr)
    if panel_top is None:
        return candidates
    return [
        candidate
        for candidate in candidates
        if candidate.player != "self" or candidate.center[1] < panel_top
    ]


def _self_action_panel_top(arr: np.ndarray) -> int | None:
    height, width = arr.shape[:2]
    crop = arr[
        int(round(height * 0.66)): int(round(height * 0.80)),
        int(round(width * 0.35)): int(round(width * 0.66)),
    ]
    if crop.size == 0:
        return None
    rgb = crop.astype(np.int16)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    red_brown = (r >= 75) & (r <= 190) & (g <= 125) & (b <= 125) & (r >= g + 5)
    if float(red_brown.sum()) / max(1, red_brown.size) < 0.12:
        return None
    return int(round(height * 0.64))


def _renumber_by_player(
    candidates: list[RiverTileCandidate],
    rois: list[RiverRoi],
) -> list[RiverTileCandidate]:
    ordered: list[RiverTileCandidate] = []
    roi_by_player = {roi.player: roi for roi in rois}
    for player in RIVER_PLAYERS:
        player_candidates = [candidate for candidate in candidates if candidate.player == player]
        roi = roi_by_player.get(player)
        if roi is None:
            continue
        ordered_candidates = _order_candidates(
            player_candidates,
            order=roi.order,
            axis_tolerance=max(8, min(36, int(roi.height * 0.11))),
        )
        for index, candidate in enumerate(ordered_candidates, start=1):
            ordered.append(replace(candidate, order_index=index))
    return ordered


def _order_candidates(
    candidates: list[RiverTileCandidate],
    *,
    order: str,
    axis_tolerance: int,
) -> list[RiverTileCandidate]:
    if order != "row_major":
        return sorted(candidates, key=lambda item: (item.center[0], item.center[1]))
    rows = _cluster_candidates_by_y(candidates, tolerance=axis_tolerance)
    ordered: list[RiverTileCandidate] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: (item.center[0], item.center[1])))
    return ordered


def _calibrate_right_river_template(
    candidates: list[RiverTileCandidate],
    rois: list[RiverRoi],
) -> list[RiverTileCandidate]:
    """Snap the lower right river to its local row/column template.

    Mahjong Soul's right discard pile is very regular, but the lower rows are
    partially stacked and their contours are noisy. Use the already-detected
    row/column centers as a template, then keep each tile's visible-face size
    and only nudge obvious lower-row drift.
    """
    right_roi = next((roi for roi in rois if roi.player == "right_opponent"), None)
    if right_roi is None:
        return candidates
    right_candidates = [candidate for candidate in candidates if candidate.player == "right_opponent"]
    if len(right_candidates) < 8:
        return candidates

    rows = _cluster_candidates_by_y(right_candidates, tolerance=max(8, min(36, int(right_roi.height * 0.11))))
    if len(rows) < 4:
        return candidates
    lower_row_start = max(3, len(rows) - 3)

    col_centers: dict[int, int] = {}
    max_columns = max(len(row) for row in rows)
    for col_index in range(max_columns):
        values = [
            sorted(row, key=lambda item: item.center[0])[col_index].center[0]
            for row in rows
            if len(row) > col_index
        ]
        if values:
            col_centers[col_index] = int(round(float(np.median(values))))

    calibrated_by_id: dict[int, RiverTileCandidate] = {}
    for row_index, row in enumerate(rows):
        sorted_row = sorted(row, key=lambda item: item.center[0])
        row_center_y = int(round(float(np.median([candidate.center[1] for candidate in sorted_row]))))
        for col_index, candidate in enumerate(sorted_row):
            if row_index < lower_row_start:
                continue
            center_x = col_centers.get(col_index, candidate.center[0])
            bbox = _template_nudged_bbox(
                candidate.bbox,
                center=(center_x, row_center_y),
                roi=right_roi,
            )
            calibrated_by_id[id(candidate)] = replace(
                candidate,
                bbox=bbox,
                quad=_perspective_quad(candidate.player, bbox),
                center=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
            )

    return [calibrated_by_id.get(id(candidate), candidate) for candidate in candidates]


def _template_nudged_bbox(
    bbox: tuple[int, int, int, int],
    *,
    center: tuple[int, int],
    roi: RiverRoi,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    current_center = ((left + right) // 2, (top + bottom) // 2)
    dx = center[0] - current_center[0]
    dy = center[1] - current_center[1]
    max_dx = max(2, int(round(width * 0.18)))
    max_dy = max(2, int(round(height * 0.18)))
    nudged_center = (
        current_center[0] + max(-max_dx, min(max_dx, dx)),
        current_center[1] + max(-max_dy, min(max_dy, dy)),
    )
    return _centered_bbox(nudged_center, width, height, roi)


def _stabilize_lower_side_visible_quads(
    candidates: list[RiverTileCandidate],
    rois: list[RiverRoi],
) -> list[RiverTileCandidate]:
    """Keep lower side-river debug quads on the visible tile face.

    The lower right river has strong overlap and orange tile bases. The contour
    fit can push a corner a few pixels outside its own connected-component box,
    making the last rows look crooked. Clamp only those lower visible-face quads
    back into their detection bbox; do not infer a hidden full tile.
    Classification crops are expanded separately.
    """
    roi_by_player = {roi.player: roi for roi in rois}
    stabilized: list[RiverTileCandidate] = []
    for candidate in candidates:
        roi = roi_by_player.get(candidate.player)
        if (
            roi is not None
            and candidate.player == "right_opponent"
            and candidate.center[1] >= roi.top + roi.height * 0.55
        ):
            stabilized.append(replace(candidate, quad=_clamp_quad_to_bbox(candidate.quad, candidate.bbox)))
        else:
            stabilized.append(candidate)
    return stabilized


def _clamp_quad_to_bbox(
    quad: Quad,
    bbox: tuple[int, int, int, int],
) -> Quad:
    left, top, right, bottom = bbox
    width = max(1, right - left)
    max_left_inset = max(3, int(round(width * 0.12)))
    left_edge_limit = left + max_left_inset
    points = [
        (
            max(left, min(right, x)),
            max(top, min(bottom, y)),
        )
        for x, y in quad
    ]
    # Lower right tiles are partially stacked, but the visible face should not
    # collapse into a narrow sliver. Keep the left edge near its component box.
    points[0] = (min(points[0][0], left_edge_limit), points[0][1])
    points[1] = (min(points[1][0], left_edge_limit), points[1][1])
    return (points[0], points[1], points[2], points[3])


def _complete_side_river_candidates(
    candidates: list[RiverTileCandidate],
    rois: list[RiverRoi],
    arr: np.ndarray,
    *,
    image_area: float,
    params: RiverDetectorParams,
) -> list[RiverTileCandidate]:
    completed = list(candidates)
    roi_by_player = {roi.player: roi for roi in rois}
    for player in ("left_opponent", "right_opponent"):
        roi = roi_by_player.get(player)
        if roi is None:
            continue
        player_candidates = [candidate for candidate in completed if candidate.player == player]
        additions = _side_player_grid_additions(
            player_candidates,
            player=player,
            roi=roi,
            arr=arr,
            image_area=image_area,
            params=params,
        )
        completed.extend(additions)
    return _dedupe_candidates(completed, iou_threshold=params.nms_iou_threshold)


def _side_player_grid_additions(
    candidates: list[RiverTileCandidate],
    *,
    player: str,
    roi: RiverRoi,
    arr: np.ndarray,
    image_area: float,
    params: RiverDetectorParams,
) -> list[RiverTileCandidate]:
    if len(candidates) < 5:
        return []
    rows = _cluster_candidates_by_y(candidates, tolerance=max(8, min(36, int(roi.height * 0.11))))
    if len(rows) < 2:
        return []

    widths = [candidate.width for candidate in candidates]
    heights = [candidate.height for candidate in candidates]
    tile_width = int(round(float(np.median(widths))))
    tile_height = int(round(float(np.median(heights))))
    if tile_width <= 0 or tile_height <= 0:
        return []

    row_centers = [_median_center(row, axis="y") for row in rows]
    row_steps = [b - a for a, b in zip(row_centers, row_centers[1:], strict=False) if b - a > tile_height * 0.35]
    row_step = int(round(float(np.median(row_steps)))) if row_steps else int(round(tile_height * 0.75))
    if row_step <= 0:
        return []

    x_steps: list[int] = []
    for row in rows:
        xs = sorted(candidate.center[0] for candidate in row)
        x_steps.extend(b - a for a, b in zip(xs, xs[1:], strict=False) if b - a > tile_width * 0.45)
    x_step = int(round(float(np.median(x_steps)))) if x_steps else int(round(tile_width * 0.88))
    if x_step <= 0:
        return []

    additions: list[RiverTileCandidate] = []
    occupied = list(candidates)
    additions.extend(
        _fill_side_column_gaps(
            rows,
            occupied=occupied,
            player=player,
            roi=roi,
            arr=arr,
            image_area=image_area,
            params=params,
            tile_width=tile_width,
            tile_height=tile_height,
            row_step=row_step,
        )
    )
    for row in rows:
        additions.extend(
            _fill_side_row_gaps(
                row,
                occupied=occupied + additions,
                player=player,
                roi=roi,
                arr=arr,
                image_area=image_area,
                params=params,
                tile_width=tile_width,
                tile_height=tile_height,
                x_step=x_step,
            )
        )
        additions.extend(
            _fill_side_row_from_reference(
                row,
                rows=rows,
                occupied=occupied + additions,
                player=player,
                roi=roi,
                arr=arr,
                image_area=image_area,
                params=params,
                tile_width=tile_width,
                tile_height=tile_height,
            )
        )

    extension_rows = _cluster_candidates_by_y(
        occupied + additions,
        tolerance=max(8, min(36, int(roi.height * 0.11))),
    )
    additions.extend(
        _extend_side_rows_downward(
            extension_rows,
            occupied=occupied + additions,
            player=player,
            roi=roi,
            arr=arr,
            image_area=image_area,
            params=params,
            tile_width=tile_width,
            tile_height=tile_height,
            row_step=row_step,
        )
    )
    return additions


def _fill_side_column_gaps(
    rows: list[list[RiverTileCandidate]],
    *,
    occupied: list[RiverTileCandidate],
    player: str,
    roi: RiverRoi,
    arr: np.ndarray,
    image_area: float,
    params: RiverDetectorParams,
    tile_width: int,
    tile_height: int,
    row_step: int,
) -> list[RiverTileCandidate]:
    if len(rows) < 2:
        return []
    additions: list[RiverTileCandidate] = []
    row_centers = [
        (
            int(round(_median_center(row, axis="x"))),
            int(round(_median_center(row, axis="y"))),
        )
        for row in rows
    ]
    for (left_x, upper_y), (right_x, lower_y) in zip(row_centers, row_centers[1:], strict=False):
        if abs(left_x - right_x) > tile_width * 0.45:
            continue
        gap = lower_y - upper_y
        if gap <= row_step * 1.45:
            continue
        missing_count = max(1, int(round(gap / row_step)) - 1)
        for offset in range(1, missing_count + 1):
            center = (
                int(round(left_x + (right_x - left_x) * offset / (missing_count + 1))),
                int(round(upper_y + gap * offset / (missing_count + 1))),
            )
            candidate = _completion_candidate_at(
                player,
                center=center,
                tile_width=tile_width,
                tile_height=tile_height,
                roi=roi,
                arr=arr,
                image_area=image_area,
                params=params,
                occupied=occupied + additions,
            )
            if candidate is not None:
                additions.append(candidate)
    return additions


def _fill_side_row_from_reference(
    row: list[RiverTileCandidate],
    *,
    rows: list[list[RiverTileCandidate]],
    occupied: list[RiverTileCandidate],
    player: str,
    roi: RiverRoi,
    arr: np.ndarray,
    image_area: float,
    params: RiverDetectorParams,
    tile_width: int,
    tile_height: int,
) -> list[RiverTileCandidate]:
    if len(row) >= 3:
        return []
    row_y = int(round(_median_center(row, axis="y")))
    row_index = rows.index(row)
    reference = _nearest_reference_row(rows, row_index)
    if reference is None or len(reference) <= len(row):
        return []
    additions: list[RiverTileCandidate] = []
    row_centers = [candidate.center for candidate in row]
    for ref_x in sorted(candidate.center[0] for candidate in reference):
        if any(_same_grid_cell((ref_x, row_y), center, tile_width=tile_width, tile_height=tile_height) for center in row_centers):
            continue
        candidate = _completion_candidate_at(
            player,
            center=(ref_x, row_y),
            tile_width=tile_width,
            tile_height=tile_height,
            roi=roi,
            arr=arr,
            image_area=image_area,
            params=params,
            occupied=occupied + additions,
        )
        if candidate is not None:
            additions.append(candidate)
    return additions


def _nearest_reference_row(
    rows: list[list[RiverTileCandidate]],
    row_index: int,
) -> list[RiverTileCandidate] | None:
    current_len = len(rows[row_index])
    best: tuple[int, int, list[RiverTileCandidate]] | None = None
    for index, row in enumerate(rows):
        if index == row_index or len(row) <= current_len:
            continue
        distance = abs(index - row_index)
        # Prefer the most complete row as the column reference; distance only
        # breaks ties. Side rivers often have a neighboring row that is also
        # missing the same edge tile.
        score = len(row)
        if best is None or score > best[0] or (score == best[0] and distance < best[1]):
            best = (score, distance, row)
    return best[2] if best is not None else None


def _fill_side_row_gaps(
    row: list[RiverTileCandidate],
    *,
    occupied: list[RiverTileCandidate],
    player: str,
    roi: RiverRoi,
    arr: np.ndarray,
    image_area: float,
    params: RiverDetectorParams,
    tile_width: int,
    tile_height: int,
    x_step: int,
) -> list[RiverTileCandidate]:
    if len(row) < 2:
        return []
    additions: list[RiverTileCandidate] = []
    row_y = int(round(_median_center(row, axis="y")))
    xs = sorted(candidate.center[0] for candidate in row)
    for left_x, right_x in zip(xs, xs[1:], strict=False):
        if right_x - left_x <= x_step * 1.45:
            continue
        missing_count = max(1, int(round((right_x - left_x) / x_step)) - 1)
        for offset in range(1, missing_count + 1):
            center_x = int(round(left_x + (right_x - left_x) * offset / (missing_count + 1)))
            candidate = _completion_candidate_at(
                player,
                center=(center_x, row_y),
                tile_width=tile_width,
                tile_height=tile_height,
                roi=roi,
                arr=arr,
                image_area=image_area,
                params=params,
                occupied=occupied + additions,
            )
            if candidate is not None:
                additions.append(candidate)
    return additions


def _extend_side_rows_downward(
    rows: list[list[RiverTileCandidate]],
    *,
    occupied: list[RiverTileCandidate],
    player: str,
    roi: RiverRoi,
    arr: np.ndarray,
    image_area: float,
    params: RiverDetectorParams,
    tile_width: int,
    tile_height: int,
    row_step: int,
) -> list[RiverTileCandidate]:
    additions: list[RiverTileCandidate] = []
    last_row = rows[-1]
    # Two visible columns are enough to probe the next side-river row; the
    # candidate-level lower-half evidence gate rejects row-above tails.
    if len(last_row) < 2:
        return []
    last_y = int(round(_median_center(last_row, axis="y")))
    xs = sorted(candidate.center[0] for candidate in last_row)
    if not xs:
        return []
    for extra_row in range(1, 2):
        center_y = last_y + row_step * extra_row
        if center_y + tile_height // 2 > roi.bottom:
            break
        row_added = 0
        for center_x in xs:
            candidate = _completion_candidate_at(
                player,
                center=(center_x, center_y),
                tile_width=tile_width,
                tile_height=tile_height,
                roi=roi,
                arr=arr,
                image_area=image_area,
                params=params,
                occupied=occupied + additions,
            )
            if candidate is not None:
                additions.append(candidate)
                row_added += 1
        if row_added == 0:
            break
    return additions


def _completion_candidate_at(
    player: str,
    *,
    center: tuple[int, int],
    tile_width: int,
    tile_height: int,
    roi: RiverRoi,
    arr: np.ndarray,
    image_area: float,
    params: RiverDetectorParams,
    occupied: list[RiverTileCandidate],
) -> RiverTileCandidate | None:
    bbox = _centered_bbox(center, tile_width, tile_height, roi)
    if any(_same_grid_cell(center, candidate.center, tile_width=tile_width, tile_height=tile_height) for candidate in occupied):
        return None
    if _looks_like_overlap_tail_only(arr, bbox, occupied):
        return None
    evidence = _tile_surface_evidence(arr, bbox)
    if evidence < 0.18:
        return None
    quad = _quad_from_bbox_region(arr, roi=roi, bbox=bbox, pad=max(1, int(round(np.sqrt(image_area) * params.pad_ratio))))
    score = round(min(0.86, max(0.45, evidence * 1.4)), 4)
    return _candidate_from_box(player, 0, (bbox[0], bbox[1], bbox[2], bbox[3], score, quad), source="river_detector_v2_completion")


def _looks_like_overlap_tail_only(
    arr: np.ndarray,
    bbox: tuple[int, int, int, int],
    occupied: list[RiverTileCandidate],
) -> bool:
    """Reject a completion that only sees the lower tail of the row above."""
    overlapping = [candidate for candidate in occupied if _horizontal_overlap_ratio(bbox, candidate.bbox) >= 0.62]
    if not overlapping:
        return False
    left, top, right, bottom = bbox
    height = max(1, bottom - top)
    upper = (left, top, right, int(round(top + height * 0.45)))
    lower = (left, int(round(top + height * 0.58)), right, bottom)
    upper_evidence = _tile_surface_evidence(arr, upper)
    lower_evidence = _tile_surface_evidence(arr, lower)
    for candidate in overlapping:
        vertical_overlap = _vertical_overlap_ratio(bbox, candidate.bbox)
        if vertical_overlap >= 0.28 and lower_evidence < 0.42 and upper_evidence >= lower_evidence * 1.35:
            return True
    return False


def _horizontal_overlap_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return overlap / max(1, min(a[2] - a[0], b[2] - b[0]))


def _vertical_overlap_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return overlap / max(1, min(a[3] - a[1], b[3] - b[1]))


def _same_grid_cell(
    a: tuple[int, int],
    b: tuple[int, int],
    *,
    tile_width: int,
    tile_height: int,
) -> bool:
    return abs(a[0] - b[0]) <= tile_width * 0.35 and abs(a[1] - b[1]) <= tile_height * 0.35


def _quad_from_bbox_region(
    arr: np.ndarray,
    *,
    roi: RiverRoi,
    bbox: tuple[int, int, int, int],
    pad: int,
) -> Quad | None:
    left, top, right, bottom = bbox
    crop = arr[top:bottom, left:right]
    if crop.size == 0:
        return None
    mask = _tile_face_mask(crop)
    local_roi = RiverRoi(
        player=roi.player,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        order=roi.order,
    )
    return _quad_from_component(mask, roi=local_roi, fallback_bbox=bbox, pad=pad)


def _tile_surface_evidence(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = bbox
    crop = arr[top:bottom, left:right]
    if crop.size == 0:
        return 0.0
    rgb = crop.astype(np.int16)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    face = _tile_face_mask(crop) > 0
    orange_side = (r >= 185) & (g >= 95) & (g <= 185) & (b <= 90) & (r >= g + 25)
    bright = (r + g + b) / 3 >= 155
    return float((face | orange_side | bright).sum()) / max(1, face.size)


def _centered_bbox(
    center: tuple[int, int],
    width: int,
    height: int,
    roi: RiverRoi,
) -> tuple[int, int, int, int]:
    cx, cy = center
    left = max(roi.left, int(round(cx - width / 2)))
    top = max(roi.top, int(round(cy - height / 2)))
    right = min(roi.right, left + width)
    bottom = min(roi.bottom, top + height)
    if right - left < width:
        left = max(roi.left, right - width)
    if bottom - top < height:
        top = max(roi.top, bottom - height)
    return (left, top, right, bottom)


def _cluster_candidates_by_y(
    candidates: list[RiverTileCandidate],
    *,
    tolerance: int,
) -> list[list[RiverTileCandidate]]:
    rows: list[list[RiverTileCandidate]] = []
    centers: list[float] = []
    for candidate in sorted(candidates, key=lambda item: (item.center[1], item.center[0])):
        cy = float(candidate.center[1])
        if centers and abs(cy - centers[-1]) <= tolerance:
            rows[-1].append(candidate)
            centers[-1] = float(np.mean([item.center[1] for item in rows[-1]]))
        else:
            rows.append([candidate])
            centers.append(cy)
    return rows


def _median_center(candidates: list[RiverTileCandidate], *, axis: str) -> float:
    index = 1 if axis == "y" else 0
    return float(np.median([candidate.center[index] for candidate in candidates]))


def _candidate_from_box(
    player: str,
    order_index: int,
    box: SurfaceBox,
    *,
    source: str = "river_detector_v2",
) -> RiverTileCandidate:
    left, top, right, bottom, score, fitted_quad = box
    bbox = (left, top, right, bottom)
    return RiverTileCandidate(
        player=player,
        order_index=order_index,
        bbox=bbox,
        quad=fitted_quad or _perspective_quad(player, bbox),
        center=((left + right) // 2, (top + bottom) // 2),
        confidence=score,
        source=source,
    )


def _quad_from_component(
    component_mask: np.ndarray,
    *,
    roi: RiverRoi,
    fallback_bbox: tuple[int, int, int, int],
    pad: int,
) -> Quad | None:
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon=max(2.0, perimeter * 0.035), closed=True)
    if len(approx) == 4:
        points = [(int(point[0][0]) + roi.left, int(point[0][1]) + roi.top) for point in approx]
        return _expand_ordered_quad(_order_quad_points(points), pad=pad)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    points = [(int(round(x)) + roi.left, int(round(y)) + roi.top) for x, y in box]
    quad = _expand_ordered_quad(_order_quad_points(points), pad=pad)
    if _quad_area(quad) < _bbox_area(fallback_bbox) * 0.25:
        return None
    return quad


def _order_quad_points(points: list[tuple[int, int]]) -> Quad:
    pts = sorted(points, key=lambda point: (point[1], point[0]))
    top = sorted(pts[:2], key=lambda point: point[0])
    bottom = sorted(pts[2:], key=lambda point: point[0])
    return (top[0], bottom[0], bottom[1], top[1])


def _expand_ordered_quad(quad: Quad, *, pad: int) -> Quad:
    if pad <= 0:
        return quad
    cx = sum(x for x, _y in quad) / 4.0
    cy = sum(y for _x, y in quad) / 4.0
    expanded: list[tuple[int, int]] = []
    for x, y in quad:
        dx = x - cx
        dy = y - cy
        length = max(1.0, float(np.hypot(dx, dy)))
        expanded.append((int(round(x + pad * dx / length)), int(round(y + pad * dy / length))))
    return (expanded[0], expanded[1], expanded[2], expanded[3])


def _quad_area(quad: Quad) -> float:
    points = np.asarray(quad, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_from_quad(quad: Quad) -> tuple[int, int, int, int]:
    xs = [x for x, _y in quad]
    ys = [y for _x, y in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def _classification_padding_px(candidate: RiverTileCandidate) -> int:
    base = max(2, int(round(min(candidate.width, candidate.height) * 0.10)))
    if candidate.player in {"left_opponent", "right_opponent"}:
        base = max(2, int(round(min(candidate.width, candidate.height) * 0.06)))
    if candidate.source == "river_detector_v2_completion":
        base = max(base, int(round(min(candidate.width, candidate.height) * 0.20)))
    return base


def _perspective_quad(
    player: str,
    bbox: tuple[int, int, int, int],
) -> Quad:
    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    if player in {"left_opponent", "right_opponent"}:
        skew_x = max(2, min(width // 5, height // 6))
        return (
            (left + skew_x, top),
            (left, bottom),
            (right - skew_x, bottom),
            (right, top),
        )
    if player == "top_opponent":
        skew_x = max(2, width // 10)
        return (
            (left + skew_x, top),
            (left, bottom),
            (right, bottom),
            (right - skew_x, top),
        )
    return (
        (left, top),
        (left, bottom),
        (right, bottom),
        (right, top),
    )


def _box_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter = max(0, right - left) * max(0, bottom - top)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _roi(
    player: str,
    width: int,
    height: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
    *,
    order: str,
) -> RiverRoi:
    return RiverRoi(
        player=player,
        left=max(0, min(width - 1, int(round(width * left)))),
        top=max(0, min(height - 1, int(round(height * top)))),
        right=max(1, min(width, int(round(width * right)))),
        bottom=max(1, min(height, int(round(height * bottom)))),
        order=order,
    )
