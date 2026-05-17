from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.perception.river_detector_v2 import (
    RIVER_PLAYERS,
    Quad,
    RiverTileCandidate,
    detect_river_tiles_v2,
    _tile_face_mask,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use MobileSAM/SAM prompts to refine Mahjong Soul river tile masks offline.",
    )
    parser.add_argument("image", type=Path, help="Input Mahjong Soul screenshot.")
    parser.add_argument("--out", type=Path, default=None, help="Output preview PNG.")
    parser.add_argument("--json-out", type=Path, default=None, help="Output JSON report.")
    parser.add_argument("--model", type=Path, default=Path("mobile_sam.pt"), help="Ultralytics SAM/MobileSAM model path.")
    parser.add_argument("--player", choices=RIVER_PLAYERS, default="right_opponent", help="River player to refine.")
    parser.add_argument("--min-order", type=int, default=7, help="Only refine candidates with order_index >= this value.")
    parser.add_argument("--max-order", type=int, default=None, help="Only refine candidates with order_index <= this value.")
    parser.add_argument("--prompt-padding", type=int, default=6, help="Pixels added around each detector bbox for SAM prompts.")
    parser.add_argument(
        "--prompt-mode",
        choices=("box-points", "points", "adaptive", "bbox", "quad"),
        default="points",
        help="How to build SAM prompt boxes from detector candidates.",
    )
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu, 0, mps.")
    parser.add_argument("--dry-run", action="store_true", help="Only draw prompt boxes; do not load SAM.")
    args = parser.parse_args()

    out_path = args.out or args.image.with_name(f"{args.image.stem}-sam-river-prompts.png")
    json_path = args.json_out or out_path.with_suffix(".json")

    with Image.open(args.image) as opened:
        image = opened.convert("RGB")

    detection = detect_river_tiles_v2(image)
    selected = _select_candidates(detection.by_player.get(args.player, []), min_order=args.min_order, max_order=args.max_order)
    prompt_boxes = build_prompt_boxes(selected, image.size, padding=args.prompt_padding, mode=args.prompt_mode, image=image)
    point_prompts = build_point_prompts(selected, mode=args.prompt_mode, positive_boxes=prompt_boxes)
    use_box_prompts = args.prompt_mode != "points"

    status = "dry_run"
    masks: list[np.ndarray] = []
    error: str | None = None
    if not args.dry_run:
        try:
            masks = _run_ultralytics_sam(
                image,
                prompt_boxes,
                point_prompts,
                model_path=args.model,
                device=str(args.device),
                use_box_prompts=use_box_prompts,
            )
            status = "sam_ok"
        except Exception as exc:  # pragma: no cover - exercised when optional deps are missing.
            status = "sam_unavailable"
            error = str(exc)

    records = []
    for candidate, prompt_box, point_prompt, mask in _zip_masks(selected, prompt_boxes, point_prompts, masks):
        mask_quad = mask_to_quad(mask, fallback_bbox=candidate.bbox) if mask is not None else None
        records.append(
            {
                "player": candidate.player,
                "order_index": candidate.order_index,
                "detector_bbox": list(candidate.bbox),
                "detector_quad": [[x, y] for x, y in candidate.quad],
                "prompt_box": list(prompt_box),
                "use_box_prompt": use_box_prompts,
                "positive_points": [[x, y] for x, y in point_prompt["positive"]],
                "negative_points": [[x, y] for x, y in point_prompt["negative"]],
                "sam_quad": [[x, y] for x, y in mask_quad] if mask_quad is not None else None,
            }
        )

    preview = _draw_preview(image, records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path)
    json_path.write_text(
        json.dumps(
            {
                "source": "experiment_sam_river_masks",
                "status": status,
                "error": error,
                "image": str(args.image),
                "model": str(args.model),
                "player": args.player,
                "prompt_mode": args.prompt_mode,
                "prompt_padding": args.prompt_padding,
                "candidate_count": len(selected),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"status={status}")
    if error:
        print(f"error={error}")
    print(f"image={args.image}")
    print(f"output={out_path}")
    print(f"json={json_path}")
    print(f"candidate_count={len(selected)}")
    return 0


def mask_to_quad(
    mask: np.ndarray,
    *,
    fallback_bbox: tuple[int, int, int, int],
) -> Quad | None:
    """Fit a four-point quad to a SAM mask, falling back to the largest contour."""
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < max(16.0, _bbox_area(fallback_bbox) * 0.08):
        return None

    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon=max(2.0, perimeter * 0.025), closed=True)
    if len(approx) == 4:
        points = [(int(point[0][0]), int(point[0][1])) for point in approx]
    else:
        rect = cv2.minAreaRect(contour)
        points = [(int(round(x)), int(round(y))) for x, y in cv2.boxPoints(rect)]
    return _order_quad_points(points)


def build_prompt_boxes(
    candidates: list[RiverTileCandidate],
    image_size: tuple[int, int],
    *,
    padding: int,
    mode: str = "adaptive",
    image: Image.Image | None = None,
) -> list[tuple[int, int, int, int]]:
    if mode in {"points", "box-points"}:
        padding = min(padding, 3)
    if mode == "quad":
        return [_expand_bbox(_bbox_from_quad(candidate.quad), image_size, padding=padding) for candidate in candidates]
    if image is not None and mode in {"box-points", "points"}:
        return _surface_prompt_boxes(candidates, image, padding=padding)
    base_boxes = [_expand_bbox(candidate.bbox, image_size, padding=padding) for candidate in candidates]
    if mode in {"bbox", "points", "box-points"} or len(candidates) < 2:
        return base_boxes
    return _clip_prompt_boxes_to_candidate_grid(candidates, base_boxes, image_size)


def build_point_prompts(
    candidates: list[RiverTileCandidate],
    *,
    mode: str = "box-points",
    positive_boxes: list[tuple[int, int, int, int]] | None = None,
) -> list[dict[str, list[tuple[int, int]]]]:
    if mode not in {"points", "box-points"}:
        return [{"positive": [], "negative": []} for _candidate in candidates]

    prompts: list[dict[str, list[tuple[int, int]]]] = []
    rows = _cluster_candidate_indexes_by_y(candidates)
    neighbor_indexes = _candidate_neighbor_indexes(candidates, rows)
    for index, candidate in enumerate(candidates):
        positive = [_box_center(positive_boxes[index]) if positive_boxes is not None and index < len(positive_boxes) else _quad_center(candidate.quad)]
        negative = [candidates[neighbor_index].center for neighbor_index in neighbor_indexes.get(index, [])]
        prompts.append({"positive": positive, "negative": negative[:4]})
    return prompts


def _candidate_neighbor_indexes(
    candidates: list[RiverTileCandidate],
    rows: list[list[int]],
) -> dict[int, list[int]]:
    neighbors: dict[int, list[int]] = {index: [] for index in range(len(candidates))}
    row_positions: dict[int, tuple[int, int]] = {}
    for row_pos, row in enumerate(rows):
        for col_pos, index in enumerate(sorted(row, key=lambda item: candidates[item].center[0])):
            row_positions[index] = (row_pos, col_pos)

    for index, (row_pos, col_pos) in row_positions.items():
        same_row = sorted(rows[row_pos], key=lambda item: candidates[item].center[0])
        if col_pos > 0:
            neighbors[index].append(same_row[col_pos - 1])
        if col_pos < len(same_row) - 1:
            neighbors[index].append(same_row[col_pos + 1])
        for other_row_pos in (row_pos - 1, row_pos + 1):
            if other_row_pos < 0 or other_row_pos >= len(rows):
                continue
            other_row = sorted(rows[other_row_pos], key=lambda item: candidates[item].center[0])
            if not other_row:
                continue
            nearest = min(other_row, key=lambda item: abs(candidates[item].center[0] - candidates[index].center[0]))
            neighbors[index].append(nearest)
    return neighbors


def _quad_center(quad: Quad) -> tuple[int, int]:
    return (
        int(round(sum(x for x, _y in quad) / 4.0)),
        int(round(sum(y for _x, y in quad) / 4.0)),
    )


def _box_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def _clip_prompt_boxes_to_candidate_grid(
    candidates: list[RiverTileCandidate],
    boxes: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    rows = _cluster_candidate_indexes_by_y(candidates)
    clipped = list(boxes)
    for row_pos, row in enumerate(rows):
        sorted_row = sorted(row, key=lambda index: candidates[index].center[0])
        centers_x = [candidates[index].center[0] for index in sorted_row]
        for col_pos, index in enumerate(sorted_row):
            left, top, right, bottom = clipped[index]
            if col_pos > 0:
                left = max(left, (centers_x[col_pos - 1] + centers_x[col_pos]) // 2)
            if col_pos < len(sorted_row) - 1:
                right = min(right, (centers_x[col_pos] + centers_x[col_pos + 1]) // 2)

            # Edge tiles need a little breathing room; their missing neighbor is
            # outside the row, not inside another prompt.
            edge_pad = max(3, int(round(candidates[index].width * 0.08)))
            if col_pos == 0:
                left = max(0, left - edge_pad)
            if col_pos == len(sorted_row) - 1:
                right = min(image_size[0], right + edge_pad)
            if row_pos == len(rows) - 1:
                bottom = min(image_size[1], bottom + edge_pad)

            # Never let prompt clipping cut into the detector's own visible
            # face. SAM can tolerate a little overlap; it cannot recover a tile
            # face that the prompt excludes.
            bbox_left, bbox_top, bbox_right, bbox_bottom = candidates[index].bbox
            left = min(left, bbox_left)
            top = min(top, bbox_top)
            right = max(right, bbox_right)
            bottom = max(bottom, bbox_bottom)
            clipped[index] = _repair_prompt_box((left, top, right, bottom), fallback=boxes[index])
    return clipped


def _cluster_candidate_indexes_by_y(candidates: list[RiverTileCandidate]) -> list[list[int]]:
    if not candidates:
        return []
    heights = [candidate.height for candidate in candidates if candidate.height > 0]
    tolerance = max(8, int(round(float(np.median(heights)) * 0.45))) if heights else 24
    rows: list[list[int]] = []
    centers: list[float] = []
    for index, candidate in sorted(enumerate(candidates), key=lambda item: (item[1].center[1], item[1].center[0])):
        cy = float(candidate.center[1])
        if centers and abs(cy - centers[-1]) <= tolerance:
            rows[-1].append(index)
            centers[-1] = float(np.mean([candidates[item].center[1] for item in rows[-1]]))
        else:
            rows.append([index])
            centers.append(cy)
    return rows


def _repair_prompt_box(
    box: tuple[int, int, int, int],
    *,
    fallback: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    if right - left >= 12 and bottom - top >= 12:
        return (left, top, right, bottom)
    return fallback


def _run_ultralytics_sam(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    point_prompts: list[dict[str, list[tuple[int, int]]]],
    *,
    model_path: Path,
    device: str,
    use_box_prompts: bool,
) -> list[np.ndarray]:
    if not boxes and not point_prompts:
        return []
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Install it, then rerun without --dry-run.") from exc
    if not model_path.exists():
        raise RuntimeError(f"SAM model not found: {model_path}. Put mobile_sam.pt there or pass --model.")

    model = SAM(str(model_path))
    arr = np.asarray(image)
    masks: list[np.ndarray] = []
    for index, box in enumerate(boxes):
        prompt = point_prompts[index] if index < len(point_prompts) else {"positive": [], "negative": []}
        points = [*prompt["positive"], *prompt["negative"]]
        labels = [1 for _point in prompt["positive"]] + [0 for _point in prompt["negative"]]
        kwargs: dict[str, Any] = {"device": device, "verbose": False}
        if points:
            kwargs["points"] = [list(point) for point in points]
            kwargs["labels"] = labels
        if use_box_prompts and prompt["positive"]:
            kwargs["bboxes"] = [list(box)]
        results = model.predict(arr, **kwargs)
        mask = _first_result_mask(results)
        if mask is not None:
            masks.append(mask)
    return masks


def _first_result_mask(results: Any) -> np.ndarray | None:
    if not results:
        return None
    masks_obj = getattr(results[0], "masks", None)
    if masks_obj is None or getattr(masks_obj, "data", None) is None:
        return None
    data = masks_obj.data
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    if len(data) == 0:
        return None
    return np.asarray(data[0], dtype=np.uint8)


def _select_candidates(
    candidates: list[RiverTileCandidate],
    *,
    min_order: int,
    max_order: int | None,
) -> list[RiverTileCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.order_index >= min_order and (max_order is None or candidate.order_index <= max_order)
    ]


def _zip_masks(
    candidates: list[RiverTileCandidate],
    boxes: list[tuple[int, int, int, int]],
    point_prompts: list[dict[str, list[tuple[int, int]]]],
    masks: list[np.ndarray],
) -> list[tuple[RiverTileCandidate, tuple[int, int, int, int], dict[str, list[tuple[int, int]]], np.ndarray | None]]:
    return [
        (
            candidate,
            box,
            point_prompts[index] if index < len(point_prompts) else {"positive": [], "negative": []},
            masks[index] if index < len(masks) else None,
        )
        for index, (candidate, box) in enumerate(zip(candidates, boxes, strict=False))
    ]


def _draw_preview(image: Image.Image, records: list[dict[str, Any]]) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for record in records:
        if record.get("use_box_prompt", True):
            prompt = record["prompt_box"]
            draw.rectangle(prompt, outline=(255, 255, 255), width=1)
        detector_points = [tuple(point) for point in record["detector_quad"]]
        draw.line(detector_points + [detector_points[0]], fill=(255, 120, 120), width=3)
        for point in record["positive_points"]:
            x, y = point
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(80, 240, 160), outline=(0, 0, 0), width=1)
        for point in record["negative_points"]:
            x, y = point
            draw.line((x - 5, y - 5, x + 5, y + 5), fill=(255, 80, 80), width=2)
            draw.line((x - 5, y + 5, x + 5, y - 5), fill=(255, 80, 80), width=2)
        if record["sam_quad"] is not None:
            sam_points = [tuple(point) for point in record["sam_quad"]]
            draw.line(sam_points + [sam_points[0]], fill=(80, 240, 160), width=4)
    return preview


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    padding: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = bbox
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _surface_prompt_boxes(
    candidates: list[RiverTileCandidate],
    image: Image.Image,
    *,
    padding: int,
) -> list[tuple[int, int, int, int]]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    image_size = image.size
    return [_surface_prompt_box(candidate, arr, image_size, padding=padding) for candidate in candidates]


def _surface_prompt_box(
    candidate: RiverTileCandidate,
    arr: np.ndarray,
    image_size: tuple[int, int],
    *,
    padding: int,
) -> tuple[int, int, int, int]:
    search_box = _expand_bbox(candidate.bbox, image_size, padding=max(4, padding))
    left, top, right, bottom = search_box
    crop = arr[top:bottom, left:right]
    if crop.size == 0:
        return _expand_bbox(_bbox_from_quad(candidate.quad), image_size, padding=padding)

    mask = _tile_face_mask(crop)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return _expand_bbox(_bbox_from_quad(candidate.quad), image_size, padding=padding)

    target_x, target_y = _quad_center(candidate.quad)
    local_target = (target_x - left, target_y - top)
    min_area = max(18, int(round(_bbox_area(candidate.bbox) * 0.08)))
    best_component: tuple[float, tuple[int, int, int, int]] | None = None
    for component_id in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[component_id]]
        if area < min_area:
            continue
        cx, cy = centroids[component_id]
        contains_target = x <= local_target[0] <= x + w and y <= local_target[1] <= y + h
        distance = float((cx - local_target[0]) ** 2 + (cy - local_target[1]) ** 2)
        score = distance - (1_000_000.0 if contains_target else 0.0)
        box = (left + x, top + y, left + x + w, top + y + h)
        if best_component is None or score < best_component[0]:
            best_component = (score, box)

    if best_component is None:
        return _expand_bbox(_bbox_from_quad(candidate.quad), image_size, padding=padding)
    return _expand_bbox(best_component[1], image_size, padding=padding)


def _bbox_from_quad(quad: Quad) -> tuple[int, int, int, int]:
    xs = [x for x, _y in quad]
    ys = [y for _x, y in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def _order_quad_points(points: list[tuple[int, int]]) -> Quad:
    pts = sorted(points, key=lambda point: (point[1], point[0]))
    top = sorted(pts[:2], key=lambda point: point[0])
    bottom = sorted(pts[2:], key=lambda point: point[0])
    return (top[0], bottom[0], bottom[1], top[1])


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


if __name__ == "__main__":
    raise SystemExit(main())
