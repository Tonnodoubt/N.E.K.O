from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .discard_layout import DiscardSlot, build_discard_layout
from .discard_quad_finder import DiscardQuadRefinement, refine_discard_slot_quad
from .roi import collect_region_metrics
from .tile_templates import TileTemplateMatch, classify_tile_from_templates


DEFAULT_MIN_DISCARD_CONFIDENCE = 0.55
AMBIGUOUS_DISCARD_TEMPLATE_PAIRS = {
    frozenset({"5p", "6p"}),
    frozenset({"6p", "7p"}),
    frozenset({"6s", "9s"}),
}
MIN_AMBIGUOUS_PAIR_CONFIDENCE = 0.78
MIN_AMBIGUOUS_PAIR_DISTANCE_MARGIN = 12.0


@dataclass
class DiscardParseResult:
    discard_piles: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    visible_tiles: list[str] = field(default_factory=list)
    raw_detections: list[dict[str, Any]] = field(default_factory=list)
    analysis_hints: dict[str, Any] = field(default_factory=dict)


def parse_discards_from_image(
    image: Image.Image,
    template_payload: dict[str, Any],
    *,
    layout: dict[str, list[DiscardSlot]] | None = None,
    min_confidence: float = DEFAULT_MIN_DISCARD_CONFIDENCE,
    include_empty_detections: bool = False,
) -> DiscardParseResult:
    if not template_payload:
        return DiscardParseResult(
            analysis_hints={
                "discard_parser_source": "template_profile",
                "discard_parser_available": False,
                "discard_parser_reason": "missing_tile_templates",
            },
        )

    layout = layout or build_discard_layout(*image.size)
    discard_piles: dict[str, list[dict[str, Any]]] = {}
    visible_tiles: list[str] = []
    raw_detections: list[dict[str, Any]] = []
    confidences: list[float] = []
    occupied_count = 0
    accepted_bboxes: list[list[int]] = []

    for player, slots in layout.items():
        for slot in slots:
            slot_metrics = _collect_discard_slot_metrics(image, slot)
            refinement = refine_discard_slot_quad(image, slot)
            occupied = is_probably_occupied_discard_slot(slot_metrics) or refinement is not None
            if occupied:
                occupied_count += 1

            detection = _base_detection(
                slot,
                slot_metrics=slot_metrics,
                occupied=occupied,
                refinement=refinement,
            )
            if not occupied:
                if include_empty_detections:
                    raw_detections.append(detection)
                continue

            match, selected_refinement = _classify_slot_with_best_crop(
                image,
                slot,
                template_payload,
                refinement=refinement,
                min_confidence=min_confidence,
            )
            if match is None:
                raw_detections.append(detection)
                continue
            detection_quad = selected_refinement.quad if selected_refinement is not None else slot.corners
            detection_bbox = selected_refinement.bbox if selected_refinement is not None else slot.bbox
            detection.update(
                {
                    "candidate_tile": match.tile,
                    "confidence": match.confidence,
                    "template_distance": match.distance,
                    "runner_up_tile": match.runner_up_tile,
                    "runner_up_distance": match.runner_up_distance,
                    "bbox": detection_bbox,
                    "quad": [[x, y] for x, y in detection_quad],
                    "quad_source": "refined_tile_surface" if selected_refinement is not None else "layout_slot",
                }
            )
            if selected_refinement is not None:
                owner_slot = _better_refinement_owner(selected_refinement.bbox, slot, slots)
                if owner_slot is not None:
                    detection["rejected_refinement_owner_slot_id"] = owner_slot.slot_id
                    raw_detections.append(detection)
                    continue
            if _overlaps_existing_detection(detection_bbox, accepted_bboxes):
                detection["suppressed_duplicate"] = True
                raw_detections.append(detection)
                continue
            rejection_reason = _template_match_rejection_reason(match, min_confidence=min_confidence)
            if rejection_reason:
                detection["accepted"] = False
                detection["rejection_reason"] = rejection_reason
                raw_detections.append(detection)
                continue
            raw_detections.append(detection)

            discard_item = {
                "tile": match.tile,
                "player": player,
                "turn_index": slot.turn_index,
                "bbox": detection_bbox,
                "quad": [[x, y] for x, y in detection_quad],
                "confidence": match.confidence,
                "orientation": slot.orientation,
                "source": "discard_template_profile",
                "slot_id": slot.slot_id,
                "quad_source": "refined_tile_surface" if selected_refinement is not None else "layout_slot",
                "template_distance": match.distance,
            }
            if selected_refinement is not None:
                discard_item["quad_confidence"] = selected_refinement.confidence
            discard_piles.setdefault(player, []).append(discard_item)
            visible_tiles.append(match.tile)
            confidences.append(match.confidence)
            accepted_bboxes.append(list(detection_bbox))

    recognized_count = len(visible_tiles)
    analysis_confidence = round(sum(confidences) / max(1, len(confidences)), 3) if confidences else 0.0
    return DiscardParseResult(
        discard_piles=discard_piles,
        visible_tiles=visible_tiles,
        raw_detections=raw_detections,
        analysis_hints={
            "discard_parser_source": "template_profile",
            "discard_parser_available": True,
            "discard_slot_count": sum(len(slots) for slots in layout.values()),
            "occupied_discard_slot_count": occupied_count,
            "recognized_discard_tile_count": recognized_count,
            "discard_analysis_confidence": analysis_confidence,
        },
    )


def crop_discard_slot(image: Image.Image, slot: DiscardSlot, *, refine: bool = True) -> Image.Image:
    if refine:
        refinement = refine_discard_slot_quad(image, slot)
        if refinement is not None:
            return crop_discard_quad(
                image,
                refinement.quad,
                output_size=refinement.output_size,
                orientation=slot.orientation,
            )
    return crop_discard_quad(
        image,
        slot.corners,
        output_size=(slot.box.width, slot.box.height),
        orientation=slot.orientation,
    )


def crop_discard_quad(
    image: Image.Image,
    quad: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    *,
    output_size: tuple[int, int],
    orientation: str,
) -> Image.Image:
    width = max(1, int(output_size[0]))
    height = max(1, int(output_size[1]))
    if len(quad) != 4:
        crop = image.crop((0, 0, width, height))
        return normalize_discard_crop(crop, orientation)
    data = tuple(float(value) for point in quad for value in point)
    transform_quad = getattr(getattr(Image, "Transform", Image), "QUAD")
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    crop = image.transform((width, height), transform_quad, data, resample=resampling)
    return normalize_discard_crop(crop, orientation)


def normalize_discard_crop(crop: Image.Image, orientation: str) -> Image.Image:
    if orientation == "top":
        return crop.rotate(180, expand=True)
    if orientation == "left":
        return crop.rotate(270, expand=True)
    if orientation == "right":
        return crop.rotate(90, expand=True)
    return crop


def _classify_slot_with_best_crop(
    image: Image.Image,
    slot: DiscardSlot,
    template_payload: dict[str, Any],
    *,
    refinement: DiscardQuadRefinement | None,
    min_confidence: float,
):
    base_crop = crop_discard_slot(image, slot, refine=False)
    base_match = classify_tile_from_templates(base_crop, template_payload)
    if refinement is None:
        return base_match, None

    should_try_refined = base_match is None or base_match.confidence < min_confidence
    if not should_try_refined:
        return base_match, None

    refined_crop = crop_discard_quad(
        image,
        refinement.quad,
        output_size=refinement.output_size,
        orientation=slot.orientation,
    )
    refined_match = classify_tile_from_templates(refined_crop, template_payload)
    if refined_match is None:
        return base_match, None
    if base_match is not None and base_match.confidence > refined_match.confidence:
        return base_match, None
    return refined_match, refinement


def is_probably_occupied_discard_slot(slot_metrics: dict[str, Any]) -> bool:
    mean_luma = _float_metric(slot_metrics, "slot_mean_luma", "mean_luma")
    bright_ratio = _float_metric(slot_metrics, "slot_bright_ratio", "bright_ratio")
    white_ratio = _float_metric(slot_metrics, "slot_white_ratio", "white_ratio")
    dark_ratio = _float_metric(slot_metrics, "slot_dark_ratio", "dark_ratio")
    stddev = _float_metric(slot_metrics, "slot_stddev", "stddev")
    return mean_luma >= 88.0 and (bright_ratio >= 0.12 or white_ratio >= 0.04) and dark_ratio <= 0.62 and stddev >= 14.0


def _collect_discard_slot_metrics(image: Image.Image, slot: DiscardSlot) -> dict[str, Any]:
    metrics = collect_region_metrics(image, slot.box, sample_step=4)
    return {
        "slot_mean_luma": metrics["mean_luma"],
        "slot_bright_ratio": metrics["bright_ratio"],
        "slot_dark_ratio": metrics["dark_ratio"],
        "slot_white_ratio": metrics["white_ratio"],
        "slot_colorful_ratio": metrics["colorful_ratio"],
        "slot_stddev": metrics["stddev"],
    }


def _base_detection(
    slot: DiscardSlot,
    *,
    slot_metrics: dict[str, Any],
    occupied: bool,
    refinement: DiscardQuadRefinement | None = None,
    source: str = "discard_template_profile",
) -> dict[str, Any]:
    detection = {
        "slot_id": slot.slot_id,
        "group": "discard",
        "player": slot.player,
        "turn_index": slot.turn_index,
        "candidate_tile": "",
        "confidence": 0.0,
        "box": slot.box.to_dict(),
        "bbox": slot.bbox,
        "quad": [[x, y] for x, y in slot.corners],
        "orientation": slot.orientation,
        "occupied": occupied,
        "source": source,
        "quad_source": "layout_slot",
        **slot_metrics,
    }
    if refinement is not None:
        detection["refined_quad_candidate"] = refinement.to_dict()
    return detection


def _float_metric(metrics: dict[str, Any], primary: str, fallback: str) -> float:
    value = metrics.get(primary, metrics.get(fallback, 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _template_match_rejection_reason(match: TileTemplateMatch, *, min_confidence: float) -> str:
    if match.confidence < min_confidence:
        return "low_confidence"
    if not match.runner_up_tile:
        return ""
    pair = frozenset({match.tile, match.runner_up_tile})
    if pair not in AMBIGUOUS_DISCARD_TEMPLATE_PAIRS:
        return ""
    if match.runner_up_distance is None:
        return "ambiguous_discard_template_pair"
    distance_margin = float(match.runner_up_distance) - float(match.distance)
    if match.confidence < MIN_AMBIGUOUS_PAIR_CONFIDENCE or distance_margin < MIN_AMBIGUOUS_PAIR_DISTANCE_MARGIN:
        return "ambiguous_discard_template_pair"
    return ""


def _overlaps_existing_detection(bbox: list[int], accepted_bboxes: list[list[int]]) -> bool:
    return any(_bbox_iou(bbox, accepted) >= 0.45 for accepted in accepted_bboxes)


def _better_refinement_owner(
    bbox: list[int],
    slot: DiscardSlot,
    slots: list[DiscardSlot],
) -> DiscardSlot | None:
    current_score = _bbox_fit_score(bbox, slot.bbox)
    best_slot = slot
    best_score = current_score
    for candidate in slots:
        score = _bbox_fit_score(bbox, candidate.bbox)
        if score > best_score:
            best_slot = candidate
            best_score = score
    if best_slot.slot_id == slot.slot_id:
        return None
    if best_score >= max(0.18, current_score + 0.07, current_score * 1.12):
        return best_slot
    return None


def _bbox_fit_score(bbox: list[int], slot_bbox: list[int]) -> float:
    if len(bbox) != 4 or len(slot_bbox) != 4:
        return 0.0
    overlap = _intersection_area(bbox, slot_bbox)
    if overlap <= 0:
        return 0.0
    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    slot_area = max(1, (slot_bbox[2] - slot_bbox[0]) * (slot_bbox[3] - slot_bbox[1]))
    return overlap / max(1, min(bbox_area, slot_area))


def _bbox_iou(first: list[int], second: list[int]) -> float:
    if len(first) != 4 or len(second) != 4:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / max(1, union)


def _intersection_area(first: list[int], second: list[int]) -> int:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0, right - left) * max(0, bottom - top)
