from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ..contracts import PerceivedGameState
from .calibration import CalibrationProfile, resolve_calibration_profile
from .discard_parser import parse_discards_from_image
from .external_discard_recognizer import load_external_discard_result
from .hand_layout import TileSlot, build_hand_layout
from .riichi_detector import detect_riichi_players
from .roi import collect_region_metrics
from .tile_templates import classify_tile_from_templates, extract_tile_signature, is_probably_occupied_hand_slot
from .tile_templates import _decode_signature, _rms_distance
from .vit_tile_classifier import (
    VitTileClassifierUnavailable,
    classify_tile_crops,
    vit_classifier_enabled,
    vit_device_from_config,
    vit_model_from_config,
    vit_top_k_from_config,
)

logger = logging.getLogger(__name__)
OPPONENT_PLAYERS = {"left_opponent", "top_opponent", "right_opponent"}
TILE_PARSE_SCENES = {"in_match", "replay", "dialog", "unknown"}
MIN_HAND_TILE_CONFIDENCE = 0.12
POST_MELD_WAITING_HAND_COUNTS = {10: 1, 7: 2, 4: 3, 1: 4}
POST_MELD_DRAW_HAND_COUNTS = {11: 1, 8: 2, 5: 3, 2: 4}
DISCARD_TURN_HAND_COUNTS = {14: 0, **POST_MELD_DRAW_HAND_COUNTS}
WAITING_HAND_COUNTS = {13: 0, **POST_MELD_WAITING_HAND_COUNTS}


@dataclass
class TileParseResult:
    hand_tiles: list[str] = field(default_factory=list)
    melds: list[list[str]] = field(default_factory=list)
    dora_indicators: list[str] = field(default_factory=list)
    riichi_players: list[str] = field(default_factory=list)
    discard_piles: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    visible_tiles: list[str] = field(default_factory=list)
    known_genbutsu_tiles: list[str] = field(default_factory=list)
    raw_detections: list[dict[str, Any]] = field(default_factory=list)
    analysis_hints: dict[str, Any] = field(default_factory=dict)

    def to_state_updates(self) -> dict[str, Any]:
        return {
            "hand_tiles": list(self.hand_tiles),
            "melds": [list(group) for group in self.melds],
            "dora_indicators": list(self.dora_indicators),
            "riichi_players": list(self.riichi_players),
            "discard_piles": {player: list(items) for player, items in self.discard_piles.items()},
            "visible_tiles": list(self.visible_tiles),
            "known_genbutsu_tiles": list(self.known_genbutsu_tiles),
            "raw_detections": list(self.raw_detections),
            "analysis_hints": dict(self.analysis_hints),
        }


def parse_tiles_from_image(
    image_path: Path,
    image: Image.Image,
    *,
    scene: str,
    metrics: dict[str, dict[str, Any]],
    calibration_dir: Path | None = None,
    fixture_mode: str = "auto",
    tile_classifier_config: dict[str, Any] | None = None,
) -> TileParseResult:
    width, height = image.size
    calibration = resolve_calibration_profile(width, height, calibration_dir=calibration_dir)
    if calibration.screen_width <= 0 or calibration.screen_height <= 0:
        calibration.screen_width = width
        calibration.screen_height = height
    layout = build_hand_layout(width, height, calibration=calibration)
    fixture = None if fixture_mode == "disabled" else _load_fixture(image_path)
    base_state = _classify_tile_level_state(scene=scene, metrics=metrics, calibration=calibration)

    if fixture is not None:
        return _from_fixture(fixture, calibration=calibration, layout=layout, base_state=base_state)

    if scene not in TILE_PARSE_SCENES:
        return _scene_suppressed_result(
            scene=scene,
            calibration=calibration,
            layout=layout,
            base_state=base_state,
        )

    vit_result = _from_vit_classifier(
        image,
        calibration=calibration,
        base_state=base_state,
        classifier_config=tile_classifier_config,
    )
    if vit_result is not None:
        vit_result = _with_visual_riichi_result(vit_result, image=image)
        return _with_external_discard_result(vit_result, image_path=image_path, image=image)

    vit_discard_result = _from_vit_discard_classifier(
        image,
        calibration=calibration,
        base_state=base_state,
        classifier_config=tile_classifier_config,
    )
    if vit_discard_result is not None:
        vit_discard_result = _with_visual_riichi_result(vit_discard_result, image=image)
        return _with_external_discard_result(vit_discard_result, image_path=image_path, image=image)

    template_result = _from_template_profile(
        image_path,
        image,
        calibration=calibration,
        base_state=base_state,
        classifier_config=tile_classifier_config,
    )
    if template_result is not None:
        template_result = _with_visual_riichi_result(template_result, image=image)
        return _with_external_discard_result(template_result, image_path=image_path, image=image)

    slot_metrics = _collect_slot_metrics(image, layout["hand"][:14])
    confidence = 0.42 if base_state == "tile_level_partial" else 0.0
    raw_detections = [
        {
            "slot_id": item["slot_id"],
            "group": "hand",
            "candidate_tile": "",
            "confidence": 0.0,
            "box": item["box"],
            "slot_mean_luma": item["slot_mean_luma"],
            "slot_colorful_ratio": item["slot_colorful_ratio"],
        }
        for item in slot_metrics[:6]
    ]
    result = TileParseResult(
        hand_tiles=[],
        melds=[],
        dora_indicators=[],
        riichi_players=[],
        raw_detections=raw_detections,
        analysis_hints={
            "analysis_version": "mahjong-core-v1",
            "tile_level_state": base_state,
            "tile_level_available": False,
            "analysis_confidence": confidence,
            "calibration_profile": calibration.profile_id,
            "calibration_enabled": calibration.enabled,
            "tile_parser_source": "heuristic_layout_only",
            "hand_slot_count": len(layout["hand"]),
        },
    )
    result = _with_visual_riichi_result(result, image=image)
    return _with_external_discard_result(result, image_path=image_path, image=image)


def _scene_suppressed_result(
    *,
    scene: str,
    calibration: CalibrationProfile,
    layout: dict[str, list[TileSlot]],
    base_state: str,
) -> TileParseResult:
    return TileParseResult(
        analysis_hints={
            "analysis_version": "mahjong-core-v1",
            "tile_level_state": base_state,
            "tile_level_available": False,
            "analysis_confidence": 0.0,
            "calibration_profile": calibration.profile_id,
            "calibration_enabled": calibration.enabled,
            "tile_parser_source": "scene_suppressed",
            "tile_parser_suppressed_scene": scene,
            "hand_slot_count": len(layout["hand"]),
        },
    )


def _from_template_profile(
    image_path: Path,
    image: Image.Image,
    *,
    calibration: CalibrationProfile,
    base_state: str,
    classifier_config: dict[str, Any] | None = None,
) -> TileParseResult | None:
    if not calibration.enabled or not calibration.hand_tile_templates:
        return None

    width, height = image.size
    hand_result = _best_template_hand_result(
        image,
        calibration=calibration,
        base_layout=build_hand_layout(width, height, calibration=calibration),
    )
    hand_tiles = hand_result["hand_tiles"]
    raw_detections = hand_result["raw_detections"]
    confidences = hand_result["confidences"]
    selected_layout = hand_result["layout"]

    discard_templates = _combined_discard_template_payload(
        discard_payload=calibration.discard_tile_templates,
        hand_payload=calibration.hand_tile_templates,
    )
    _ = image_path
    discard_result = parse_discards_from_image(image, discard_templates, classifier_config=classifier_config)
    raw_detections.extend(discard_result.raw_detections)
    analysis_confidence = round(sum(confidences) / max(1, len(confidences)), 3) if confidences else 0.0
    reliable_tile_count = len(hand_tiles) >= 10 or len(hand_tiles) in DISCARD_TURN_HAND_COUNTS
    tile_level_state = "tile_level_reliable" if reliable_tile_count and analysis_confidence >= 0.55 else base_state
    inferred_meld_count = _infer_meld_count_from_hand_count(len(hand_tiles))
    analysis_hints = {
        "analysis_version": "mahjong-core-v1",
        "tile_level_state": tile_level_state,
        "tile_level_available": bool(hand_tiles),
        "analysis_confidence": analysis_confidence,
        "calibration_profile": calibration.profile_id,
        "calibration_enabled": calibration.enabled,
        "tile_parser_source": "template_profile",
        "hand_slot_count": len(selected_layout["hand"]),
        "recognized_hand_tile_count": len(hand_tiles),
        "min_hand_tile_confidence": MIN_HAND_TILE_CONFIDENCE,
        "hand_layout_draw_slot_index": hand_result["draw_slot_index"],
        "hand_tile_slots": _hand_tile_slots_from_detections(raw_detections, hand_tiles),
    }
    if inferred_meld_count:
        analysis_hints["recognized_meld_group_count"] = inferred_meld_count
        analysis_hints["post_meld_hand_shape"] = _hand_shape_from_count(len(hand_tiles))
    analysis_hints.update(discard_result.analysis_hints)
    analysis_hints["discard_template_source"] = _discard_template_source(calibration)
    if discard_result.visible_tiles:
        analysis_hints["visible_tiles"] = list(discard_result.visible_tiles)
        if _discard_state_reliable(discard_result.analysis_hints):
            analysis_hints["deck_state_complete"] = True
            analysis_hints["deck_state_source"] = "discard_parser"

    return TileParseResult(
        hand_tiles=hand_tiles,
        melds=[],
        dora_indicators=[],
        riichi_players=[],
        discard_piles=discard_result.discard_piles,
        visible_tiles=discard_result.visible_tiles,
        raw_detections=raw_detections,
        analysis_hints=analysis_hints,
    )


def _from_vit_classifier(
    image: Image.Image,
    *,
    calibration: CalibrationProfile,
    base_state: str,
    classifier_config: dict[str, Any] | None,
) -> TileParseResult | None:
    if not vit_classifier_enabled(classifier_config, area="hand"):
        return None

    try:
        hand_result = _best_vit_hand_result(
            image,
            calibration=calibration,
            classifier_config=classifier_config,
        )
    except VitTileClassifierUnavailable as exc:
        logger.info("mahjong ViT tile classifier unavailable; falling back to templates: %s", exc)
        return None
    except Exception as exc:
        logger.warning("mahjong ViT hand classification failed; falling back to templates: %s", exc)
        return None

    hand_tiles = hand_result["hand_tiles"]
    raw_detections = hand_result["raw_detections"]
    confidences = hand_result["confidences"]
    selected_layout = hand_result["layout"]
    analysis_confidence = round(sum(confidences) / max(1, len(confidences)), 3) if confidences else 0.0
    if not _vit_hand_result_reliable(hand_tiles, analysis_confidence, classifier_config=classifier_config):
        return None

    discard_templates = _combined_discard_template_payload(
        discard_payload=calibration.discard_tile_templates,
        hand_payload=calibration.hand_tile_templates,
    )
    discard_result = parse_discards_from_image(
        image,
        discard_templates,
        classifier_config=classifier_config,
    )
    tile_level_state = "tile_level_reliable" if analysis_confidence >= _vit_min_mean_confidence(classifier_config) else base_state
    inferred_meld_count = _infer_meld_count_from_hand_count(len(hand_tiles))
    analysis_hints = {
        "analysis_version": "mahjong-core-v1",
        "tile_level_state": tile_level_state,
        "tile_level_available": bool(hand_tiles),
        "analysis_confidence": analysis_confidence,
        "calibration_profile": calibration.profile_id,
        "calibration_enabled": calibration.enabled,
        "tile_parser_source": "vit_classifier",
        "vit_model": vit_model_from_config(classifier_config),
        "vit_min_confidence": _vit_min_confidence(classifier_config),
        "vit_min_mean_confidence": _vit_min_mean_confidence(classifier_config),
        "hand_slot_count": len(selected_layout["hand"]),
        "recognized_hand_tile_count": len(hand_tiles),
        "hand_layout_draw_slot_index": hand_result["draw_slot_index"],
        "hand_tile_slots": _hand_tile_slots_from_detections(raw_detections, hand_tiles),
    }
    if inferred_meld_count:
        analysis_hints["recognized_meld_group_count"] = inferred_meld_count
        analysis_hints["post_meld_hand_shape"] = _hand_shape_from_count(len(hand_tiles))
    analysis_hints.update(discard_result.analysis_hints)
    if discard_result.visible_tiles:
        analysis_hints["visible_tiles"] = list(discard_result.visible_tiles)
        if _discard_state_reliable(discard_result.analysis_hints):
            analysis_hints["deck_state_complete"] = True
            analysis_hints["deck_state_source"] = discard_result.analysis_hints.get(
                "deck_state_source",
                "discard_parser",
            )

    return TileParseResult(
        hand_tiles=hand_tiles,
        melds=[],
        dora_indicators=[],
        riichi_players=[],
        discard_piles=discard_result.discard_piles,
        visible_tiles=discard_result.visible_tiles,
        raw_detections=raw_detections + discard_result.raw_detections,
        analysis_hints=analysis_hints,
    )


def _from_vit_discard_classifier(
    image: Image.Image,
    *,
    calibration: CalibrationProfile,
    base_state: str,
    classifier_config: dict[str, Any] | None,
) -> TileParseResult | None:
    if not vit_classifier_enabled(classifier_config, area="discard"):
        return None

    discard_result = parse_discards_from_image(
        image,
        {},
        classifier_config=classifier_config,
    )
    if not discard_result.visible_tiles:
        return None

    analysis_hints = {
        "analysis_version": "mahjong-core-v1",
        "tile_level_state": base_state,
        "tile_level_available": False,
        "analysis_confidence": 0.0,
        "calibration_profile": calibration.profile_id,
        "calibration_enabled": calibration.enabled,
        "tile_parser_source": "vit_discard_only",
        "vit_model": vit_model_from_config(classifier_config),
        "visible_tiles": list(discard_result.visible_tiles),
    }
    analysis_hints.update(discard_result.analysis_hints)
    if _discard_state_reliable(discard_result.analysis_hints):
        analysis_hints["deck_state_complete"] = True
        analysis_hints["deck_state_source"] = "vit_discard_parser"

    return TileParseResult(
        discard_piles=discard_result.discard_piles,
        visible_tiles=discard_result.visible_tiles,
        raw_detections=discard_result.raw_detections,
        analysis_hints=analysis_hints,
    )


def _best_vit_hand_result(
    image: Image.Image,
    *,
    calibration: CalibrationProfile,
    classifier_config: dict[str, Any] | None,
) -> dict[str, Any]:
    width, height = image.size
    base_layout = build_hand_layout(width, height, calibration=calibration)
    all_crops: list[Image.Image] = []
    candidates = [
        _prepare_vit_hand_candidate(
            image,
            layout=base_layout if draw_slot_index == 14 else build_hand_layout(
                width,
                height,
                calibration=calibration,
                draw_slot_index=draw_slot_index,
            ),
            draw_slot_index=draw_slot_index,
            all_crops=all_crops,
        )
        for draw_slot_index in _candidate_draw_slot_indices()
    ]
    predictions = classify_tile_crops(
        all_crops,
        model=vit_model_from_config(classifier_config),
        device=vit_device_from_config(classifier_config),
        top_k=vit_top_k_from_config(classifier_config),
    )
    for candidate in candidates:
        _finalize_vit_hand_candidate(
            candidate,
            predictions=predictions,
            classifier_config=classifier_config,
        )
    return max(candidates, key=_hand_layout_score)


def _prepare_vit_hand_candidate(
    image: Image.Image,
    *,
    layout: dict[str, list[TileSlot]],
    draw_slot_index: int,
    all_crops: list[Image.Image],
) -> dict[str, Any]:
    raw_detections: list[dict[str, Any]] = []
    crop_refs: list[tuple[int, int]] = []
    seen_occupied = False
    for slot, slot_metrics in zip(layout["hand"][:14], _collect_slot_metrics(image, layout["hand"][:14]), strict=True):
        occupied = is_probably_occupied_hand_slot(slot_metrics)
        detection = {
            "slot_id": slot.slot_id,
            "group": slot.group,
            "candidate_tile": "",
            "confidence": 0.0,
            "box": slot.box.to_dict(),
            "slot_mean_luma": slot_metrics.get("slot_mean_luma"),
            "slot_colorful_ratio": slot_metrics.get("slot_colorful_ratio"),
            "occupied": occupied,
            "source": "vit_classifier",
        }
        raw_detections.append(detection)
        if not occupied:
            if seen_occupied:
                break
            continue

        seen_occupied = True
        crop = image.crop((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom))
        crop_refs.append((len(raw_detections) - 1, len(all_crops)))
        all_crops.append(crop)

    return {
        "hand_tiles": [],
        "raw_detections": raw_detections,
        "confidences": [],
        "draw_slot_index": draw_slot_index,
        "layout": layout,
        "crop_refs": crop_refs,
    }


def _finalize_vit_hand_candidate(
    candidate: dict[str, Any],
    *,
    predictions: list[Any],
    classifier_config: dict[str, Any] | None,
) -> None:
    hand_tiles: list[str] = []
    confidences: list[float] = []
    raw_detections = candidate["raw_detections"]
    min_confidence = _vit_min_confidence(classifier_config)
    for detection_index, crop_index in candidate.get("crop_refs", []):
        if crop_index >= len(predictions):
            continue
        prediction = predictions[crop_index]
        if prediction is None:
            continue
        detection = raw_detections[detection_index]
        detection.update(prediction.to_detection_fields())
        detection["vit_model"] = vit_model_from_config(classifier_config)
        if prediction.confidence < min_confidence:
            detection["accepted"] = False
            detection["rejection_reason"] = "low_confidence"
            continue
        detection["accepted"] = True
        hand_tiles.append(prediction.tile)
        confidences.append(prediction.confidence)
    candidate["hand_tiles"] = hand_tiles
    candidate["confidences"] = confidences
    candidate.pop("crop_refs", None)


def _vit_hand_result_reliable(
    hand_tiles: list[str],
    analysis_confidence: float,
    *,
    classifier_config: dict[str, Any] | None,
) -> bool:
    if not hand_tiles:
        return False
    if _config_bool(classifier_config, "allow_partial", default=False):
        return analysis_confidence >= _vit_min_confidence(classifier_config)
    count = len(hand_tiles)
    reliable_tile_count = count >= 10 or count in DISCARD_TURN_HAND_COUNTS or count in WAITING_HAND_COUNTS
    return reliable_tile_count and analysis_confidence >= _vit_min_mean_confidence(classifier_config)


def _vit_min_confidence(classifier_config: dict[str, Any] | None) -> float:
    env_value = os.environ.get("MAHJONG_COMPANION_VIT_MIN_CONFIDENCE")
    if env_value is not None:
        return _coerce_float(env_value, default=0.65)
    if isinstance(classifier_config, dict):
        return _coerce_float(classifier_config.get("min_confidence"), default=0.65)
    return 0.65


def _vit_min_mean_confidence(classifier_config: dict[str, Any] | None) -> float:
    env_value = os.environ.get("MAHJONG_COMPANION_VIT_MIN_MEAN_CONFIDENCE")
    if env_value is not None:
        return _coerce_float(env_value, default=0.70)
    if isinstance(classifier_config, dict):
        return _coerce_float(classifier_config.get("min_mean_confidence"), default=0.70)
    return 0.70


def _config_bool(config: dict[str, Any] | None, key: str, *, default: bool) -> bool:
    if not isinstance(config, dict) or key not in config:
        return default
    value = config.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "disabled"}


def _best_template_hand_result(
    image: Image.Image,
    *,
    calibration: CalibrationProfile,
    base_layout: dict[str, list[TileSlot]],
) -> dict[str, Any]:
    width, height = image.size
    draw_indices = _candidate_draw_slot_indices()

    def _evaluate_layout(draw_slot_index: int) -> dict[str, Any]:
        layout = (
            base_layout
            if draw_slot_index == 14
            else build_hand_layout(width, height, calibration=calibration, draw_slot_index=draw_slot_index)
        )
        result = _classify_hand_from_layout(image, calibration=calibration, layout=layout)
        result["draw_slot_index"] = draw_slot_index
        result["layout"] = layout
        return result

    candidates = [_evaluate_layout(i) for i in draw_indices]

    return max(candidates, key=_hand_layout_score)


def _candidate_draw_slot_indices() -> list[int]:
    return [14, 11, 8, 5, 2]


def _classify_hand_from_layout(
    image: Image.Image,
    *,
    calibration: CalibrationProfile,
    layout: dict[str, list[TileSlot]],
) -> dict[str, Any]:
    hand_tiles: list[str] = []
    raw_detections: list[dict[str, Any]] = []
    confidences: list[float] = []
    template_payload = _combined_hand_template_payload(
        hand_payload=calibration.hand_tile_templates,
        discard_payload=calibration.discard_tile_templates,
    )

    for slot, slot_metrics in zip(layout["hand"][:14], _collect_slot_metrics(image, layout["hand"][:14]), strict=True):
        occupied = is_probably_occupied_hand_slot(slot_metrics)
        detection = {
            "slot_id": slot.slot_id,
            "group": slot.group,
            "candidate_tile": "",
            "confidence": 0.0,
            "box": slot.box.to_dict(),
            "slot_mean_luma": slot_metrics.get("slot_mean_luma"),
            "slot_colorful_ratio": slot_metrics.get("slot_colorful_ratio"),
            "occupied": occupied,
            "source": "template_profile",
        }
        if not occupied:
            raw_detections.append(detection)
            if hand_tiles:
                break
            continue

        crop = image.crop((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom))
        match = classify_tile_from_templates(crop, template_payload)
        if match is None:
            raw_detections.append(detection)
            continue

        detection.update(
            {
                "candidate_tile": match.tile,
                "confidence": match.confidence,
                "template_distance": match.distance,
                "runner_up_tile": match.runner_up_tile,
                "runner_up_distance": match.runner_up_distance,
            },
        )
        rejection_reason = _match_rejection_reason(match)
        if not rejection_reason:
            rejection_reason = _cross_check_ambiguous_partners(match, crop, template_payload)
        if rejection_reason:
            detection["accepted"] = False
            detection["rejection_reason"] = rejection_reason
            raw_detections.append(detection)
            continue

        hand_tiles.append(match.tile)
        confidences.append(match.confidence)
        detection["accepted"] = True
        raw_detections.append(detection)

    return {
        "hand_tiles": hand_tiles,
        "raw_detections": raw_detections,
        "confidences": confidences,
    }


def _hand_layout_score(result: dict[str, Any]) -> tuple[int, int, float, int]:
    tiles = result.get("hand_tiles") if isinstance(result.get("hand_tiles"), list) else []
    confidences = result.get("confidences") if isinstance(result.get("confidences"), list) else []
    count = len(tiles)
    mean_confidence = sum(float(value or 0.0) for value in confidences) / max(1, len(confidences))
    shape_score = 2 if count in (DISCARD_TURN_HAND_COUNTS | WAITING_HAND_COUNTS) else 0
    if count >= 12:
        shape_score += 1
    return (shape_score, count, mean_confidence, -abs(int(result.get("draw_slot_index", 14) or 14) - 14))


_AMBIGUOUS_PAIRS: list[tuple[set[str], float, float]] = [
    ({"6s", "9s"}, 0.08, 0.72),
    ({"4p", "5p"}, 0.16, 0.70),
    ({"5p", "6p"}, 0.18, 0.76),
    ({"6p", "7p"}, 0.12, 0.70),
    ({"2p", "3p"}, 0.14, 0.68),
    ({"1m", "7m"}, 0.12, 0.70),
]


def _match_rejection_reason(match: Any) -> str:
    if float(getattr(match, "confidence", 0.0) or 0.0) < MIN_HAND_TILE_CONFIDENCE:
        return "low_confidence"
    tile = str(getattr(match, "tile", "") or "")
    runner = str(getattr(match, "runner_up_tile", "") or "")
    pair = {tile, runner}
    for ambiguous_set, margin_max, conf_max in _AMBIGUOUS_PAIRS:
        if pair == ambiguous_set:
            distance = _coerce_float(getattr(match, "distance", None), default=0.0)
            runner_distance = _coerce_float(getattr(match, "runner_up_distance", None), default=0.0)
            if runner_distance > 0:
                margin_ratio = (runner_distance - distance) / runner_distance
                confidence = float(getattr(match, "confidence", 0.0) or 0.0)
                if margin_ratio < margin_max and confidence < conf_max:
                    return f"ambiguous_{tile}_{runner}".replace(" ", "_")
            break
    return ""


def _cross_check_ambiguous_partners(
    match: Any,
    crop: Image.Image,
    payload: dict[str, Any],
) -> str:
    tile = str(getattr(match, "tile", "") or "")
    best_distance = float(getattr(match, "distance", 0.0) or 0.0)
    confidence = float(getattr(match, "confidence", 0.0) or 0.0)
    if not tile or best_distance <= 0:
        return ""

    for ambiguous_set, margin_max, conf_max in _AMBIGUOUS_PAIRS:
        if tile not in ambiguous_set:
            continue
        partner = next(iter(ambiguous_set - {tile}), "")
        if not partner:
            continue
        templates = payload.get("templates", {})
        partner_sigs = templates.get(partner)
        if isinstance(partner_sigs, dict):
            partner_sigs = partner_sigs.get("signatures")
        if not isinstance(partner_sigs, list) or not partner_sigs:
            continue

        try:
            query = extract_tile_signature(
                crop,
                inner_bounds=payload.get("inner_bounds"),
                width=payload.get("width"),
                height=payload.get("height"),
            )
        except Exception:
            return ""

        expected_signature_length = (
            _coerce_int(payload.get("width"), default=16)
            * _coerce_int(payload.get("height"), default=24)
            * 3
        )
        min_partner_dist = float("inf")
        for sig in partner_sigs:
            decoded = _decode_signature(sig, expected_length=expected_signature_length)
            if not decoded:
                continue
            dist = _rms_distance(query, decoded)
            if dist < min_partner_dist:
                min_partner_dist = dist

        if min_partner_dist == float("inf"):
            continue

        max_dist = max(best_distance, min_partner_dist)
        min_dist = min(best_distance, min_partner_dist)
        margin_ratio = (max_dist - min_dist) / max_dist if max_dist > 0 else 0.0
        if margin_ratio < margin_max and confidence < conf_max:
            return f"ambiguous_{tile}_{partner}".replace(" ", "_")
    return ""


def _with_external_discard_result(
    result: TileParseResult,
    *,
    image_path: Path,
    image: Image.Image,
) -> TileParseResult:
    external_piles, hints = load_external_discard_result(image_path, image)
    result.analysis_hints.update(hints)
    if not external_piles:
        return result

    result.discard_piles = external_piles
    result.visible_tiles = _visible_tiles_from_discard_piles(external_piles)
    result.raw_detections.extend(
        _raw_discard_detections_from_piles(
            external_piles,
            analysis_confidence=1.0,
        ),
    )
    result.analysis_hints["visible_tiles"] = list(result.visible_tiles)
    result.analysis_hints["discard_parser_source"] = "external_discard_recognizer"
    result.analysis_hints["recognized_discard_tile_count"] = len(result.visible_tiles)
    if result.visible_tiles:
        result.analysis_hints["deck_state_complete"] = True
        result.analysis_hints["deck_state_source"] = "external_discard_recognizer"
    return result


def _with_visual_riichi_result(result: TileParseResult, *, image: Image.Image) -> TileParseResult:
    players, detections = detect_riichi_players(image)
    result.analysis_hints["riichi_detector_source"] = "riichi_stick_detector"
    result.analysis_hints["riichi_stick_count"] = len(detections)
    if detections:
        result.analysis_hints["riichi_stick_detections"] = detections
        result.raw_detections.extend(detections)
    if players and not result.riichi_players:
        result.riichi_players = players
    return result


def _combined_discard_template_payload(
    *,
    discard_payload: dict[str, Any],
    hand_payload: dict[str, Any],
) -> dict[str, Any]:
    if not discard_payload:
        return hand_payload
    if not hand_payload:
        return discard_payload
    discard_templates = discard_payload.get("templates")
    hand_templates = hand_payload.get("templates")
    if not isinstance(discard_templates, dict) or not isinstance(hand_templates, dict):
        return discard_payload
    if discard_payload.get("version") != hand_payload.get("version"):
        return discard_payload
    if discard_payload.get("signature_version") != hand_payload.get("signature_version"):
        return discard_payload

    merged = dict(hand_payload)
    merged_templates: dict[str, dict[str, Any]] = {
        str(tile): dict(item)
        for tile, item in hand_templates.items()
        if isinstance(item, dict)
    }
    max_samples_per_tile = _coerce_int(
        discard_payload.get("max_samples_per_tile") or hand_payload.get("max_samples_per_tile"),
        default=12,
    )
    for tile, item in discard_templates.items():
        if not isinstance(item, dict):
            continue
        tile_key = str(tile)
        hand_item = merged_templates.get(tile_key, {})
        discard_signatures = _signature_list(item.get("signatures"))
        hand_signatures = _signature_list(hand_item.get("signatures"))
        signatures = (discard_signatures + hand_signatures)[: max(1, max_samples_per_tile)]
        if not signatures:
            continue
        merged_templates[tile_key] = {
            **hand_item,
            "count": _coerce_int(item.get("count"), default=len(discard_signatures))
            + _coerce_int(hand_item.get("count"), default=len(hand_signatures)),
            "signatures": signatures,
        }

    merged["templates"] = merged_templates
    merged["source_sample_count"] = _coerce_int(discard_payload.get("source_sample_count"), default=0) + _coerce_int(
        hand_payload.get("source_sample_count"),
        default=0,
    )
    merged["stored_sample_count"] = sum(
        len(_signature_list(item.get("signatures")))
        for item in merged_templates.values()
        if isinstance(item, dict)
    )
    merged["source"] = "discard_and_hand_tile_templates"
    return merged


def _combined_hand_template_payload(
    *,
    hand_payload: dict[str, Any],
    discard_payload: dict[str, Any],
) -> dict[str, Any]:
    if not discard_payload:
        return hand_payload
    if not hand_payload:
        return discard_payload
    discard_templates = discard_payload.get("templates")
    hand_templates = hand_payload.get("templates")
    if not isinstance(discard_templates, dict) or not isinstance(hand_templates, dict):
        return hand_payload
    if discard_payload.get("version") != hand_payload.get("version"):
        return hand_payload
    if discard_payload.get("signature_version") != hand_payload.get("signature_version"):
        return hand_payload

    merged = dict(hand_payload)
    merged_templates: dict[str, dict[str, Any]] = {
        str(tile): dict(item)
        for tile, item in hand_templates.items()
        if isinstance(item, dict)
    }
    max_samples_per_tile = _coerce_int(
        hand_payload.get("max_samples_per_tile") or discard_payload.get("max_samples_per_tile"),
        default=12,
    )
    for tile, item in discard_templates.items():
        if not isinstance(item, dict):
            continue
        tile_key = str(tile)
        hand_item = merged_templates.get(tile_key, {})
        hand_signatures = _signature_list(hand_item.get("signatures"))
        discard_signatures = _signature_list(item.get("signatures"))
        signatures = (hand_signatures + discard_signatures)[: max(1, max_samples_per_tile)]
        if not signatures:
            continue
        merged_templates[tile_key] = {
            **hand_item,
            "count": _coerce_int(hand_item.get("count"), default=len(hand_signatures))
            + _coerce_int(item.get("count"), default=len(discard_signatures)),
            "signatures": signatures,
        }

    merged["templates"] = merged_templates
    merged["source_sample_count"] = _coerce_int(hand_payload.get("source_sample_count"), default=0) + _coerce_int(
        discard_payload.get("source_sample_count"),
        default=0,
    )
    merged["stored_sample_count"] = sum(
        len(_signature_list(item.get("signatures")))
        for item in merged_templates.values()
        if isinstance(item, dict)
    )
    merged["source"] = "hand_and_discard_tile_templates"
    return merged


def _discard_template_source(calibration: CalibrationProfile) -> str:
    if calibration.discard_tile_templates and calibration.hand_tile_templates:
        return "discard_and_hand_tile_templates"
    if calibration.discard_tile_templates:
        return "discard_tile_templates"
    return "hand_tile_templates"


def _signature_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def enrich_perceived_state_with_tiles(
    perceived: PerceivedGameState,
    image_path: Path,
    image: Image.Image,
    *,
    metrics: dict[str, dict[str, Any]],
    calibration_dir: Path | None = None,
    fixture_mode: str = "auto",
    tile_classifier_config: dict[str, Any] | None = None,
) -> PerceivedGameState:
    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene=perceived.scene,
        metrics=metrics,
        calibration_dir=calibration_dir,
        fixture_mode=fixture_mode,
        tile_classifier_config=tile_classifier_config,
    )
    payload = perceived.to_dict()
    updates = parsed.to_state_updates()
    riichi_players = parsed.riichi_players or list(perceived.riichi_players)
    known_genbutsu_tiles = list(parsed.known_genbutsu_tiles)
    inferred_user_turn = (
        perceived.scene in {"in_match", "unknown"}
        and not perceived.buttons
        and not perceived.is_user_turn
        and len(parsed.hand_tiles) in DISCARD_TURN_HAND_COUNTS
    )
    if riichi_players and parsed.discard_piles and not known_genbutsu_tiles:
        known_genbutsu_tiles = _derive_known_genbutsu_tiles(
            riichi_players=riichi_players,
            discard_piles=parsed.discard_piles,
        )
    updates["riichi_players"] = riichi_players
    updates["known_genbutsu_tiles"] = known_genbutsu_tiles
    if inferred_user_turn:
        updates["scene"] = "in_match"
        updates["is_user_turn"] = True
        notes = list(perceived.notes)
        notes.append(f"{len(parsed.hand_tiles)} recognized hand tiles imply user discard turn")
        updates["notes"] = notes
        analysis_hints = dict(updates.get("analysis_hints", {}))
        analysis_hints["user_turn_inferred_from_hand_count"] = len(parsed.hand_tiles)
        updates["analysis_hints"] = analysis_hints
    if known_genbutsu_tiles:
        analysis_hints = dict(updates.get("analysis_hints", {}))
        analysis_hints["known_genbutsu_tiles"] = list(known_genbutsu_tiles)
        updates["analysis_hints"] = analysis_hints
    payload.update(updates)
    return PerceivedGameState(**payload)


def _from_fixture(
    fixture: dict[str, Any],
    *,
    calibration: CalibrationProfile,
    layout: dict[str, list[TileSlot]],
    base_state: str,
) -> TileParseResult:
    hand_tiles = _normalize_tile_list(fixture.get("hand_tiles"))
    melds = _normalize_group_list(fixture.get("melds"))
    dora_indicators = _normalize_tile_list(fixture.get("dora_indicators"))
    riichi_players = _normalize_tile_list(fixture.get("riichi_players"))
    discard_piles = _normalize_discard_piles(fixture.get("discard_piles"))
    visible_tiles = _normalize_tile_list(fixture.get("visible_tiles")) or _visible_tiles_from_discard_piles(
        discard_piles,
    )
    known_genbutsu_tiles = _normalize_tile_list(
        fixture.get("known_genbutsu_tiles")
        or fixture.get("genbutsu_tiles")
        or fixture.get("confirmed_safe_tiles"),
    )
    if not known_genbutsu_tiles:
        known_genbutsu_tiles = _derive_known_genbutsu_tiles(
            riichi_players=riichi_players,
            discard_piles=discard_piles,
        )
    analysis_confidence = float(fixture.get("analysis_confidence", 0.86) or 0.86)
    tile_level_state = str(fixture.get("tile_level_state", "")).strip() or (
        "tile_level_reliable" if hand_tiles else base_state
    )
    raw_detections = fixture.get("raw_detections")
    if not isinstance(raw_detections, list):
        raw_detections = _raw_detections_from_label(fixture, hand_tiles)
    if not isinstance(raw_detections, list):
        fixture_layout = build_hand_layout(
            calibration.screen_width,
            calibration.screen_height,
            calibration=calibration,
            draw_slot_index=_draw_slot_index_for_hand_count(len(hand_tiles)),
        )
        raw_detections = [
            {
                "slot_id": slot.slot_id,
                "group": slot.group,
                "candidate_tile": hand_tiles[index] if index < len(hand_tiles) else "",
                "confidence": analysis_confidence,
                "box": slot.box.to_dict(),
            }
            for index, slot in enumerate(fixture_layout["hand"][: len(hand_tiles)])
        ]
    if discard_piles:
        raw_detections.extend(
            _raw_discard_detections_from_piles(discard_piles, analysis_confidence=analysis_confidence),
        )

    fixture_hints = fixture.get("analysis_hints") if isinstance(fixture.get("analysis_hints"), dict) else {}
    analysis_hints = {
        **fixture_hints,
        "analysis_version": "mahjong-core-v1",
        "tile_level_state": tile_level_state,
        "tile_level_available": bool(hand_tiles),
        "analysis_confidence": analysis_confidence,
        "calibration_profile": calibration.profile_id,
        "calibration_enabled": calibration.enabled,
        "tile_parser_source": "fixture",
        "recognized_hand_tile_count": len(hand_tiles),
        "hand_tile_slots": _hand_tile_slots_from_detections(raw_detections, hand_tiles),
    }
    inferred_meld_count = _infer_meld_count_from_hand_count(len(hand_tiles))
    if inferred_meld_count and not analysis_hints.get("recognized_meld_group_count"):
        analysis_hints["recognized_meld_group_count"] = inferred_meld_count
        analysis_hints["post_meld_hand_shape"] = _hand_shape_from_count(len(hand_tiles))
    if visible_tiles:
        analysis_hints["visible_tiles"] = list(visible_tiles)
        if fixture_hints.get("deck_state_complete") is not False:
            analysis_hints["deck_state_complete"] = True
            analysis_hints.setdefault("deck_state_source", "fixture")
    if known_genbutsu_tiles:
        analysis_hints["known_genbutsu_tiles"] = list(known_genbutsu_tiles)

    return TileParseResult(
        hand_tiles=hand_tiles,
        melds=melds,
        dora_indicators=dora_indicators,
        riichi_players=riichi_players,
        discard_piles=discard_piles,
        visible_tiles=visible_tiles,
        known_genbutsu_tiles=known_genbutsu_tiles,
        raw_detections=[item for item in raw_detections if isinstance(item, dict)],
        analysis_hints=analysis_hints,
    )


def _classify_tile_level_state(
    *,
    scene: str,
    metrics: dict[str, dict[str, Any]],
    calibration: CalibrationProfile,
) -> str:
    hand_metrics = metrics.get("bottom_hand_area", {})
    if scene not in TILE_PARSE_SCENES:
        return "tile_level_unavailable"
    if not calibration.enabled:
        return "tile_level_partial"
    if float(hand_metrics.get("colorful_ratio", 0.0) or 0.0) >= 0.42:
        return "tile_level_partial"
    if scene in {"unknown", "dialog"}:
        return "tile_level_partial"
    return "tile_level_unavailable"


def _collect_slot_metrics(image: Image.Image, slots: list[TileSlot]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for slot in slots:
        slot_metrics = collect_region_metrics(image, slot.box, sample_step=4)
        metrics.append(
            {
                "slot_id": slot.slot_id,
                "box": slot.box.to_dict(),
                "slot_mean_luma": slot_metrics.get("mean_luma"),
                "slot_colorful_ratio": slot_metrics.get("colorful_ratio"),
                "slot_bright_ratio": slot_metrics.get("bright_ratio"),
                "slot_dark_ratio": slot_metrics.get("dark_ratio"),
                "slot_stddev": slot_metrics.get("stddev"),
            }
        )
    return metrics


def _hand_tile_slots_from_detections(
    raw_detections: list[dict[str, Any]],
    hand_tiles: list[str],
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    accepted = [
        item
        for item in raw_detections
        if isinstance(item, dict)
        and item.get("group") == "hand"
        and item.get("accepted", True) is not False
        and str(item.get("candidate_tile", "")).strip()
    ]
    for index, item in enumerate(accepted[: len(hand_tiles)]):
        box = item.get("box")
        if not isinstance(box, dict):
            continue
        tile = str(item.get("candidate_tile", "")).strip()
        if not tile:
            continue
        slots.append(
            {
                "slot_id": str(item.get("slot_id", f"hand_{index + 1}")),
                "tile": tile,
                "index": index,
                "box": dict(box),
                "confidence": _coerce_float(item.get("confidence"), default=0.0),
                "source": str(item.get("source", "")).strip() or "tile_parser",
            }
        )
    return slots


def _infer_meld_count_from_hand_count(hand_count: int) -> int:
    return DISCARD_TURN_HAND_COUNTS.get(hand_count) or WAITING_HAND_COUNTS.get(hand_count, 0)


def _hand_shape_from_count(hand_count: int) -> str:
    if hand_count in DISCARD_TURN_HAND_COUNTS:
        return "discard_turn"
    if hand_count in WAITING_HAND_COUNTS:
        return "waiting"
    return ""


def _draw_slot_index_for_hand_count(hand_count: int) -> int:
    if hand_count in DISCARD_TURN_HAND_COUNTS:
        return hand_count
    return 14


def _load_fixture(image_path: Path) -> dict[str, Any] | None:
    candidates = [
        image_path.with_name(f"{image_path.stem}-tiles.json"),
        image_path.with_suffix(".tiles.json"),
        image_path.with_suffix(".label.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("failed to load tile fixture %s: %s", candidate, exc)
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _raw_detections_from_label(fixture: dict[str, Any], hand_tiles: list[str]) -> list[dict[str, Any]] | None:
    layout = fixture.get("layout")
    if not isinstance(layout, dict):
        return None
    hand_slots = layout.get("hand_slots")
    if not isinstance(hand_slots, list):
        return None

    raw_detections: list[dict[str, Any]] = []
    for index, slot in enumerate(hand_slots):
        if not isinstance(slot, dict):
            continue
        tile = str(slot.get("tile", "")).strip()
        if not tile and index < len(hand_tiles):
            tile = hand_tiles[index]
        raw_detections.append(
            {
                "slot_id": str(slot.get("slot_id", f"hand_{index + 1}")),
                "group": "hand",
                "candidate_tile": tile,
                "confidence": float(fixture.get("analysis_confidence", 0.86) or 0.86),
                "box": slot.get("box") if isinstance(slot.get("box"), dict) else {},
            }
        )
    return raw_detections


def _normalize_tile_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_group_list(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    groups: list[list[str]] = []
    for item in value:
        if not isinstance(item, list):
            continue
        group = _normalize_tile_list(item)
        if group:
            groups.append(group)
    return groups


def _normalize_discard_piles(value: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        return {}
    piles: dict[str, list[dict[str, Any]]] = {}
    for player, raw_items in value.items():
        player_key = str(player).strip()
        if not player_key or not isinstance(raw_items, list):
            continue
        normalized_items: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            tile = str(item.get("tile", "")).strip()
            if not tile:
                continue
            payload = {
                "tile": tile,
                "player": str(item.get("player") or player_key),
                "turn_index": _coerce_int(item.get("turn_index"), default=index + 1),
                "confidence": _coerce_float(item.get("confidence"), default=1.0),
                "orientation": str(item.get("orientation", "")).strip() or player_key,
                "source": str(item.get("source", "")).strip() or "fixture",
            }
            bbox = item.get("bbox")
            if isinstance(bbox, list | tuple) and len(bbox) == 4:
                try:
                    payload["bbox"] = [int(part) for part in bbox]
                except (TypeError, ValueError):
                    pass
            quad = item.get("quad")
            if isinstance(quad, list | tuple) and len(quad) == 4:
                try:
                    payload["quad"] = [[int(point[0]), int(point[1])] for point in quad]
                except (TypeError, ValueError, IndexError):
                    pass
            normalized_items.append(payload)
        if normalized_items:
            piles[player_key] = normalized_items
    return piles


def _discard_state_reliable(hints: dict[str, Any]) -> bool:
    try:
        recognized_count = int(hints.get("recognized_discard_tile_count", 0) or 0)
    except (TypeError, ValueError):
        recognized_count = 0
    try:
        confidence = float(hints.get("discard_analysis_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    source = str(hints.get("discard_parser_source", "")).strip()
    if source == "external_discard_recognizer":
        return recognized_count >= 1
    return recognized_count >= 4 and confidence >= 0.62


def _visible_tiles_from_discard_piles(discard_piles: dict[str, list[dict[str, Any]]]) -> list[str]:
    tiles: list[str] = []
    for pile in discard_piles.values():
        for item in pile:
            tile = str(item.get("tile", "")).strip()
            if tile:
                tiles.append(tile)
    return tiles


def _derive_known_genbutsu_tiles(
    *,
    riichi_players: list[str],
    discard_piles: dict[str, list[dict[str, Any]]],
) -> list[str]:
    if not riichi_players:
        return []
    tiles: list[str] = []
    for player in riichi_players:
        if player not in OPPONENT_PLAYERS:
            continue
        for item in discard_piles.get(player, []):
            tile = str(item.get("tile", "")).strip()
            if tile:
                tiles.append(tile)
    return _dedupe_text(tiles)


def _raw_discard_detections_from_piles(
    discard_piles: dict[str, list[dict[str, Any]]],
    *,
    analysis_confidence: float,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for player, pile in discard_piles.items():
        for item in pile:
            tile = str(item.get("tile", "")).strip()
            if not tile:
                continue
            detections.append({
                "slot_id": f"discard_{player}_{item.get('turn_index', len(detections) + 1)}",
                "group": "discard",
                "player": player,
                "candidate_tile": tile,
                "confidence": float(item.get("confidence", analysis_confidence) or analysis_confidence),
                "box": item.get("bbox", []),
                "quad": item.get("quad", []),
                "orientation": item.get("orientation", ""),
                "source": item.get("source", "fixture"),
            })
    return detections


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
