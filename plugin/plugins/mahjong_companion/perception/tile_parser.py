from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ..contracts import PerceivedGameState
from .bottom_hand_detector import detect_bottom_hand_tiles
from .calibration import CalibrationProfile, resolve_calibration_profile
from .discard_layout import DiscardSlot, build_discard_layout
from .discard_parser import parse_discards_from_image
from .external_discard_recognizer import load_external_discard_result
from .fixture_loader import (
    load_fixture as _load_fixture,
    normalize_discard_piles as _normalize_discard_piles,
    normalize_group_list as _normalize_group_list,
    normalize_tile_list as _normalize_tile_list,
    raw_detections_from_label as _raw_detections_from_label,
    raw_discard_detections_from_piles as _raw_discard_detections_from_piles,
)
from .hand_baseline import detect_hand_baseline
from .hand_layout import TileSlot, build_hand_layout
from .riichi_detector import detect_riichi_players
from .roi import collect_region_metrics
from .tile_classifier_dispatch import classify_hand_tile
from .tile_templates import extract_tile_signature, is_probably_occupied_hand_slot
from .tile_templates import _decode_signature, _rms_distance

logger = logging.getLogger(__name__)
OPPONENT_PLAYERS = {"left_opponent", "top_opponent", "right_opponent"}
TILE_PARSE_SCENES = {"in_match", "replay", "dialog", "unknown"}
MIN_HAND_TILE_CONFIDENCE = 0.12
POST_MELD_WAITING_HAND_COUNTS = {10: 1, 7: 2, 4: 3, 1: 4}
POST_MELD_DRAW_HAND_COUNTS = {11: 1, 8: 2, 5: 3, 2: 4}
DISCARD_TURN_HAND_COUNTS = {14: 0, **POST_MELD_DRAW_HAND_COUNTS}
WAITING_HAND_COUNTS = {13: 0, **POST_MELD_WAITING_HAND_COUNTS}


def classify_tile_from_templates(crop: Image.Image, payload: dict[str, Any]):
    return classify_hand_tile(crop, payload)


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
) -> TileParseResult:
    width, height = image.size
    calibration = resolve_calibration_profile(width, height, calibration_dir=calibration_dir)
    if calibration.screen_width <= 0 or calibration.screen_height <= 0:
        calibration.screen_width = width
        calibration.screen_height = height
    baseline = detect_hand_baseline(image)
    layout = build_hand_layout(width, height, calibration=calibration, baseline=baseline)
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

    template_result = _from_template_profile(
        image_path,
        image,
        calibration=calibration,
        base_state=base_state,
        baseline=baseline,
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
    result = _with_bottom_hand_detector_result(result, image=image)
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
    baseline: object | None = None,
) -> TileParseResult | None:
    if not calibration.enabled or not calibration.hand_tile_templates:
        return None

    width, height = image.size
    hand_result = _best_template_hand_result(
        image,
        calibration=calibration,
        base_layout=build_hand_layout(width, height, calibration=calibration, baseline=baseline),
        baseline=baseline,
    )
    hand_tiles = hand_result["hand_tiles"]
    raw_detections = hand_result["raw_detections"]
    confidences = hand_result["confidences"]
    selected_layout = hand_result["layout"]

    discard_templates = calibration.discard_tile_templates or calibration.hand_tile_templates
    _ = image_path
    discard_layout = build_discard_layout(width, height, calibration=calibration, baseline=baseline)
    discard_result = parse_discards_from_image(image, discard_templates, layout=discard_layout)
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
        "anchor_driven_layout": baseline is not None,
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


def _best_template_hand_result(
    image: Image.Image,
    *,
    calibration: CalibrationProfile,
    base_layout: dict[str, list[TileSlot]],
    baseline: object | None = None,
    relaxed: bool = False,
) -> dict[str, Any]:
    width, height = image.size
    draw_indices = _candidate_draw_slot_indices()

    def _evaluate_layout(draw_slot_index: int) -> dict[str, Any]:
        layout = (
            base_layout
            if draw_slot_index == 14
            else build_hand_layout(width, height, calibration=calibration, draw_slot_index=draw_slot_index, baseline=baseline)
        )
        result = _classify_hand_from_layout(image, calibration=calibration, layout=layout, relaxed=relaxed)
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
    relaxed: bool = False,
) -> dict[str, Any]:
    hand_tiles: list[str] = []
    raw_detections: list[dict[str, Any]] = []
    confidences: list[float] = []
    template_payload = calibration.hand_tile_templates or calibration.discard_tile_templates

    for slot, slot_metrics in zip(layout["hand"][:14], _collect_slot_metrics(image, layout["hand"][:14]), strict=True):
        occupied = is_probably_occupied_hand_slot(slot_metrics, relaxed=relaxed)
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
    result.analysis_hints["discard_parser_source"] = str(
        hints.get("discard_parser_source") or "external_discard_recognizer"
    )
    result.analysis_hints["recognized_discard_tile_count"] = len(result.visible_tiles)
    if result.visible_tiles:
        unknown_count = _coerce_int(result.analysis_hints.get("model_river_unknown_count"), default=0)
        result.analysis_hints["deck_state_complete"] = unknown_count == 0
        if unknown_count > 0:
            result.analysis_hints["deck_state_partial"] = True
            result.analysis_hints["deck_state_unknown_tile_count"] = unknown_count
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


def _with_bottom_hand_detector_result(result: TileParseResult, *, image: Image.Image) -> TileParseResult:
    if result.hand_tiles:
        return result

    detection = detect_bottom_hand_tiles(image)
    result.analysis_hints["bottom_hand_detector_source"] = detection.source
    result.analysis_hints["bottom_hand_raw_slots"] = [slot.to_dict() for slot in detection.slots]
    if detection.anchor is not None:
        result.analysis_hints["bottom_hand_anchor"] = detection.anchor.to_dict()
    if not detection.hand_tiles:
        return result

    hand_count = len(detection.hand_tiles)
    result.analysis_hints["bottom_hand_recognized_tile_count"] = hand_count
    if hand_count not in DISCARD_TURN_HAND_COUNTS and hand_count not in WAITING_HAND_COUNTS:
        result.analysis_hints["bottom_hand_unsupported_count"] = hand_count
        return result

    result.hand_tiles = list(detection.hand_tiles)
    result.raw_detections.extend(
        {
            "slot_id": slot.slot_id,
            "group": "hand",
            "candidate_tile": slot.tile,
            "confidence": slot.confidence,
            "box": {
                "left": slot.bbox[0],
                "top": slot.bbox[1],
                "right": slot.bbox[2],
                "bottom": slot.bbox[3],
            },
            "accepted": slot.accepted,
            "source": detection.source,
        }
        for slot in detection.slots
        if slot.accepted and slot.tile
    )
    inferred_meld_count = _infer_meld_count_from_hand_count(hand_count)
    result.analysis_hints.update(
        {
            "tile_level_state": "tile_level_reliable" if detection.confidence >= 0.55 else "tile_level_partial",
            "tile_level_available": True,
            "analysis_confidence": detection.confidence,
            "tile_parser_source": detection.source,
            "recognized_hand_tile_count": hand_count,
            "hand_tile_slots": _hand_tile_slots_from_detections(result.raw_detections, result.hand_tiles),
        }
    )
    if inferred_meld_count:
        result.analysis_hints["recognized_meld_group_count"] = inferred_meld_count
        result.analysis_hints["post_meld_hand_shape"] = _hand_shape_from_count(hand_count)
    return result


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
) -> PerceivedGameState:
    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene=perceived.scene,
        metrics=metrics,
        calibration_dir=calibration_dir,
        fixture_mode=fixture_mode,
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
