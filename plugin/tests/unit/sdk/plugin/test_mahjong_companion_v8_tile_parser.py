from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.decision.generator import build_decision
from plugin.plugins.mahjong_companion.perception import tile_parser as tile_parser_module
from plugin.plugins.mahjong_companion.perception.calibration import CalibrationProfile, resolve_calibration_profile
from plugin.plugins.mahjong_companion.perception.discard_parser import DiscardParseResult
from plugin.plugins.mahjong_companion.perception.tile_parser import enrich_perceived_state_with_tiles, parse_tiles_from_image
from plugin.plugins.mahjong_companion.perception.tile_templates import TileTemplateMatch


def test_resolve_calibration_profile_returns_builtin_fallback_when_missing() -> None:
    profile = resolve_calibration_profile(1280, 720)

    assert profile.profile_id == "default-1280x720"
    assert profile.enabled is False
    assert profile.confidence == 0.18


def test_parse_tiles_from_image_uses_fixture_and_marks_reliable(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture-frame.png"
    Image.new("RGB", (1280, 720), color=(40, 80, 140)).save(image_path)
    (tmp_path / "fixture-frame-tiles.json").write_text(
        json.dumps(
            {
                "hand_tiles": ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "9m", "5z"],
                "dora_indicators": ["4p"],
                "analysis_confidence": 0.83,
                "tile_level_state": "tile_level_reliable",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with Image.open(image_path) as image:
        parsed = parse_tiles_from_image(
            image_path,
            image.convert("RGB"),
            scene="in_match",
            metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        )

    assert parsed.hand_tiles[:3] == ["1m", "2m", "3m"]
    assert parsed.analysis_hints["tile_level_available"] is True
    assert parsed.analysis_hints["tile_level_state"] == "tile_level_reliable"
    assert parsed.analysis_hints["analysis_confidence"] == 0.83


def test_parse_tiles_from_fixture_derives_discard_genbutsu_hints(tmp_path: Path) -> None:
    image_path = tmp_path / "discard-frame.png"
    Image.new("RGB", (1280, 720), color=(40, 80, 140)).save(image_path)
    (tmp_path / "discard-frame-tiles.json").write_text(
        json.dumps(
            {
                "hand_tiles": ["1m", "2m", "3m"],
                "riichi_players": ["right_opponent"],
                "discard_piles": {
                    "right_opponent": [
                        {"tile": "9m", "bbox": [850, 310, 890, 360], "turn_index": 1},
                        {"tile": "5z", "bbox": [895, 310, 935, 360], "turn_index": 2},
                    ],
                    "left_opponent": [
                        {"tile": "1p", "bbox": [300, 310, 340, 360], "turn_index": 1},
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with Image.open(image_path) as image:
        parsed = parse_tiles_from_image(
            image_path,
            image.convert("RGB"),
            scene="in_match",
            metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        )

    assert parsed.discard_piles["right_opponent"][0]["tile"] == "9m"
    assert parsed.visible_tiles == ["9m", "5z", "1p"]
    assert parsed.known_genbutsu_tiles == ["9m", "5z"]
    assert parsed.analysis_hints["known_genbutsu_tiles"] == ["9m", "5z"]
    assert any(item.get("group") == "discard" for item in parsed.raw_detections)


def test_enrich_perceived_state_with_tiles_emits_partial_hints_without_fixture(tmp_path: Path) -> None:
    image_path = tmp_path / "partial-frame.png"
    image = Image.new("RGB", (1280, 720), color=(80, 120, 180))
    image.save(image_path)
    perceived = PerceivedGameState(
        scene="in_match",
        confidence=0.84,
        is_user_turn=True,
    )

    enriched = enrich_perceived_state_with_tiles(
        perceived,
        image_path,
        image,
        metrics={"bottom_hand_area": {"colorful_ratio": 0.51}},
    )

    assert enriched.analysis_hints["tile_level_state"] == "tile_level_partial"
    assert enriched.analysis_hints["tile_level_available"] is False
    assert enriched.analysis_hints["analysis_version"] == "mahjong-core-v1"
    assert enriched.raw_detections


def test_enrich_perceived_state_infers_user_turn_from_14_hand_tiles(tmp_path: Path) -> None:
    image_path = tmp_path / "draw-turn-frame.png"
    image = Image.new("RGB", (1280, 720), color=(40, 80, 140))
    image.save(image_path)
    (tmp_path / "draw-turn-frame-tiles.json").write_text(
        json.dumps(
            {
                "hand_tiles": ["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p", "2s", "3s", "4s", "5z", "9m"],
                "analysis_confidence": 0.83,
                "tile_level_state": "tile_level_reliable",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    perceived = PerceivedGameState(
        scene="in_match",
        confidence=0.84,
        is_user_turn=False,
        buttons=[],
    )

    enriched = enrich_perceived_state_with_tiles(
        perceived,
        image_path,
        image,
        metrics={"bottom_hand_area": {"colorful_ratio": 0.51}},
    )

    assert enriched.is_user_turn is True
    assert enriched.analysis_hints["user_turn_inferred_from_hand_count"] == 14
    assert any("14 recognized hand tiles" in note for note in enriched.notes)
    assert build_decision(enriched).decision_type == "tile_efficiency_hint"


def test_enrich_perceived_state_infers_post_meld_user_turn_from_11_hand_tiles(tmp_path: Path) -> None:
    image_path = tmp_path / "post-meld-draw-turn-frame.png"
    image = Image.new("RGB", (1280, 720), color=(40, 80, 140))
    image.save(image_path)
    (tmp_path / "post-meld-draw-turn-frame-tiles.json").write_text(
        json.dumps(
            {
                "hand_tiles": ["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p", "5z", "9m"],
                "analysis_confidence": 0.83,
                "tile_level_state": "tile_level_reliable",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    perceived = PerceivedGameState(
        scene="unknown",
        confidence=0.48,
        is_user_turn=False,
        buttons=[],
    )

    enriched = enrich_perceived_state_with_tiles(
        perceived,
        image_path,
        image,
        metrics={"bottom_hand_area": {"colorful_ratio": 0.12}},
    )

    assert enriched.scene == "in_match"
    assert enriched.is_user_turn is True
    assert enriched.analysis_hints["user_turn_inferred_from_hand_count"] == 11
    assert enriched.analysis_hints["recognized_meld_group_count"] == 1
    assert build_decision(enriched).decision_type == "tile_efficiency_hint"


def test_parse_tiles_from_image_suppresses_lobby_scene_before_templates(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "lobby-frame.png"
    image = Image.new("RGB", (1280, 720), color=(200, 210, 225))
    image.save(image_path)

    def fail_template_parse(*args, **kwargs):
        raise AssertionError("template parsing should not run outside match scenes")

    monkeypatch.setattr(tile_parser_module, "_from_template_profile", fail_template_parse)

    with Image.open(image_path) as opened:
        parsed = parse_tiles_from_image(
            image_path,
            opened.convert("RGB"),
            scene="lobby",
            metrics={"bottom_hand_area": {"colorful_ratio": 0.78}},
        )

    assert parsed.hand_tiles == []
    assert parsed.discard_piles == {}
    assert parsed.visible_tiles == []
    assert parsed.raw_detections == []
    assert parsed.analysis_hints["tile_parser_source"] == "scene_suppressed"
    assert parsed.analysis_hints["tile_level_state"] == "tile_level_unavailable"
    assert parsed.analysis_hints["tile_level_available"] is False


def test_build_decision_propagates_v8_analysis_confidence_and_tile_level_state() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.88,
        is_user_turn=True,
        hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "9m", "5z"],
        dora_indicators=["4p"],
        analysis_hints={
            "analysis_confidence": 0.85,
            "tile_level_state": "tile_level_reliable",
        },
    )

    decision = build_decision(state)

    assert decision.mahjong_analysis["analysis_confidence"] == 0.85
    assert decision.mahjong_analysis["tile_level_state"] == "tile_level_reliable"
    assert decision.engine_meta["analysis_confidence"] == 0.85
    assert decision.engine_meta["tile_level_state"] == "tile_level_reliable"


def test_build_decision_exposes_v8_defense_alerts_under_riichi_pressure() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.86,
        is_user_turn=True,
        hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "9m", "5z"],
        dora_indicators=["4p"],
        riichi_players=["right_opponent"],
        analysis_hints={
            "analysis_confidence": 0.82,
            "tile_level_state": "tile_level_reliable",
        },
    )

    decision = build_decision(state)

    assert decision.mahjong_analysis["defense_alerts"]
    assert "立直" in decision.mahjong_analysis["defense_alerts"][0]
    assert decision.engine_meta["defense_alert_count"] >= 1


def test_build_decision_requires_14_recognized_tiles_for_discard_hint() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "9m", "5z"],
        dora_indicators=["4p"],
        analysis_hints={
            "analysis_confidence": 0.85,
            "tile_level_state": "tile_level_reliable",
            "recognized_hand_tile_count": 13,
        },
    )

    decision = build_decision(state)

    assert decision.decision_type == "scene_update"
    assert decision.recommended_focus == "turn_observe"


def test_build_decision_allows_post_meld_discard_hint_when_hand_count_matches_turn_shape() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "5z"],
        melds=[["2m", "3m", "4m"]],
        dora_indicators=["4p"],
        analysis_hints={
            "analysis_confidence": 0.85,
            "tile_level_state": "tile_level_reliable",
            "recognized_hand_tile_count": 11,
        },
    )

    decision = build_decision(state)

    assert decision.decision_type == "tile_efficiency_hint"
    assert decision.recommended_focus == "tile_efficiency"


def test_build_decision_promotes_unknown_scene_when_tile_evidence_exists() -> None:
    state = PerceivedGameState(
        scene="unknown",
        confidence=0.5,
        is_user_turn=True,
        hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "5z"],
        analysis_hints={
            "analysis_confidence": 0.85,
            "tile_level_state": "tile_level_reliable",
            "recognized_hand_tile_count": 11,
            "recognized_meld_group_count": 1,
        },
    )

    decision = build_decision(state)

    assert decision.scene == "in_match"
    assert decision.decision_type == "tile_efficiency_hint"


def test_parse_tiles_from_image_filters_low_confidence_hand_match(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "template-frame.png"
    image = Image.new("RGB", (1280, 720), color=(210, 210, 210))
    image.save(image_path)

    def fake_collect_slot_metrics(image_obj, slots):
        metrics = []
        for index, slot in enumerate(slots):
            metrics.append(
                {
                    "slot_id": slot.slot_id,
                    "box": slot.box.to_dict(),
                    "slot_mean_luma": 120.0 if index < 2 else 0.0,
                    "slot_colorful_ratio": 0.2,
                    "slot_bright_ratio": 0.3 if index < 2 else 0.0,
                    "slot_dark_ratio": 0.2 if index < 2 else 1.0,
                    "slot_stddev": 24.0 if index < 2 else 0.0,
                }
            )
        return metrics

    matches = iter(
        [
            TileTemplateMatch(tile="4p", confidence=0.027, distance=78.9, runner_up_tile="5p", runner_up_distance=82.961),
            TileTemplateMatch(tile="5p", confidence=0.63, distance=44.2, runner_up_tile="4p", runner_up_distance=61.4),
        ]
    )

    def fake_classify_tile_from_templates(crop, payload):
        del crop, payload
        return next(matches, None)

    monkeypatch.setattr(
        tile_parser_module,
        "resolve_calibration_profile",
        lambda width, height, calibration_dir=None: CalibrationProfile(
            profile_id="test-profile",
            enabled=True,
            screen_width=width,
            screen_height=height,
            hand_tile_templates={"templates": {"4p": {"signatures": ["stub"]}, "5p": {"signatures": ["stub"]}}},
        ),
    )
    monkeypatch.setattr(tile_parser_module, "_collect_slot_metrics", fake_collect_slot_metrics)
    monkeypatch.setattr(tile_parser_module, "_candidate_draw_slot_indices", lambda: [14])
    monkeypatch.setattr(tile_parser_module, "classify_tile_from_templates", fake_classify_tile_from_templates)
    monkeypatch.setattr(
        tile_parser_module,
        "parse_discards_from_image",
        lambda image_obj, template_payload, **kwargs: DiscardParseResult(
            discard_piles={},
            visible_tiles=[],
            raw_detections=[],
            analysis_hints={},
        ),
    )

    with Image.open(image_path) as opened:
        parsed = parse_tiles_from_image(
            image_path,
            opened.convert("RGB"),
            scene="in_match",
            metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        )

    assert parsed.hand_tiles == ["5p"]
    assert parsed.analysis_hints["recognized_hand_tile_count"] == 1
    low_confidence_detection = next(item for item in parsed.raw_detections if item.get("candidate_tile") == "4p")
    assert low_confidence_detection["accepted"] is False
    assert low_confidence_detection["rejection_reason"] == "low_confidence"


def test_parse_tiles_from_image_rejects_ambiguous_6s_9s_hand_match(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "ambiguous-frame.png"
    image = Image.new("RGB", (1280, 720), color=(210, 210, 210))
    image.save(image_path)

    def fake_collect_slot_metrics(image_obj, slots):
        metrics = []
        for index, slot in enumerate(slots):
            metrics.append(
                {
                    "slot_id": slot.slot_id,
                    "box": slot.box.to_dict(),
                    "slot_mean_luma": 120.0 if index == 0 else 0.0,
                    "slot_colorful_ratio": 0.2,
                    "slot_bright_ratio": 0.3 if index == 0 else 0.0,
                    "slot_dark_ratio": 0.2 if index == 0 else 1.0,
                    "slot_stddev": 24.0 if index == 0 else 0.0,
                }
            )
        return metrics

    def fake_classify_tile_from_templates(crop, payload):
        del crop, payload
        return TileTemplateMatch(
            tile="9s",
            confidence=0.63,
            distance=40.0,
            runner_up_tile="6s",
            runner_up_distance=42.0,
        )

    monkeypatch.setattr(
        tile_parser_module,
        "resolve_calibration_profile",
        lambda width, height, calibration_dir=None: CalibrationProfile(
            profile_id="test-profile",
            enabled=True,
            screen_width=width,
            screen_height=height,
            hand_tile_templates={"templates": {"6s": {"signatures": ["stub"]}, "9s": {"signatures": ["stub"]}}},
        ),
    )
    monkeypatch.setattr(tile_parser_module, "_collect_slot_metrics", fake_collect_slot_metrics)
    monkeypatch.setattr(tile_parser_module, "classify_tile_from_templates", fake_classify_tile_from_templates)
    monkeypatch.setattr(
        tile_parser_module,
        "parse_discards_from_image",
        lambda image_obj, template_payload, **kwargs: DiscardParseResult(
            discard_piles={},
            visible_tiles=[],
            raw_detections=[],
            analysis_hints={},
        ),
    )

    with Image.open(image_path) as opened:
        parsed = parse_tiles_from_image(
            image_path,
            opened.convert("RGB"),
            scene="unknown",
            metrics={"bottom_hand_area": {"colorful_ratio": 0.12}},
        )

    assert parsed.hand_tiles == []
    ambiguous_detection = next(item for item in parsed.raw_detections if item.get("candidate_tile") == "9s")
    assert ambiguous_detection["accepted"] is False
    assert ambiguous_detection["rejection_reason"].startswith("ambiguous_")
    assert {"6s", "9s"} <= set(ambiguous_detection["rejection_reason"].split("_"))
