from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.perception.model_river_adapter import (
    ModelRiverCandidate,
    ModelRiverConfig,
    ModelRiverDetection,
    assign_model_detections_to_rivers,
    classify_model_river_candidates,
    fuse_model_with_v2_gaps,
    load_manual_template_payload,
    model_candidates_to_parse_result,
    normalize_tile_label,
    parse_model_river_from_json,
    parse_roboflow_predictions,
    save_unknown_model_river_crops,
)
from plugin.plugins.mahjong_companion.perception.river_detector_v2 import RiverDetectionResult, RiverTileCandidate
from plugin.plugins.mahjong_companion.perception.tile_templates import TileTemplateMatch


def test_parse_roboflow_predictions_rescales_boxes():
    payload = {
        "_scale_x": 2.0,
        "_scale_y": 3.0,
        "predictions": [
            {"x": 100, "y": 80, "width": 40, "height": 20, "confidence": 0.7, "class": "8D"}
        ],
    }

    detections = parse_roboflow_predictions(payload)

    assert detections[0].bbox == (160, 210, 240, 270)
    assert detections[0].label == "8D"


def test_normalize_tile_label_maps_external_model_classes():
    assert normalize_tile_label("8D") == "8p"
    assert normalize_tile_label("4B") == "4s"
    assert normalize_tile_label("RD") == "7z"


def test_assign_model_detections_to_rivers_uses_current_river_rois():
    detections = [
        ModelRiverDetection(bbox=(1020, 780, 1100, 860), confidence=0.9, label="1C"),
        ModelRiverDetection(bbox=(1510, 430, 1610, 500), confidence=0.9, label="8D"),
        ModelRiverDetection(bbox=(30, 30, 80, 80), confidence=0.9, label="9D"),
    ]

    assigned = assign_model_detections_to_rivers(detections, (2560, 1440))

    assert sorted((item.player, item.turn_index, item.detection.label) for item in assigned) == [
        ("right_opponent", 1, "8D"),
        ("self", 1, "1C"),
    ]


def test_fuse_model_with_v2_gaps_adds_uncovered_v2_candidate():
    assigned = [
        ModelRiverCandidate(
            player="right_opponent",
            turn_index=1,
            detection=ModelRiverDetection(bbox=(100, 100, 160, 160), confidence=0.9, label="8D"),
        )
    ]
    v2_result = RiverDetectionResult(
        candidates=[
            RiverTileCandidate(
                player="right_opponent",
                order_index=1,
                bbox=(102, 102, 162, 162),
                quad=((102, 102), (102, 162), (162, 162), (162, 102)),
                center=(132, 132),
                confidence=0.8,
            ),
            RiverTileCandidate(
                player="right_opponent",
                order_index=2,
                bbox=(180, 100, 240, 160),
                quad=((180, 100), (180, 160), (240, 160), (240, 100)),
                center=(210, 130),
                confidence=0.7,
            ),
        ],
        image_size=(320, 240),
    )

    fused = fuse_model_with_v2_gaps(assigned, v2_result)

    assert [(item.turn_index, item.detection.source) for item in fused] == [
        (1, "model"),
        (2, "river_detector_v2_fallback"),
    ]


def test_model_candidates_to_parse_result_reports_tile_overflow():
    candidates = [
        ModelRiverCandidate(
            player="self",
            turn_index=index,
            detection=ModelRiverDetection(bbox=(index * 10, 10, index * 10 + 8, 18), confidence=0.9),
            tile="8p",
            tile_confidence=0.9,
        )
        for index in range(1, 6)
    ]

    result = model_candidates_to_parse_result(
        candidates,
        v2_result=RiverDetectionResult(image_size=(100, 100)),
        source_json=Path(__file__),
    )

    assert result.analysis_hints["model_river_tile_overflow_counts"] == {"8p": 5}
    assert len(result.discard_piles["self"]) == 5


def test_model_candidates_to_parse_result_reports_unknown_counts():
    candidates = [
        ModelRiverCandidate(
            player="left_opponent",
            turn_index=1,
            detection=ModelRiverDetection(bbox=(10, 10, 20, 20), confidence=0.8),
            tile="unknown",
            tile_confidence=0.4,
        ),
        ModelRiverCandidate(
            player="self",
            turn_index=1,
            detection=ModelRiverDetection(bbox=(20, 10, 30, 20), confidence=0.8),
            tile="8p",
            tile_confidence=0.9,
        ),
    ]

    result = model_candidates_to_parse_result(
        candidates,
        v2_result=RiverDetectionResult(image_size=(100, 100)),
        source_json=Path(__file__),
    )

    assert result.analysis_hints["model_river_unknown_count"] == 1
    assert result.analysis_hints["model_river_known_count"] == 1
    assert result.analysis_hints["model_river_unknown_by_player"] == {"left_opponent": 1}
    assert result.visible_tiles == ["8p"]


def test_model_candidates_to_parse_result_ignores_empty_candidates():
    candidates = [
        ModelRiverCandidate(
            player="left_opponent",
            turn_index=1,
            detection=ModelRiverDetection(bbox=(10, 10, 20, 20), confidence=0.8),
            tile="empty",
            tile_confidence=1.0,
        ),
        ModelRiverCandidate(
            player="self",
            turn_index=1,
            detection=ModelRiverDetection(bbox=(20, 10, 30, 20), confidence=0.8),
            tile="8p",
            tile_confidence=0.9,
        ),
    ]

    result = model_candidates_to_parse_result(
        candidates,
        v2_result=RiverDetectionResult(image_size=(100, 100)),
        source_json=Path(__file__),
    )

    assert result.analysis_hints["model_river_empty_count"] == 1
    assert result.analysis_hints["model_river_unknown_count"] == 0
    assert result.visible_tiles == ["8p"]


def test_save_unknown_model_river_crops_writes_only_unknown(tmp_path):
    image = Image.new("RGB", (80, 80), (220, 210, 190))
    candidates = [
        ModelRiverCandidate(
            player="left_opponent",
            turn_index=1,
            detection=ModelRiverDetection(bbox=(10, 10, 50, 60), confidence=0.8),
            tile="unknown",
            tile_confidence=0.35,
        ),
        ModelRiverCandidate(
            player="self",
            turn_index=1,
            detection=ModelRiverDetection(bbox=(50, 10, 70, 60), confidence=0.8),
            tile="8p",
            tile_confidence=0.9,
        ),
    ]

    saved = save_unknown_model_river_crops(image, Path("frame.png"), candidates, tmp_path)

    assert saved == 1
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_classify_model_river_candidates_rejects_low_confidence_crop(monkeypatch):
    image = Image.new("RGB", (80, 80), (220, 210, 190))
    candidates = [
        ModelRiverCandidate(
            player="left_opponent",
            turn_index=1,
            detection=ModelRiverDetection(bbox=(10, 10, 50, 60), confidence=0.8),
        )
    ]
    match = TileTemplateMatch(tile="1s", confidence=0.35, distance=100.0)
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.perception.model_river_adapter.classify_tiles_batch",
        lambda _crops, _payload: [match],
    )

    classified = classify_model_river_candidates(image, candidates)

    assert classified[0].tile == "unknown"
    assert classified[0].tile_confidence == 0.35


def test_classify_model_river_candidates_uses_manual_template_for_low_confidence(monkeypatch):
    image = Image.new("RGB", (80, 80), (220, 210, 190))
    candidates = [
        ModelRiverCandidate(
            player="left_opponent",
            turn_index=1,
            detection=ModelRiverDetection(bbox=(10, 10, 50, 60), confidence=0.8),
        )
    ]
    match = TileTemplateMatch(tile="1s", confidence=0.35, distance=100.0)
    manual_match = TileTemplateMatch(tile="7p", confidence=0.72, distance=20.0)
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.perception.model_river_adapter.classify_tiles_batch",
        lambda _crops, _payload: [match],
    )
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.perception.model_river_adapter.classify_tile_from_templates",
        lambda _crop, _payload: manual_match,
    )

    classified = classify_model_river_candidates(image, candidates, manual_template_payload={"templates": {}})

    assert classified[0].tile == "7p"
    assert classified[0].tile_confidence == 0.72


def test_load_manual_template_payload_keeps_empty_samples(tmp_path):
    crop = Image.new("RGB", (20, 20), (220, 210, 190))
    crop_path = tmp_path / "empty.png"
    crop.save(crop_path)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps([{"file": str(crop_path), "label": "empty"}]), encoding="utf-8")

    payload = load_manual_template_payload(labels_path)

    assert "empty" in payload["templates"]


def test_classify_model_river_candidates_uses_strong_manual_template_before_overflow_cap(monkeypatch):
    image = Image.new("RGB", (120, 80), (220, 210, 190))
    candidates = [
        ModelRiverCandidate(
            player="self",
            turn_index=index,
            detection=ModelRiverDetection(bbox=(index * 10, 10, index * 10 + 8, 30), confidence=0.8),
        )
        for index in range(1, 6)
    ]
    matches = [
        TileTemplateMatch(tile="3m", confidence=confidence, distance=10.0)
        for confidence in (0.99, 0.98, 0.97, 0.96, 0.60)
    ]
    manual_matches = iter([None, None, None, None, TileTemplateMatch(tile="2m", confidence=0.99, distance=0.0)])
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.perception.model_river_adapter.classify_tiles_batch",
        lambda _crops, _payload: matches,
    )
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.perception.model_river_adapter.classify_tile_from_templates",
        lambda _crop, _payload: next(manual_matches),
    )

    classified = classify_model_river_candidates(image, candidates, manual_template_payload={"templates": {}})

    assert [item.tile for item in classified] == ["3m", "3m", "3m", "3m", "2m"]


def test_classify_model_river_candidates_marks_onnx_none_as_empty(monkeypatch):
    image = Image.new("RGB", (80, 80), (220, 210, 190))
    candidates = [
        ModelRiverCandidate(
            player="left_opponent",
            turn_index=1,
            detection=ModelRiverDetection(bbox=(10, 10, 50, 60), confidence=0.8),
        )
    ]
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.perception.model_river_adapter.classify_tiles_batch",
        lambda _crops, _payload: [None],
    )
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.perception.model_river_adapter.onnx_backend_available",
        lambda: True,
    )

    classified = classify_model_river_candidates(image, candidates)

    assert classified[0].tile == "empty"
    assert classified[0].tile_confidence == 1.0


def test_classify_model_river_candidates_caps_tile_overflow(monkeypatch):
    image = Image.new("RGB", (120, 80), (220, 210, 190))
    candidates = [
        ModelRiverCandidate(
            player="self",
            turn_index=index,
            detection=ModelRiverDetection(bbox=(index * 10, 10, index * 10 + 8, 30), confidence=0.8),
        )
        for index in range(1, 6)
    ]
    matches = [
        TileTemplateMatch(tile="3m", confidence=confidence, distance=10.0)
        for confidence in (0.99, 0.98, 0.97, 0.96, 0.60)
    ]
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.perception.model_river_adapter.classify_tiles_batch",
        lambda _crops, _payload: matches,
    )

    classified = classify_model_river_candidates(image, candidates)

    assert [item.tile for item in classified].count("3m") == 4
    assert [item.tile for item in classified].count("unknown") == 1


def test_parse_model_river_from_json_reads_cache_and_fuses(tmp_path):
    image_path = tmp_path / "frame.png"
    image = Image.new("RGB", (320, 240), (35, 70, 110))
    ImageDraw.Draw(image).rectangle((130, 120, 180, 170), fill=(220, 210, 190))
    image.save(image_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "frame.json").write_text(
        json.dumps({"predictions": [{"x": 155, "y": 145, "width": 50, "height": 50, "confidence": 0.8, "class": "8D"}]}),
        encoding="utf-8",
    )

    result = parse_model_river_from_json(
        image,
        image_path,
        config=ModelRiverConfig(detector_json_dir=cache, classify_crops=False, fuse_v2_gaps=False),
    )

    assert result.analysis_hints["discard_parser_available"]
    assert result.visible_tiles == ["8p"]
