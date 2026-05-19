from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from plugin.plugins.mahjong_companion.scripts.spike_river_model_detector import (
    AssignedDetection,
    ModelDetection,
    _json_payload_path,
    _normalize_tile_label,
    _roboflow_image_payload,
    _roboflow_cache_path,
    _select_contact_items,
    _redact_secrets,
    _image_paths,
    assign_detections_to_rivers,
    classify_assigned_detections,
    fuse_model_with_v2_gaps,
    parse_roboflow_predictions,
)
from plugin.plugins.mahjong_companion.perception.river_detector_v2 import RiverDetectionResult, RiverTileCandidate
from plugin.plugins.mahjong_companion.perception.tile_templates import TileTemplateMatch


def test_parse_roboflow_predictions_converts_center_boxes():
    payload = {
        "_scale_x": 2.0,
        "_scale_y": 3.0,
        "predictions": [
            {
                "x": 100.0,
                "y": 80.0,
                "width": 40.0,
                "height": 20.0,
                "confidence": 0.88,
                "class": "8p",
            }
        ]
    }

    detections = parse_roboflow_predictions(payload)

    assert len(detections) == 1
    assert detections[0].bbox == (160, 210, 240, 270)
    assert detections[0].confidence == 0.88
    assert detections[0].label == "8p"
    assert detections[0].source == "roboflow"


def test_assign_detections_to_rivers_uses_river_rois():
    detections = [
        ModelDetection(bbox=(1020, 780, 1100, 860), confidence=0.9, label="1m"),
        ModelDetection(bbox=(1510, 430, 1610, 500), confidence=0.9, label="8p"),
        ModelDetection(bbox=(30, 30, 80, 80), confidence=0.9, label="9m"),
    ]

    assigned = assign_detections_to_rivers(detections, (2560, 1440))

    assert sorted((item.player, item.order_index, item.detection.label) for item in assigned) == [
        ("right_opponent", 1, "8p"),
        ("self", 1, "1m"),
    ]


def test_json_payload_path_matches_frame_stem(tmp_path):
    payload = tmp_path / "20260505-frame.json"
    payload.write_text("{}", encoding="utf-8")

    Args = SimpleNamespace(detector_json_dir=tmp_path, detector_json=None)
    assert _json_payload_path(tmp_path / "20260505-frame.png", Args) == payload


def test_roboflow_cache_path_uses_image_stem(tmp_path):
    Args = SimpleNamespace(roboflow_cache_dir=tmp_path / "cache")

    assert _roboflow_cache_path(tmp_path / "20260505-frame.png", Args) == tmp_path / "cache" / "20260505-frame.json"


def test_redact_secrets_removes_roboflow_api_key():
    text = "https://detect.roboflow.com/model/1?api_key=secret&confidence=0.25"

    assert _redact_secrets(text) == "https://detect.roboflow.com/model/1?api_key=<redacted>&confidence=0.25"


def test_normalize_tile_label_maps_kim_mahjong_classes():
    assert _normalize_tile_label("8D") == "8p"
    assert _normalize_tile_label("1C") == "1m"
    assert _normalize_tile_label("RD") == "7z"


def test_roboflow_image_payload_resizes_and_reports_scale(tmp_path):
    from PIL import Image

    image_path = tmp_path / "large.png"
    Image.new("RGB", (200, 100), (1, 2, 3)).save(image_path)

    payload, scale = _roboflow_image_payload(image_path, max_size=100)

    assert payload
    assert scale == (2.0, 2.0)


def test_fuse_model_with_v2_gaps_adds_uncovered_candidate():
    assigned = [
        AssignedDetection(
            player="right_opponent",
            order_index=1,
            detection=ModelDetection(bbox=(100, 100, 160, 160), confidence=0.9, label="8p"),
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

    assert [(item.order_index, item.detection.source) for item in fused] == [
        (1, "model"),
        (2, "river_detector_v2_fallback"),
    ]


def test_fuse_model_with_v2_gaps_does_not_exceed_v2_target():
    assigned = [
        AssignedDetection(
            player="right_opponent",
            order_index=index,
            detection=ModelDetection(bbox=(index * 20, 100, index * 20 + 10, 120), confidence=0.9),
        )
        for index in range(1, 4)
    ]
    v2_result = RiverDetectionResult(
        candidates=[
            RiverTileCandidate(
                player="right_opponent",
                order_index=1,
                bbox=(200, 100, 240, 160),
                quad=((200, 100), (200, 160), (240, 160), (240, 100)),
                center=(220, 130),
                confidence=0.8,
            )
        ],
        image_size=(320, 240),
    )

    fused = fuse_model_with_v2_gaps(assigned, v2_result)

    assert len(fused) == 3


def test_classify_assigned_detections_rejects_low_confidence_crop(monkeypatch):
    image = Image.new("RGB", (80, 80), (220, 210, 190))
    assigned = [
        AssignedDetection(
            player="left_opponent",
            order_index=1,
            detection=ModelDetection(bbox=(10, 10, 50, 60), confidence=0.8),
        )
    ]
    match = TileTemplateMatch(tile="1s", confidence=0.35, distance=100.0)
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.scripts.spike_river_model_detector.classify_tiles_batch",
        lambda _crops, _payload: [match],
    )

    classified = classify_assigned_detections(image, assigned)

    assert classified[0].tile == "unknown"
    assert classified[0].tile_confidence == 0.35


def test_classify_assigned_detections_marks_onnx_none_as_empty(monkeypatch):
    image = Image.new("RGB", (80, 80), (220, 210, 190))
    assigned = [
        AssignedDetection(
            player="left_opponent",
            order_index=1,
            detection=ModelDetection(bbox=(10, 10, 50, 60), confidence=0.8),
        )
    ]
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.scripts.spike_river_model_detector.classify_tiles_batch",
        lambda _crops, _payload: [None],
    )
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.scripts.spike_river_model_detector.onnx_backend_available",
        lambda: True,
    )

    classified = classify_assigned_detections(image, assigned)

    assert classified[0].tile == "empty"
    assert classified[0].tile_confidence == 1.0


def test_classify_assigned_detections_caps_tile_overflow(monkeypatch):
    image = Image.new("RGB", (120, 80), (220, 210, 190))
    assigned = [
        AssignedDetection(
            player="self",
            order_index=index,
            detection=ModelDetection(bbox=(index * 10, 10, index * 10 + 8, 30), confidence=0.8),
        )
        for index in range(1, 6)
    ]
    matches = [
        TileTemplateMatch(tile="3m", confidence=confidence, distance=10.0)
        for confidence in (0.99, 0.98, 0.97, 0.96, 0.60)
    ]
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.scripts.spike_river_model_detector.classify_tiles_batch",
        lambda _crops, _payload: matches,
    )

    classified = classify_assigned_detections(image, assigned)

    assert [item.tile for item in classified].count("3m") == 4
    assert [item.tile for item in classified].count("unknown") == 1


def test_select_contact_items_prefers_high_model_count():
    items = [
        ("a", None, None, [], {"model_candidate_count": 1, "v2_candidate_count": 1}),
        ("b", None, None, [], {"model_candidate_count": 10, "v2_candidate_count": 9}),
    ]

    selected = _select_contact_items(items, "model-count", limit=1)

    assert selected[0][0] == "b"


def test_select_contact_items_can_sort_by_v2_count():
    items = [
        ("a", None, None, [], {"model_candidate_count": 10, "v2_candidate_count": 1}),
        ("b", None, None, [], {"model_candidate_count": 0, "v2_candidate_count": 9}),
    ]

    selected = _select_contact_items(items, "v2-count", limit=1)

    assert selected[0][0] == "b"


def test_image_paths_respects_batch_glob(tmp_path):
    (tmp_path / "a.png").write_text("", encoding="utf-8")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")

    Args = SimpleNamespace(batch_dir=tmp_path, batch_glob="*.png", batch_limit=0, image=None)

    assert _image_paths(Args) == [tmp_path / "a.png"]
