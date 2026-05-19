from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.perception import pipeline
from plugin.plugins.mahjong_companion.scripts.debug_river_detector_v2 import _batch_frame_row
from plugin.plugins.mahjong_companion.scripts import run_river_eval


def test_batch_frame_row_records_tiles_by_player(tmp_path):
    candidate = SimpleNamespace(player="self")
    result = SimpleNamespace(
        candidates=[candidate],
        by_player={
            "self": [candidate],
            "left_opponent": [],
            "top_opponent": [],
            "right_opponent": [],
        },
    )

    row = _batch_frame_row(
        tmp_path / "frame.png",
        result,
        {id(candidate): {"tile": "8p", "tile_confidence": 0.9}},
    )

    assert row["tiles_by_player"]["self"] == ["8p"]
    assert row["tile_counts"]["8p"] == 1


def test_strategy_debug_payload_builds_river_only_analysis():
    payload = run_river_eval._strategy_debug_payload(
        {
            "batch_dir": "fixtures",
            "frames": [
                {
                    "image": "frame.png",
                    "candidate_count": 4,
                    "unknown_count": 0,
                    "empty_count": 0,
                    "tiles_by_player": {
                        "self": ["8p"],
                        "left_opponent": ["1z"],
                        "top_opponent": ["2m"],
                        "right_opponent": ["3s"],
                    },
                    "tile_overflow_counts": {},
                }
            ],
        }
    )

    assert payload["frame_count"] == 1
    assert payload["strategy_ready_frame_count"] == 1
    assert payload["frames"][0]["known_count"] == 4
    assert any("牌河识别健康度够用" in item for item in payload["frames"][0]["teaching_points"])


def test_strategy_debug_payload_uses_bottom_hand_fallback(monkeypatch, tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1280, 720), "white").save(image_path)

    def analyze_without_hand(*args, **kwargs):
        return (
            PerceivedGameState(scene="in_match", confidence=0.7, hand_tiles=[]),
            {},
        )

    monkeypatch.setattr(pipeline, "analyze_image_path", analyze_without_hand)
    monkeypatch.setattr(
        run_river_eval,
        "_bottom_hand_detection_payload",
        lambda path: {
            "hand_tiles": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "2p", "3p", "4p", "5s", "5s", "6s", "7s"],
            "melds": [],
            "dora_indicators": [],
            "riichi_players": [],
            "source": "bottom_hand_detector",
            "confidence": 0.88,
            "analysis_hints": {"tile_parser_source": "bottom_hand_detector", "analysis_confidence": 0.88},
            "error": "",
        },
    )

    payload = run_river_eval._strategy_debug_payload(
        {
            "batch_dir": "fixtures",
            "frames": [
                {
                    "image": str(image_path),
                    "candidate_count": 4,
                    "unknown_count": 0,
                    "empty_count": 0,
                    "tiles_by_player": {
                        "self": ["8p"],
                        "left_opponent": ["1z"],
                        "top_opponent": ["2m"],
                        "right_opponent": ["3s"],
                    },
                    "tile_overflow_counts": {},
                }
            ],
        }
    )

    frame = payload["frames"][0]
    assert payload["hand_ready_frame_count"] == 1
    assert payload["candidate_discard_frame_count"] == 1
    assert frame["hand_parser_source"] == "bottom_hand_detector"
    assert frame["hand_count"] == 14
    assert frame["candidate_discards"]


def test_strategy_debug_payload_uses_open_hand_count_for_discard_advice(monkeypatch, tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1280, 720), "white").save(image_path)

    monkeypatch.setattr(
        pipeline,
        "analyze_image_path",
        lambda *args, **kwargs: (PerceivedGameState(scene="in_match", confidence=0.7, hand_tiles=[]), {}),
    )
    monkeypatch.setattr(
        run_river_eval,
        "_bottom_hand_detection_payload",
        lambda path: {
            "hand_tiles": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "2p", "3p", "4p"],
            "melds": [],
            "dora_indicators": [],
            "riichi_players": [],
            "source": "bottom_hand_detector",
            "confidence": 0.88,
            "analysis_hints": {"tile_parser_source": "bottom_hand_detector", "analysis_confidence": 0.88},
            "error": "",
        },
    )

    payload = run_river_eval._strategy_debug_payload(
        {
            "batch_dir": "fixtures",
            "frames": [
                {
                    "image": str(image_path),
                    "candidate_count": 4,
                    "unknown_count": 0,
                    "empty_count": 0,
                    "tiles_by_player": {
                        "self": ["8p"],
                        "left_opponent": ["1z"],
                        "top_opponent": ["2m"],
                        "right_opponent": ["3s"],
                    },
                    "tile_overflow_counts": {},
                }
            ],
        }
    )

    frame = payload["frames"][0]
    assert payload["hand_ready_frame_count"] == 1
    assert payload["candidate_discard_frame_count"] == 1
    assert frame["hand_count"] == 10
    assert frame["candidate_discards"]


def test_strategy_debug_payload_keeps_unsupported_hand_count_out_of_discard_advice(monkeypatch, tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1280, 720), "white").save(image_path)

    monkeypatch.setattr(
        pipeline,
        "analyze_image_path",
        lambda *args, **kwargs: (PerceivedGameState(scene="in_match", confidence=0.7, hand_tiles=[]), {}),
    )
    monkeypatch.setattr(
        run_river_eval,
        "_bottom_hand_detection_payload",
        lambda path: {
            "hand_tiles": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"],
            "melds": [],
            "dora_indicators": [],
            "riichi_players": [],
            "source": "bottom_hand_detector",
            "confidence": 0.88,
            "analysis_hints": {"tile_parser_source": "bottom_hand_detector", "analysis_confidence": 0.88},
            "error": "",
        },
    )

    payload = run_river_eval._strategy_debug_payload(
        {
            "batch_dir": "fixtures",
            "frames": [
                {
                    "image": str(image_path),
                    "candidate_count": 4,
                    "unknown_count": 0,
                    "empty_count": 0,
                    "tiles_by_player": {
                        "self": ["8p"],
                        "left_opponent": ["1z"],
                        "top_opponent": ["2m"],
                        "right_opponent": ["3s"],
                    },
                    "tile_overflow_counts": {},
                }
            ],
        }
    )

    frame = payload["frames"][0]
    assert payload["hand_ready_frame_count"] == 0
    assert payload["candidate_discard_frame_count"] == 0
    assert frame["hand_count"] == 9
    assert frame["candidate_discards"] == []
