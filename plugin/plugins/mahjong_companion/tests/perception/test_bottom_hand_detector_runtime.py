from __future__ import annotations

from PIL import Image

from plugin.plugins.mahjong_companion.perception.bottom_hand_detector import BottomHandDetection, BottomHandSlot
from plugin.plugins.mahjong_companion.perception.tile_parser import TileParseResult, _with_bottom_hand_detector_result


def test_bottom_hand_detector_result_promotes_supported_open_hand(monkeypatch):
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.perception.tile_parser.detect_bottom_hand_tiles",
        lambda image: BottomHandDetection(
            hand_tiles=["1m", "2m", "3m", "4m", "5m", "6m", "7m", "2p", "3p", "4p"],
            slots=[
                BottomHandSlot(
                    slot_id="bottom_hand_1",
                    tile="1m",
                    confidence=0.9,
                    accepted=True,
                    bbox=[10, 20, 40, 80],
                )
            ],
            confidence=0.9,
        ),
    )

    result = _with_bottom_hand_detector_result(TileParseResult(), image=Image.new("RGB", (100, 100)))

    assert len(result.hand_tiles) == 10
    assert result.analysis_hints["tile_parser_source"] == "bottom_hand_detector"
    assert result.analysis_hints["recognized_meld_group_count"] == 1
    assert result.analysis_hints["post_meld_hand_shape"] == "waiting"


def test_bottom_hand_detector_result_rejects_unsupported_hand_count(monkeypatch):
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.perception.tile_parser.detect_bottom_hand_tiles",
        lambda image: BottomHandDetection(
            hand_tiles=["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"],
            confidence=0.9,
        ),
    )

    result = _with_bottom_hand_detector_result(TileParseResult(), image=Image.new("RGB", (100, 100)))

    assert result.hand_tiles == []
    assert result.analysis_hints["bottom_hand_unsupported_count"] == 9
