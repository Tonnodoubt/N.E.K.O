from __future__ import annotations

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_coach.perception import discard_parser
from plugin.plugins.mahjong_coach.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_coach.perception.tile_templates import TileTemplateMatch


def test_parse_discards_accepts_onnx_batch_result(monkeypatch) -> None:
    image = Image.new("RGB", (1920, 1080), (24, 28, 30))
    slot = build_discard_layout(*image.size)["self"][0]
    draw = ImageDraw.Draw(image)
    draw.rectangle((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom), fill=(235, 228, 212))
    draw.rectangle((slot.box.left + 18, slot.box.top + 16, slot.box.left + 40, slot.box.top + 50), fill=(20, 20, 20))

    monkeypatch.setattr(discard_parser, "onnx_discard_available", lambda: True)
    monkeypatch.setattr(
        discard_parser,
        "classify_discard_tiles_batch",
        lambda crops, _payload: [TileTemplateMatch(tile="7p", confidence=0.97, distance=2.46) for _crop in crops],
    )

    result = discard_parser.parse_discards_from_image(
        image,
        {},
        layout={"self": [slot]},
        min_confidence=0.90,
    )

    assert result.visible_tiles == ["7p"]
    assert result.discard_piles["self"][0]["tile"] == "7p"
    assert result.analysis_hints["recognized_discard_tile_count"] == 1


def test_parse_discards_rejects_empty_class(monkeypatch) -> None:
    image = Image.new("RGB", (1920, 1080), (24, 28, 30))
    slot = build_discard_layout(*image.size)["self"][0]
    draw = ImageDraw.Draw(image)
    draw.rectangle((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom), fill=(235, 228, 212))
    draw.rectangle((slot.box.left + 18, slot.box.top + 16, slot.box.left + 40, slot.box.top + 50), fill=(20, 20, 20))

    monkeypatch.setattr(discard_parser, "onnx_discard_available", lambda: True)
    monkeypatch.setattr(
        discard_parser,
        "classify_discard_tiles_batch",
        lambda crops, _payload: [TileTemplateMatch(tile="empty", confidence=0.99, distance=0.82) for _crop in crops],
    )

    result = discard_parser.parse_discards_from_image(
        image,
        {},
        layout={"self": [slot]},
        min_confidence=0.90,
    )

    assert result.visible_tiles == []
    assert result.raw_detections[0]["rejection_reason"] == "empty_tile_class"
