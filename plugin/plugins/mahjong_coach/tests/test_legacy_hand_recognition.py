from __future__ import annotations

import base64
import json
from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_coach.perception import action_detector, meld_state
from plugin.plugins.mahjong_coach.perception.action_detector import detect_action_buttons_fast
from plugin.plugins.mahjong_coach.perception.calibration import CalibrationProfile
from plugin.plugins.mahjong_coach.perception.fast_hand_path import detect_fast_hand_path
from plugin.plugins.mahjong_coach.perception.hand_layout import build_hand_layout
from plugin.plugins.mahjong_coach.perception.meld_state import build_self_meld_layout, detect_meld_state_path
from plugin.plugins.mahjong_coach.perception.riichi_detector import detect_riichi_sticks
from plugin.plugins.mahjong_coach.perception import tile_classifier_dispatch
from plugin.plugins.mahjong_coach.perception.tile_classifier_dispatch import classify_hand_tile
from plugin.plugins.mahjong_coach.perception.tile_templates import TileTemplateMatch, extract_tile_signature


def test_fast_hand_path_reuses_legacy_templates(tmp_path: Path) -> None:
    width, height = 1920, 1080
    tiles = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]
    image = Image.new("RGB", (width, height), (35, 80, 110))
    layout = build_hand_layout(width, height, calibration=CalibrationProfile(screen_width=width, screen_height=height))
    samples: dict[str, Image.Image] = {}

    for slot, tile in zip(layout["hand"], tiles, strict=True):
        crop = _tile_crop(slot.box.width, slot.box.height, tile)
        image.paste(crop, (slot.box.left, slot.box.top))
        samples.setdefault(tile, crop)

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "synthetic-1920x1080.json").write_text(
        json.dumps(
            {
                "profile_id": "synthetic-1920x1080",
                "enabled": True,
                "screen_width": width,
                "screen_height": height,
                "confidence": 0.99,
                "hand_offsets": {},
                "hand_tile_templates": _template_payload(samples),
            }
        ),
        encoding="utf-8",
    )
    frame_path = tmp_path / "frame.png"
    image.save(frame_path)

    result = detect_fast_hand_path(frame_path, calibration_dir=profile_dir)

    assert result.ok is True
    assert result.hand_tiles == tiles
    assert result.reason == "matched_14_hand_tiles"


def test_fast_hand_path_accepts_open_hand_count_with_lower_threshold(tmp_path: Path) -> None:
    width, height = 1920, 1080
    tiles = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s"]
    image = Image.new("RGB", (width, height), (35, 80, 110))
    layout = build_hand_layout(width, height, calibration=CalibrationProfile(screen_width=width, screen_height=height))
    samples: dict[str, Image.Image] = {}

    for slot, tile in zip(layout["hand"], tiles):
        crop = _tile_crop(slot.box.width, slot.box.height, tile)
        image.paste(crop, (slot.box.left, slot.box.top))
        samples.setdefault(tile, crop)

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "synthetic-1920x1080.json").write_text(
        json.dumps(
            {
                "profile_id": "synthetic-1920x1080",
                "enabled": True,
                "screen_width": width,
                "screen_height": height,
                "confidence": 0.99,
                "hand_offsets": {},
                "hand_tile_templates": _template_payload(samples),
            }
        ),
        encoding="utf-8",
    )
    frame_path = tmp_path / "open_hand_frame.png"
    image.save(frame_path)

    default_result = detect_fast_hand_path(frame_path, calibration_dir=profile_dir)
    open_result = detect_fast_hand_path(frame_path, calibration_dir=profile_dir, min_hand_tiles=4)

    assert default_result.ok is False
    assert default_result.reason == "unstable_hand_count"
    assert open_result.ok is True
    assert open_result.hand_tiles == tiles
    assert open_result.reason == "matched_open_8_hand_tiles"


def test_meld_state_uses_onnx_classifier_for_self_melds(tmp_path: Path, monkeypatch) -> None:
    image = Image.new("RGB", (1920, 1080), (35, 80, 110))
    slots = build_self_meld_layout(*image.size)
    for slot in slots[:3]:
        image.paste(_tile_crop(slot.box.width, slot.box.height, "5z"), (slot.box.left, slot.box.top))

    calls: dict[str, int] = {}

    def fake_classify(crops):
        calls["crop_count"] = len(crops)
        return [TileTemplateMatch(tile="5z", confidence=0.96, distance=3.0) for _ in crops]

    monkeypatch.setattr(meld_state, "onnx_discard_available", lambda: True)
    monkeypatch.setattr(meld_state, "classify_discard_tiles_batch", fake_classify)
    frame_path = tmp_path / "meld_frame.png"
    image.save(frame_path)

    result = detect_meld_state_path(frame_path)

    assert result.ok is True
    assert result.reason == "recognized_self_melds"
    assert result.open_meld_count == 1
    assert result.tiles == ["5z", "5z", "5z"]
    assert calls["crop_count"] == 3
    assert result.analysis_hints["meld_parser_source"] == "onnx_tile_classifier"


def test_meld_layout_moves_up_for_ultrawide_captures() -> None:
    slots = build_self_meld_layout(2048, 721)

    assert slots
    assert slots[0].box.top < 480


def test_red_five_post_processing_with_template_result() -> None:
    crop = _tile_crop(72, 116, "5m", red_center=True)
    payload = _template_payload({"5m": crop})

    match = classify_hand_tile(crop, payload)

    assert match is not None
    assert match.tile == "0m"


def test_hand_classifier_keeps_onnx_opt_in(monkeypatch) -> None:
    crop = _tile_crop(72, 116, "5m")
    payload = _template_payload({"5m": crop})

    monkeypatch.delenv("MAHJONG_COACH_ONNX_HAND_ENABLED", raising=False)
    monkeypatch.setattr(tile_classifier_dispatch, "onnx_discard_available", lambda: True)
    monkeypatch.setattr(
        tile_classifier_dispatch,
        "_classify_hand_onnx",
        lambda _crop: TileTemplateMatch(tile="9z", confidence=0.99, distance=1.0),
    )

    default_match = classify_hand_tile(crop, payload)
    enabled_match = classify_hand_tile(crop, payload, use_onnx=True)

    assert default_match is not None
    assert default_match.tile == "5m"
    assert enabled_match is not None
    assert enabled_match.tile == "9z"


def test_action_button_template_scan_detects_pon(tmp_path: Path) -> None:
    template_path = (
        Path(__file__).resolve().parents[1]
        / "perception"
        / "templates"
        / "1920x1080"
        / "pon.png"
    )
    template = Image.open(template_path).convert("RGB")
    image = Image.new("RGB", (1920, 1080), (20, 30, 40))
    image.paste(template, (800, 660))
    frame_path = tmp_path / "pon_frame.png"
    image.save(frame_path)

    buttons, meta = detect_action_buttons_fast(frame_path)

    assert "pon" in buttons
    assert meta["templates"]["available"] is True


def test_action_button_template_scan_detects_riichi(tmp_path: Path) -> None:
    template_path = (
        Path(__file__).resolve().parents[1]
        / "perception"
        / "templates"
        / "1920x1080"
        / "riichi.png"
    )
    template = Image.open(template_path).convert("RGB")
    image = Image.new("RGB", (1920, 1080), (20, 30, 40))
    image.paste(template, (800, 660))
    frame_path = tmp_path / "riichi_frame.png"
    image.save(frame_path)

    buttons, meta = detect_action_buttons_fast(frame_path)

    assert "riichi" in buttons
    assert meta["templates"]["available"] is True


def test_action_detector_rejects_conflicting_button_sets(tmp_path: Path, monkeypatch) -> None:
    image = Image.new("RGB", (1920, 1080), (40, 120, 210))
    frame_path = tmp_path / "desktop_like.png"
    image.save(frame_path)

    monkeypatch.setattr(
        action_detector,
        "_detect_template_buttons",
        lambda _image: (["pon", "kan", "chi", "ron", "tsumo", "riichi"], {"available": True}),
    )

    buttons, meta = detect_action_buttons_fast(frame_path)

    assert buttons == []
    assert meta["button_filter"]["rejected"] is True


def test_action_detector_does_not_promote_color_metrics_without_templates(tmp_path: Path, monkeypatch) -> None:
    image = Image.new("RGB", (1920, 1080), (20, 30, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((500, 650, 820, 760), fill=(220, 40, 40))
    draw.rectangle((850, 650, 1170, 760), fill=(30, 170, 90))
    draw.rectangle((1200, 650, 1520, 760), fill=(220, 165, 45))
    frame_path = tmp_path / "colored_bar.png"
    image.save(frame_path)
    monkeypatch.setattr(action_detector, "_detect_template_buttons", lambda _image: ([], {"available": True}))

    buttons, meta = detect_action_buttons_fast(frame_path)

    assert buttons == []
    assert meta["button_filter"]["rejected"] is False


def test_riichi_stick_counter_reads_zero(tmp_path: Path) -> None:
    frame_path = tmp_path / "riichi_counter_zero.png"
    _counter_frame("0").save(frame_path)

    result = detect_riichi_sticks(frame_path)

    assert result.riichi_players == []
    assert result.stick_count == 0


def test_riichi_stick_counter_reads_nonzero(tmp_path: Path) -> None:
    frame_path = tmp_path / "riichi_counter_one.png"
    _counter_frame("1").save(frame_path)

    result = detect_riichi_sticks(frame_path)

    assert result.riichi_players == ["unknown"]
    assert result.stick_count == 1


def _counter_frame(digit: str) -> Image.Image:
    image = Image.new("RGB", (1920, 1080), (18, 30, 45))
    draw = ImageDraw.Draw(image)
    draw.line((88, 145, 105, 165), fill=(240, 240, 235), width=6)
    draw.line((105, 145, 88, 165), fill=(240, 240, 235), width=6)
    if digit == "0":
        draw.ellipse((120, 132, 144, 171), outline=(240, 240, 235), width=7)
    else:
        draw.rounded_rectangle((129, 132, 140, 172), radius=3, fill=(240, 240, 235))
        draw.polygon([(126, 140), (139, 130), (142, 138), (130, 148)], fill=(240, 240, 235))
    return image


def _tile_crop(width: int, height: int, label: str, *, red_center: bool = False) -> Image.Image:
    crop = Image.new("RGB", (width, height), (244, 244, 238))
    draw = ImageDraw.Draw(crop)
    draw.rectangle((1, 1, width - 2, height - 2), outline=(180, 180, 170), width=2)
    draw.rectangle((int(width * 0.22), int(height * 0.28), int(width * 0.78), int(height * 0.72)), fill=(30, 30, 30))
    if red_center:
        draw.rectangle((int(width * 0.38), int(height * 0.40), int(width * 0.62), int(height * 0.60)), fill=(220, 20, 20))
    draw.text((max(2, width // 5), max(2, height // 8)), label, fill=(10, 10, 10))
    return crop


def _template_payload(samples: dict[str, Image.Image]) -> dict[str, object]:
    templates = {}
    for tile, crop in samples.items():
        encoded = base64.b64encode(extract_tile_signature(crop)).decode("ascii")
        templates[tile] = {"count": 1, "signatures": [encoded]}
    return {
        "version": "mahjong-hand-template-v1",
        "signature_version": "rgb-inner-16x24-v1",
        "width": 16,
        "height": 24,
        "inner_bounds": [0.06, 0.06, 0.94, 0.82],
        "max_rms_distance": 82.0,
        "source_sample_count": len(samples),
        "stored_sample_count": len(samples),
        "templates": templates,
    }
