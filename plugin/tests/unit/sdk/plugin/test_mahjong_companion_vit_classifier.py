from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.perception import discard_parser, tile_parser, vit_template_training, vit_tile_classifier
from plugin.plugins.mahjong_companion import adapters
from plugin.plugins.mahjong_companion.adapters import DefaultPerceptionAdapter
from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.perception.calibration import CalibrationProfile, load_calibration_profile, save_calibration_profile
from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_companion.perception.discard_parser import DiscardParseResult, parse_discards_from_image
from plugin.plugins.mahjong_companion.perception.hand_layout import build_hand_layout
from plugin.plugins.mahjong_companion.perception.tile_parser import parse_tiles_from_image
from plugin.plugins.mahjong_companion.perception.vit_tile_classifier import VitTilePrediction


def test_parse_tiles_from_image_can_use_vit_hand_result_without_templates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "frame.png"
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    layout = build_hand_layout(*image.size)
    image.save(image_path)
    hand_tiles = ["1m"] * 14

    def fake_best_vit_hand_result(*_, **__):
        raw_detections = [
            {
                "slot_id": slot.slot_id,
                "group": "hand",
                "candidate_tile": hand_tiles[index],
                "confidence": 0.84,
                "box": slot.box.to_dict(),
                "accepted": True,
                "source": "vit_classifier",
            }
            for index, slot in enumerate(layout["hand"][:14])
        ]
        return {
            "hand_tiles": hand_tiles,
            "raw_detections": raw_detections,
            "confidences": [0.84] * 14,
            "layout": layout,
            "draw_slot_index": 14,
        }

    monkeypatch.setattr(tile_parser, "_best_vit_hand_result", fake_best_vit_hand_result)

    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene="in_match",
        metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        fixture_mode="disabled",
        tile_classifier_config={
            "enabled": True,
            "hand_enabled": True,
            "discard_enabled": False,
            "min_confidence": 0.65,
            "min_mean_confidence": 0.70,
        },
    )

    assert parsed.hand_tiles == hand_tiles
    assert parsed.analysis_hints["tile_parser_source"] == "vit_classifier"
    assert parsed.analysis_hints["recognized_hand_tile_count"] == 14
    assert parsed.analysis_hints["hand_tile_slots"][0]["source"] == "vit_classifier"


def test_parse_discards_from_image_can_use_vit_without_templates(monkeypatch) -> None:
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    layout = build_discard_layout(*image.size)
    slot = layout["self"][0]
    _paste_upright_tile(image, slot, _tile_image((210, 52, 58)))

    def fake_classify_tile_crops(crops, **_):
        return [
            VitTilePrediction(
                tile="5z",
                label="wd",
                confidence=0.92,
                top_k=[{"label": "wd", "tile": "5z", "score": 0.92}],
            )
            for _crop in crops
        ]

    monkeypatch.setattr(discard_parser, "classify_tile_crops", fake_classify_tile_crops)

    parsed = parse_discards_from_image(
        image,
        {},
        layout={"self": [slot]},
        classifier_config={
            "enabled": True,
            "hand_enabled": False,
            "discard_enabled": True,
            "min_confidence": 0.65,
        },
    )

    assert parsed.discard_piles["self"][0]["tile"] == "5z"
    assert parsed.discard_piles["self"][0]["source"] == "discard_vit_classifier"
    assert parsed.analysis_hints["discard_parser_source"] == "vit_classifier"
    assert parsed.analysis_hints["recognized_discard_tile_count"] == 1
    assert parsed.raw_detections[0]["vit_label"] == "wd"


def test_parse_tiles_from_image_can_emit_vit_discard_only_result(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    image.save(image_path)

    def fake_parse_discards_from_image(*_, **__):
        return DiscardParseResult(
            discard_piles={
                "self": [
                    {
                        "tile": "7z",
                        "player": "self",
                        "turn_index": 1,
                        "confidence": 0.88,
                        "source": "discard_vit_classifier",
                    }
                ]
            },
            visible_tiles=["7z"],
            raw_detections=[
                {
                    "slot_id": "discard_self_01",
                    "group": "discard",
                    "candidate_tile": "7z",
                    "confidence": 0.88,
                    "source": "discard_vit_classifier",
                }
            ],
            analysis_hints={
                "discard_parser_source": "vit_classifier",
                "recognized_discard_tile_count": 1,
                "discard_analysis_confidence": 0.88,
            },
        )

    monkeypatch.setattr(tile_parser, "parse_discards_from_image", fake_parse_discards_from_image)

    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene="in_match",
        metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        fixture_mode="disabled",
        tile_classifier_config={
            "enabled": True,
            "hand_enabled": False,
            "discard_enabled": True,
        },
    )

    assert parsed.hand_tiles == []
    assert parsed.visible_tiles == ["7z"]
    assert parsed.analysis_hints["tile_parser_source"] == "vit_discard_only"
    assert parsed.analysis_hints["discard_parser_source"] == "vit_classifier"


def test_vit_classifier_can_require_accelerator(monkeypatch) -> None:
    monkeypatch.setattr(vit_tile_classifier, "_accelerator_available", lambda: False)

    assert not vit_tile_classifier.vit_classifier_enabled(
        {"enabled": True, "require_accelerator": True},
        area="discard",
    )

    monkeypatch.setenv("MAHJONG_COMPANION_VIT_ENABLED", "1")
    assert vit_tile_classifier.vit_classifier_enabled(
        {"enabled": True, "require_accelerator": True},
        area="discard",
    )


def test_default_perception_adapter_disables_vit_for_live_calls(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_analyze_image_path(_image_path: Path, **kwargs):
        calls.append(dict(kwargs.get("tile_classifier_config") or {}))
        return PerceivedGameState(), {}

    monkeypatch.setattr(adapters, "analyze_image_path", fake_analyze_image_path)
    adapter = DefaultPerceptionAdapter(
        tile_classifier_config={
            "enabled": True,
            "live_enabled": False,
            "discard_enabled": True,
        },
    )

    adapter.analyze(Path("frame.png"), live=True)
    adapter.analyze(Path("frame.png"), live=False)

    assert calls[0]["enabled"] is False
    assert calls[0]["force_disabled"] is True
    assert calls[0]["disabled_reason"] == "live_tile_classifier_disabled"
    assert calls[1]["enabled"] is True


def test_force_disabled_beats_global_vit_env(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_COMPANION_VIT_ENABLED", "1")

    assert not vit_tile_classifier.vit_classifier_enabled(
        {"force_disabled": True, "enabled": True},
        area="discard",
    )


def test_train_profile_templates_from_vit_crops_writes_discard_templates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop_root = tmp_path / "sample_crops"
    crop_root.mkdir()
    _tile_image((210, 52, 58)).save(crop_root / "self_01.png")
    base_profile_path = tmp_path / "base.json"
    output_profile_path = tmp_path / "trained.json"
    output_report_path = tmp_path / "report.json"
    save_calibration_profile(
        CalibrationProfile(
            profile_id="base-1920x1080",
            enabled=True,
            screen_width=1920,
            screen_height=1080,
            confidence=0.9,
        ),
        base_profile_path,
    )

    def fake_classify_tile_crops(crops, **_):
        return [
            VitTilePrediction(
                tile="5z",
                label="wd",
                confidence=0.91,
                top_k=[
                    {"label": "wd", "tile": "5z", "score": 0.91},
                    {"label": "rd", "tile": "7z", "score": 0.02},
                ],
            )
            for _crop in crops
        ]

    monkeypatch.setattr(vit_template_training, "classify_tile_crops", fake_classify_tile_crops)

    report = vit_template_training.train_profile_templates_from_vit_crops(
        [crop_root],
        base_profile_path=base_profile_path,
        output_profile_path=output_profile_path,
        output_report_path=output_report_path,
        target="discard",
    )

    trained = load_calibration_profile(output_profile_path)
    assert report.accepted_crops == 1
    assert report.accepted_by_tile == {"5z": 1}
    assert output_report_path.exists()
    assert "5z" in trained.discard_tile_templates["templates"]
    assert trained.profile_id == "base-1920x1080-vit-discard"


def _tile_image(color: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (39, 47), color=(238, 236, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 32, 33), fill=color)
    draw.rectangle((6, 36, 32, 41), fill=(218, 138, 28))
    return image


def _paste_upright_tile(image: Image.Image, slot, tile: Image.Image) -> None:
    box = slot.box
    image.paste(tile.resize((box.width, box.height)), (box.left, box.top))
