from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.perception import vit_template_training, vit_tile_classifier
from plugin.plugins.mahjong_companion.perception.calibration import CalibrationProfile, load_calibration_profile, save_calibration_profile
from plugin.plugins.mahjong_companion.perception.vit_tile_classifier import VitTilePrediction


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
