from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.perception.calibration import CalibrationProfile, save_calibration_profile
from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_companion.perception.hand_layout import build_hand_layout
from plugin.plugins.mahjong_companion.perception.tile_templates import build_hand_tile_template_payload
from plugin.plugins.mahjong_companion.scripts.prepare_discard_labeling_batch import (
    main as prepare_discard_labeling_batch_main,
    prepare_discard_labeling_batch,
)


def test_prepare_discard_labeling_batch_writes_review_and_candidate_overlay(tmp_path: Path) -> None:
    input_dir = tmp_path / "screenshots"
    output_dir = tmp_path / "labeling"
    calibration_dir = tmp_path / "calibration"
    image_path = input_dir / "round-1.png"
    _write_sample_image_and_profile(image_path, calibration_dir)

    result = prepare_discard_labeling_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        calibration_dir=calibration_dir,
    )

    assert result["ok"] is True
    assert result["image_count"] == 1
    assert result["candidate_count"] == 1
    assert result["label_scope"] == "partial"
    assert result["cases"][0]["label_scope"] == "partial"
    assert result["cases"][0]["candidates"][0]["tile"] == "1m"
    assert result["cases"][0]["candidates"][0]["player"] == "self"
    assert Path(result["cases"][0]["overlay_path"]).exists()
    assert Path(result["cases"][0]["sheet_path"]).exists()
    assert Path(result["review_json_path"]).exists()
    assert Path(result["review_markdown_path"]).exists()
    review = json.loads(Path(result["review_json_path"]).read_text(encoding="utf-8"))
    assert review["candidate_count"] == 1


def test_prepare_discard_labeling_batch_cli(tmp_path: Path, capsys) -> None:
    input_dir = tmp_path / "screenshots"
    output_dir = tmp_path / "labeling"
    calibration_dir = tmp_path / "calibration"
    image_path = input_dir / "round-1.png"
    _write_sample_image_and_profile(image_path, calibration_dir)

    exit_code = prepare_discard_labeling_batch_main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--calibration-dir",
            str(calibration_dir),
            "--pretty",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] == 1
    assert payload["label_scope"] == "partial"


def _write_sample_image_and_profile(image_path: Path, calibration_dir: Path) -> None:
    image_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    hand_tile = _tile_image((54, 92, 205))
    discard_tile = _tile_image((210, 52, 58))
    hand_slot = build_hand_layout(*image.size)["hand"][0]
    discard_slot = build_discard_layout(*image.size)["self"][0]
    image.paste(hand_tile.resize((hand_slot.box.width, hand_slot.box.height)), (hand_slot.box.left, hand_slot.box.top))
    image.paste(
        discard_tile.resize((discard_slot.box.width, discard_slot.box.height)),
        (discard_slot.box.left, discard_slot.box.top),
    )
    image.save(image_path)
    save_calibration_profile(
        CalibrationProfile(
            profile_id="test-1280x720",
            enabled=True,
            screen_width=1280,
            screen_height=720,
            confidence=0.9,
            hand_tile_templates=build_hand_tile_template_payload([("2p", hand_tile)]),
            discard_tile_templates=build_hand_tile_template_payload([("1m", discard_tile)]),
        ),
        calibration_dir / "profiles" / "test-1280x720.json",
    )


def _tile_image(color: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (39, 47), color=(238, 236, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 32, 33), fill=color)
    draw.rectangle((6, 36, 32, 41), fill=(218, 138, 28))
    return image
