from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.perception.calibration import (
    CalibrationProfile,
    label_sidecar_path,
    resolve_calibration_profile,
    save_calibration_profile,
    train_calibration_profile,
)
from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_companion.perception.hand_layout import build_hand_layout
from plugin.plugins.mahjong_companion.perception.tile_parser import parse_tiles_from_image
from plugin.plugins.mahjong_companion.scripts.label_calibration import main as label_calibration_main


def test_label_calibration_cli_writes_sidecar_and_trained_profile(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw_dir = tmp_path / "data" / "calibration" / "raw" / "1280x720" / "in_match"
    raw_dir.mkdir(parents=True)
    image_path = raw_dir / "frame.png"
    Image.new("RGB", (1280, 720), color=(40, 80, 140)).save(image_path)
    profile_path = tmp_path / "data" / "calibration" / "profiles" / "majsoul-pc-2026.04-1280x720.json"

    exit_code = label_calibration_main(
        [
            "--raw-dir",
            str(raw_dir),
            "--image",
            str(image_path),
            "--hand-tiles",
            "1m,E,R5s",
            "--client-version",
            "2026.04",
            "--train-output",
            str(profile_path),
            "--min-samples",
            "1",
            "--pretty",
        ],
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["profile_written"] == str(profile_path)
    label_payload = json.loads(label_sidecar_path(image_path).read_text(encoding="utf-8"))
    assert label_payload["schema_version"] == "mahjong-calibration-label-v1"
    assert label_payload["hand_tiles"] == ["1m", "1z", "R5s"]
    assert len(label_payload["layout"]["hand_slots"]) == 14

    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile_payload["enabled"] is True
    assert profile_payload["screen_width"] == 1280
    assert profile_payload["screen_height"] == 720
    assert profile_payload["hand_tile_templates"]["source_sample_count"] == 3


def test_train_calibration_profile_records_median_hand_offsets(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "1280x720"
    raw_dir.mkdir(parents=True)
    default_first = build_hand_layout(1280, 720)["hand"][0].box

    for index, delta_x in enumerate([8, 12, 10], start=1):
        image_path = raw_dir / f"frame-{index}.png"
        Image.new("RGB", (1280, 720), color=(30, 70, 120)).save(image_path)
        label_calibration_main(
            [
                "--raw-dir",
                str(raw_dir),
                "--image",
                str(image_path),
                "--hand-tiles",
                "1m 2m 3m 4p 5p 6p 7s 8s 9s 1z 1z 9m 5z",
                "--hand-left",
                str(default_first.left + delta_x),
                "--hand-top",
                str(default_first.top - 6),
                "--tile-width",
                str(default_first.width + 3),
            "--tile-height",
            str(default_first.height + 4),
            "--tile-gap",
            "2",
            "--draw-gap",
            "36",
            "--min-samples",
            "1",
        ],
        )

    profile = train_calibration_profile(raw_dir, client_version="2026.04", min_samples=3)

    assert profile.profile_id == "majsoul-pc-2026.04-1280x720"
    assert profile.enabled is True
    assert profile.hand_offsets.x_px == 10
    assert profile.hand_offsets.y_px == -6
    assert profile.hand_offsets.width_px == 3
    assert profile.hand_offsets.height_px == 4
    assert profile.hand_offsets.gap_px == 2 - int((default_first.width + 3) * 0.12)
    assert profile.hand_offsets.draw_gap_px == 36


def test_resolve_calibration_profile_loads_named_profile_from_profiles_dir(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_path = raw_dir / "frame.png"
    Image.new("RGB", (1280, 720), color=(30, 70, 120)).save(image_path)
    label_calibration_main(
        [
            "--raw-dir",
            str(raw_dir),
            "--image",
            str(image_path),
            "--hand-tiles",
            "1m 2m 3m",
            "--min-samples",
            "1",
        ],
    )
    profile = train_calibration_profile(raw_dir, client_version="2026.04", min_samples=1)
    profile_dir = tmp_path / "calibration" / "profiles"
    profile_path = profile_dir / "majsoul-pc-2026.04-1280x720.json"
    save_calibration_profile(profile, profile_path)

    resolved = resolve_calibration_profile(1280, 720, calibration_dir=tmp_path / "calibration")

    assert resolved.profile_id == "majsoul-pc-2026.04-1280x720"
    assert resolved.enabled is True
    assert resolved.hand_tile_templates


def test_resolve_calibration_profile_merges_specialized_template_profiles(tmp_path: Path) -> None:
    profile_dir = tmp_path / "calibration" / "profiles"
    manual = CalibrationProfile(
        profile_id="majsoul-pc-manual-2026.05-1920x1080",
        source="manual labels",
        enabled=True,
        screen_width=1920,
        screen_height=1080,
        confidence=0.95,
        hand_tile_templates=_template_payload("manual_hand", stored_sample_count=12),
        discard_tile_templates=_template_payload("manual_discard", stored_sample_count=4),
    )
    vit_discard = CalibrationProfile(
        profile_id="majsoul-pc-manual-2026.05-vit-discard-1920x1080",
        source="vit_template_training",
        enabled=True,
        screen_width=1920,
        screen_height=1080,
        confidence=0.96,
        hand_tile_templates=_template_payload("vit_labeled_hand", stored_sample_count=64),
        discard_tile_templates=_template_payload("vit_labeled_discard", stored_sample_count=128),
    )
    save_calibration_profile(manual, profile_dir / "majsoul-pc-manual-2026.05-1920x1080.json")
    save_calibration_profile(vit_discard, profile_dir / "majsoul-pc-manual-2026.05-vit-discard-1920x1080.json")

    resolved = resolve_calibration_profile(1920, 1080, calibration_dir=tmp_path / "calibration")

    assert resolved.profile_id == "majsoul-pc-manual-2026.05-vit-discard-1920x1080-merged"
    assert resolved.hand_tile_templates["source"] == "manual_hand"
    assert resolved.discard_tile_templates["source"] == "vit_labeled_discard"
    assert "merged_hand_templates_from_profile=majsoul-pc-manual-2026.05-1920x1080" in resolved.notes


def test_resolve_calibration_profile_scales_same_aspect_profile(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_path = raw_dir / "frame.png"
    Image.new("RGB", (1280, 720), color=(30, 70, 120)).save(image_path)
    label_calibration_main(
        [
            "--raw-dir",
            str(raw_dir),
            "--image",
            str(image_path),
            "--hand-tiles",
            "1m 2m 3m",
            "--hand-left",
            str(build_hand_layout(1280, 720)["hand"][0].box.left + 12),
            "--tile-width",
            str(build_hand_layout(1280, 720)["hand"][0].box.width + 6),
            "--draw-gap",
            "30",
            "--min-samples",
            "1",
        ],
    )
    profile = train_calibration_profile(raw_dir, client_version="2026.04", min_samples=1)
    profile_dir = tmp_path / "calibration" / "profiles"
    save_calibration_profile(profile, profile_dir / "majsoul-pc-2026.04-1280x720.json")

    resolved = resolve_calibration_profile(2560, 1440, calibration_dir=tmp_path / "calibration")

    assert resolved.profile_id == "majsoul-pc-2026.04-1280x720-scaled-2560x1440"
    assert resolved.screen_width == 2560
    assert resolved.screen_height == 1440
    assert resolved.hand_offsets.x_px == profile.hand_offsets.x_px * 2
    assert resolved.hand_offsets.width_px == profile.hand_offsets.width_px * 2
    assert resolved.hand_offsets.draw_gap_px == profile.hand_offsets.draw_gap_px * 2
    assert resolved.hand_tile_templates == profile.hand_tile_templates
    assert "scaled_from_resolution=1280x720" in resolved.notes


def _template_payload(source: str, *, stored_sample_count: int) -> dict[str, object]:
    return {
        "version": "mahjong-hand-template-v1",
        "signature_version": "rgb-inner-16x24-v1",
        "source": source,
        "source_sample_count": stored_sample_count,
        "stored_sample_count": stored_sample_count,
        "templates": {"1m": {"count": stored_sample_count, "signatures": ["dummy"]}},
    }


def test_parse_tiles_from_image_uses_trained_template_profile_without_sidecar(tmp_path: Path) -> None:
    calibration_dir = tmp_path / "calibration"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_path = raw_dir / "frame.png"
    probe_path = raw_dir / "probe.png"
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    draw = ImageDraw.Draw(image)
    slots = build_hand_layout(1280, 720)["hand"]
    for slot, color in zip(slots[:3], [(214, 64, 70), (56, 156, 90), (54, 92, 205)], strict=True):
        box = slot.box
        draw.rectangle((box.left, box.top, box.right, box.bottom), fill=(238, 236, 220))
        draw.rectangle(
            (box.left + 8, box.top + 8, box.right - 8, box.bottom - 18),
            fill=color,
        )
    image.save(image_path)
    image.save(probe_path)

    label_calibration_main(
        [
            "--raw-dir",
            str(raw_dir),
            "--image",
            str(image_path),
            "--hand-tiles",
            "1m 2p 3s",
            "--client-version",
            "2026.04",
            "--train-output",
            str(calibration_dir / "profiles" / "majsoul-pc-2026.04-1280x720.json"),
            "--min-samples",
            "1",
        ],
    )

    with Image.open(probe_path) as opened:
        parsed = parse_tiles_from_image(
            probe_path,
            opened.convert("RGB"),
            scene="in_match",
            metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
            calibration_dir=calibration_dir,
            fixture_mode="disabled",
        )

    assert parsed.hand_tiles == ["1m", "2p", "3s"]
    assert parsed.analysis_hints["tile_parser_source"] == "template_profile"
    assert parsed.analysis_hints["tile_level_available"] is True


def test_train_calibration_profile_builds_discard_tile_templates(tmp_path: Path) -> None:
    calibration_dir = tmp_path / "calibration"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_path = raw_dir / "frame.png"
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

    label_calibration_main(
        [
            "--raw-dir",
            str(raw_dir),
            "--image",
            str(image_path),
            "--hand-tiles",
            "2p",
            "--discard",
            "self:1:1m",
            "--riichi-players",
            "self",
            "--min-samples",
            "1",
        ],
    )
    label_path = label_sidecar_path(image_path)
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert payload["riichi_players"] == ["self"]
    assert payload["discard_piles"]["self"][0]["tile"] == "1m"
    assert payload["discard_piles"]["self"][0]["bbox"] == discard_slot.bbox

    profile = train_calibration_profile(raw_dir, client_version="2026.04", min_samples=1)
    save_calibration_profile(profile, calibration_dir / "profiles" / "majsoul-pc-2026.04-1280x720.json")
    resolved = resolve_calibration_profile(1280, 720, calibration_dir=calibration_dir)

    assert resolved.hand_tile_templates["source_sample_count"] == 1
    assert resolved.discard_tile_templates["source_sample_count"] == 1
    assert any(note.startswith("discard_tile_template_samples=") for note in resolved.notes)


def test_label_calibration_train_output_includes_extra_label_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_dir = tmp_path / "raw"
    extra_dir = tmp_path / "eval" / "discard_recognition" / "1280x720" / "reviewed"
    profile_path = tmp_path / "calibration" / "profiles" / "majsoul-pc-2026.04-1280x720.json"
    raw_dir.mkdir()
    extra_dir.mkdir(parents=True)

    raw_image_path = raw_dir / "raw-frame.png"
    raw_image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    hand_tile = _tile_image((54, 92, 205))
    hand_slot = build_hand_layout(*raw_image.size)["hand"][0]
    raw_image.paste(hand_tile.resize((hand_slot.box.width, hand_slot.box.height)), (hand_slot.box.left, hand_slot.box.top))
    raw_image.save(raw_image_path)
    label_calibration_main(
        [
            "--raw-dir",
            str(raw_dir),
            "--image",
            str(raw_image_path),
            "--hand-tiles",
            "2p",
            "--min-samples",
            "1",
        ],
    )
    capsys.readouterr()

    extra_image_path = extra_dir / "frame.png"
    extra_image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    discard_tile = _tile_image((210, 52, 58))
    discard_slot = build_discard_layout(*extra_image.size)["self"][0]
    extra_image.paste(
        discard_tile.resize((discard_slot.box.width, discard_slot.box.height)),
        (discard_slot.box.left, discard_slot.box.top),
    )
    extra_image.save(extra_image_path)
    (extra_dir / "frame.label.json").write_text(
        json.dumps(
            {
                "schema_version": "mahjong-discard-recognition-label-v1",
                "image": {"path": "frame.png"},
                "discard_piles": {
                    "self": [
                        {
                            "tile": "7p",
                            "turn_index": 1,
                            "bbox": discard_slot.bbox,
                            "orientation": "bottom",
                        }
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = label_calibration_main(
        [
            "--raw-dir",
            str(raw_dir),
            "--train-extra-root",
            str(extra_dir),
            "--client-version",
            "2026.04",
            "--train-output",
            str(profile_path),
            "--min-samples",
            "1",
            "--pretty",
        ],
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["train_extra_roots"] == [str(extra_dir)]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["hand_tile_templates"]["source_sample_count"] == 1
    assert profile["discard_tile_templates"]["source_sample_count"] == 1
    assert "7p" in profile["discard_tile_templates"]["templates"]


def test_parse_tiles_from_image_uses_calibration_label_sidecar(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_path = raw_dir / "frame.png"
    Image.new("RGB", (1280, 720), color=(40, 80, 140)).save(image_path)
    label_calibration_main(
        [
            "--raw-dir",
            str(raw_dir),
            "--image",
            str(image_path),
            "--hand-tiles",
            "1m 2m 3m",
            "--min-samples",
            "1",
        ],
    )

    with Image.open(image_path) as image:
        parsed = parse_tiles_from_image(
            image_path,
            image.convert("RGB"),
            scene="in_match",
            metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        )

    assert parsed.hand_tiles == ["1m", "2m", "3m"]
    assert parsed.analysis_hints["tile_level_available"] is True
    assert parsed.raw_detections[0]["candidate_tile"] == "1m"


def _tile_image(color: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (39, 47), color=(238, 236, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 32, 33), fill=color)
    draw.rectangle((6, 36, 32, 41), fill=(218, 138, 28))
    return image
