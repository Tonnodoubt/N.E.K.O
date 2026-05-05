from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from plugin.plugins.mahjong_companion.scripts.prepare_discard_fixture import (
    main as prepare_discard_fixture_main,
    prepare_discard_fixture,
)


def test_prepare_discard_fixture_writes_overlay_and_label(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output_dir = tmp_path / "labeling"
    Image.new("RGB", (1280, 720), color=(30, 70, 120)).save(source)

    result = prepare_discard_fixture(
        image_path=source,
        case_id="round-1",
        output_dir=output_dir,
        discard_items=[{"player": "self", "turn_index": 1, "tile": "5z"}],
        riichi_players=["right_opponent"],
    )

    assert result["ok"] is True
    assert result["write_fixture"] is False
    assert result["label_scope"] == "partial"
    assert result["discard_count"] == 1
    overlay_path = Path(result["overlay_path"])
    sheet_path = Path(result["sheet_path"])
    assert overlay_path.exists()
    assert sheet_path.exists()
    assert ImageChops.difference(Image.open(source).convert("RGB"), Image.open(overlay_path).convert("RGB")).getbbox()
    assert Image.open(sheet_path).size == (82 * 18, 112 * 4)

    label = json.loads(Path(result["label_path"]).read_text(encoding="utf-8"))
    assert label["label_scope"] == "partial"
    assert label["image"]["path"] == "frame.png"
    assert label["image"]["width"] == 1280
    assert label["image"]["height"] == 720
    assert label["image"]["resolution"] == "1280x720"
    assert label["riichi_players"] == ["right_opponent"]
    assert label["discard_piles"]["self"][0]["tile"] == "5z"
    assert label["discard_piles"]["self"][0]["turn_index"] == 1
    assert len(label["layout"]["discard_slots"]) == 72


def test_prepare_discard_fixture_can_write_eval_case_from_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.png"
    eval_dir = tmp_path / "eval"
    Image.new("RGB", (1280, 720), color=(30, 70, 120)).save(source)

    exit_code = prepare_discard_fixture_main(
        [
            "--image",
            str(source),
            "--case-id",
            "round-2",
            "--discard",
            "self:1:E",
            "--riichi-players",
            "right_opponent",
            "--eval-dir",
            str(eval_dir),
            "--write-fixture",
            "--label-scope",
            "full",
            "--pretty",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["write_fixture"] is True
    label_path = eval_dir / "discard_recognition" / "1280x720" / "round-2" / "frame.label.json"
    label = json.loads(label_path.read_text(encoding="utf-8"))
    assert label["label_scope"] == "full"
    assert label["discard_piles"]["self"][0]["tile"] == "1z"


def test_prepare_discard_fixture_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output_dir = tmp_path / "labeling"
    Image.new("RGB", (1280, 720), color=(30, 70, 120)).save(source)
    prepare_discard_fixture(image_path=source, case_id="round-1", output_dir=output_dir)

    with pytest.raises(FileExistsError):
        prepare_discard_fixture(image_path=source, case_id="round-1", output_dir=output_dir)
