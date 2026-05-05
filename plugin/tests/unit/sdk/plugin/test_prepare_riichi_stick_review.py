from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.scripts.prepare_riichi_stick_review import (
    main as prepare_riichi_stick_review_main,
    prepare_riichi_stick_review,
)


def test_prepare_riichi_stick_review_writes_sheet_json_and_markdown(tmp_path: Path) -> None:
    input_dir = tmp_path / "screens"
    output_dir = tmp_path / "review"
    input_dir.mkdir()
    _write_frame(input_dir / "round-1.png", top=True, self_stick=True)
    _write_frame(input_dir / "round-2.png", right=True)
    _write_frame(input_dir / "round-3.png")

    result = prepare_riichi_stick_review(input_dir=input_dir, output_dir=output_dir, columns=2)

    assert result["ok"] is True
    assert result["image_count"] == 3
    assert result["detected_case_count"] == 2
    assert result["detection_count"] == 3
    assert result["counts_by_player"] == {
        "top_opponent": 1,
        "self": 1,
        "right_opponent": 1,
    }
    assert Path(result["sheet_path"]).exists()
    assert Path(result["review_json_path"]).exists()
    assert Path(result["review_markdown_path"]).exists()
    assert result["cases"][0]["review_crop_bbox"] == [720, 270, 1165, 585]
    markdown = Path(result["review_markdown_path"]).read_text(encoding="utf-8")
    assert "top_opponent" in markdown
    assert "round-3" in markdown


def test_prepare_riichi_stick_review_cli_accepts_single_image(tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "frame.png"
    output_dir = tmp_path / "review"
    _write_frame(image_path, left=True)

    exit_code = prepare_riichi_stick_review_main(
        [
            "--image",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--pretty",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts_by_player"] == {"left_opponent": 1}
    assert Path(payload["sheet_path"]).exists()


def _write_frame(
    path: Path,
    *,
    top: bool = False,
    self_stick: bool = False,
    left: bool = False,
    right: bool = False,
) -> None:
    image = Image.new("RGB", (1920, 1080), color=(28, 58, 104))
    if top:
        _draw_stick(image, (900, 322, 1015, 340))
    if self_stick:
        _draw_stick(image, (890, 516, 1030, 534))
    if left:
        _draw_stick(image, (820, 350, 838, 465))
    if right:
        _draw_stick(image, (1088, 350, 1106, 465))
    image.save(path)


def _draw_stick(image: Image.Image, bbox: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, fill=(236, 236, 232))
    left, top, right, bottom = bbox
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    draw.ellipse((center_x - 5, center_y - 5, center_x + 5, center_y + 5), fill=(228, 18, 18))
