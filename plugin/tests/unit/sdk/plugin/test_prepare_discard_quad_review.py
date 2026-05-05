from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_companion.scripts.prepare_discard_quad_review import (
    main as prepare_discard_quad_review_main,
    prepare_discard_quad_review,
)


def test_prepare_discard_quad_review_writes_overlay_sheet_and_json(tmp_path: Path) -> None:
    image_path = tmp_path / "screens" / "round-1.png"
    _write_quad_sample(image_path)
    output_dir = tmp_path / "review"

    result = prepare_discard_quad_review(
        image_path=image_path,
        output_dir=output_dir,
        include_empty_slots=True,
    )

    assert result["ok"] is True
    assert result["case_count"] == 1
    assert result["quad_count"] >= 2
    case = result["cases"][0]
    assert Path(case["overlay_path"]).exists()
    assert Path(case["sheet_path"]).exists()
    assert Path(case["json_path"]).exists()
    assert ImageChops.difference(
        Image.open(image_path).convert("RGB"),
        Image.open(case["overlay_path"]).convert("RGB"),
    ).getbbox()

    payload = json.loads(Path(case["json_path"]).read_text(encoding="utf-8"))
    refined = [slot for slot in payload["slots"] if slot["refined"]]
    assert payload["schema_version"] == "mahjong-discard-quad-review-v1"
    assert len(refined) >= 2
    assert all(len(slot["quad"]) == 4 for slot in refined)
    assert {"self", "right_opponent"}.issubset({slot["player"] for slot in refined})


def test_prepare_discard_quad_review_batch_cli(tmp_path: Path, capsys) -> None:
    input_dir = tmp_path / "screens"
    _write_quad_sample(input_dir / "round-1.png")
    output_dir = tmp_path / "review"

    exit_code = prepare_discard_quad_review_main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--include-empty-slots",
            "--pretty",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case_count"] == 1
    assert payload["quad_count"] >= 2
    assert Path(payload["review_json_path"]).exists()
    assert Path(payload["review_markdown_path"]).exists()


def _write_quad_sample(image_path: Path) -> None:
    image_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    layout = build_discard_layout(*image.size)
    draw = ImageDraw.Draw(image)
    _draw_slanted_tile(draw, layout["self"][0], fill=(238, 236, 220))
    _draw_slanted_tile(draw, layout["right_opponent"][0], fill=(238, 236, 220))
    image.save(image_path)


def _draw_slanted_tile(draw: ImageDraw.ImageDraw, slot, *, fill: tuple[int, int, int]) -> None:
    box = slot.box
    draw.polygon(
        [
            (box.left - 5, box.top - 4),
            (box.left + 1, box.bottom + 6),
            (box.right + 7, box.bottom - 2),
            (box.right + 1, box.top - 9),
        ],
        fill=fill,
    )
    draw.rectangle((box.left + 8, box.top + 8, box.right - 8, box.bottom - 16), fill=(210, 52, 58))
