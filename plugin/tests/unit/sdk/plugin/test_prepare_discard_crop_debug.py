from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.scripts.prepare_discard_crop_debug import (
    main as prepare_discard_crop_debug_main,
    prepare_discard_crop_debug,
)


def test_prepare_discard_crop_debug_writes_per_case_sheets(tmp_path: Path) -> None:
    quad_dir = tmp_path / "quad-review"
    output_dir = tmp_path / "crop-debug"
    _write_quad_case(quad_dir / "1920x1080" / "round-1")

    result = prepare_discard_crop_debug(quad_review_dir=quad_dir, output_dir=output_dir)

    assert result["ok"] is True
    assert result["case_count"] == 1
    assert result["candidate_count"] == 3
    assert result["candidate_counts_by_player"] == {
        "left_opponent": 1,
        "top_opponent": 1,
        "right_opponent": 1,
    }
    case = result["cases"][0]
    assert Path(case["overlay_path"]).exists()
    assert case["all_slots_path"].endswith("frame.quad-slots.png")
    assert all(Path(path).exists() for path in case["player_sheets"].values())
    assert Path(result["summary_json_path"]).exists()
    assert Path(result["summary_markdown_path"]).exists()
    markdown = Path(result["summary_markdown_path"]).read_text(encoding="utf-8")
    assert "left_opponent" in markdown
    assert "accepted overlay" in markdown


def test_prepare_discard_crop_debug_cli_can_include_rejected(tmp_path: Path, capsys) -> None:
    quad_dir = tmp_path / "quad-review"
    output_dir = tmp_path / "crop-debug"
    _write_quad_case(quad_dir / "1920x1080" / "round-1")

    exit_code = prepare_discard_crop_debug_main(
        [
            "--quad-review-dir",
            str(quad_dir),
            "--output-dir",
            str(output_dir),
            "--player",
            "top_opponent",
            "--include-rejected",
            "--pretty",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] == 2
    assert payload["candidate_counts_by_player"] == {"top_opponent": 2}


def _write_quad_case(case_dir: Path) -> None:
    case_dir.mkdir(parents=True)
    image = Image.new("RGB", (360, 260), color=(28, 58, 104))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 50, 90, 120), fill=(238, 236, 220))
    draw.rectangle((150, 30, 205, 85), fill=(238, 236, 220))
    draw.rectangle((260, 50, 310, 120), fill=(238, 236, 220))
    image.save(case_dir / "frame.png")
    image.save(case_dir / "frame.quad-slots.png")
    (case_dir / "frame.quads.json").write_text(
        json.dumps(
            {
                "schema_version": "mahjong-discard-quad-review-v1",
                "case_id": "round-1",
                "image": {"path": "frame.png", "width": 360, "height": 260},
                "slots": [
                    _slot("discard_left_opponent_01", "left_opponent", 1, "left", "5z", True, [40, 50, 90, 120]),
                    _slot("discard_top_opponent_02", "top_opponent", 2, "top", "2m", True, [150, 30, 205, 85]),
                    _slot("discard_top_opponent_03", "top_opponent", 3, "top", "4p", False, [210, 30, 260, 85]),
                    _slot("discard_right_opponent_01", "right_opponent", 1, "right", "5z", True, [260, 50, 310, 120]),
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _slot(
    slot_id: str,
    player: str,
    turn_index: int,
    orientation: str,
    tile: str,
    accepted: bool,
    bbox: list[int],
) -> dict:
    left, top, right, bottom = bbox
    return {
        "slot_id": slot_id,
        "player": player,
        "turn_index": turn_index,
        "orientation": orientation,
        "candidate_tile": tile,
        "tile_confidence": 0.72 if accepted else 0.22,
        "accepted": accepted,
        "refined": True,
        "bbox": bbox,
        "quad": [[left, top], [left, bottom], [right, bottom], [right, top]],
    }
