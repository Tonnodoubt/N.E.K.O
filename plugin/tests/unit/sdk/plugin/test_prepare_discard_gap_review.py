from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.scripts.prepare_discard_gap_review import (
    main as prepare_discard_gap_review_main,
    prepare_discard_gap_review,
)


def test_prepare_discard_gap_review_writes_top_candidate_sheet_and_json(tmp_path: Path) -> None:
    quad_dir = tmp_path / "quad-review"
    output_dir = tmp_path / "gap-review"
    eval_dir = tmp_path / "eval"
    _write_quad_case(quad_dir / "1920x1080" / "round-1")

    result = prepare_discard_gap_review(quad_review_dir=quad_dir, output_dir=output_dir, eval_dir=eval_dir)

    assert result["ok"] is True
    assert result["eval_dir"] == str(eval_dir)
    assert result["candidate_count"] == 1
    assert result["candidate_counts_by_player"] == {"top_opponent": 1}
    assert Path(result["sheet_path"]).exists()
    assert Path(result["crop_dir"]).exists()
    assert Path(result["review_json_path"]).exists()
    assert Path(result["review_markdown_path"]).exists()
    candidate = result["candidates"][0]
    assert candidate["candidate_id"] == "top-001"
    assert candidate["confirm_spec"] == "top_opponent:3:5z"
    assert Path(candidate["crop_path"]).exists()
    assert "prepare_discard_fixture" in candidate["prepare_fixture_command"]
    markdown = Path(result["review_markdown_path"]).read_text(encoding="utf-8")
    assert "top-001" in markdown
    assert str(eval_dir) in markdown
    assert "round-1-top-reviewed" in markdown


def test_prepare_discard_gap_review_can_include_rejected_candidates_from_cli(
    tmp_path: Path,
    capsys,
) -> None:
    quad_dir = tmp_path / "quad-review"
    output_dir = tmp_path / "gap-review"
    _write_quad_case(quad_dir / "1920x1080" / "round-1")

    exit_code = prepare_discard_gap_review_main(
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
    assert payload["candidate_counts_by_case"] == {"round-1": 2}
    assert [candidate["candidate_id"] for candidate in payload["candidates"]] == ["top-001", "top-002"]


def test_prepare_discard_gap_review_can_exclude_owner_rejected_candidates(tmp_path: Path) -> None:
    quad_dir = tmp_path / "quad-review"
    output_dir = tmp_path / "gap-review"
    _write_quad_case(quad_dir / "1920x1080" / "round-1")
    _append_owner_rejected_candidate(quad_dir / "1920x1080" / "round-1" / "frame.quads.json")

    result = prepare_discard_gap_review(
        quad_review_dir=quad_dir,
        output_dir=output_dir,
        players=["top_opponent"],
        accepted_only=False,
        exclude_owner_rejected=True,
    )

    assert result["candidate_count"] == 2
    assert result["exclude_owner_rejected"] is True
    assert {candidate["confirm_spec"] for candidate in result["candidates"]} == {
        "top_opponent:3:5z",
        "top_opponent:4:7p",
    }


def _write_quad_case(case_dir: Path) -> None:
    case_dir.mkdir(parents=True)
    image = Image.new("RGB", (320, 240), color=(28, 58, 104))
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 40, 150, 90), fill=(238, 236, 220))
    draw.rectangle((170, 40, 220, 90), fill=(238, 236, 220))
    image.save(case_dir / "frame.png")
    (case_dir / "frame.quads.json").write_text(
        json.dumps(
            {
                "schema_version": "mahjong-discard-quad-review-v1",
                "case_id": "round-1",
                "image": {"path": "frame.png", "width": 320, "height": 240},
                "slots": [
                    {
                        "slot_id": "discard_top_opponent_03",
                        "player": "top_opponent",
                        "turn_index": 3,
                        "orientation": "top",
                        "candidate_tile": "5z",
                        "tile_confidence": 0.82,
                        "template_distance": 12.3,
                        "accepted": True,
                        "refined": True,
                        "bbox": [100, 40, 150, 90],
                        "quad": [[100, 40], [100, 90], [150, 90], [150, 40]],
                    },
                    {
                        "slot_id": "discard_top_opponent_04",
                        "player": "top_opponent",
                        "turn_index": 4,
                        "orientation": "top",
                        "candidate_tile": "7p",
                        "tile_confidence": 0.21,
                        "template_distance": 55.1,
                        "accepted": False,
                        "refined": True,
                        "bbox": [170, 40, 220, 90],
                        "quad": [[170, 40], [170, 90], [220, 90], [220, 40]],
                    },
                    {
                        "slot_id": "discard_self_01",
                        "player": "self",
                        "turn_index": 1,
                        "orientation": "bottom",
                        "candidate_tile": "1m",
                        "tile_confidence": 0.9,
                        "accepted": True,
                        "refined": True,
                        "bbox": [10, 180, 50, 220],
                        "quad": [[10, 180], [10, 220], [50, 220], [50, 180]],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _append_owner_rejected_candidate(quad_path: Path) -> None:
    payload = json.loads(quad_path.read_text(encoding="utf-8"))
    payload["slots"].append(
        {
            "slot_id": "discard_top_opponent_05",
            "player": "top_opponent",
            "turn_index": 5,
            "orientation": "top",
            "candidate_tile": "5z",
            "tile_confidence": 0.72,
            "template_distance": 18.0,
            "accepted": False,
            "refined": True,
            "rejected_refinement_owner_slot_id": "discard_top_opponent_04",
            "bbox": [172, 42, 220, 90],
            "quad": [[172, 42], [172, 90], [220, 90], [220, 42]],
        }
    )
    quad_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
