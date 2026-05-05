from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from plugin.plugins.mahjong_companion.scripts.apply_discard_candidate_confirmations import (
    apply_discard_candidate_confirmations,
    main as apply_discard_candidate_confirmations_main,
)


def test_apply_discard_candidate_confirmations_writes_grouped_fixtures(tmp_path: Path) -> None:
    review_path = _write_review(tmp_path)
    eval_dir = tmp_path / "eval"

    result = apply_discard_candidate_confirmations(
        review_path=review_path,
        accept_ids=["top-001"],
        correction_specs={"top-002": "7p"},
        reject_ids=["right-001"],
        eval_dir=eval_dir,
        case_suffix="manual-reviewed",
        overwrite=True,
    )

    assert result["ok"] is True
    assert result["selected_count"] == 2
    assert result["fixture_count"] == 1
    fixture_path = Path(result["fixture_results"][0]["fixture_label_path"])
    label = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert label["case_id"] == "round-1-manual-reviewed"
    pile = label["discard_piles"]["top_opponent"]
    assert [item["tile"] for item in pile] == ["2m", "7p"]
    assert "right_opponent" not in label["discard_piles"]


def test_apply_discard_candidate_confirmations_writes_template_and_reads_confirmation_file(
    tmp_path: Path,
    capsys,
) -> None:
    review_path = _write_review(tmp_path)
    confirmation_path = tmp_path / "confirmations.json"
    template_path = tmp_path / "template.json"
    eval_dir = tmp_path / "eval"
    confirmation_path.write_text(
        json.dumps(
            {
                "confirmations": [
                    {"candidate_id": "top-001", "status": "accept"},
                    {"candidate_id": "right-001", "status": "correct", "tile": "P"},
                    {"candidate_id": "top-002", "status": "reject"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = apply_discard_candidate_confirmations_main(
        [
            "--review",
            str(review_path),
            "--confirmations",
            str(confirmation_path),
            "--write-template",
            str(template_path),
            "--eval-dir",
            str(eval_dir),
            "--overwrite",
            "--pretty",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["template_path"]).exists()
    assert payload["selected_count"] == 2
    assert payload["fixture_count"] == 1
    fixture_path = Path(payload["fixture_results"][0]["fixture_label_path"])
    label = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert label["discard_piles"]["top_opponent"][0]["tile"] == "2m"
    assert label["discard_piles"]["right_opponent"][0]["tile"] == "5z"


def _write_review(tmp_path: Path) -> Path:
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (1280, 720), color=(30, 70, 120)).save(frame_path)
    review_path = tmp_path / "discard-gap-review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": "mahjong-discard-gap-review-v1",
                "candidates": [
                    _candidate("top-001", frame_path, "round-1", "top_opponent", 1, "2m"),
                    _candidate("top-002", frame_path, "round-1", "top_opponent", 2, "4p"),
                    _candidate("right-001", frame_path, "round-1", "right_opponent", 3, "6p"),
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return review_path


def _candidate(
    candidate_id: str,
    frame_path: Path,
    case_id: str,
    player: str,
    turn_index: int,
    tile: str,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "case_id": case_id,
        "frame_path": str(frame_path),
        "player": player,
        "turn_index": turn_index,
        "candidate_tile": tile,
        "confirm_spec": f"{player}:{turn_index}:{tile}",
        "crop_path": str(frame_path.with_name(f"{candidate_id}.png")),
    }
