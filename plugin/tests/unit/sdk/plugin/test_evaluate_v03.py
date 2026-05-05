from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.perception.hand_layout import build_hand_layout
from plugin.plugins.mahjong_companion.scripts import evaluate_v03 as evaluate_v03_module
from plugin.plugins.mahjong_companion.scripts.evaluate_v03 import evaluate_v03, main as evaluate_v03_main
from plugin.plugins.mahjong_companion.scripts.label_calibration import main as label_calibration_main


HAND = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "9m", "5z"]


def test_evaluate_v03_disables_hand_sidecars_by_default(tmp_path: Path) -> None:
    hand_dir = tmp_path / "hand_recognition" / "1280x720"
    hand_dir.mkdir(parents=True)
    image_path = hand_dir / "frame.png"
    Image.new("RGB", (1280, 720), color=(40, 80, 140)).save(image_path)
    (hand_dir / "frame.label.json").write_text(
        json.dumps({"hand_tiles": HAND, "image": {"path": "frame.png"}, "scene": "in_match"}),
        encoding="utf-8",
    )

    report = evaluate_v03(eval_dir=tmp_path)

    assert report["ok"] is True
    assert report["metrics"]["hand_recognition"]["case_count"] == 1
    assert report["metrics"]["hand_recognition"]["tile_accuracy"] == 0.0
    assert "case_details" not in report["metrics"]["hand_recognition"]


def test_evaluate_v03_can_include_hand_recognition_diagnostics(tmp_path: Path) -> None:
    hand_dir = tmp_path / "hand_recognition" / "1280x720"
    hand_dir.mkdir(parents=True)
    image_path = hand_dir / "frame.png"
    Image.new("RGB", (1280, 720), color=(40, 80, 140)).save(image_path)
    (hand_dir / "frame.label.json").write_text(
        json.dumps({"hand_tiles": HAND[:3], "image": {"path": "frame.png"}, "scene": "in_match"}),
        encoding="utf-8",
    )

    report = evaluate_v03(eval_dir=tmp_path, include_details=True)
    hand = report["metrics"]["hand_recognition"]

    assert hand["mismatch_case_count"] == 1
    assert hand["missing_prediction_tiles"] == 3
    assert hand["confusion_top"][0] == {"expected": "1m", "predicted": "", "count": 1}
    assert hand["case_details"][0]["expected_hand_tiles"] == HAND[:3]
    assert hand["case_details"][0]["predicted_hand_tiles"] == []
    assert hand["case_details"][0]["mismatches"] == [
        {"slot": 1, "expected": "1m", "predicted": ""},
        {"slot": 2, "expected": "2m", "predicted": ""},
        {"slot": 3, "expected": "3m", "predicted": ""},
    ]


def test_evaluate_v03_allows_fixture_sidecars_when_explicit(tmp_path: Path) -> None:
    hand_dir = tmp_path / "hand_recognition" / "1280x720"
    hand_dir.mkdir(parents=True)
    image_path = hand_dir / "frame.png"
    Image.new("RGB", (1280, 720), color=(40, 80, 140)).save(image_path)
    (hand_dir / "frame.label.json").write_text(
        json.dumps(
            {
                "hand_tiles": HAND,
                "image": {"path": "frame.png"},
                "scene": "in_match",
                "analysis_confidence": 0.86,
            },
        ),
        encoding="utf-8",
    )

    report = evaluate_v03(eval_dir=tmp_path, allow_fixture_sidecars=True)

    assert report["ok"] is True
    assert report["metrics"]["hand_recognition"]["tile_accuracy"] == 1.0
    assert report["metrics"]["hand_recognition"]["full_hand_accuracy"] == 1.0


def test_evaluate_v03_reports_red5_normalized_hand_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hand_dir = tmp_path / "hand_recognition" / "1280x720"
    hand_dir.mkdir(parents=True)
    image_path = hand_dir / "frame.png"
    Image.new("RGB", (1280, 720), color=(40, 80, 140)).save(image_path)
    (hand_dir / "frame.label.json").write_text(
        json.dumps({"hand_tiles": ["R5m", "5m", "4p"], "image": {"path": "frame.png"}, "scene": "in_match"}),
        encoding="utf-8",
    )

    def fake_parse_tiles_from_image(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            hand_tiles=["5m", "5m", "4p"],
            analysis_hints={
                "tile_parser_source": "test-double",
                "analysis_confidence": 0.91,
                "calibration_profile": "test-profile",
            },
        )

    monkeypatch.setattr(evaluate_v03_module, "parse_tiles_from_image", fake_parse_tiles_from_image)

    report = evaluate_v03(eval_dir=tmp_path, include_details=True)
    hand = report["metrics"]["hand_recognition"]
    detail = hand["case_details"][0]

    assert hand["tile_accuracy"] == 0.6667
    assert hand["full_hand_accuracy"] == 0.0
    assert hand["red5_normalized_tile_accuracy"] == 1.0
    assert hand["red5_normalized_full_hand_accuracy"] == 1.0
    assert hand["red5_only_mismatch_count"] == 1
    assert detail["red5_normalized_mismatches"] == []


def test_evaluate_v03_hand_holdout_trains_without_answer_sidecars(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    slots = build_hand_layout(1280, 720)["hand"]
    colors = [(214, 64, 70), (56, 156, 90), (54, 92, 205)]
    for index in range(3):
        image_path = raw_dir / f"frame-{index}.png"
        image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
        draw = ImageDraw.Draw(image)
        for slot, color in zip(slots[:3], colors, strict=True):
            box = slot.box
            draw.rectangle((box.left, box.top, box.right, box.bottom), fill=(238, 236, 220))
            draw.rectangle((box.left + 8, box.top + 8, box.right - 8, box.bottom - 18), fill=color)
        image.save(image_path)
        label_calibration_main(
            [
                "--raw-dir",
                str(raw_dir),
                "--image",
                str(image_path),
                "--hand-tiles",
                "1m 2p 3s",
                "--min-samples",
                "1",
            ],
        )

    report = evaluate_v03(
        eval_dir=tmp_path / "eval",
        strict_hand=True,
        hand_holdout_dir=raw_dir,
        holdout_folds=3,
        holdout_min_train_samples=1,
        include_details=True,
    )
    holdout = report["metrics"]["hand_holdout"]

    assert report["ok"] is True
    assert holdout["case_count"] == 3
    assert holdout["fold_count"] == 3
    assert holdout["tile_accuracy"] == 1.0
    assert holdout["full_hand_accuracy"] == 1.0
    assert holdout["coverage_blocked_tiles"] == 0
    assert holdout["coverage_adjusted_tile_accuracy"] == 1.0
    assert holdout["red5_normalized_tile_accuracy"] == 1.0
    assert holdout["red5_normalized_full_hand_accuracy"] == 1.0
    assert holdout["red5_normalized_coverage_adjusted_tile_accuracy"] == 1.0
    assert holdout["case_details"][0]["tile_parser_source"] == "template_profile"


def test_evaluate_v03_strict_accepts_hand_holdout_with_checked_in_hand_fixtures(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    slots = build_hand_layout(1280, 720)["hand"]
    colors = [(214, 64, 70), (56, 156, 90), (54, 92, 205)]
    for index in range(3):
        image_path = raw_dir / f"frame-{index}.png"
        image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
        draw = ImageDraw.Draw(image)
        for slot, color in zip(slots[:3], colors, strict=True):
            box = slot.box
            draw.rectangle((box.left, box.top, box.right, box.bottom), fill=(238, 236, 220))
            draw.rectangle((box.left + 8, box.top + 8, box.right - 8, box.bottom - 18), fill=color)
        image.save(image_path)
        label_calibration_main(
            [
                "--raw-dir",
                str(raw_dir),
                "--image",
                str(image_path),
                "--hand-tiles",
                "1m 2p 3s",
                "--min-samples",
                "1",
            ],
        )

    report = evaluate_v03(
        eval_dir=Path("plugin/tests/data/mahjong_companion/eval"),
        strict=True,
        hand_holdout_dir=raw_dir,
        holdout_folds=3,
        holdout_min_train_samples=1,
    )

    assert report["ok"] is True
    assert report["metrics"]["hand_recognition"]["case_count"] == 21
    assert report["metrics"]["hand_recognition"]["tile_accuracy"] == 1.0
    assert report["metrics"]["hand_holdout"]["case_count"] == 3


def test_evaluate_v03_scores_decision_and_risk_cases(tmp_path: Path) -> None:
    decision_dir = tmp_path / "decision_top1"
    decision_dir.mkdir(parents=True)
    (decision_dir / "case-1.json").write_text(
        json.dumps(
            {
                "hand_tiles": HAND,
                "expected_top1": "5z",
                "scene": "in_match",
                "is_user_turn": True,
                "confidence": 0.9,
            },
        ),
        encoding="utf-8",
    )
    risk_dir = tmp_path / "risk_detection"
    risk_dir.mkdir(parents=True)
    (risk_dir / "case-1.json").write_text(
        json.dumps(
            {
                "hand_tiles": HAND,
                "riichi_players": ["right_opponent"],
                "expected_genbutsu": True,
                "expected_genbutsu_tiles": ["5z"],
            },
        ),
        encoding="utf-8",
    )

    report = evaluate_v03(eval_dir=tmp_path)

    assert report["metrics"]["decision_top1"]["match_rate"] == 1.0
    assert report["metrics"]["decision_top1"]["top3_match_rate"] == 1.0
    assert report["metrics"]["risk_detection"]["genbutsu_recall"] == 1.0
    assert report["metrics"]["decision_latency_p95_ms"] is not None


def test_evaluate_v03_scores_review_structure_and_repeated_patterns(tmp_path: Path) -> None:
    review_dir = tmp_path / "review_patterns"
    review_dir.mkdir(parents=True)
    (review_dir / "case-1.json").write_text(
        json.dumps(
            {
                "session_id": "mahjong-eval",
                "candidates": [
                    {
                        "captured_at": "2026-04-15T00:00:00+00:00",
                        "scene": "in_match",
                        "decision_type": "tile_efficiency_hint",
                        "priority": 72,
                        "risk_level": "medium",
                        "summary": "这一巡更适合先走稳一点的牌效率路线。",
                        "recommended_focus": "tile_efficiency",
                        "review_tags": ["tile_efficiency", "mid_round_choice"],
                    }
                ],
                "review_summaries": [
                    {
                        "session_id": "mahjong-a",
                        "generated_at": "2026-04-15T00:00:00+00:00",
                        "training_points": ["中盘牌效率弃牌优先级"],
                    },
                    {
                        "session_id": "mahjong-b",
                        "generated_at": "2026-04-16T00:00:00+00:00",
                        "training_points": ["中盘牌效率弃牌优先级"],
                    },
                ],
                "expected_repeated_patterns": ["tile_efficiency"],
            },
        ),
        encoding="utf-8",
    )

    report = evaluate_v03(eval_dir=tmp_path)

    assert report["metrics"]["review_patterns"]["structured_summary_pass_rate"] == 1.0
    assert report["metrics"]["review_patterns"]["repeated_pattern_match_rate"] == 1.0


def test_checked_in_v03_json_fixtures_score_cleanly() -> None:
    eval_dir = Path("plugin/tests/data/mahjong_companion/eval")

    report = evaluate_v03(
        eval_dir=eval_dir,
        calibration_dir=Path("plugin/plugins/mahjong_companion/data/calibration"),
    )

    assert report["ok"] is True
    assert report["metrics"]["decision_top1"]["case_count"] >= 6
    assert report["metrics"]["decision_top1"]["match_rate"] == 1.0
    assert report["metrics"]["decision_top1"]["top3_match_rate"] == 1.0
    assert report["metrics"]["risk_detection"]["expected_genbutsu_cases"] >= 3
    assert report["metrics"]["risk_detection"]["genbutsu_recall"] == 1.0
    assert report["metrics"]["review_patterns"]["case_count"] >= 2
    assert report["metrics"]["review_patterns"]["structured_summary_pass_rate"] == 1.0


def test_evaluate_v03_strict_json_passes_checked_in_json_fixtures(tmp_path: Path) -> None:
    report_path = tmp_path / "strict-json-report.json"

    exit_code = evaluate_v03_main(
        [
            "--eval-dir",
            "plugin/tests/data/mahjong_companion/eval",
            "--calibration-dir",
            "plugin/plugins/mahjong_companion/data/calibration",
            "--strict-json",
            "--report",
            str(report_path),
        ],
    )

    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["strict_json"] is True
    assert report["metrics"]["hand_recognition"]["case_count"] == 21
    assert report["metrics"]["hand_recognition"]["full_hand_accuracy"] == 1.0


def test_evaluate_v03_strict_fails_when_eval_sets_are_missing(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"

    exit_code = evaluate_v03_main(["--eval-dir", str(tmp_path), "--report", str(report_path), "--strict"])

    assert exit_code == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert "hand_recognition or hand_holdout has no eval cases" in report["failures"]
