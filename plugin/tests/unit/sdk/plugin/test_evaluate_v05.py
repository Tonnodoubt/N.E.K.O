from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.perception.calibration import CalibrationProfile, save_calibration_profile
from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_companion.perception.hand_layout import build_hand_layout
from plugin.plugins.mahjong_companion.perception.tile_templates import build_hand_tile_template_payload
from plugin.plugins.mahjong_companion.scripts.evaluate_v05 import evaluate_v05, main as evaluate_v05_main


def test_evaluate_v05_scores_discard_and_genbutsu_metrics(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    calibration_dir = tmp_path / "calibration"
    _write_discard_case(eval_dir, calibration_dir)
    _write_genbutsu_case(eval_dir)

    report = evaluate_v05(eval_dir=eval_dir, calibration_dir=calibration_dir, strict=True, include_details=True)

    assert report["ok"] is True
    assert report["metrics"]["discard_recognition"]["tile_accuracy"] == 1.0
    assert report["metrics"]["discard_recognition"]["recall"] == 1.0
    assert report["metrics"]["discard_recognition"]["false_positive_rate"] == 0.0
    assert report["metrics"]["discard_recognition"]["by_player"]["self"]["recall"] == 1.0
    assert report["metrics"]["discard_recognition"]["by_orientation"]["bottom"]["tile_accuracy"] == 1.0
    assert report["metrics"]["discard_recognition"]["coverage_warnings"] == []
    assert report["metrics"]["genbutsu_hint"]["recall"] == 1.0
    assert report["metrics"]["decision_latency_p95_ms"] is not None


def test_evaluate_v05_cli_writes_json_report(tmp_path: Path, capsys) -> None:
    eval_dir = tmp_path / "eval"
    calibration_dir = tmp_path / "calibration"
    report_path = tmp_path / "v05-report.json"
    _write_discard_case(eval_dir, calibration_dir)
    _write_genbutsu_case(eval_dir)

    exit_code = evaluate_v05_main(
        [
            "--eval-dir",
            str(eval_dir),
            "--calibration-dir",
            str(calibration_dir),
            "--strict",
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert json.loads(report_path.read_text(encoding="utf-8"))["metrics"]["genbutsu_hint"]["recall"] == 1.0


def test_evaluate_v05_partial_discard_labels_ignore_unlabeled_predictions(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    calibration_dir = tmp_path / "calibration"
    _write_discard_case(eval_dir, calibration_dir, label_scope="partial", add_unlabeled_discard=True)

    report = evaluate_v05(eval_dir=eval_dir, calibration_dir=calibration_dir, strict=True, include_details=True)

    discard = report["metrics"]["discard_recognition"]
    assert report["ok"] is True
    assert discard["partial_case_count"] == 1
    assert discard["expected_count"] == 1
    assert discard["predicted_count"] == 2
    assert discard["matched_slot_count"] == 1
    assert discard["ignored_unlabeled_prediction_count"] == 1
    assert discard["false_positive_count"] == 0
    assert discard["false_positive_rate"] is None
    assert discard["by_player"]["self"]["ignored_unlabeled_prediction_count"] == 1
    assert discard["by_player"]["self"]["false_positive_count"] == 0
    assert discard["by_orientation"]["bottom"]["ignored_unlabeled_prediction_count"] == 1
    assert [warning["kind"] for warning in discard["coverage_warnings"]] == ["partial_only_labels"]
    assert discard["details"][0]["ignored_unlabeled_prediction_count"] == 1


def test_evaluate_v05_full_discard_labels_count_unlabeled_predictions(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    calibration_dir = tmp_path / "calibration"
    _write_discard_case(eval_dir, calibration_dir, add_unlabeled_discard=True)

    report = evaluate_v05(eval_dir=eval_dir, calibration_dir=calibration_dir, strict=True)

    discard = report["metrics"]["discard_recognition"]
    assert report["ok"] is True
    assert discard["full_case_count"] == 1
    assert discard["predicted_count"] == 2
    assert discard["false_positive_count"] == 1
    assert discard["false_positive_rate"] == round(1 / 71, 6)
    assert discard["by_player"]["self"]["false_positive_count"] == 1
    assert discard["by_player"]["self"]["false_positive_rate"] == round(1 / 17, 6)
    assert discard["by_orientation"]["bottom"]["false_positive_count"] == 1
    assert discard["coverage_warnings"] == []


def _write_discard_case(
    eval_dir: Path,
    calibration_dir: Path,
    *,
    label_scope: str | None = None,
    add_unlabeled_discard: bool = False,
) -> None:
    case_dir = eval_dir / "discard_recognition" / "1280x720" / "basic"
    case_dir.mkdir(parents=True)
    image_path = case_dir / "frame.png"
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
    if add_unlabeled_discard:
        extra_discard_slot = build_discard_layout(*image.size)["self"][1]
        image.paste(
            discard_tile.resize((extra_discard_slot.box.width, extra_discard_slot.box.height)),
            (extra_discard_slot.box.left, extra_discard_slot.box.top),
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
    payload = {
        "image": {"path": "frame.png"},
        "scene": "in_match",
        "discard_piles": {
            "self": [
                {
                    "tile": "1m",
                    "turn_index": 1,
                    "bbox": discard_slot.bbox,
                    "orientation": "bottom",
                }
            ]
        },
    }
    if label_scope:
        payload["label_scope"] = label_scope
    (case_dir / "frame.label.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_genbutsu_case(eval_dir: Path) -> None:
    case_dir = eval_dir / "genbutsu_hint"
    case_dir.mkdir(parents=True)
    (case_dir / "right-riichi.json").write_text(
        json.dumps(
            {
                "state": {
                    "scene": "in_match",
                    "confidence": 0.9,
                    "is_user_turn": True,
                    "hand_tiles": [
                        "1m",
                        "2m",
                        "3m",
                        "4m",
                        "5m",
                        "6m",
                        "7p",
                        "8p",
                        "9p",
                        "2s",
                        "3s",
                        "4s",
                        "5z",
                        "9m",
                    ],
                    "riichi_players": ["right_opponent"],
                    "discard_piles": {"right_opponent": [{"tile": "9m", "turn_index": 1}]},
                },
                "expected_known_genbutsu_tiles": ["9m"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _tile_image(color: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (39, 47), color=(238, 236, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 32, 33), fill=color)
    draw.rectangle((6, 36, 32, 41), fill=(218, 138, 28))
    return image
