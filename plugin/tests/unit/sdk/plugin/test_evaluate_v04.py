from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.scripts.evaluate_v04 import evaluate_v04, main as evaluate_v04_main


def _write_button_case(root: Path) -> None:
    case_dir = root / "button_localization" / "1920x1080" / "kan-skip"
    case_dir.mkdir(parents=True)
    template_dir = Path("plugin/plugins/mahjong_companion/perception/templates/1920x1080")
    kan = Image.open(template_dir / "kan.png").convert("RGB")
    skip = Image.open(template_dir / "skip.png").convert("RGB")
    image = Image.new("RGB", (1920, 1080), (31, 66, 112))
    draw = ImageDraw.Draw(image)
    draw.rectangle((360, 865, 1560, 990), fill=(84, 71, 82))
    draw.rectangle((430, 930, 620, 970), fill=(211, 130, 58))
    image.paste(kan, (860, 760))
    image.paste(skip, (1130, 760))
    image.save(case_dir / "frame.png")
    (case_dir / "frame.label.json").write_text(
        json.dumps(
            {
                "image": {"path": "frame.png"},
                "buttons": [
                    {"button_type": "kan", "bbox": [860, 760, 1155, 890]},
                    {"button_type": "skip", "bbox": [1130, 760, 1505, 890]},
                ],
            },
        ),
        encoding="utf-8",
    )


def _write_audit_case(root: Path) -> None:
    audit_dir = root / "audit_chain"
    audit_dir.mkdir(parents=True)
    (audit_dir / "complete.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "action_id": "dialog_confirm",
                        "executed_at": "2026-05-01T00:00:00+00:00",
                        "ok": True,
                        "locator_source": "button_candidate",
                        "button_region": {
                            "button_type": "confirm",
                            "bbox": [300, 240, 420, 300],
                            "confidence": 0.94,
                        },
                        "target_x": 360,
                        "target_y": 270,
                        "risk_level": "dangerous",
                        "confirmation_chain": [
                            {
                                "step": "user_explicit",
                                "at": "2026-05-01T00:00:00+00:00",
                                "value": True,
                            }
                        ],
                    }
                ]
            },
        ),
        encoding="utf-8",
    )


def test_evaluate_v04_scores_button_guard_and_audit_metrics(tmp_path: Path) -> None:
    _write_button_case(tmp_path)
    _write_audit_case(tmp_path)

    report = evaluate_v04(
        eval_dir=tmp_path,
        template_dir=Path("plugin/plugins/mahjong_companion/perception/templates"),
        strict=True,
        include_details=True,
    )

    assert report["ok"] is True
    assert report["metrics"]["template_inventory"]["available_button_types"] == [
        "chi",
        "kan",
        "pon",
        "riichi",
        "ron",
        "skip",
        "tsumo",
    ]
    assert report["metrics"]["template_inventory"]["missing_button_types"] == []
    assert report["metrics"]["button_localization"]["precision"] == 1.0
    assert report["metrics"]["button_localization"]["recall"] == 1.0
    assert report["metrics"]["button_localization"]["per_button_type"] == [
        {
            "button_type": "kan",
            "expected_count": 1,
            "detected_count": 1,
            "matched_count": 1,
            "iou_pass_count": 1,
            "false_positive_count": 0,
            "iou_pass_rate": 1.0,
            "precision": 1.0,
            "recall": 1.0,
        },
        {
            "button_type": "skip",
            "expected_count": 1,
            "detected_count": 1,
            "matched_count": 1,
            "iou_pass_count": 1,
            "false_positive_count": 0,
            "iou_pass_rate": 1.0,
            "precision": 1.0,
            "recall": 1.0,
        },
    ]
    assert report["metrics"]["assist_action"]["false_abort_rate"] == 0.0
    assert report["metrics"]["audit_chain"]["completeness"] == 1.0
    assert report["metrics"]["button_localization"]["case_details"][0]["matches"][0]["pass"] is True


def test_evaluate_v04_strict_requires_button_cases(tmp_path: Path) -> None:
    _write_audit_case(tmp_path)

    report = evaluate_v04(
        eval_dir=tmp_path,
        template_dir=Path("plugin/plugins/mahjong_companion/perception/templates"),
        strict=True,
    )

    assert report["ok"] is False
    assert any("button_localization requires" in item for item in report["failures"])


def test_evaluate_v04_strict_templates_passes_when_assets_are_complete(tmp_path: Path) -> None:
    _write_button_case(tmp_path)
    _write_audit_case(tmp_path)

    report = evaluate_v04(
        eval_dir=tmp_path,
        template_dir=Path("plugin/plugins/mahjong_companion/perception/templates"),
        strict=True,
        strict_templates=True,
    )

    assert report["ok"] is True
    assert report["metrics"]["template_inventory"]["coverage"] == 1.0
    assert report["failures"] == []


def test_evaluate_v04_main_writes_report(tmp_path: Path) -> None:
    _write_button_case(tmp_path)
    _write_audit_case(tmp_path)
    report_path = tmp_path / "report.json"

    exit_code = evaluate_v04_main(
        [
            "--eval-dir",
            str(tmp_path),
            "--template-dir",
            "plugin/plugins/mahjong_companion/perception/templates",
            "--strict",
            "--report",
            str(report_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
