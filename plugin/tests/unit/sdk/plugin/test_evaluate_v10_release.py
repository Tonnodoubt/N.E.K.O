from __future__ import annotations

import json
from pathlib import Path

from plugin.plugins.mahjong_companion.scripts import evaluate_v10_release as module


def test_evaluate_v10_release_combines_existing_gates(monkeypatch) -> None:
    monkeypatch.setattr(module, "evaluate_v03", lambda **_: _v03_report())
    monkeypatch.setattr(module, "evaluate_v04", lambda **_: _v04_report())
    monkeypatch.setattr(module, "evaluate_v05", lambda **_: _v05_report())

    report = module.evaluate_v10_release(include_smoke=False)

    assert report["ok"] is True
    assert report["summary"]["v03"]["decision_top1_match_rate"] == 0.8
    assert report["summary"]["v04"]["template_coverage"] == 1.0
    assert report["summary"]["v05"]["discard_tile_accuracy"] == 1.0
    assert report["summary"]["smoke"]["skipped"] is True


def test_evaluate_v10_release_reports_subgate_failures(monkeypatch) -> None:
    failed = _v05_report()
    failed["ok"] = False
    failed["failures"] = ["discard_recognition.tile_accuracy 0.5 < 0.7"]
    monkeypatch.setattr(module, "evaluate_v03", lambda **_: _v03_report())
    monkeypatch.setattr(module, "evaluate_v04", lambda **_: _v04_report())
    monkeypatch.setattr(module, "evaluate_v05", lambda **_: failed)

    report = module.evaluate_v10_release(include_smoke=False)

    assert report["ok"] is False
    assert "v05 gate failed" in report["failures"]
    assert "v05: discard_recognition.tile_accuracy 0.5 < 0.7" in report["failures"]


def test_evaluate_v10_release_cli_writes_report(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(module, "evaluate_v03", lambda **_: _v03_report())
    monkeypatch.setattr(module, "evaluate_v04", lambda **_: _v04_report())
    monkeypatch.setattr(module, "evaluate_v05", lambda **_: _v05_report())
    report_path = tmp_path / "release-report.json"

    exit_code = module.main(["--skip-smoke", "--report", str(report_path), "--pretty"])

    assert exit_code == 0
    stdout = json.loads(capsys.readouterr().out)
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout["ok"] is True
    assert written["schema_version"] == "mahjong-companion-release-gate-v1"


def _v03_report() -> dict:
    return {
        "ok": True,
        "metrics": {
            "decision_top1": {"match_rate": 0.8},
            "risk_detection": {"genbutsu_recall": 1.0},
            "review_patterns": {"schema_pass_rate": 1.0},
            "decision_latency_p95_ms": 250.0,
        },
        "failures": [],
    }


def _v04_report() -> dict:
    return {
        "ok": True,
        "metrics": {
            "template_inventory": {"coverage": 1.0},
            "button_localization": {"iou_pass_rate": 1.0, "precision": 1.0, "recall": 1.0},
            "assist_action": {"false_abort_rate": 0.0},
            "audit_chain": {"completeness": 1.0},
        },
        "failures": [],
    }


def _v05_report() -> dict:
    return {
        "ok": True,
        "metrics": {
            "discard_recognition": {
                "expected_count": 71,
                "tile_accuracy": 1.0,
                "recall": 1.0,
                "by_player": {
                    "top_opponent": {"tile_accuracy": 1.0},
                    "left_opponent": {"tile_accuracy": 1.0},
                    "right_opponent": {"tile_accuracy": 1.0},
                },
            },
            "decision_latency_p95_ms": 330.0,
        },
        "failures": [],
    }
