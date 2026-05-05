from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from ..smoke_test import run_v1_to_v9_smoke
from ..storage import write_json_atomic
from .evaluate_v03 import evaluate_v03
from .evaluate_v04 import DEFAULT_REQUIRED_BUTTON_TYPES, evaluate_v04
from .evaluate_v05 import evaluate_v05


DEFAULT_EVAL_DIR = Path("plugin/tests/data/mahjong_companion/eval")
DEFAULT_CALIBRATION_DIR = Path("plugin/plugins/mahjong_companion/data/calibration")
DEFAULT_TEMPLATE_DIR = Path("plugin/plugins/mahjong_companion/perception/templates")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = evaluate_v10_release(
        eval_dir=Path(args.eval_dir),
        calibration_dir=Path(args.calibration_dir),
        template_dir=Path(args.template_dir),
        include_smoke=not bool(args.skip_smoke),
        include_details=bool(args.details),
        max_details=int(args.max_details),
    )
    if args.report:
        write_json_atomic(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if report["ok"] else 1


def evaluate_v10_release(
    *,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR,
    template_dir: Path = DEFAULT_TEMPLATE_DIR,
    include_smoke: bool = True,
    include_details: bool = False,
    max_details: int = 20,
) -> dict[str, Any]:
    v03 = evaluate_v03(
        eval_dir=eval_dir,
        calibration_dir=calibration_dir,
        strict_json=True,
        # v1.1 ratchets hand_recognition back up after the v1.0.1 relaxed gate.
        strict_hand=True,
        include_details=include_details,
        max_details=max_details,
    )
    v04 = evaluate_v04(
        eval_dir=eval_dir,
        calibration_dir=calibration_dir,
        template_dir=template_dir,
        strict=True,
        strict_templates=True,
        required_button_types=DEFAULT_REQUIRED_BUTTON_TYPES,
        include_details=include_details,
        max_details=max_details,
    )
    v05 = evaluate_v05(
        eval_dir=eval_dir,
        calibration_dir=calibration_dir,
        strict=True,
        include_details=include_details,
        max_details=max_details,
    )
    smoke = asyncio.run(run_v1_to_v9_smoke()) if include_smoke else {"ok": True, "skipped": True}

    failures = _collect_release_failures(v03=v03, v04=v04, v05=v05, smoke=smoke)
    return {
        "ok": not failures,
        "schema_version": "mahjong-companion-release-gate-v1",
        "eval_dir": str(eval_dir),
        "calibration_dir": str(calibration_dir),
        "template_dir": str(template_dir),
        "include_smoke": include_smoke,
        "summary": {
            "v03": _summarize_v03(v03),
            "v04": _summarize_v04(v04),
            "v05": _summarize_v05(v05),
            "smoke": _summarize_smoke(smoke),
        },
        "reports": {
            "v03": v03,
            "v04": v04,
            "v05": v05,
            "smoke": smoke,
        },
        "failures": failures,
    }


def _collect_release_failures(
    *,
    v03: dict[str, Any],
    v04: dict[str, Any],
    v05: dict[str, Any],
    smoke: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for name, report in [("v03", v03), ("v04", v04), ("v05", v05), ("smoke", smoke)]:
        if report.get("ok") is not True:
            failures.append(f"{name} gate failed")
        for item in report.get("failures", []):
            failures.append(f"{name}: {item}")
    return failures


def _summarize_v03(report: dict[str, Any]) -> dict[str, Any]:
    metrics = _dict(report.get("metrics"))
    decision = _dict(metrics.get("decision_top1"))
    risk = _dict(metrics.get("risk_detection"))
    review = _dict(metrics.get("review_patterns"))
    return {
        "ok": bool(report.get("ok")),
        "decision_top1_match_rate": decision.get("match_rate"),
        "risk_genbutsu_recall": risk.get("genbutsu_recall"),
        "review_structured_summary_pass_rate": review.get("structured_summary_pass_rate"),
        "review_repeated_pattern_match_rate": review.get("repeated_pattern_match_rate"),
        "decision_latency_p95_ms": metrics.get("decision_latency_p95_ms"),
    }


def _summarize_v04(report: dict[str, Any]) -> dict[str, Any]:
    metrics = _dict(report.get("metrics"))
    button = _dict(metrics.get("button_localization"))
    inventory = _dict(metrics.get("template_inventory"))
    assist = _dict(metrics.get("assist_action"))
    audit = _dict(metrics.get("audit_chain"))
    return {
        "ok": bool(report.get("ok")),
        "template_coverage": inventory.get("coverage"),
        "button_iou_pass_rate": button.get("iou_pass_rate"),
        "button_precision": button.get("precision"),
        "button_recall": button.get("recall"),
        "assist_false_abort_rate": assist.get("false_abort_rate"),
        "audit_completeness": audit.get("completeness"),
    }


def _summarize_v05(report: dict[str, Any]) -> dict[str, Any]:
    metrics = _dict(report.get("metrics"))
    discard = _dict(metrics.get("discard_recognition"))
    by_player = _dict(discard.get("by_player"))
    top = _dict(by_player.get("top_opponent"))
    left = _dict(by_player.get("left_opponent"))
    right = _dict(by_player.get("right_opponent"))
    return {
        "ok": bool(report.get("ok")),
        "discard_expected_count": discard.get("expected_count"),
        "discard_tile_accuracy": discard.get("tile_accuracy"),
        "discard_recall": discard.get("recall"),
        "top_opponent_tile_accuracy": top.get("tile_accuracy"),
        "left_opponent_tile_accuracy": left.get("tile_accuracy"),
        "right_opponent_tile_accuracy": right.get("tile_accuracy"),
        "decision_latency_p95_ms": metrics.get("decision_latency_p95_ms"),
    }


def _summarize_smoke(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results")
    return {
        "ok": bool(report.get("ok")),
        "skipped": bool(report.get("skipped", False)),
        "result_count": len(results) if isinstance(results, list) else 0,
        "message_count": report.get("message_count"),
        "status_count": report.get("status_count"),
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mahjong Companion release gate checks.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--calibration-dir", default=str(DEFAULT_CALIBRATION_DIR))
    parser.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--report", default="")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
