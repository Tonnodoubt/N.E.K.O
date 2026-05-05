from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Sequence

from PIL import Image

from ..contracts import PerceivedGameState
from ..decision.generator import build_decision
from ..perception.calibration import (
    iter_calibration_label_paths,
    save_calibration_profile,
    train_calibration_profile_from_paths,
)
from ..perception.tile_parser import parse_tiles_from_image
from ..review.summarizer import build_review_summary
from ..review.trend_aggregator import build_trend_summary
from ..storage import load_json_payload, write_json_atomic


THRESHOLDS = {
    "hand_recognition.tile_accuracy": 0.92,
    "hand_recognition.full_hand_accuracy": 0.50,
    "decision_top1.match_rate": 0.95,
    "decision_top1.top3_match_rate": 0.80,
    "risk_detection.genbutsu_recall": 0.95,
    "decision_latency_p95_ms": 800.0,
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = evaluate_v03(
        eval_dir=Path(args.eval_dir),
        strict=bool(args.strict),
        strict_json=bool(args.strict_json),
        strict_hand=bool(args.strict_hand),
        allow_fixture_sidecars=bool(args.allow_fixture_sidecars),
        calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
        hand_holdout_dir=Path(args.hand_holdout_dir) if args.hand_holdout_dir else None,
        holdout_folds=int(args.holdout_folds),
        holdout_min_train_samples=int(args.holdout_min_train_samples),
        holdout_client_version=str(args.holdout_client_version),
        include_details=bool(args.details),
        max_details=int(args.max_details),
    )
    if args.report:
        write_json_atomic(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if report["ok"] else 1


def evaluate_v03(
    *,
    eval_dir: Path,
    strict: bool = False,
    strict_json: bool = False,
    strict_hand: bool = False,
    allow_fixture_sidecars: bool = False,
    calibration_dir: Path | None = None,
    hand_holdout_dir: Path | None = None,
    holdout_folds: int = 5,
    holdout_min_train_samples: int = 1,
    holdout_client_version: str = "holdout",
    include_details: bool = False,
    max_details: int = 20,
) -> dict[str, Any]:
    latencies: list[float] = []
    hand = _evaluate_hand_recognition(
        eval_dir / "hand_recognition",
        allow_fixture_sidecars=allow_fixture_sidecars,
        calibration_dir=calibration_dir,
        include_details=include_details,
        max_details=max_details,
        latencies=latencies,
    )
    decision = _evaluate_decision_top1(eval_dir / "decision_top1", latencies=latencies)
    risk = _evaluate_risk_detection(eval_dir / "risk_detection", latencies=latencies)
    review = _evaluate_review_patterns(eval_dir / "review_patterns", latencies=latencies)
    metrics = {
        "hand_recognition": hand,
        "decision_top1": decision,
        "risk_detection": risk,
        "review_patterns": review,
        "decision_latency_p95_ms": _round_or_none(_percentile(latencies, 95)),
    }
    hand_holdout: dict[str, Any] | None = None
    if hand_holdout_dir is not None:
        hand_holdout = _evaluate_hand_holdout(
            hand_holdout_dir,
            folds=holdout_folds,
            min_train_samples=holdout_min_train_samples,
            client_version=holdout_client_version,
            include_details=include_details,
            max_details=max_details,
            latencies=latencies,
        )
        metrics["hand_holdout"] = hand_holdout
        metrics["decision_latency_p95_ms"] = _round_or_none(_percentile(latencies, 95))

    failures = _collect_failures(
        metrics,
        strict=strict,
        strict_json=strict_json,
        strict_hand=strict_hand,
        eval_dir=eval_dir,
    )
    if (strict or strict_json or strict_hand) and allow_fixture_sidecars:
        failures.append("strict modes cannot be combined with --allow-fixture-sidecars")
    failures.extend(hand.pop("_errors", []))
    if hand_holdout is not None:
        failures.extend(hand_holdout.pop("_errors", []))
    failures.extend(decision.pop("_errors", []))
    failures.extend(risk.pop("_errors", []))
    failures.extend(review.pop("_errors", []))
    return {
        "ok": not failures,
        "strict": strict,
        "strict_json": strict_json,
        "strict_hand": strict_hand,
        "eval_dir": str(eval_dir),
        "calibration_dir": str(calibration_dir) if calibration_dir is not None else "",
        "hand_holdout_dir": str(hand_holdout_dir) if hand_holdout_dir is not None else "",
        "allow_fixture_sidecars": allow_fixture_sidecars,
        "thresholds": dict(THRESHOLDS),
        "metrics": metrics,
        "failures": failures,
    }


def _evaluate_hand_recognition(
    root: Path,
    *,
    allow_fixture_sidecars: bool,
    calibration_dir: Path | None,
    include_details: bool,
    max_details: int,
    latencies: list[float],
) -> dict[str, Any]:
    errors: list[str] = []
    cases = _load_hand_cases(root, errors=errors)
    total_tiles = 0
    correct_tiles = 0
    full_hand_correct = 0
    red5_normalized_correct_tiles = 0
    red5_normalized_full_hand_correct = 0
    red5_normalized_mismatch_case_count = 0
    red5_only_mismatch_count = 0
    usable_cases = 0
    mismatch_case_count = 0
    missing_prediction_tiles = 0
    extra_prediction_tiles = 0
    confusion_counts: dict[tuple[str, str], int] = {}
    case_details: list[dict[str, Any]] = []

    for case in cases:
        image_path = case.get("image_path")
        expected = _normalize_tiles(case.get("expected_hand_tiles"))
        if not isinstance(image_path, Path) or not image_path.exists() or not expected:
            errors.append(f"invalid hand recognition case: {case.get('case_id', '<unknown>')}")
            continue

        try:
            started = perf_counter()
            with Image.open(image_path) as opened:
                parsed = parse_tiles_from_image(
                    image_path,
                    opened.convert("RGB"),
                    scene=str(case.get("scene") or "in_match"),
                    metrics=_metrics_from_case(case),
                    calibration_dir=_optional_path(case.get("calibration_dir")) or calibration_dir,
                    fixture_mode="auto" if allow_fixture_sidecars else "disabled",
                )
            latencies.append((perf_counter() - started) * 1000.0)
        except Exception as exc:
            errors.append(f"{case.get('case_id', image_path)}: {exc}")
            continue

        predicted = _normalize_tiles(parsed.hand_tiles)
        mismatches = _tile_mismatches(expected, predicted)
        correct_for_case = _count_correct_tiles(expected, predicted)
        red5_expected = _normalize_red_five_tiles(expected)
        red5_predicted = _normalize_red_five_tiles(predicted)
        red5_mismatches = _tile_mismatches(red5_expected, red5_predicted)
        red5_correct_for_case = _count_correct_tiles(red5_expected, red5_predicted)
        red5_full_correct = red5_predicted == red5_expected
        usable_cases += 1
        total_tiles += len(expected)
        correct_tiles += correct_for_case
        red5_normalized_correct_tiles += red5_correct_for_case
        red5_normalized_full_hand_correct += 1 if red5_full_correct else 0
        if mismatches:
            mismatch_case_count += 1
        if red5_mismatches:
            red5_normalized_mismatch_case_count += 1
        red5_only_mismatch_count += _red5_only_mismatch_count(mismatches)
        missing_prediction_tiles += max(0, len(expected) - len(predicted))
        extra_prediction_tiles += max(0, len(predicted) - len(expected))
        for item in mismatches:
            expected_tile = str(item.get("expected", ""))
            predicted_tile = str(item.get("predicted", ""))
            confusion_counts[(expected_tile, predicted_tile)] = confusion_counts.get((expected_tile, predicted_tile), 0) + 1
        if predicted == expected:
            full_hand_correct += 1
        if include_details and len(case_details) < max(0, int(max_details)):
            case_details.append(
                {
                    "case_id": str(case.get("case_id", image_path)),
                    "image_path": str(image_path),
                    "expected_hand_tiles": expected,
                    "predicted_hand_tiles": predicted,
                    "correct_tiles": correct_for_case,
                    "total_tiles": len(expected),
                    "tile_accuracy": _ratio(correct_for_case, len(expected)),
                    "full_hand_correct": predicted == expected,
                    "red5_normalized_correct_tiles": red5_correct_for_case,
                    "red5_normalized_tile_accuracy": _ratio(red5_correct_for_case, len(expected)),
                    "red5_normalized_full_hand_correct": red5_full_correct,
                    "red5_only_mismatch_count": _red5_only_mismatch_count(mismatches),
                    "mismatches": mismatches,
                    "red5_normalized_mismatches": red5_mismatches,
                    "tile_parser_source": parsed.analysis_hints.get("tile_parser_source", ""),
                    "analysis_confidence": parsed.analysis_hints.get("analysis_confidence"),
                    "calibration_profile": parsed.analysis_hints.get("calibration_profile", ""),
                }
            )

    result = {
        "case_count": usable_cases,
        "tile_accuracy": _ratio(correct_tiles, total_tiles),
        "full_hand_accuracy": _ratio(full_hand_correct, usable_cases),
        "correct_tiles": correct_tiles,
        "total_tiles": total_tiles,
        "red5_normalized_tile_accuracy": _ratio(red5_normalized_correct_tiles, total_tiles),
        "red5_normalized_full_hand_accuracy": _ratio(red5_normalized_full_hand_correct, usable_cases),
        "red5_normalized_correct_tiles": red5_normalized_correct_tiles,
        "red5_normalized_total_tiles": total_tiles,
        "red5_normalized_mismatch_case_count": red5_normalized_mismatch_case_count,
        "red5_only_mismatch_count": red5_only_mismatch_count,
        "mismatch_case_count": mismatch_case_count,
        "missing_prediction_tiles": missing_prediction_tiles,
        "extra_prediction_tiles": extra_prediction_tiles,
        "confusion_top": _format_confusion_counts(confusion_counts, limit=10),
        "_errors": errors,
    }
    if include_details:
        result["case_details"] = case_details
    return result


def _evaluate_hand_holdout(
    label_root: Path,
    *,
    folds: int,
    min_train_samples: int,
    client_version: str,
    include_details: bool,
    max_details: int,
    latencies: list[float],
) -> dict[str, Any]:
    errors: list[str] = []
    label_paths = iter_calibration_label_paths(label_root)
    if len(label_paths) < 2:
        return _empty_hand_holdout_result(label_root=label_root, errors=[f"hand holdout requires at least 2 labels under {label_root}"])

    folds = max(2, min(int(folds or 2), len(label_paths)))
    fold_groups = [label_paths[index::folds] for index in range(folds)]
    fold_groups = [group for group in fold_groups if group]

    total_tiles = 0
    correct_tiles = 0
    full_hand_correct = 0
    red5_normalized_correct_tiles = 0
    red5_normalized_full_hand_correct = 0
    red5_normalized_mismatch_case_count = 0
    red5_only_mismatch_count = 0
    usable_cases = 0
    mismatch_case_count = 0
    missing_prediction_tiles = 0
    extra_prediction_tiles = 0
    coverage_blocked_tiles = 0
    coverage_adjusted_total_tiles = 0
    coverage_adjusted_correct_tiles = 0
    coverage_adjusted_full_hand_correct = 0
    red5_normalized_coverage_blocked_tiles = 0
    red5_normalized_coverage_adjusted_total_tiles = 0
    red5_normalized_coverage_adjusted_correct_tiles = 0
    red5_normalized_coverage_adjusted_full_hand_correct = 0
    confusion_counts: dict[tuple[str, str], int] = {}
    fold_reports: list[dict[str, Any]] = []
    case_details: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="mahjong-v03-holdout-") as temp_dir:
        temp_root = Path(temp_dir)
        for fold_index, test_paths in enumerate(fold_groups, start=1):
            test_set = set(test_paths)
            train_paths = [path for path in label_paths if path not in test_set]
            if not train_paths:
                errors.append(f"fold {fold_index}: no train labels")
                continue
            train_tile_counts = _tile_counts_for_labels(train_paths)
            train_red5_tile_counts = _normalize_red_five_tile_counts(train_tile_counts)
            fold_calibration_dir = temp_root / f"fold-{fold_index}"
            profile = train_calibration_profile_from_paths(
                train_paths,
                label_root=label_root,
                client_version=client_version,
                min_samples=min_train_samples,
                profile_id=f"holdout-fold-{fold_index}-{_label_resolution_slug(train_paths)}",
            )
            save_calibration_profile(profile, fold_calibration_dir / "profiles" / f"{profile.profile_id}.json")

            fold_total = 0
            fold_correct = 0
            fold_full_correct = 0
            fold_coverage_blocked = 0
            fold_adjusted_total = 0
            fold_adjusted_correct = 0
            fold_adjusted_full_correct = 0
            fold_red5_correct = 0
            fold_red5_full_correct = 0
            fold_red5_coverage_blocked = 0
            fold_red5_adjusted_total = 0
            fold_red5_adjusted_correct = 0
            fold_red5_adjusted_full_correct = 0
            fold_red5_mismatch_cases = 0
            fold_red5_only_mismatch_count = 0
            fold_cases = 0
            for label_path in test_paths:
                case = _hand_case_from_label_path(label_path, root=label_root)
                image_path = case.get("image_path")
                expected = _normalize_tiles(case.get("expected_hand_tiles"))
                if not isinstance(image_path, Path) or not image_path.exists() or not expected:
                    errors.append(f"fold {fold_index}: invalid hand holdout label: {label_path}")
                    continue

                try:
                    started = perf_counter()
                    with Image.open(image_path) as opened:
                        parsed = parse_tiles_from_image(
                            image_path,
                            opened.convert("RGB"),
                            scene=str(case.get("scene") or "in_match"),
                            metrics=_metrics_from_case(case),
                            calibration_dir=fold_calibration_dir,
                            fixture_mode="disabled",
                        )
                    latencies.append((perf_counter() - started) * 1000.0)
                except Exception as exc:
                    errors.append(f"fold {fold_index}: {label_path}: {exc}")
                    continue

                predicted = _normalize_tiles(parsed.hand_tiles)
                mismatches = _tile_mismatches(expected, predicted)
                coverage_blocked = _coverage_blocked_mismatches(mismatches, train_tile_counts)
                correct_for_case = _count_correct_tiles(expected, predicted)
                red5_expected = _normalize_red_five_tiles(expected)
                red5_predicted = _normalize_red_five_tiles(predicted)
                red5_mismatches = _tile_mismatches(red5_expected, red5_predicted)
                red5_coverage_blocked = _coverage_blocked_mismatches(red5_mismatches, train_red5_tile_counts)
                red5_correct_for_case = _count_correct_tiles(red5_expected, red5_predicted)
                adjusted_total_for_case = len(expected) - len(coverage_blocked)
                adjusted_correct_for_case = correct_for_case
                adjusted_full_correct = len(mismatches) == len(coverage_blocked)
                red5_adjusted_total_for_case = len(expected) - len(red5_coverage_blocked)
                red5_adjusted_correct_for_case = red5_correct_for_case
                red5_adjusted_full_correct = len(red5_mismatches) == len(red5_coverage_blocked)
                full_correct = predicted == expected
                red5_full_correct = red5_predicted == red5_expected
                red5_only_for_case = _red5_only_mismatch_count(mismatches)
                fold_cases += 1
                fold_total += len(expected)
                fold_correct += correct_for_case
                fold_full_correct += 1 if full_correct else 0
                fold_coverage_blocked += len(coverage_blocked)
                fold_adjusted_total += adjusted_total_for_case
                fold_adjusted_correct += adjusted_correct_for_case
                fold_adjusted_full_correct += 1 if adjusted_full_correct else 0
                fold_red5_correct += red5_correct_for_case
                fold_red5_full_correct += 1 if red5_full_correct else 0
                fold_red5_coverage_blocked += len(red5_coverage_blocked)
                fold_red5_adjusted_total += red5_adjusted_total_for_case
                fold_red5_adjusted_correct += red5_adjusted_correct_for_case
                fold_red5_adjusted_full_correct += 1 if red5_adjusted_full_correct else 0
                fold_red5_mismatch_cases += 1 if red5_mismatches else 0
                fold_red5_only_mismatch_count += red5_only_for_case
                usable_cases += 1
                total_tiles += len(expected)
                correct_tiles += correct_for_case
                full_hand_correct += 1 if full_correct else 0
                red5_normalized_correct_tiles += red5_correct_for_case
                red5_normalized_full_hand_correct += 1 if red5_full_correct else 0
                coverage_blocked_tiles += len(coverage_blocked)
                coverage_adjusted_total_tiles += adjusted_total_for_case
                coverage_adjusted_correct_tiles += adjusted_correct_for_case
                coverage_adjusted_full_hand_correct += 1 if adjusted_full_correct else 0
                red5_normalized_coverage_blocked_tiles += len(red5_coverage_blocked)
                red5_normalized_coverage_adjusted_total_tiles += red5_adjusted_total_for_case
                red5_normalized_coverage_adjusted_correct_tiles += red5_adjusted_correct_for_case
                red5_normalized_coverage_adjusted_full_hand_correct += 1 if red5_adjusted_full_correct else 0
                if mismatches:
                    mismatch_case_count += 1
                if red5_mismatches:
                    red5_normalized_mismatch_case_count += 1
                red5_only_mismatch_count += red5_only_for_case
                missing_prediction_tiles += max(0, len(expected) - len(predicted))
                extra_prediction_tiles += max(0, len(predicted) - len(expected))
                for item in mismatches:
                    expected_tile = str(item.get("expected", ""))
                    predicted_tile = str(item.get("predicted", ""))
                    confusion_counts[(expected_tile, predicted_tile)] = (
                        confusion_counts.get((expected_tile, predicted_tile), 0) + 1
                    )
                if include_details and len(case_details) < max(0, int(max_details)):
                    case_details.append(
                        {
                            "fold": fold_index,
                            "case_id": str(case.get("case_id", label_path)),
                            "image_path": str(image_path),
                            "expected_hand_tiles": expected,
                            "predicted_hand_tiles": predicted,
                            "correct_tiles": correct_for_case,
                            "total_tiles": len(expected),
                            "tile_accuracy": _ratio(correct_for_case, len(expected)),
                            "full_hand_correct": full_correct,
                            "red5_normalized_correct_tiles": red5_correct_for_case,
                            "red5_normalized_tile_accuracy": _ratio(red5_correct_for_case, len(expected)),
                            "red5_normalized_full_hand_correct": red5_full_correct,
                            "red5_only_mismatch_count": red5_only_for_case,
                            "mismatches": mismatches,
                            "red5_normalized_mismatches": red5_mismatches,
                            "coverage_blocked_mismatches": coverage_blocked,
                            "coverage_adjusted_tile_accuracy": _ratio(adjusted_correct_for_case, adjusted_total_for_case),
                            "coverage_adjusted_full_hand_correct": adjusted_full_correct,
                            "red5_normalized_coverage_blocked_mismatches": red5_coverage_blocked,
                            "red5_normalized_coverage_adjusted_tile_accuracy": _ratio(
                                red5_adjusted_correct_for_case,
                                red5_adjusted_total_for_case,
                            ),
                            "red5_normalized_coverage_adjusted_full_hand_correct": red5_adjusted_full_correct,
                            "tile_parser_source": parsed.analysis_hints.get("tile_parser_source", ""),
                            "analysis_confidence": parsed.analysis_hints.get("analysis_confidence"),
                            "calibration_profile": parsed.analysis_hints.get("calibration_profile", ""),
                        }
                    )

            fold_reports.append(
                {
                    "fold": fold_index,
                    "train_case_count": len(train_paths),
                    "test_case_count": fold_cases,
                    "tile_accuracy": _ratio(fold_correct, fold_total),
                    "full_hand_accuracy": _ratio(fold_full_correct, fold_cases),
                    "coverage_adjusted_tile_accuracy": _ratio(fold_adjusted_correct, fold_adjusted_total),
                    "coverage_adjusted_full_hand_accuracy": _ratio(fold_adjusted_full_correct, fold_cases),
                    "red5_normalized_tile_accuracy": _ratio(fold_red5_correct, fold_total),
                    "red5_normalized_full_hand_accuracy": _ratio(fold_red5_full_correct, fold_cases),
                    "red5_normalized_coverage_adjusted_tile_accuracy": _ratio(
                        fold_red5_adjusted_correct,
                        fold_red5_adjusted_total,
                    ),
                    "red5_normalized_coverage_adjusted_full_hand_accuracy": _ratio(
                        fold_red5_adjusted_full_correct,
                        fold_cases,
                    ),
                    "correct_tiles": fold_correct,
                    "total_tiles": fold_total,
                    "coverage_blocked_tiles": fold_coverage_blocked,
                    "coverage_adjusted_correct_tiles": fold_adjusted_correct,
                    "coverage_adjusted_total_tiles": fold_adjusted_total,
                    "red5_normalized_correct_tiles": fold_red5_correct,
                    "red5_normalized_total_tiles": fold_total,
                    "red5_normalized_coverage_blocked_tiles": fold_red5_coverage_blocked,
                    "red5_normalized_coverage_adjusted_correct_tiles": fold_red5_adjusted_correct,
                    "red5_normalized_coverage_adjusted_total_tiles": fold_red5_adjusted_total,
                    "red5_normalized_mismatch_case_count": fold_red5_mismatch_cases,
                    "red5_only_mismatch_count": fold_red5_only_mismatch_count,
                    "missing_train_tiles": _missing_train_tiles_for_labels(test_paths, train_tile_counts),
                    "train_tile_class_count": len(train_tile_counts),
                    "profile_enabled": profile.enabled,
                    "profile_confidence": profile.confidence,
                }
            )

    result = {
        "label_root": str(label_root),
        "fold_count": len(fold_groups),
        "case_count": usable_cases,
        "tile_accuracy": _ratio(correct_tiles, total_tiles),
        "full_hand_accuracy": _ratio(full_hand_correct, usable_cases),
        "coverage_adjusted_tile_accuracy": _ratio(coverage_adjusted_correct_tiles, coverage_adjusted_total_tiles),
        "coverage_adjusted_full_hand_accuracy": _ratio(coverage_adjusted_full_hand_correct, usable_cases),
        "red5_normalized_tile_accuracy": _ratio(red5_normalized_correct_tiles, total_tiles),
        "red5_normalized_full_hand_accuracy": _ratio(red5_normalized_full_hand_correct, usable_cases),
        "red5_normalized_coverage_adjusted_tile_accuracy": _ratio(
            red5_normalized_coverage_adjusted_correct_tiles,
            red5_normalized_coverage_adjusted_total_tiles,
        ),
        "red5_normalized_coverage_adjusted_full_hand_accuracy": _ratio(
            red5_normalized_coverage_adjusted_full_hand_correct,
            usable_cases,
        ),
        "correct_tiles": correct_tiles,
        "total_tiles": total_tiles,
        "coverage_blocked_tiles": coverage_blocked_tiles,
        "coverage_adjusted_correct_tiles": coverage_adjusted_correct_tiles,
        "coverage_adjusted_total_tiles": coverage_adjusted_total_tiles,
        "red5_normalized_correct_tiles": red5_normalized_correct_tiles,
        "red5_normalized_total_tiles": total_tiles,
        "red5_normalized_coverage_blocked_tiles": red5_normalized_coverage_blocked_tiles,
        "red5_normalized_coverage_adjusted_correct_tiles": red5_normalized_coverage_adjusted_correct_tiles,
        "red5_normalized_coverage_adjusted_total_tiles": red5_normalized_coverage_adjusted_total_tiles,
        "red5_normalized_mismatch_case_count": red5_normalized_mismatch_case_count,
        "red5_only_mismatch_count": red5_only_mismatch_count,
        "mismatch_case_count": mismatch_case_count,
        "missing_prediction_tiles": missing_prediction_tiles,
        "extra_prediction_tiles": extra_prediction_tiles,
        "confusion_top": _format_confusion_counts(confusion_counts, limit=10),
        "folds": fold_reports,
        "_errors": errors,
    }
    if include_details:
        result["case_details"] = case_details
    return result


def _empty_hand_holdout_result(*, label_root: Path, errors: list[str]) -> dict[str, Any]:
    return {
        "label_root": str(label_root),
        "fold_count": 0,
        "case_count": 0,
        "tile_accuracy": None,
        "full_hand_accuracy": None,
        "coverage_adjusted_tile_accuracy": None,
        "coverage_adjusted_full_hand_accuracy": None,
        "red5_normalized_tile_accuracy": None,
        "red5_normalized_full_hand_accuracy": None,
        "red5_normalized_coverage_adjusted_tile_accuracy": None,
        "red5_normalized_coverage_adjusted_full_hand_accuracy": None,
        "correct_tiles": 0,
        "total_tiles": 0,
        "coverage_blocked_tiles": 0,
        "coverage_adjusted_correct_tiles": 0,
        "coverage_adjusted_total_tiles": 0,
        "red5_normalized_correct_tiles": 0,
        "red5_normalized_total_tiles": 0,
        "red5_normalized_coverage_blocked_tiles": 0,
        "red5_normalized_coverage_adjusted_correct_tiles": 0,
        "red5_normalized_coverage_adjusted_total_tiles": 0,
        "red5_normalized_mismatch_case_count": 0,
        "red5_only_mismatch_count": 0,
        "mismatch_case_count": 0,
        "missing_prediction_tiles": 0,
        "extra_prediction_tiles": 0,
        "confusion_top": [],
        "folds": [],
        "_errors": errors,
    }


def _evaluate_decision_top1(root: Path, *, latencies: list[float]) -> dict[str, Any]:
    errors: list[str] = []
    cases = _load_json_cases(root, errors=errors)
    match_count = 0
    top3_match_count = 0
    usable_cases = 0

    for case_path, payload in cases:
        expected_top1 = str(payload.get("expected_top1", "")).strip()
        if not expected_top1:
            errors.append(f"{case_path}: missing expected_top1")
            continue
        state = _state_from_payload(payload)
        started = perf_counter()
        decision = build_decision(state)
        latencies.append((perf_counter() - started) * 1000.0)
        candidates = decision.mahjong_analysis.get("candidate_discards")
        if not isinstance(candidates, list):
            candidates = []
        top_tiles = [str(item.get("tile", "")).strip() for item in candidates if isinstance(item, dict)]
        top_tiles = [tile for tile in top_tiles if tile]
        usable_cases += 1
        if top_tiles[:1] == [expected_top1]:
            match_count += 1
        if expected_top1 in top_tiles[:3]:
            top3_match_count += 1

    return {
        "case_count": usable_cases,
        "match_rate": _ratio(match_count, usable_cases),
        "top3_match_rate": _ratio(top3_match_count, usable_cases),
        "matches": match_count,
        "top3_matches": top3_match_count,
        "_errors": errors,
    }


def _evaluate_risk_detection(root: Path, *, latencies: list[float]) -> dict[str, Any]:
    errors: list[str] = []
    cases = _load_json_cases(root, errors=errors)
    expected_genbutsu_cases = 0
    detected_genbutsu_cases = 0
    usable_cases = 0

    for case_path, payload in cases:
        expected_tiles = _normalize_tiles(payload.get("expected_genbutsu_tiles") or payload.get("genbutsu_tiles"))
        expects_genbutsu = bool(payload.get("expected_genbutsu", expected_tiles))
        state = _state_from_payload(payload)
        started = perf_counter()
        decision = build_decision(state)
        latencies.append((perf_counter() - started) * 1000.0)
        alerts = decision.mahjong_analysis.get("defense_alerts")
        alert_text = " ".join(str(item) for item in alerts if str(item).strip()) if isinstance(alerts, list) else ""

        usable_cases += 1
        if not expects_genbutsu:
            continue
        expected_genbutsu_cases += 1
        if "现物" in alert_text or any(tile in alert_text for tile in expected_tiles):
            detected_genbutsu_cases += 1
        elif not expected_tiles:
            errors.append(f"{case_path}: expected_genbutsu requires expected_genbutsu_tiles")

    return {
        "case_count": usable_cases,
        "expected_genbutsu_cases": expected_genbutsu_cases,
        "detected_genbutsu_cases": detected_genbutsu_cases,
        "genbutsu_recall": _ratio(detected_genbutsu_cases, expected_genbutsu_cases),
        "_errors": errors,
    }


def _evaluate_review_patterns(root: Path, *, latencies: list[float]) -> dict[str, Any]:
    errors: list[str] = []
    cases = _load_json_cases(root, errors=errors)
    structured_summary_passes = 0
    repeated_pattern_matches = 0
    expected_pattern_cases = 0
    usable_cases = 0

    for case_path, payload in cases:
        usable_cases += 1
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            started = perf_counter()
            summary = build_review_summary(
                session_id=str(payload.get("session_id", case_path.stem)),
                candidates=[item for item in candidates if isinstance(item, dict)],
            )
            latencies.append((perf_counter() - started) * 1000.0)
            if _summary_has_structured_fields(summary):
                structured_summary_passes += 1
            else:
                errors.append(f"{case_path}: structured review summary fields are incomplete")

        review_summaries = payload.get("review_summaries")
        expected_patterns = _string_list(payload.get("expected_repeated_patterns"))
        if isinstance(review_summaries, list) and expected_patterns:
            expected_pattern_cases += 1
            started = perf_counter()
            trend = build_trend_summary(
                review_summaries=[item for item in review_summaries if isinstance(item, dict)],
                pending_memories=[],
                session_window=int(payload.get("session_window", 3) or 3),
            )
            latencies.append((perf_counter() - started) * 1000.0)
            actual_patterns = {
                str(item.get("pattern_id", "")).strip()
                for item in trend.get("repeated_mistake_patterns", [])
                if isinstance(item, dict)
            }
            if set(expected_patterns).issubset(actual_patterns):
                repeated_pattern_matches += 1

    return {
        "case_count": usable_cases,
        "structured_summary_pass_rate": _ratio(structured_summary_passes, usable_cases),
        "repeated_pattern_match_rate": _ratio(repeated_pattern_matches, expected_pattern_cases),
        "structured_summary_passes": structured_summary_passes,
        "expected_pattern_cases": expected_pattern_cases,
        "repeated_pattern_matches": repeated_pattern_matches,
        "_errors": errors,
    }


def _load_hand_cases(root: Path, *, errors: list[str]) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    cases: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        payload = load_json_payload(path, default={}, expected_type=dict)
        if not payload:
            continue
        image_path = _resolve_image_path(path, payload, root)
        expected = payload.get("expected_hand_tiles") or payload.get("hand_tiles")
        cases.append(
            {
                "case_id": payload.get("case_id") or str(path.relative_to(root)),
                "image_path": image_path,
                "expected_hand_tiles": expected,
                "scene": payload.get("scene", "in_match"),
                "metrics": payload.get("metrics"),
                "calibration_dir": _resolve_optional_case_path(path, payload.get("calibration_dir"), root),
            }
        )
    return cases


def _hand_case_from_label_path(label_path: Path, *, root: Path) -> dict[str, Any]:
    payload = load_json_payload(label_path, default={}, expected_type=dict)
    image_path = _resolve_image_path(label_path, payload, root)
    expected = payload.get("expected_hand_tiles") or payload.get("hand_tiles")
    return {
        "case_id": payload.get("case_id") or str(_relative_case_id(label_path, root)),
        "image_path": image_path,
        "expected_hand_tiles": expected,
        "scene": payload.get("scene", "in_match"),
        "metrics": payload.get("metrics"),
        "calibration_dir": _resolve_optional_case_path(label_path, payload.get("calibration_dir"), root),
    }


def _load_json_cases(root: Path, *, errors: list[str]) -> list[tuple[Path, dict[str, Any]]]:
    if not root.exists():
        return []
    cases: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*.json")):
        payload = load_json_payload(path, default={}, expected_type=dict)
        if payload:
            cases.append((path, payload))
        else:
            errors.append(f"{path}: empty or invalid JSON case")
    return cases


def _state_from_payload(payload: dict[str, Any]) -> PerceivedGameState:
    state_payload = payload.get("state")
    if not isinstance(state_payload, dict):
        state_payload = payload
    return PerceivedGameState(
        scene=str(state_payload.get("scene", "in_match") or "in_match"),
        confidence=float(state_payload.get("confidence", 0.9) or 0.9),
        is_user_turn=bool(state_payload.get("is_user_turn", True)),
        buttons=_string_list(state_payload.get("buttons")),
        notes=_string_list(state_payload.get("notes")),
        roi_hits=state_payload.get("roi_hits") if isinstance(state_payload.get("roi_hits"), dict) else {},
        hand_tiles=_normalize_tiles(state_payload.get("hand_tiles")),
        melds=_tile_groups(state_payload.get("melds")),
        dora_indicators=_normalize_tiles(state_payload.get("dora_indicators")),
        riichi_players=_string_list(state_payload.get("riichi_players")),
        raw_detections=state_payload.get("raw_detections") if isinstance(state_payload.get("raw_detections"), list) else [],
        analysis_hints=state_payload.get("analysis_hints") if isinstance(state_payload.get("analysis_hints"), dict) else {},
    )


def _collect_failures(
    metrics: dict[str, Any],
    *,
    strict: bool,
    strict_json: bool,
    strict_hand: bool,
    eval_dir: Path,
) -> list[str]:
    failures: list[str] = []
    if not (strict or strict_json or strict_hand):
        return failures
    hand_cases = int(metrics.get("hand_recognition", {}).get("case_count", 0) or 0)
    holdout_cases = int(metrics.get("hand_holdout", {}).get("case_count", 0) or 0)
    requires_eval_dir = strict or strict_json or (strict_hand and hand_cases > 0) or (strict_hand and holdout_cases <= 0)
    if requires_eval_dir and not eval_dir.exists():
        failures.append(f"eval dir does not exist: {eval_dir}")

    if strict or strict_json:
        _require_at_least_one(metrics["decision_top1"], "decision_top1", failures)
        _require_at_least_one(metrics["risk_detection"], "risk_detection", failures)
        _require_at_least_one(metrics["review_patterns"], "review_patterns", failures)
        _check_threshold(metrics, "decision_top1.match_rate", failures)
        _check_threshold(metrics, "decision_top1.top3_match_rate", failures)
        _check_threshold(metrics, "risk_detection.genbutsu_recall", failures)

    if strict:
        if hand_cases <= 0 and holdout_cases <= 0:
            failures.append("hand_recognition or hand_holdout has no eval cases")
        if hand_cases > 0:
            _check_threshold(metrics, "hand_recognition.tile_accuracy", failures)
            _check_threshold(metrics, "hand_recognition.full_hand_accuracy", failures)
        if holdout_cases > 0:
            _check_threshold(metrics, "hand_holdout.tile_accuracy", failures, threshold=THRESHOLDS["hand_recognition.tile_accuracy"])
            _check_threshold(
                metrics,
                "hand_holdout.full_hand_accuracy",
                failures,
                threshold=THRESHOLDS["hand_recognition.full_hand_accuracy"],
            )

    if strict_hand:
        if hand_cases <= 0 and holdout_cases <= 0:
            failures.append("hand_recognition or hand_holdout has no eval cases")
        if hand_cases > 0:
            _check_threshold(metrics, "hand_recognition.tile_accuracy", failures)
            _check_threshold(metrics, "hand_recognition.full_hand_accuracy", failures)
        if holdout_cases > 0:
            _check_threshold(metrics, "hand_holdout.tile_accuracy", failures, threshold=THRESHOLDS["hand_recognition.tile_accuracy"])
            _check_threshold(
                metrics,
                "hand_holdout.full_hand_accuracy",
                failures,
                threshold=THRESHOLDS["hand_recognition.full_hand_accuracy"],
            )

    _check_threshold(metrics, "decision_latency_p95_ms", failures)
    review_patterns = metrics.get("review_patterns")
    if isinstance(review_patterns, dict) and int(review_patterns.get("case_count", 0) or 0) > 0:
        summary_rate = review_patterns.get("structured_summary_pass_rate")
        if isinstance(summary_rate, int | float) and float(summary_rate) < 1.0:
            failures.append(f"review_patterns.structured_summary_pass_rate={summary_rate} below 1.0")
        pattern_rate = review_patterns.get("repeated_pattern_match_rate")
        if isinstance(pattern_rate, int | float) and float(pattern_rate) < 1.0:
            failures.append(f"review_patterns.repeated_pattern_match_rate={pattern_rate} below 1.0")
    return failures


def _check_threshold(
    metrics: dict[str, Any],
    key: str,
    failures: list[str],
    *,
    threshold: float | None = None,
) -> None:
    threshold = THRESHOLDS.get(key) if threshold is None else threshold
    if threshold is None:
        return
    value = _metric_value(metrics, key)
    if value is None:
        failures.append(f"{key} is missing")
        return
    if key == "decision_latency_p95_ms":
        if value > threshold:
            failures.append(f"{key}={value} exceeds {threshold}")
    elif value < threshold:
        failures.append(f"{key}={value} below {threshold}")


def _require_at_least_one(metric: dict[str, Any], name: str, failures: list[str]) -> None:
    if int(metric.get("case_count", 0) or 0) <= 0:
        failures.append(f"{name} has no eval cases")


def _metric_value(metrics: dict[str, Any], dotted: str) -> float | None:
    parts = dotted.split(".")
    current: Any = metrics
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    if isinstance(current, int | float) and not isinstance(current, bool):
        if math.isnan(float(current)):
            return None
        return float(current)
    return None


def _resolve_image_path(case_path: Path, payload: dict[str, Any], root: Path) -> Path | None:
    explicit = payload.get("image_path")
    if isinstance(explicit, str) and explicit.strip():
        return _resolve_case_path(case_path, explicit, root)
    image = payload.get("image")
    if isinstance(image, dict):
        image_path = image.get("path")
        if isinstance(image_path, str) and image_path.strip():
            return _resolve_case_path(case_path, image_path, root)
    for suffix in IMAGE_EXTENSIONS:
        candidate = case_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    if case_path.name.endswith(".label.json"):
        stem = case_path.name.removesuffix(".label.json")
        for suffix in IMAGE_EXTENSIONS:
            candidate = case_path.with_name(f"{stem}{suffix}")
            if candidate.exists():
                return candidate
    return None


def _resolve_optional_case_path(case_path: Path, value: Any, root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _resolve_case_path(case_path, value, root)


def _resolve_case_path(case_path: Path, value: str, root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in [case_path.parent, root]:
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return case_path.parent / candidate


def _relative_case_id(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _optional_path(value: Any) -> Path | None:
    return value if isinstance(value, Path) else None


def _metrics_from_case(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = case.get("metrics")
    if isinstance(metrics, dict):
        return {str(key): value for key, value in metrics.items() if isinstance(value, dict)}
    return {"bottom_hand_area": {"colorful_ratio": 0.58}}


def _summary_has_structured_fields(summary: dict[str, Any]) -> bool:
    for field in ["facts", "risks", "suggestions", "training_points"]:
        value = summary.get(field)
        if not isinstance(value, list) or not value:
            return False
    return True


def _normalize_tiles(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _string_list(value: Any) -> list[str]:
    return _normalize_tiles(value)


def _tile_groups(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    groups: list[list[str]] = []
    for item in value:
        group = _normalize_tiles(item)
        if group:
            groups.append(group)
    return groups


def _tile_mismatches(expected: list[str], predicted: list[str]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for index in range(max(len(expected), len(predicted))):
        expected_tile = expected[index] if index < len(expected) else ""
        predicted_tile = predicted[index] if index < len(predicted) else ""
        if expected_tile == predicted_tile:
            continue
        mismatches.append(
            {
                "slot": index + 1,
                "expected": expected_tile,
                "predicted": predicted_tile,
            }
        )
    return mismatches


def _count_correct_tiles(expected: list[str], predicted: list[str]) -> int:
    return sum(
        1
        for index, expected_tile in enumerate(expected)
        if index < len(predicted) and predicted[index] == expected_tile
    )


def _normalize_red_five_tiles(tiles: list[str]) -> list[str]:
    return [_normalize_red_five_tile(tile) for tile in tiles]


def _normalize_red_five_tile(tile: str) -> str:
    text = str(tile).strip()
    lower = text.lower()
    if len(lower) == 3 and lower[0] == "r" and lower[1] == "5" and lower[2] in {"m", "p", "s"}:
        return f"5{lower[2]}"
    return text


def _red5_only_mismatch_count(mismatches: list[dict[str, Any]]) -> int:
    return sum(
        1
        for item in mismatches
        if _normalize_red_five_tile(str(item.get("expected", "")))
        == _normalize_red_five_tile(str(item.get("predicted", "")))
    )


def _coverage_blocked_mismatches(
    mismatches: list[dict[str, Any]],
    train_tile_counts: dict[str, int],
) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    for item in mismatches:
        expected = str(item.get("expected", "")).strip()
        if not expected or train_tile_counts.get(expected, 0) > 0:
            continue
        blocked_item = dict(item)
        blocked_item["reason"] = "expected_tile_absent_from_train_fold"
        blocked.append(blocked_item)
    return blocked


def _tile_counts_for_labels(label_paths: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label_path in label_paths:
        payload = load_json_payload(label_path, default={}, expected_type=dict)
        for tile in _normalize_tiles(payload.get("hand_tiles") or payload.get("expected_hand_tiles")):
            counts[tile] = counts.get(tile, 0) + 1
    return counts


def _normalize_red_five_tile_counts(counts: dict[str, int]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for tile, count in counts.items():
        key = _normalize_red_five_tile(tile)
        normalized[key] = normalized.get(key, 0) + count
    return normalized


def _missing_train_tiles_for_labels(label_paths: list[Path], train_tile_counts: dict[str, int]) -> list[dict[str, Any]]:
    missing: dict[str, int] = {}
    for label_path in label_paths:
        payload = load_json_payload(label_path, default={}, expected_type=dict)
        for tile in _normalize_tiles(payload.get("hand_tiles") or payload.get("expected_hand_tiles")):
            if train_tile_counts.get(tile, 0) > 0:
                continue
            missing[tile] = missing.get(tile, 0) + 1
    return [
        {
            "tile": tile,
            "test_count": count,
        }
        for tile, count in sorted(missing.items(), key=lambda item: (-item[1], item[0]))
    ]


def _format_confusion_counts(counts: dict[tuple[str, str], int], *, limit: int) -> list[dict[str, Any]]:
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [
        {
            "expected": expected,
            "predicted": predicted,
            "count": count,
        }
        for (expected, predicted), count in items[: max(0, limit)]
    ]


def _label_resolution_slug(label_paths: list[Path]) -> str:
    for label_path in label_paths:
        payload = load_json_payload(label_path, default={}, expected_type=dict)
        image = payload.get("image")
        if isinstance(image, dict):
            resolution = str(image.get("resolution", "")).strip().lower().replace("×", "x")
            if resolution:
                return resolution
            width = int(image.get("width", 0) or 0)
            height = int(image.get("height", 0) or 0)
            if width > 0 and height > 0:
                return f"{width}x{height}"
    return "unknown"


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _round_or_none(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mahjong_companion v0.3 release metrics.")
    parser.add_argument(
        "--eval-dir",
        default="plugin/tests/data/mahjong_companion/eval",
        help="Evaluation fixture root.",
    )
    parser.add_argument("--report", help="Optional JSON report output path.")
    parser.add_argument("--strict", action="store_true", help="Fail when any v0.3 metric is missing or below target.")
    parser.add_argument("--strict-json", action="store_true", help="Fail when decision/risk/review JSON metrics miss targets.")
    parser.add_argument("--strict-hand", action="store_true", help="Fail when hand recognition or holdout metrics miss targets.")
    parser.add_argument("--calibration-dir", help="Default calibration directory for hand recognition cases.")
    parser.add_argument("--hand-holdout-dir", help="Calibration label directory for k-fold hand recognition holdout.")
    parser.add_argument("--holdout-folds", type=int, default=5, help="Number of folds for --hand-holdout-dir.")
    parser.add_argument(
        "--holdout-min-train-samples",
        type=int,
        default=1,
        help="Minimum training labels required to enable each temporary holdout profile.",
    )
    parser.add_argument("--holdout-client-version", default="holdout", help="Client version slug for holdout profiles.")
    parser.add_argument("--details", action="store_true", help="Include per-case hand recognition diagnostics.")
    parser.add_argument("--max-details", type=int, default=20, help="Maximum hand case diagnostics to include.")
    parser.add_argument(
        "--allow-fixture-sidecars",
        action="store_true",
        help="Allow *.tiles.json/*.label.json sidecars while parsing hand cases; intended for unit tests only.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON report.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
