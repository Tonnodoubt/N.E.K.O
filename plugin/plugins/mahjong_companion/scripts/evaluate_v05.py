from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from PIL import Image

from ..contracts import PerceivedGameState
from ..decision.generator import build_decision
from ..perception.roi import build_default_rois, collect_region_metrics
from ..perception.tile_parser import parse_tiles_from_image
from ..storage import load_json_payload, write_json_atomic
from ..tile_labels import format_tile_label


THRESHOLDS = {
    "discard_recognition.tile_accuracy": 0.70,
    "discard_recognition.recall": 0.75,
    "discard_recognition.false_positive_rate": 0.10,
    "genbutsu_hint.recall": 0.95,
    "decision_latency_p95_ms": 900.0,
}
DISCARD_SLOT_COUNT = 72
DISCARD_SLOTS_PER_PLAYER = 18
DISCARD_PLAYERS = ("self", "left_opponent", "top_opponent", "right_opponent")
OPPONENT_PLAYERS = {"left_opponent", "top_opponent", "right_opponent"}
DISCARD_ORIENTATION_BY_PLAYER = {
    "self": "bottom",
    "left_opponent": "left",
    "top_opponent": "top",
    "right_opponent": "right",
}
DISCARD_ORIENTATIONS = ("bottom", "left", "top", "right")
DEFAULT_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "data" / "calibration"
FULL_LABEL_SCOPE = "full"
PARTIAL_LABEL_SCOPE = "partial"
PARTIAL_LABEL_SCOPE_VALUES = {"partial", "positive_only", "selected", "incomplete"}
FULL_LABEL_SCOPE_VALUES = {"", "full", "complete", "exhaustive"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    calibration_dir = Path(args.calibration_dir) if args.calibration_dir else _default_calibration_dir()
    report = evaluate_v05(
        eval_dir=Path(args.eval_dir),
        calibration_dir=calibration_dir,
        strict=bool(args.strict),
        strict_json=bool(args.strict_json),
        include_details=bool(args.details),
        max_details=int(args.max_details),
    )
    if args.report:
        write_json_atomic(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if report["ok"] else 1


def evaluate_v05(
    *,
    eval_dir: Path,
    calibration_dir: Path | None = None,
    strict: bool = False,
    strict_json: bool = False,
    include_details: bool = False,
    max_details: int = 20,
) -> dict[str, Any]:
    calibration_dir = calibration_dir or _default_calibration_dir()
    discard = _evaluate_discard_recognition(
        eval_dir / "discard_recognition",
        calibration_dir=calibration_dir,
        include_details=include_details,
        max_details=max_details,
    )
    genbutsu = _evaluate_genbutsu_hints(
        eval_dir / "genbutsu_hint",
        include_details=include_details,
        max_details=max_details,
    )
    metrics = {
        "discard_recognition": discard,
        "genbutsu_hint": genbutsu,
        "decision_latency_p95_ms": _p95(discard.pop("_latencies_ms", []) + genbutsu.pop("_latencies_ms", [])),
    }
    failures = _collect_failures(metrics, strict=strict, strict_json=strict_json)
    failures.extend(discard.pop("_errors", []))
    failures.extend(genbutsu.pop("_errors", []))
    return {
        "ok": not failures,
        "strict": strict,
        "strict_json": strict_json,
        "eval_dir": str(eval_dir),
        "calibration_dir": str(calibration_dir) if calibration_dir is not None else "",
        "thresholds": dict(THRESHOLDS),
        "metrics": metrics,
        "failures": failures,
    }


def _default_calibration_dir() -> Path | None:
    return DEFAULT_CALIBRATION_DIR if DEFAULT_CALIBRATION_DIR.exists() else None


def _evaluate_discard_recognition(
    root: Path,
    *,
    calibration_dir: Path | None,
    include_details: bool,
    max_details: int,
) -> dict[str, Any]:
    errors: list[str] = []
    cases = _load_discard_cases(root, errors=errors)
    expected_count = 0
    predicted_count = 0
    matched_slot_count = 0
    correct_tile_count = 0
    false_positive_count = 0
    ignored_unlabeled_prediction_count = 0
    partial_case_count = 0
    full_case_count = 0
    evaluable_empty_slot_count = 0
    by_player = {player: _empty_discard_bucket() for player in DISCARD_PLAYERS}
    by_orientation = {orientation: _empty_discard_bucket() for orientation in DISCARD_ORIENTATIONS}
    latencies_ms: list[float] = []
    details: list[dict[str, Any]] = []

    for case in cases:
        image_path = case.get("image_path")
        expected_items = case.get("discard_items")
        if not isinstance(image_path, Path) or not image_path.exists() or not isinstance(expected_items, list):
            errors.append(f"invalid discard recognition case: {case.get('case_id', '<unknown>')}")
            continue

        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            metrics = _frame_metrics(image)
            started_at = perf_counter()
            parsed = parse_tiles_from_image(
                image_path,
                image,
                scene=str(case.get("scene", "in_match") or "in_match"),
                metrics=metrics,
                calibration_dir=calibration_dir,
                fixture_mode="disabled",
            )
            latencies_ms.append((perf_counter() - started_at) * 1000.0)
        except Exception as exc:
            errors.append(f"{case.get('case_id', image_path)}: {exc}")
            continue

        expected_by_slot = {
            _slot_key(item): item for item in expected_items if isinstance(item, dict) and _slot_key(item)
        }
        case_expected_player_counts: dict[str, int] = {}
        for expected in expected_by_slot.values():
            _increment_discard_buckets(
                by_player=by_player,
                by_orientation=by_orientation,
                item=expected,
                field="expected_count",
            )
            player = _item_player(expected)
            case_expected_player_counts[player] = case_expected_player_counts.get(player, 0) + 1
        label_scope = _normalize_discard_label_scope(case.get("label_scope"))
        is_partial_label = label_scope == PARTIAL_LABEL_SCOPE
        if is_partial_label:
            partial_case_count += 1
        else:
            full_case_count += 1
        empty_slot_keys = {
            key
            for key in case.get("empty_slot_keys", [])
            if isinstance(key, tuple) and len(key) == 2 and key not in expected_by_slot
        }
        if is_partial_label:
            evaluable_empty_slot_count += len(empty_slot_keys)
            for player, turn_index in empty_slot_keys:
                _increment_discard_bucket(
                    by_player,
                    player,
                    "evaluable_empty_slot_count",
                )
                _increment_discard_bucket(
                    by_orientation,
                    _orientation_from_player(player),
                    "evaluable_empty_slot_count",
                )
                _ = turn_index
        else:
            evaluable_empty_slot_count += max(0, DISCARD_SLOT_COUNT - len(expected_by_slot))
            for player in DISCARD_PLAYERS:
                player_empty_count = max(0, DISCARD_SLOTS_PER_PLAYER - case_expected_player_counts.get(player, 0))
                _increment_discard_bucket(by_player, player, "evaluable_empty_slot_count", player_empty_count)
                _increment_discard_bucket(
                    by_orientation,
                    _orientation_from_player(player),
                    "evaluable_empty_slot_count",
                    player_empty_count,
                )

        predicted_items = [
            item for pile in parsed.discard_piles.values() for item in pile if isinstance(item, dict) and _slot_key(item)
        ]
        expected_count += len(expected_by_slot)
        predicted_count += len(predicted_items)
        for predicted in predicted_items:
            _increment_discard_buckets(
                by_player=by_player,
                by_orientation=by_orientation,
                item=predicted,
                field="predicted_count",
            )

        case_matches: list[dict[str, Any]] = []
        case_false_positives = 0
        case_ignored_unlabeled_predictions = 0
        for predicted in predicted_items:
            key = _slot_key(predicted)
            expected = expected_by_slot.get(key)
            if expected is None:
                if is_partial_label and key not in empty_slot_keys:
                    ignored_unlabeled_prediction_count += 1
                    case_ignored_unlabeled_predictions += 1
                    _increment_discard_buckets(
                        by_player=by_player,
                        by_orientation=by_orientation,
                        item=predicted,
                        field="ignored_unlabeled_prediction_count",
                    )
                    continue
                false_positive_count += 1
                case_false_positives += 1
                _increment_discard_buckets(
                    by_player=by_player,
                    by_orientation=by_orientation,
                    item=predicted,
                    field="false_positive_count",
                )
                continue
            matched_slot_count += 1
            _increment_discard_buckets(
                by_player=by_player,
                by_orientation=by_orientation,
                item=expected,
                field="matched_slot_count",
            )
            tile_ok = str(predicted.get("tile", "")).strip() == str(expected.get("tile", "")).strip()
            if tile_ok:
                correct_tile_count += 1
                _increment_discard_buckets(
                    by_player=by_player,
                    by_orientation=by_orientation,
                    item=expected,
                    field="correct_tile_count",
                )
            case_matches.append({"slot": key, "expected": expected, "predicted": predicted, "tile_ok": tile_ok})

        if include_details and len(details) < max(0, int(max_details)):
            details.append(
                {
                    "case_id": str(case.get("case_id", image_path)),
                    "image_path": str(image_path),
                    "label_scope": label_scope,
                    "expected_count": len(expected_by_slot),
                    "predicted_count": len(predicted_items),
                    "matches": case_matches,
                    "false_positive_count": case_false_positives,
                    "ignored_unlabeled_prediction_count": case_ignored_unlabeled_predictions,
                }
            )

    finalized_by_player = _finalize_discard_buckets(by_player)
    finalized_by_orientation = _finalize_discard_buckets(by_orientation)
    coverage_warnings = _discard_coverage_warnings(
        by_player=finalized_by_player,
        by_orientation=finalized_by_orientation,
        partial_case_count=partial_case_count,
        full_case_count=full_case_count,
    )
    result = {
        "case_count": len(cases),
        "full_case_count": full_case_count,
        "partial_case_count": partial_case_count,
        "expected_count": expected_count,
        "predicted_count": predicted_count,
        "evaluated_prediction_count": max(0, predicted_count - ignored_unlabeled_prediction_count),
        "matched_slot_count": matched_slot_count,
        "correct_tile_count": correct_tile_count,
        "false_positive_count": false_positive_count,
        "ignored_unlabeled_prediction_count": ignored_unlabeled_prediction_count,
        "evaluable_empty_slot_count": evaluable_empty_slot_count,
        "tile_accuracy": _ratio(correct_tile_count, matched_slot_count),
        "recall": _ratio(matched_slot_count, expected_count),
        "false_positive_rate": _ratio(false_positive_count, evaluable_empty_slot_count),
        "by_player": finalized_by_player,
        "by_orientation": finalized_by_orientation,
        "coverage_warnings": coverage_warnings,
        "_errors": errors,
        "_latencies_ms": latencies_ms,
    }
    if include_details:
        result["details"] = details
    return result


def _evaluate_genbutsu_hints(
    root: Path,
    *,
    include_details: bool,
    max_details: int,
) -> dict[str, Any]:
    errors: list[str] = []
    cases = _load_genbutsu_cases(root, errors=errors)
    expected_case_count = 0
    detected_case_count = 0
    latencies_ms: list[float] = []
    details: list[dict[str, Any]] = []

    for case in cases:
        state_payload = case.get("state")
        if not isinstance(state_payload, dict):
            errors.append(f"invalid genbutsu case: {case.get('case_id', '<unknown>')}")
            continue

        expected_tiles = _normalize_text_list(
            case.get("expected_known_genbutsu_tiles")
            or case.get("expected_genbutsu_tiles")
            or state_payload.get("known_genbutsu_tiles"),
        )
        if expected_tiles:
            expected_case_count += 1

        state_payload = dict(state_payload)
        if expected_tiles and not state_payload.get("known_genbutsu_tiles"):
            state_payload["known_genbutsu_tiles"] = _derive_known_genbutsu_tiles(
                riichi_players=_normalize_text_list(state_payload.get("riichi_players")),
                discard_piles=state_payload.get("discard_piles"),
            )
        try:
            state = PerceivedGameState(**state_payload)
            started_at = perf_counter()
            decision = build_decision(state)
            latencies_ms.append((perf_counter() - started_at) * 1000.0)
        except Exception as exc:
            errors.append(f"{case.get('case_id', '<unknown>')}: {exc}")
            continue

        alerts = " ".join(str(item) for item in decision.mahjong_analysis.get("defense_alerts", []))
        detected = bool(expected_tiles) and "现物" in alerts and all(
            _tile_mentioned_in_alerts(tile, alerts) for tile in expected_tiles
        )
        if detected:
            detected_case_count += 1
        if include_details and len(details) < max(0, int(max_details)):
            details.append(
                {
                    "case_id": str(case.get("case_id", "<unknown>")),
                    "expected_tiles": expected_tiles,
                    "detected": detected,
                    "defense_alerts": decision.mahjong_analysis.get("defense_alerts", []),
                }
            )

    result = {
        "case_count": len(cases),
        "expected_genbutsu_cases": expected_case_count,
        "detected_genbutsu_cases": detected_case_count,
        "recall": _ratio(detected_case_count, expected_case_count),
        "_errors": errors,
        "_latencies_ms": latencies_ms,
    }
    if include_details:
        result["details"] = details
    return result


def _tile_mentioned_in_alerts(tile: str, alerts: str) -> bool:
    raw_tile = str(tile).strip()
    if not raw_tile:
        return False
    label = format_tile_label(raw_tile)
    return raw_tile in alerts or bool(label and label in alerts)


def _load_discard_cases(root: Path, *, errors: list[str]) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    cases: list[dict[str, Any]] = []
    for label_path in sorted(root.rglob("*.label.json")):
        payload = load_json_payload(label_path, default={}, expected_type=dict)
        image_path = _resolve_image_path(label_path, payload)
        discard_items = _discard_items_from_payload(payload)
        if image_path is None:
            errors.append(f"{label_path}: missing image path")
            continue
        cases.append(
            {
                "case_id": str(payload.get("case_id") or label_path.parent.name or label_path.stem),
                "image_path": image_path,
                "scene": payload.get("scene", "in_match"),
                "label_scope": _discard_label_scope_from_payload(payload),
                "discard_items": discard_items,
                "empty_slot_keys": _empty_discard_slot_keys_from_payload(payload),
            }
        )
    return cases


def _load_genbutsu_cases(root: Path, *, errors: list[str]) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    cases: list[dict[str, Any]] = []
    for case_path in sorted(root.rglob("*.json")):
        if case_path.name.endswith(".label.json"):
            continue
        payload = load_json_payload(case_path, default={}, expected_type=dict)
        if not payload:
            errors.append(f"{case_path}: empty or invalid JSON case")
            continue
        payload["case_id"] = str(payload.get("case_id") or case_path.stem)
        cases.append(payload)
    return cases


def _discard_items_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_piles = payload.get("discard_piles")
    if not isinstance(raw_piles, dict):
        return []
    items: list[dict[str, Any]] = []
    for player, raw_items in raw_piles.items():
        player_key = str(player).strip()
        if not player_key or not isinstance(raw_items, list):
            continue
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            tile = str(item.get("tile", "")).strip()
            if not tile:
                continue
            items.append(
                {
                    "tile": tile,
                    "player": str(item.get("player") or player_key),
                    "turn_index": int(item.get("turn_index", index + 1) or index + 1),
                    "orientation": str(item.get("orientation") or _orientation_from_player(player_key)),
                }
            )
    return items


def _empty_discard_bucket() -> dict[str, int]:
    return {
        "expected_count": 0,
        "predicted_count": 0,
        "matched_slot_count": 0,
        "correct_tile_count": 0,
        "false_positive_count": 0,
        "ignored_unlabeled_prediction_count": 0,
        "evaluable_empty_slot_count": 0,
    }


def _increment_discard_buckets(
    *,
    by_player: dict[str, dict[str, int]],
    by_orientation: dict[str, dict[str, int]],
    item: dict[str, Any],
    field: str,
    amount: int = 1,
) -> None:
    player = _item_player(item)
    _increment_discard_bucket(by_player, player, field, amount)
    _increment_discard_bucket(by_orientation, _item_orientation(item), field, amount)


def _increment_discard_bucket(
    buckets: dict[str, dict[str, int]],
    key: str,
    field: str,
    amount: int = 1,
) -> None:
    if key not in buckets:
        buckets[key] = _empty_discard_bucket()
    buckets[key][field] = int(buckets[key].get(field, 0)) + int(amount)


def _finalize_discard_buckets(buckets: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for key, bucket in buckets.items():
        predicted = int(bucket.get("predicted_count", 0))
        ignored = int(bucket.get("ignored_unlabeled_prediction_count", 0))
        matched = int(bucket.get("matched_slot_count", 0))
        correct = int(bucket.get("correct_tile_count", 0))
        expected = int(bucket.get("expected_count", 0))
        false_positive = int(bucket.get("false_positive_count", 0))
        empty_slots = int(bucket.get("evaluable_empty_slot_count", 0))
        finalized[key] = {
            **bucket,
            "evaluated_prediction_count": max(0, predicted - ignored),
            "tile_accuracy": _ratio(correct, matched),
            "recall": _ratio(matched, expected),
            "false_positive_rate": _ratio(false_positive, empty_slots),
        }
    return finalized


def _discard_coverage_warnings(
    *,
    by_player: dict[str, dict[str, Any]],
    by_orientation: dict[str, dict[str, Any]],
    partial_case_count: int,
    full_case_count: int,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if partial_case_count > 0 and full_case_count == 0:
        warnings.append(
            {
                "scope": "discard_recognition",
                "kind": "partial_only_labels",
                "message": "discard recognition currently has only partial labels; unlabelled predictions are not false-positive claims.",
                "partial_case_count": partial_case_count,
                "full_case_count": full_case_count,
            }
        )

    for player in DISCARD_PLAYERS:
        bucket = by_player.get(player, {})
        if int(bucket.get("expected_count", 0)) == 0 and int(bucket.get("ignored_unlabeled_prediction_count", 0)) > 0:
            warnings.append(
                {
                    "scope": "by_player",
                    "key": player,
                    "kind": "unscored_predictions",
                    "message": f"{player} has predictions but no labelled expected slots, so quality is not scored yet.",
                    "predicted_count": int(bucket.get("predicted_count", 0)),
                    "ignored_unlabeled_prediction_count": int(bucket.get("ignored_unlabeled_prediction_count", 0)),
                }
            )

    for orientation in DISCARD_ORIENTATIONS:
        bucket = by_orientation.get(orientation, {})
        if int(bucket.get("expected_count", 0)) == 0 and int(bucket.get("ignored_unlabeled_prediction_count", 0)) > 0:
            warnings.append(
                {
                    "scope": "by_orientation",
                    "key": orientation,
                    "kind": "unscored_predictions",
                    "message": f"{orientation} orientation has predictions but no labelled expected slots, so quality is not scored yet.",
                    "predicted_count": int(bucket.get("predicted_count", 0)),
                    "ignored_unlabeled_prediction_count": int(bucket.get("ignored_unlabeled_prediction_count", 0)),
                }
            )
    return warnings


def _item_player(item: dict[str, Any]) -> str:
    return str(item.get("player", "")).strip() or "unknown"


def _item_orientation(item: dict[str, Any]) -> str:
    orientation = str(item.get("orientation", "")).strip()
    if orientation:
        return orientation
    return _orientation_from_player(_item_player(item))


def _orientation_from_player(player: str) -> str:
    return DISCARD_ORIENTATION_BY_PLAYER.get(str(player).strip(), "unknown")


def _discard_label_scope_from_payload(payload: dict[str, Any]) -> str:
    raw_scope = (
        payload.get("discard_label_scope")
        or payload.get("label_scope")
        or payload.get("annotation_scope")
        or payload.get("discard_recognition_label_scope")
    )
    if payload.get("ignore_unlabeled_predictions") is True:
        raw_scope = PARTIAL_LABEL_SCOPE
    return _normalize_discard_label_scope(raw_scope)


def _normalize_discard_label_scope(value: Any) -> str:
    scope = str(value or "").strip().lower().replace("-", "_")
    if scope in PARTIAL_LABEL_SCOPE_VALUES:
        return PARTIAL_LABEL_SCOPE
    if scope in FULL_LABEL_SCOPE_VALUES:
        return FULL_LABEL_SCOPE
    return FULL_LABEL_SCOPE


def _empty_discard_slot_keys_from_payload(payload: dict[str, Any]) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for field_name in (
        "empty_discard_slots",
        "negative_discard_slots",
        "known_empty_discard_slots",
    ):
        raw_slots = payload.get(field_name)
        if not isinstance(raw_slots, list):
            continue
        for raw_slot in raw_slots:
            key = _slot_key_from_empty_slot(raw_slot)
            if key is not None:
                keys.append(key)
    return sorted(set(keys))


def _slot_key_from_empty_slot(value: Any) -> tuple[str, int] | None:
    if isinstance(value, dict):
        return _slot_key(value)
    if not isinstance(value, str):
        return None
    player, separator, raw_turn_index = value.strip().partition(":")
    if not separator:
        player, separator, raw_turn_index = value.strip().partition("#")
    if not separator:
        return None
    try:
        turn_index = int(raw_turn_index)
    except ValueError:
        return None
    player = player.strip()
    if not player or turn_index <= 0:
        return None
    return player, turn_index


def _frame_metrics(image: Image.Image) -> dict[str, dict[str, Any]]:
    rois = build_default_rois(*image.size)
    metrics = {name: collect_region_metrics(image, roi) for name, roi in rois.items()}
    metrics["full_frame"] = collect_region_metrics(image, None)
    return metrics


def _resolve_image_path(label_path: Path, payload: dict[str, Any]) -> Path | None:
    image = payload.get("image")
    if isinstance(image, dict):
        raw_path = image.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            candidate = Path(raw_path)
            return candidate if candidate.is_absolute() else label_path.parent / candidate
    stem = label_path.name.removesuffix(".label.json")
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = label_path.with_name(f"{stem}{suffix}")
        if candidate.exists():
            return candidate
    return None


def _slot_key(item: dict[str, Any]) -> tuple[str, int] | None:
    player = str(item.get("player", "")).strip()
    try:
        turn_index = int(item.get("turn_index", 0) or 0)
    except (TypeError, ValueError):
        return None
    if not player or turn_index <= 0:
        return None
    return player, turn_index


def _derive_known_genbutsu_tiles(*, riichi_players: list[str], discard_piles: Any) -> list[str]:
    if not riichi_players or not isinstance(discard_piles, dict):
        return []
    tiles: list[str] = []
    for player in riichi_players:
        if player not in OPPONENT_PLAYERS:
            continue
        raw_items = discard_piles.get(player)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            tile = str(item.get("tile", "")).strip()
            if tile:
                tiles.append(tile)
    return _dedupe_text(tiles)


def _collect_failures(metrics: dict[str, Any], *, strict: bool, strict_json: bool) -> list[str]:
    failures: list[str] = []
    if not (strict or strict_json):
        return failures

    genbutsu_recall = metrics["genbutsu_hint"].get("recall")
    if genbutsu_recall is not None and genbutsu_recall < THRESHOLDS["genbutsu_hint.recall"]:
        failures.append(
            f"genbutsu_hint.recall {genbutsu_recall:.3f} < {THRESHOLDS['genbutsu_hint.recall']:.3f}",
        )
    latency = metrics.get("decision_latency_p95_ms")
    if latency is not None and latency > THRESHOLDS["decision_latency_p95_ms"]:
        failures.append(f"decision_latency_p95_ms {latency:.3f} > {THRESHOLDS['decision_latency_p95_ms']:.3f}")

    if not strict:
        return failures

    discard = metrics["discard_recognition"]
    for key in ("tile_accuracy", "recall"):
        value = discard.get(key)
        threshold = THRESHOLDS[f"discard_recognition.{key}"]
        if value is not None and value < threshold:
            failures.append(f"discard_recognition.{key} {value:.3f} < {threshold:.3f}")
    false_positive_rate = discard.get("false_positive_rate")
    threshold = THRESHOLDS["discard_recognition.false_positive_rate"]
    if false_positive_rate is not None and false_positive_rate > threshold:
        failures.append(f"discard_recognition.false_positive_rate {false_positive_rate:.3f} > {threshold:.3f}")
    return failures


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mahjong Companion v0.5 discard and genbutsu gates.")
    parser.add_argument("--eval-dir", default="plugin/tests/data/mahjong_companion/eval")
    parser.add_argument("--calibration-dir", default="")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--strict-json", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--report", default="")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return round(ordered[index], 3)


def _normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


if __name__ == "__main__":
    raise SystemExit(main())
