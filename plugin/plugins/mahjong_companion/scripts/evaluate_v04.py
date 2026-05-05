from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..action.human_override_guard import HumanOverrideGuard
from ..perception.action_detector import detect_button_regions
from ..perception.calibration import resolve_calibration_profile
from ..perception.roi import build_default_rois, collect_region_metrics
from ..storage import load_json_payload, write_json_atomic


THRESHOLDS = {
    "button_localization.iou_pass_rate": 0.95,
    "button_localization.precision": 0.95,
    "button_localization.recall": 0.85,
    "assist_action.false_abort_rate": 0.05,
    "audit_chain.completeness": 0.99,
    "template_inventory.coverage": 1.0,
}
DEFAULT_REQUIRED_BUTTON_TYPES = ["chi", "pon", "kan", "riichi", "ron", "tsumo", "skip"]
REQUIRED_AUDIT_FIELDS = {
    "action_id",
    "executed_at",
    "ok",
    "locator_source",
    "target_x",
    "target_y",
    "risk_level",
    "confirmation_chain",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = evaluate_v04(
        eval_dir=Path(args.eval_dir),
        calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
        template_dir=Path(args.template_dir) if args.template_dir else None,
        strict=bool(args.strict),
        strict_templates=bool(args.strict_templates),
        required_button_types=_csv_values(args.required_button_types) or DEFAULT_REQUIRED_BUTTON_TYPES,
        include_details=bool(args.details),
        max_details=int(args.max_details),
    )
    if args.report:
        write_json_atomic(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if report["ok"] else 1


def evaluate_v04(
    *,
    eval_dir: Path,
    calibration_dir: Path | None = None,
    template_dir: Path | None = None,
    strict: bool = False,
    strict_templates: bool = False,
    required_button_types: list[str] | None = None,
    include_details: bool = False,
    max_details: int = 20,
) -> dict[str, Any]:
    template_dir = template_dir or Path(__file__).resolve().parents[1] / "perception" / "templates"
    required_button_types = required_button_types or DEFAULT_REQUIRED_BUTTON_TYPES
    button = _evaluate_button_localization(
        eval_dir / "button_localization",
        calibration_dir=calibration_dir,
        template_dir=template_dir,
        include_details=include_details,
        max_details=max_details,
    )
    assist = _evaluate_assist_action_guard(trial_count=100)
    audit = _evaluate_audit_chain(eval_dir / "audit_chain", include_details=include_details, max_details=max_details)
    inventory = _evaluate_template_inventory(template_dir, required_button_types=required_button_types)
    metrics = {
        "template_inventory": inventory,
        "button_localization": button,
        "assist_action": assist,
        "audit_chain": audit,
    }
    failures = _collect_failures(metrics, strict=strict, strict_templates=strict_templates)
    failures.extend(button.pop("_errors", []))
    failures.extend(audit.pop("_errors", []))
    return {
        "ok": not failures,
        "strict": strict,
        "strict_templates": strict_templates,
        "eval_dir": str(eval_dir),
        "calibration_dir": str(calibration_dir) if calibration_dir is not None else "",
        "template_dir": str(template_dir),
        "thresholds": dict(THRESHOLDS),
        "metrics": metrics,
        "failures": failures,
    }


def _evaluate_button_localization(
    root: Path,
    *,
    calibration_dir: Path | None,
    template_dir: Path,
    include_details: bool,
    max_details: int,
) -> dict[str, Any]:
    errors: list[str] = []
    cases = _load_button_cases(root, errors=errors)
    expected_count = 0
    detected_count = 0
    matched_count = 0
    iou_pass_count = 0
    false_positive_count = 0
    per_button: dict[str, dict[str, int]] = {}
    details: list[dict[str, Any]] = []

    for case in cases:
        image_path = case.get("image_path")
        expected_buttons = case.get("buttons", [])
        if not isinstance(image_path, Path) or not image_path.exists() or not isinstance(expected_buttons, list):
            errors.append(f"invalid button localization case: {case.get('case_id', '<unknown>')}")
            continue

        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            width, height = image.size
            rois = build_default_rois(width, height)
            metrics = {name: collect_region_metrics(image, roi) for name, roi in rois.items()}
            metrics["full_frame"] = collect_region_metrics(image, None)
            profile = resolve_calibration_profile(width, height, calibration_dir=calibration_dir)
            detected = [
                region.to_dict()
                for region in detect_button_regions(
                    image,
                    metrics,
                    profile=profile,
                    template_dir=template_dir,
                )
            ]
        except Exception as exc:
            errors.append(f"{case.get('case_id', image_path)}: {exc}")
            continue

        expected = [_normalize_button(item) for item in expected_buttons if isinstance(item, dict)]
        expected = [item for item in expected if item is not None]
        matches, false_positives = _match_button_regions(expected, detected)
        for item in expected:
            _button_counter(per_button, str(item["button_type"]))["expected_count"] += 1
        for item in detected:
            _button_counter(per_button, str(item.get("button_type", "")))["detected_count"] += 1
        for item in matches:
            counter = _button_counter(per_button, str(item["button_type"]))
            counter["matched_count"] += 1
            if item["iou"] >= 0.70:
                counter["iou_pass_count"] += 1
        expected_count += len(expected)
        detected_count += len(detected)
        matched_count += len(matches)
        false_positive_count += false_positives
        iou_pass_count += sum(1 for item in matches if item["iou"] >= 0.70)
        if include_details and len(details) < max(0, int(max_details)):
            details.append({
                "case_id": str(case.get("case_id", image_path)),
                "image_path": str(image_path),
                "expected": expected,
                "detected": detected,
                "matches": matches,
                "false_positives": false_positives,
            })

    result = {
        "case_count": len(cases),
        "expected_count": expected_count,
        "detected_count": detected_count,
        "matched_count": matched_count,
        "iou_pass_count": iou_pass_count,
        "false_positive_count": false_positive_count,
        "iou_pass_rate": _ratio(iou_pass_count, expected_count),
        "precision": _ratio(matched_count, detected_count),
        "recall": _ratio(matched_count, expected_count),
        "per_button_type": _per_button_type_metrics(per_button),
        "_errors": errors,
    }
    if include_details:
        result["case_details"] = details
    return result


def _evaluate_assist_action_guard(*, trial_count: int) -> dict[str, Any]:
    false_aborts = 0
    trial_count = max(1, int(trial_count))
    for index in range(trial_count):
        guard = HumanOverrideGuard(
            pointer_provider=lambda: (100, 100),
            focus_provider=_AlwaysFocusedProvider(),
        )
        guard.arm(
            enabled=True,
            active_window_sec=1.0,
            movement_threshold_px=18,
            expected_window_title="Mahjong Soul",
            check_window_focus=True,
            now_monotonic=float(index),
        )
        decision = guard.evaluate(pointer=(100, 100), now_monotonic=float(index) + 0.1)
        if decision.should_abort:
            false_aborts += 1
    return {
        "trial_count": trial_count,
        "false_abort_count": false_aborts,
        "false_abort_rate": _ratio(false_aborts, trial_count),
    }


def _evaluate_audit_chain(
    root: Path,
    *,
    include_details: bool,
    max_details: int,
) -> dict[str, Any]:
    errors: list[str] = []
    entries = _load_audit_entries(root, errors=errors)
    complete_count = 0
    details: list[dict[str, Any]] = []
    for entry in entries:
        missing = sorted(_missing_audit_fields(entry))
        if not missing:
            complete_count += 1
        if include_details and len(details) < max(0, int(max_details)):
            details.append({
                "action_id": str(entry.get("action_id", "")),
                "complete": not missing,
                "missing": missing,
                "confirmation_steps": [
                    str(item.get("step", ""))
                    for item in entry.get("confirmation_chain", [])
                    if isinstance(item, dict)
                ],
            })
    result = {
        "case_count": len(entries),
        "complete_count": complete_count,
        "completeness": _ratio(complete_count, len(entries)),
        "_errors": errors,
    }
    if include_details:
        result["case_details"] = details
    return result


def _evaluate_template_inventory(
    template_dir: Path,
    *,
    required_button_types: list[str],
) -> dict[str, Any]:
    available = _available_template_button_types(template_dir)
    required = _dedupe_text(required_button_types)
    missing = [item for item in required if item not in available]
    return {
        "required_button_types": required,
        "available_button_types": sorted(available),
        "missing_button_types": missing,
        "required_count": len(required),
        "available_required_count": len(required) - len(missing),
        "coverage": _ratio(len(required) - len(missing), len(required)),
        "complete": not missing,
    }


def _collect_failures(metrics: dict[str, Any], *, strict: bool, strict_templates: bool) -> list[str]:
    if not strict:
        return []
    failures: list[str] = []
    if strict_templates:
        inventory = metrics["template_inventory"]
        coverage = inventory["coverage"]
        if coverage is None or coverage < THRESHOLDS["template_inventory.coverage"]:
            failures.append(
                "template_inventory.coverage="
                f"{coverage} below threshold {THRESHOLDS['template_inventory.coverage']} "
                f"(missing={inventory['missing_button_types']})"
            )
    button = metrics["button_localization"]
    if button["case_count"] <= 0:
        failures.append("button_localization requires at least one case")
    for key in (
        "button_localization.iou_pass_rate",
        "button_localization.precision",
        "button_localization.recall",
    ):
        metric = _nested_metric(metrics, key)
        threshold = THRESHOLDS[key]
        if metric is None or metric < threshold:
            failures.append(f"{key}={metric} below threshold {threshold}")
    false_abort_rate = metrics["assist_action"]["false_abort_rate"]
    if false_abort_rate is None or false_abort_rate > THRESHOLDS["assist_action.false_abort_rate"]:
        failures.append(
            "assist_action.false_abort_rate="
            f"{false_abort_rate} above threshold {THRESHOLDS['assist_action.false_abort_rate']}"
        )
    audit = metrics["audit_chain"]
    if audit["case_count"] <= 0:
        failures.append("audit_chain requires at least one case")
    completeness = audit["completeness"]
    if completeness is None or completeness < THRESHOLDS["audit_chain.completeness"]:
        failures.append(
            f"audit_chain.completeness={completeness} below threshold {THRESHOLDS['audit_chain.completeness']}"
        )
    return failures


def _load_button_cases(root: Path, *, errors: list[str]) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    cases: list[dict[str, Any]] = []
    for label_path in sorted(root.rglob("*.label.json")):
        payload = load_json_payload(label_path, default={}, expected_type=dict)
        image_path = _resolve_case_image_path(label_path, payload)
        if image_path is None:
            errors.append(f"{label_path}: missing image.path")
            continue
        cases.append({
            "case_id": str(payload.get("case_id") or label_path.relative_to(root)),
            "image_path": image_path,
            "buttons": payload.get("buttons", []),
        })
    return cases


def _load_audit_entries(root: Path, *, errors: list[str]) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        payload = load_json_payload(path, default={}, expected_type=(dict, list))
        if isinstance(payload, list):
            raw_entries = payload
        else:
            raw_entries = payload.get("entries", [payload])
        if not isinstance(raw_entries, list):
            errors.append(f"{path}: audit entries must be a list or object")
            continue
        entries.extend(item for item in raw_entries if isinstance(item, dict))
    return entries


def _available_template_button_types(template_dir: Path) -> set[str]:
    meta_path = template_dir / "meta.json"
    payload = load_json_payload(meta_path, default={}, expected_type=dict)
    templates = payload.get("templates")
    available: set[str] = set()
    if isinstance(templates, dict):
        for template_id, template_payload in templates.items():
            if not isinstance(template_payload, dict):
                continue
            button_type = str(template_payload.get("button_type") or template_id).strip()
            file_value = str(template_payload.get("file", "")).strip()
            if button_type and file_value and (template_dir / file_value).exists():
                available.add(button_type)
    for path in template_dir.rglob("*.png"):
        available.add(path.stem)
    return available


def _resolve_case_image_path(label_path: Path, payload: dict[str, Any]) -> Path | None:
    image_payload = payload.get("image", {})
    raw_path = image_payload.get("path") if isinstance(image_payload, dict) else payload.get("image_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = label_path.parent / candidate
    return candidate.resolve()


def _normalize_button(item: dict[str, Any]) -> dict[str, Any] | None:
    button_type = str(item.get("button_type", "")).strip()
    bbox = item.get("bbox")
    if not button_type or not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    try:
        normalized_bbox = [int(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    return {
        "button_type": button_type,
        "bbox": normalized_bbox,
    }


def _match_button_regions(
    expected: list[dict[str, Any]],
    detected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    unmatched = set(range(len(detected)))
    matches: list[dict[str, Any]] = []
    for expected_index, expected_item in enumerate(expected):
        best_index: int | None = None
        best_iou = 0.0
        for detected_index in list(unmatched):
            detected_item = detected[detected_index]
            if detected_item.get("button_type") != expected_item.get("button_type"):
                continue
            iou = _bbox_iou(expected_item["bbox"], detected_item.get("bbox", []))
            if iou > best_iou:
                best_index = detected_index
                best_iou = iou
        if best_index is None:
            continue
        unmatched.remove(best_index)
        matches.append({
            "expected_index": expected_index,
            "detected_index": best_index,
            "button_type": expected_item["button_type"],
            "iou": round(best_iou, 4),
            "pass": best_iou >= 0.70,
        })
    return matches, len(unmatched)


def _button_counter(per_button: dict[str, dict[str, int]], button_type: str) -> dict[str, int]:
    key = str(button_type or "unknown").strip() or "unknown"
    counter = per_button.get(key)
    if counter is None:
        counter = {
            "expected_count": 0,
            "detected_count": 0,
            "matched_count": 0,
            "iou_pass_count": 0,
        }
        per_button[key] = counter
    return counter


def _per_button_type_metrics(per_button: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for button_type in sorted(per_button):
        counter = per_button[button_type]
        expected_count = counter["expected_count"]
        detected_count = counter["detected_count"]
        matched_count = counter["matched_count"]
        iou_pass_count = counter["iou_pass_count"]
        metrics.append({
            "button_type": button_type,
            "expected_count": expected_count,
            "detected_count": detected_count,
            "matched_count": matched_count,
            "iou_pass_count": iou_pass_count,
            "false_positive_count": max(0, detected_count - matched_count),
            "iou_pass_rate": _ratio(iou_pass_count, expected_count),
            "precision": _ratio(matched_count, detected_count),
            "recall": _ratio(matched_count, expected_count),
        })
    return metrics


def _bbox_iou(left_box: list[Any], right_box: list[Any]) -> float:
    if len(left_box) != 4 or len(right_box) != 4:
        return 0.0
    try:
        l1, t1, r1, b1 = [int(value) for value in left_box]
        l2, t2, r2, b2 = [int(value) for value in right_box]
    except (TypeError, ValueError):
        return 0.0
    inter_left = max(l1, l2)
    inter_top = max(t1, t2)
    inter_right = min(r1, r2)
    inter_bottom = min(b1, b2)
    inter_width = max(0, inter_right - inter_left)
    inter_height = max(0, inter_bottom - inter_top)
    intersection = inter_width * inter_height
    left_area = max(0, r1 - l1) * max(0, b1 - t1)
    right_area = max(0, r2 - l2) * max(0, b2 - t2)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _missing_audit_fields(entry: dict[str, Any]) -> set[str]:
    missing = {field for field in REQUIRED_AUDIT_FIELDS if field not in entry}
    chain = entry.get("confirmation_chain")
    if not isinstance(chain, list) or not chain:
        missing.add("confirmation_chain")
    elif any(
        not isinstance(item, dict) or not item.get("step") or "value" not in item or not item.get("at")
        for item in chain
    ):
        missing.add("confirmation_chain")
    if entry.get("locator_source") == "button_candidate" and not isinstance(entry.get("button_region"), dict):
        missing.add("button_region")
    return missing


def _nested_metric(metrics: dict[str, Any], key: str) -> Any:
    group, name = key.split(".", 1)
    payload = metrics.get(group, {})
    if not isinstance(payload, dict):
        return None
    return payload.get(name)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


class _AlwaysFocusedProvider:
    def is_target_window_focused(self, expected_title: str) -> bool:
        return True


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mahjong Companion v0.4 metrics.")
    parser.add_argument("--eval-dir", default="plugin/tests/data/mahjong_companion/eval")
    parser.add_argument("--calibration-dir", default="")
    parser.add_argument("--template-dir", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--strict-templates", action="store_true")
    parser.add_argument("--required-button-types", default=",".join(DEFAULT_REQUIRED_BUTTON_TYPES))
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--max-details", type=int, default=20)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
