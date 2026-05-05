from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ..storage import load_json_payload, write_json_atomic
from .prepare_discard_fixture import DEFAULT_EVAL_DIR, LABEL_SCOPES, _normalize_tile, prepare_discard_fixture


CONFIRMATION_SCHEMA_VERSION = "mahjong-discard-candidate-confirmations-v1"
APPLY_SCHEMA_VERSION = "mahjong-discard-candidate-apply-v1"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = apply_discard_candidate_confirmations(
        review_path=Path(args.review),
        accept_ids=_parse_id_values(args.accept),
        correction_specs=_parse_corrections(args.correct),
        reject_ids=_parse_id_values(args.reject),
        confirmations_path=Path(args.confirmations) if args.confirmations else None,
        write_template_path=Path(args.write_template) if args.write_template else None,
        accept_all=bool(args.accept_all),
        dry_run=bool(args.dry_run),
        eval_dir=Path(args.eval_dir),
        label_scope=args.label_scope,
        case_suffix=args.case_suffix,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result["ok"] else 1


def apply_discard_candidate_confirmations(
    *,
    review_path: Path,
    accept_ids: list[str] | None = None,
    correction_specs: dict[str, str] | None = None,
    reject_ids: list[str] | None = None,
    confirmations_path: Path | None = None,
    write_template_path: Path | None = None,
    accept_all: bool = False,
    dry_run: bool = False,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    label_scope: str = "partial",
    case_suffix: str = "candidate-reviewed",
    overwrite: bool = False,
) -> dict[str, Any]:
    review = load_json_payload(review_path, default={}, expected_type=dict)
    candidates = _candidate_index(review.get("candidates"))
    if not candidates:
        raise ValueError(f"review has no candidates: {review_path}")

    template_path = ""
    if write_template_path is not None:
        template_path = str(_write_confirmation_template(write_template_path, review_path=review_path, candidates=candidates))

    file_accepts, file_corrections, file_rejects = _load_confirmation_file(confirmations_path) if confirmations_path else ([], {}, [])
    active_accept_ids = _dedupe_ids([*(accept_ids or []), *file_accepts])
    active_reject_ids = set(_dedupe_ids([*(reject_ids or []), *file_rejects]))
    active_corrections = {**file_corrections, **(correction_specs or {})}
    if accept_all:
        active_accept_ids = _dedupe_ids([*active_accept_ids, *candidates.keys()])

    selected = _selected_confirmations(
        candidates,
        accept_ids=active_accept_ids,
        correction_specs=active_corrections,
        reject_ids=active_reject_ids,
    )
    groups = _group_confirmations(selected)

    fixture_results: list[dict[str, Any]] = []
    for group in groups:
        if dry_run:
            fixture_results.append(group)
            continue
        fixture = prepare_discard_fixture(
            image_path=Path(group["frame_path"]),
            case_id=f"{group['case_id']}-{case_suffix}",
            discard_items=group["discard_items"],
            eval_dir=eval_dir,
            write_fixture=True,
            label_scope=label_scope,
            overwrite=overwrite,
        )
        fixture_results.append(
            {
                **group,
                "fixture_label_path": fixture["label_path"],
                "fixture_overlay_path": fixture["overlay_path"],
                "fixture_sheet_path": fixture["sheet_path"],
                "fixture_case_id": fixture["case_id"],
            }
        )

    return {
        "ok": True,
        "schema_version": APPLY_SCHEMA_VERSION,
        "review_path": str(review_path),
        "template_path": template_path,
        "dry_run": dry_run,
        "accept_all": accept_all,
        "candidate_count": len(candidates),
        "accepted_id_count": len(active_accept_ids),
        "corrected_id_count": len(active_corrections),
        "rejected_id_count": len(active_reject_ids),
        "selected_count": len(selected),
        "fixture_count": len(fixture_results),
        "fixture_results": fixture_results,
        "unknown_ids": _unknown_ids(candidates, active_accept_ids, active_corrections, active_reject_ids),
    }


def _write_confirmation_template(
    path: Path,
    *,
    review_path: Path,
    candidates: dict[str, dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "review_path": str(review_path),
        "instructions": [
            "Set status to accept, correct, reject, or leave todo.",
            "For correct, set tile to the corrected tile, e.g. 8p or E.",
            "Only accept/correct rows are written into partial fixtures.",
        ],
        "confirmations": [
            {
                "candidate_id": candidate_id,
                "status": "todo",
                "tile": str(candidate.get("candidate_tile", "")).strip(),
                "confirm_spec": str(candidate.get("confirm_spec", "")).strip(),
                "case_id": str(candidate.get("case_id", "")).strip(),
                "crop_path": str(candidate.get("crop_path", "")).strip(),
            }
            for candidate_id, candidate in candidates.items()
        ],
    }
    write_json_atomic(path, payload)
    return path


def _load_confirmation_file(path: Path) -> tuple[list[str], dict[str, str], list[str]]:
    payload = load_json_payload(path, default={}, expected_type=dict)
    if not payload:
        raise ValueError(f"empty confirmation file: {path}")
    accept_ids = _parse_id_values(payload.get("accept", []))
    reject_ids = _parse_id_values(payload.get("reject", []))
    corrections: dict[str, str] = {}
    raw_correct = payload.get("correct")
    if isinstance(raw_correct, dict):
        for candidate_id, tile in raw_correct.items():
            normalized = _normalize_tile(tile)
            if normalized:
                corrections[str(candidate_id).strip()] = normalized
    elif isinstance(raw_correct, list):
        corrections.update(_parse_corrections(raw_correct))

    raw_rows = payload.get("confirmations")
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            candidate_id = str(row.get("candidate_id", "")).strip()
            status = str(row.get("status", "")).strip().lower()
            if not candidate_id or status in {"", "todo", "pending"}:
                continue
            if status in {"accept", "accepted", "ok"}:
                accept_ids.append(candidate_id)
            elif status in {"reject", "rejected", "bad", "skip"}:
                reject_ids.append(candidate_id)
            elif status in {"correct", "corrected", "fix"}:
                tile = _normalize_tile(row.get("tile", ""))
                if tile:
                    corrections[candidate_id] = tile
    return _dedupe_ids(accept_ids), corrections, _dedupe_ids(reject_ids)


def _selected_confirmations(
    candidates: dict[str, dict[str, Any]],
    *,
    accept_ids: list[str],
    correction_specs: dict[str, str],
    reject_ids: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate_id in _dedupe_ids([*accept_ids, *correction_specs.keys()]):
        if candidate_id in reject_ids:
            continue
        candidate = candidates.get(candidate_id)
        if candidate is None:
            continue
        tile = correction_specs.get(candidate_id) or _normalize_tile(candidate.get("candidate_tile", ""))
        if not tile:
            continue
        selected.append(_confirmation_from_candidate(candidate_id, candidate, tile=tile))
    return selected


def _confirmation_from_candidate(candidate_id: str, candidate: dict[str, Any], *, tile: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "case_id": str(candidate.get("case_id", "")).strip(),
        "frame_path": str(candidate.get("frame_path", "")).strip(),
        "player": str(candidate.get("player", "")).strip(),
        "turn_index": _positive_int(candidate.get("turn_index"), default=0),
        "tile": tile,
        "crop_path": str(candidate.get("crop_path", "")).strip(),
    }


def _group_confirmations(confirmations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in confirmations:
        case_id = str(item.get("case_id", "")).strip()
        frame_path = str(item.get("frame_path", "")).strip()
        player = str(item.get("player", "")).strip()
        tile = str(item.get("tile", "")).strip()
        turn_index = _positive_int(item.get("turn_index"), default=0)
        if not case_id or not frame_path or not player or not tile or turn_index <= 0:
            continue
        group = grouped.setdefault(
            (frame_path, case_id),
            {
                "case_id": case_id,
                "frame_path": frame_path,
                "candidate_ids": [],
                "discard_items": [],
            },
        )
        group["candidate_ids"].append(str(item.get("candidate_id", "")).strip())
        _upsert_discard_item(group["discard_items"], {"player": player, "turn_index": turn_index, "tile": tile})
    return [
        {
            **group,
            "candidate_ids": [candidate_id for candidate_id in group["candidate_ids"] if candidate_id],
            "discard_items": sorted(group["discard_items"], key=lambda item: (item["player"], item["turn_index"])),
        }
        for group in grouped.values()
    ]


def _upsert_discard_item(items: list[dict[str, Any]], item: dict[str, Any]) -> None:
    key = (item["player"], int(item["turn_index"]))
    for index, existing in enumerate(items):
        if (existing["player"], int(existing["turn_index"])) == key:
            items[index] = item
            return
    items.append(item)


def _candidate_index(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    candidates: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "")).strip()
        if candidate_id:
            candidates[candidate_id] = item
    return candidates


def _unknown_ids(
    candidates: dict[str, dict[str, Any]],
    accept_ids: list[str],
    correction_specs: dict[str, str],
    reject_ids: set[str],
) -> list[str]:
    known = set(candidates)
    return sorted({candidate_id for candidate_id in [*accept_ids, *correction_specs.keys(), *reject_ids] if candidate_id not in known})


def _parse_id_values(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list | tuple | set):
        return []
    parsed: list[str] = []
    for value in values:
        for item in str(value).replace(",", " ").split():
            candidate_id = item.strip()
            if candidate_id:
                parsed.append(candidate_id)
    return _dedupe_ids(parsed)


def _parse_corrections(values: Any) -> dict[str, str]:
    if values is None:
        return {}
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list | tuple | set):
        return {}
    corrections: dict[str, str] = {}
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        separator = "=" if "=" in text else ":"
        parts = text.split(separator, 1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("correction must be candidate_id:tile or candidate_id=tile")
        tile = _normalize_tile(parts[1])
        if not tile:
            raise argparse.ArgumentTypeError(f"invalid corrected tile: {parts[1]}")
        corrections[parts[0].strip()] = tile
    return corrections


def _dedupe_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply confirmed discard candidate IDs from a gap review JSON.")
    parser.add_argument("--review", required=True, help="discard-gap-review.json generated by prepare_discard_gap_review.")
    parser.add_argument("--accept", action="append", default=[], help="Candidate IDs to accept; comma/space separated.")
    parser.add_argument("--correct", action="append", default=[], help="Correction as candidate_id:tile or candidate_id=tile.")
    parser.add_argument("--reject", action="append", default=[], help="Candidate IDs to skip; comma/space separated.")
    parser.add_argument("--confirmations", default="", help="JSON confirmation file with accept/correct/reject or rows.")
    parser.add_argument("--write-template", default="", help="Write a JSON confirmation template and continue.")
    parser.add_argument("--accept-all", action="store_true", help="Accept every candidate in the review. Use only after manual review.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--label-scope", choices=sorted(LABEL_SCOPES), default="partial")
    parser.add_argument("--case-suffix", default="candidate-reviewed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
