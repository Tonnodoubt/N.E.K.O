from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Any, Sequence

from PIL import Image, ImageDraw

from ..perception.discard_parser import crop_discard_quad
from ..storage import load_json_payload, write_json_atomic


DEFAULT_OUTPUT_DIR = Path("plugin/plugins/mahjong_companion/plans/artifacts/discard_gap_review")
DEFAULT_EVAL_DIR = Path("plugin/tests/data/mahjong_companion/eval")
KNOWN_PLAYERS = {"self", "left_opponent", "top_opponent", "right_opponent"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare_discard_gap_review(
        quad_review_dir=Path(args.quad_review_dir),
        output_dir=Path(args.output_dir),
        players=_parse_players(args.player),
        accepted_only=not bool(args.include_rejected),
        min_tile_confidence=float(args.min_tile_confidence),
        exclude_accepted=bool(args.exclude_accepted),
        exclude_owner_rejected=bool(args.exclude_owner_rejected),
        limit=int(args.limit),
        columns=int(args.columns),
        eval_dir=Path(args.eval_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result["ok"] else 1


def prepare_discard_gap_review(
    *,
    quad_review_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    players: list[str] | None = None,
    accepted_only: bool = True,
    min_tile_confidence: float = 0.0,
    exclude_accepted: bool = False,
    exclude_owner_rejected: bool = False,
    limit: int = 0,
    columns: int = 8,
    eval_dir: Path = DEFAULT_EVAL_DIR,
) -> dict[str, Any]:
    players = players or ["top_opponent"]
    quad_paths = _discover_quad_paths(quad_review_dir)
    if not quad_paths:
        raise FileNotFoundError(f"no frame.quads.json files found under {quad_review_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    for quad_path in quad_paths:
        try:
            candidates.extend(
                _candidates_from_quad_payload(
                    quad_path,
                    players=players,
                    accepted_only=accepted_only,
                    min_tile_confidence=min_tile_confidence,
                    exclude_accepted=exclude_accepted,
                    exclude_owner_rejected=exclude_owner_rejected,
                    eval_dir=eval_dir,
                )
            )
        except Exception as exc:
            errors.append(f"{quad_path}: {exc}")

    candidates.sort(
        key=lambda item: (
            str(item.get("player", "")),
            str(item.get("case_id", "")),
            int(item.get("turn_index", 0) or 0),
        )
    )
    if limit > 0:
        candidates = candidates[:limit]
    _assign_candidate_ids(candidates)

    sheet_path = output_dir / "discard-gap-candidates.png"
    crop_dir = output_dir / "candidates"
    _write_candidate_crops(candidates, crop_dir=crop_dir)
    _write_candidate_sheet(candidates, sheet_path=sheet_path, columns=columns)
    review = {
        "ok": not errors,
        "schema_version": "mahjong-discard-gap-review-v1",
        "quad_review_dir": str(quad_review_dir),
        "output_dir": str(output_dir),
        "players": players,
        "eval_dir": str(eval_dir),
        "accepted_only": accepted_only,
        "min_tile_confidence": round(float(min_tile_confidence), 3),
        "exclude_accepted": exclude_accepted,
        "exclude_owner_rejected": exclude_owner_rejected,
        "quad_json_count": len(quad_paths),
        "candidate_count": len(candidates),
        "candidate_counts_by_player": _counts_by_field(candidates, "player"),
        "candidate_counts_by_case": _counts_by_field(candidates, "case_id"),
        "sheet_path": str(sheet_path),
        "crop_dir": str(crop_dir),
        "candidates": candidates,
        "errors": errors,
    }
    review_json_path = output_dir / "discard-gap-review.json"
    review_md_path = output_dir / "discard-gap-review.md"
    write_json_atomic(review_json_path, review)
    review_md_path.write_text(_render_markdown(review), encoding="utf-8")
    review["review_json_path"] = str(review_json_path)
    review["review_markdown_path"] = str(review_md_path)
    return review


def _candidates_from_quad_payload(
    quad_path: Path,
    *,
    players: list[str],
    accepted_only: bool,
    min_tile_confidence: float,
    exclude_accepted: bool,
    exclude_owner_rejected: bool,
    eval_dir: Path,
) -> list[dict[str, Any]]:
    payload = load_json_payload(quad_path, default={}, expected_type=dict)
    if not payload:
        return []
    case_id = str(payload.get("case_id") or quad_path.parent.name)
    frame_path = _resolve_frame_path(quad_path, payload)
    records: list[dict[str, Any]] = []
    for slot in payload.get("slots", []):
        if not isinstance(slot, dict):
            continue
        player = str(slot.get("player", "")).strip()
        if player not in players:
            continue
        if accepted_only and not bool(slot.get("accepted")):
            continue
        if exclude_accepted and bool(slot.get("accepted")):
            continue
        if not accepted_only and not (slot.get("accepted") or slot.get("candidate_tile") or slot.get("refined")):
            continue
        if exclude_owner_rejected and str(slot.get("rejected_refinement_owner_slot_id", "")).strip():
            continue
        tile_confidence = _float_value(slot.get("tile_confidence"))
        if tile_confidence < min_tile_confidence:
            continue
        turn_index = _positive_int(slot.get("turn_index"), default=0)
        if turn_index <= 0:
            continue
        candidate_tile = str(slot.get("candidate_tile", "")).strip()
        confirm_tile = candidate_tile or "<tile>"
        records.append(
            {
                "case_id": case_id,
                "quad_json_path": str(quad_path),
                "frame_path": str(frame_path),
                "player": player,
                "turn_index": turn_index,
                "orientation": str(slot.get("orientation", "")).strip(),
                "candidate_tile": candidate_tile,
                "confirm_spec": f"{player}:{turn_index}:{confirm_tile}",
                "tile_confidence": round(tile_confidence, 3),
                "template_distance": slot.get("template_distance"),
                "accepted": bool(slot.get("accepted")),
                "refined": bool(slot.get("refined")),
                "rejected_refinement_owner_slot_id": str(slot.get("rejected_refinement_owner_slot_id", "")).strip(),
                "bbox": _int_list(slot.get("bbox") or slot.get("layout_bbox"), length=4),
                "quad": _quad_list(slot.get("quad") or slot.get("layout_quad")),
                "prepare_fixture_command": _prepare_fixture_command(
                    frame_path=frame_path,
                    case_id=case_id,
                    player=player,
                    turn_index=turn_index,
                    tile=confirm_tile,
                    eval_dir=eval_dir,
                ),
            }
        )
    return records


def _assign_candidate_ids(candidates: list[dict[str, Any]]) -> None:
    counters: dict[str, int] = {}
    for candidate in candidates:
        prefix = _player_prefix(str(candidate.get("player", "")))
        counters[prefix] = counters.get(prefix, 0) + 1
        candidate["candidate_id"] = f"{prefix}-{counters[prefix]:03d}"


def _write_candidate_crops(candidates: list[dict[str, Any]], *, crop_dir: Path) -> None:
    crop_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", "candidate")).strip() or "candidate"
        case_id = _clean_filename(str(candidate.get("case_id", "case")))
        tile = _clean_filename(str(candidate.get("candidate_tile", "tile") or "tile"))
        turn_index = _positive_int(candidate.get("turn_index"), default=0)
        crop_path = crop_dir / f"{candidate_id}-{case_id}-t{turn_index:02d}-{tile}.png"
        _candidate_crop(candidate).save(crop_path, optimize=True)
        candidate["crop_path"] = str(crop_path)


def _write_candidate_sheet(candidates: list[dict[str, Any]], *, sheet_path: Path, columns: int) -> None:
    cell_width = 132
    cell_height = 152
    columns = max(1, int(columns))
    rows = max(1, (len(candidates) + columns - 1) // columns)
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), color=(22, 28, 34))
    draw = ImageDraw.Draw(sheet)

    for index, candidate in enumerate(candidates):
        column = index % columns
        row = index // columns
        left = column * cell_width
        top = row * cell_height
        draw.rectangle((left, top, left + cell_width - 1, top + cell_height - 1), outline=(80, 220, 255), width=2)
        label = _candidate_label(candidate)
        draw.text((left + 5, top + 4), label, fill=(180, 235, 255))
        crop = _candidate_crop(candidate)
        crop.thumbnail((cell_width - 16, cell_height - 42))
        paste_left = left + (cell_width - crop.width) // 2
        paste_top = top + 34 + (cell_height - 42 - crop.height) // 2
        sheet.paste(crop, (paste_left, paste_top))

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, optimize=True)


def _candidate_crop(candidate: dict[str, Any]) -> Image.Image:
    frame_path = Path(str(candidate.get("frame_path", "")))
    if not frame_path.exists():
        return Image.new("RGB", (80, 80), color=(40, 44, 50))
    with Image.open(frame_path) as opened:
        image = opened.convert("RGB")

    quad = _quad_points(candidate.get("quad"))
    bbox = _int_list(candidate.get("bbox"), length=4)
    orientation = str(candidate.get("orientation", "")).strip()
    if quad and bbox:
        return crop_discard_quad(
            image,
            (quad[0], quad[1], quad[2], quad[3]),
            output_size=(max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])),
            orientation=orientation,
        )
    if bbox:
        return image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    return image.copy()


def _render_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Discard Gap Review",
        "",
        f"- quad_review_dir: `{review['quad_review_dir']}`",
        f"- players: `{review['players']}`",
        f"- eval_dir: `{review.get('eval_dir', DEFAULT_EVAL_DIR)}`",
        f"- accepted_only: {review['accepted_only']}",
        f"- exclude_accepted: {review.get('exclude_accepted', False)}",
        f"- exclude_owner_rejected: {review.get('exclude_owner_rejected', False)}",
        f"- candidate_count: {review['candidate_count']}",
        f"- sheet: `{review['sheet_path']}`",
        "",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in review["candidates"]:
        grouped.setdefault(str(candidate["case_id"]), []).append(candidate)
    for case_id, candidates in grouped.items():
        lines.append(f"## {case_id}")
        lines.append("")
        lines.append(f"- frame: `{candidates[0]['frame_path']}`")
        for candidate in candidates:
            owner_rejection = str(candidate.get("rejected_refinement_owner_slot_id", "")).strip()
            owner_note = f" owner_rejection=`{owner_rejection}`" if owner_rejection else ""
            lines.append(
                "- candidate: "
                f"`{candidate.get('candidate_id', '')}` "
                f"`{candidate['confirm_spec']}` "
                f"confidence={candidate['tile_confidence']} "
                f"accepted={candidate['accepted']} "
                f"crop=`{candidate.get('crop_path', '')}`"
                f"{owner_note}"
            )
        command = _case_prepare_command(candidates, eval_dir=Path(str(review.get("eval_dir", DEFAULT_EVAL_DIR))))
        if command:
            lines.append("")
            lines.append("Suggested fixture command after manual confirmation:")
            lines.append("")
            lines.append(f"```bash\n{command}\n```")
        lines.append("")
    if review["errors"]:
        lines.append("## Errors")
        lines.append("")
        for error in review["errors"]:
            lines.append(f"- {error}")
        lines.append("")
    return "\n".join(lines)


def _case_prepare_command(candidates: list[dict[str, Any]], *, eval_dir: Path) -> str:
    if not candidates:
        return ""
    frame_path = Path(str(candidates[0].get("frame_path", "")))
    case_id = str(candidates[0].get("case_id", "discard-gap")).strip()
    players = sorted({str(candidate.get("player", "")).strip() for candidate in candidates})
    review_prefix = _player_prefix(players[0]) if len(players) == 1 else "multi"
    parts = [
        ".venv/bin/python",
        "-m",
        "plugin.plugins.mahjong_companion.scripts.prepare_discard_fixture",
        "--image",
        str(frame_path),
        "--case-id",
        f"{case_id}-{review_prefix}-reviewed",
    ]
    for candidate in candidates:
        parts.extend(["--discard", str(candidate.get("confirm_spec", ""))])
    parts.extend(
        [
            "--eval-dir",
            str(eval_dir),
            "--write-fixture",
            "--label-scope",
            "partial",
            "--overwrite",
            "--pretty",
        ]
    )
    return " ".join(shlex.quote(part) for part in parts)


def _prepare_fixture_command(
    *,
    frame_path: Path,
    case_id: str,
    player: str,
    turn_index: int,
    tile: str,
    eval_dir: Path,
) -> str:
    parts = [
        ".venv/bin/python",
        "-m",
        "plugin.plugins.mahjong_companion.scripts.prepare_discard_fixture",
        "--image",
        str(frame_path),
        "--case-id",
        f"{case_id}-{player}-{turn_index:02d}",
        "--discard",
        f"{player}:{turn_index}:{tile}",
        "--eval-dir",
        str(eval_dir),
        "--write-fixture",
        "--label-scope",
        "partial",
        "--overwrite",
        "--pretty",
    ]
    return " ".join(shlex.quote(part) for part in parts)


def _discover_quad_paths(quad_review_dir: Path) -> list[Path]:
    if quad_review_dir.is_file():
        return [quad_review_dir] if quad_review_dir.name == "frame.quads.json" else []
    if not quad_review_dir.exists() or not quad_review_dir.is_dir():
        raise FileNotFoundError(f"quad review directory not found: {quad_review_dir}")
    return sorted(path for path in quad_review_dir.rglob("frame.quads.json") if path.is_file())


def _resolve_frame_path(quad_path: Path, payload: dict[str, Any]) -> Path:
    image_payload = payload.get("image")
    if isinstance(image_payload, dict):
        raw_path = image_payload.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            candidate = Path(raw_path)
            resolved = candidate if candidate.is_absolute() else quad_path.parent / candidate
            if resolved.exists():
                return resolved
    candidate = quad_path.with_name("frame.png")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"frame image not found for {quad_path}")


def _parse_players(values: list[str]) -> list[str]:
    players: list[str] = []
    for value in values or ["top_opponent"]:
        for item in str(value).replace(",", " ").split():
            player = item.strip()
            if player and player not in players:
                players.append(player)
    unknown = [player for player in players if player not in KNOWN_PLAYERS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown player(s): {', '.join(unknown)}")
    return players or ["top_opponent"]


def _counts_by_field(candidates: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = str(candidate.get(field, "")).strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _candidate_label(candidate: dict[str, Any]) -> str:
    tile = str(candidate.get("candidate_tile", "")).strip() or "?"
    case_id = str(candidate.get("case_id", "")).replace("屏幕截图-", "#")
    candidate_id = str(candidate.get("candidate_id", "")).strip()
    return f"{candidate_id}\n{case_id} t{candidate.get('turn_index')}={tile}"


def _player_prefix(player: str) -> str:
    return {
        "self": "self",
        "left_opponent": "left",
        "top_opponent": "top",
        "right_opponent": "right",
    }.get(player, "candidate")


def _clean_filename(value: str) -> str:
    clean = str(value or "").strip().replace("_", "-")
    clean = "".join(char if char.isalnum() or char in {"-", "."} else "-" for char in clean)
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-") or "candidate"


def _quad_points(value: Any) -> list[tuple[int, int]]:
    quad = _quad_list(value)
    return [(point[0], point[1]) for point in quad]


def _quad_list(value: Any) -> list[list[int]]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return []
    points: list[list[int]] = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            return []
        try:
            x, y = [int(part) for part in point]
        except (TypeError, ValueError):
            return []
        points.append([x, y])
    return points


def _int_list(value: Any, *, length: int) -> list[int]:
    if not isinstance(value, list | tuple) or len(value) != length:
        return []
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a focused review sheet for discard coverage gaps.")
    parser.add_argument("--quad-review-dir", required=True, help="Directory containing frame.quads.json files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--player",
        action="append",
        default=[],
        help="Player to include; can be repeated or comma-separated. Defaults to top_opponent.",
    )
    parser.add_argument("--include-rejected", action="store_true", help="Include non-accepted refined/candidate slots.")
    parser.add_argument("--min-tile-confidence", type=float, default=0.0)
    parser.add_argument("--exclude-accepted", action="store_true", help="Drop already accepted parser results.")
    parser.add_argument(
        "--exclude-owner-rejected",
        action="store_true",
        help="Drop slots whose refined crop was reassigned to a better neighboring slot.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
