from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

from ..perception.discard_parser import crop_discard_quad
from ..storage import load_json_payload, write_json_atomic


DEFAULT_OUTPUT_DIR = Path("plugin/plugins/mahjong_companion/plans/artifacts/discard_crop_debug")
KNOWN_PLAYERS = {"self", "left_opponent", "top_opponent", "right_opponent"}
DEFAULT_PLAYERS = ["left_opponent", "top_opponent", "right_opponent"]
PLAYER_COLORS = {
    "self": (255, 80, 80),
    "left_opponent": (255, 210, 20),
    "top_opponent": (40, 205, 255),
    "right_opponent": (90, 255, 90),
}
PLAYER_PREFIX = {
    "self": "self",
    "left_opponent": "left",
    "top_opponent": "top",
    "right_opponent": "right",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare_discard_crop_debug(
        quad_review_dir=Path(args.quad_review_dir),
        output_dir=Path(args.output_dir),
        players=_parse_players(args.player),
        accepted_only=not bool(args.include_rejected),
        limit=int(args.limit),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result["ok"] else 1


def prepare_discard_crop_debug(
    *,
    quad_review_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    players: list[str] | None = None,
    accepted_only: bool = True,
    limit: int = 0,
) -> dict[str, Any]:
    players = players or list(DEFAULT_PLAYERS)
    quad_paths = _discover_quad_paths(quad_review_dir)
    if limit > 0:
        quad_paths = quad_paths[:limit]
    if not quad_paths:
        raise FileNotFoundError(f"no frame.quads.json files found under {quad_review_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    for quad_path in quad_paths:
        try:
            cases.append(
                _prepare_case(
                    quad_path,
                    output_dir=output_dir,
                    players=players,
                    accepted_only=accepted_only,
                )
            )
        except Exception as exc:
            errors.append(f"{quad_path}: {exc}")

    summary = {
        "ok": not errors,
        "schema_version": "mahjong-discard-crop-debug-v1",
        "quad_review_dir": str(quad_review_dir),
        "output_dir": str(output_dir),
        "players": players,
        "accepted_only": accepted_only,
        "quad_json_count": len(quad_paths),
        "case_count": len(cases),
        "candidate_count": sum(case["candidate_count"] for case in cases),
        "candidate_counts_by_player": _sum_counts(cases, "candidate_counts_by_player"),
        "cases": cases,
        "errors": errors,
    }
    summary_path = output_dir / "discard-crop-debug.json"
    markdown_path = output_dir / "discard-crop-debug.md"
    write_json_atomic(summary_path, summary)
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    summary["summary_json_path"] = str(summary_path)
    summary["summary_markdown_path"] = str(markdown_path)
    return summary


def _prepare_case(
    quad_path: Path,
    *,
    output_dir: Path,
    players: list[str],
    accepted_only: bool,
) -> dict[str, Any]:
    payload = load_json_payload(quad_path, default={}, expected_type=dict)
    if not payload:
        raise ValueError(f"empty quad payload: {quad_path}")
    frame_path = _resolve_frame_path(quad_path, payload)
    with Image.open(frame_path) as opened:
        image = opened.convert("RGB")

    case_id = _clean_filename(str(payload.get("case_id") or quad_path.parent.name))
    case_dir = output_dir / f"{image.width}x{image.height}" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    slots = _filtered_slots(payload.get("slots"), players=players, accepted_only=accepted_only)
    overlay_path = case_dir / "accepted-quads-on-frame.png"
    _write_overlay(image, slots).save(overlay_path, optimize=True)

    all_slots_source = quad_path.with_name("frame.quad-slots.png")
    all_slots_path = ""
    if all_slots_source.exists():
        all_slots_path = str(all_slots_source)

    player_sheets: dict[str, str] = {}
    counts_by_player: dict[str, int] = {}
    for player in players:
        player_slots = [slot for slot in slots if slot.get("player") == player]
        counts_by_player[player] = len(player_slots)
        sheet_path = case_dir / f"{PLAYER_PREFIX.get(player, player)}-accepted-crops.png"
        _write_player_sheet(image, player_slots, player=player).save(sheet_path, optimize=True)
        player_sheets[player] = str(sheet_path)

    return {
        "case_id": case_id,
        "quad_json_path": str(quad_path),
        "frame_path": str(frame_path),
        "overlay_path": str(overlay_path),
        "all_slots_path": all_slots_path,
        "player_sheets": player_sheets,
        "candidate_count": len(slots),
        "candidate_counts_by_player": counts_by_player,
    }


def _filtered_slots(value: Any, *, players: list[str], accepted_only: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    slots: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        player = str(item.get("player", "")).strip()
        if player not in players:
            continue
        if accepted_only and not bool(item.get("accepted")):
            continue
        if not accepted_only and not (item.get("accepted") or item.get("candidate_tile") or item.get("refined")):
            continue
        slots.append(item)
    return sorted(
        slots,
        key=lambda slot: (
            players.index(str(slot.get("player", ""))) if str(slot.get("player", "")) in players else 999,
            _positive_int(slot.get("turn_index"), default=0),
            str(slot.get("slot_id", "")),
        ),
    )


def _write_overlay(image: Image.Image, slots: list[dict[str, Any]]) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for slot in slots:
        player = str(slot.get("player", "")).strip()
        color = PLAYER_COLORS.get(player, (255, 255, 255))
        quad = _quad_points(slot.get("quad") or slot.get("layout_quad"))
        if not quad:
            continue
        draw.line(quad + [quad[0]], fill=color, width=5)
        left = min(point[0] for point in quad)
        top = min(point[1] for point in quad)
        label = _slot_label(slot, short_player=True)
        text_bbox = draw.textbbox((left + 4, top - 22), label)
        draw.rectangle((left, top - 24, max(left + 110, text_bbox[2] + 4), top), fill=(0, 0, 0))
        draw.text((left + 4, top - 21), label, fill=color)
    return overlay


def _write_player_sheet(image: Image.Image, slots: list[dict[str, Any]], *, player: str) -> Image.Image:
    cell_width = 170
    cell_height = 150
    color = PLAYER_COLORS.get(player, (255, 255, 255))
    if not slots:
        sheet = Image.new("RGB", (cell_width, cell_height), color=(22, 28, 34))
        draw = ImageDraw.Draw(sheet)
        draw.rectangle((0, 0, cell_width - 1, cell_height - 1), outline=color, width=3)
        draw.text((8, 8), f"{player}\nno accepted crops", fill=color)
        return sheet

    columns = min(6, max(1, len(slots)))
    rows = (len(slots) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), color=(22, 28, 34))
    draw = ImageDraw.Draw(sheet)
    for index, slot in enumerate(slots):
        column = index % columns
        row = index // columns
        left = column * cell_width
        top = row * cell_height
        draw.rectangle((left, top, left + cell_width - 1, top + cell_height - 1), outline=color, width=3)
        draw.text((left + 5, top + 5), _slot_label(slot), fill=color)
        crop = _slot_crop(image, slot)
        crop.thumbnail((cell_width - 24, cell_height - 56))
        paste_left = left + (cell_width - crop.width) // 2
        paste_top = top + 48 + (cell_height - 56 - crop.height) // 2
        sheet.paste(crop, (paste_left, paste_top))
    return sheet


def _slot_crop(image: Image.Image, slot: dict[str, Any]) -> Image.Image:
    quad = _quad_tuple(slot.get("quad") or slot.get("layout_quad"))
    bbox = _int_list(slot.get("bbox") or slot.get("layout_bbox"), length=4)
    orientation = str(slot.get("orientation", "")).strip()
    if quad and bbox:
        return crop_discard_quad(
            image,
            quad,
            output_size=(max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])),
            orientation=orientation,
        )
    if bbox:
        return image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    return Image.new("RGB", (80, 80), color=(40, 44, 50))


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Discard Crop Debug",
        "",
        f"- quad_review_dir: `{summary['quad_review_dir']}`",
        f"- players: `{summary['players']}`",
        f"- accepted_only: {summary['accepted_only']}",
        f"- case_count: {summary['case_count']}",
        f"- candidate_count: {summary['candidate_count']}",
        f"- counts_by_player: `{summary['candidate_counts_by_player']}`",
        "",
    ]
    for case in summary["cases"]:
        lines.append(f"## {case['case_id']}")
        lines.append("")
        lines.append(f"- frame: `{case['frame_path']}`")
        lines.append(f"- accepted overlay: `{case['overlay_path']}`")
        if case.get("all_slots_path"):
            lines.append(f"- all slots sheet: `{case['all_slots_path']}`")
        lines.append(f"- counts: `{case['candidate_counts_by_player']}`")
        for player, sheet_path in case["player_sheets"].items():
            lines.append(f"- {player}: `{sheet_path}`")
        lines.append("")
    if summary["errors"]:
        lines.append("## Errors")
        lines.append("")
        for error in summary["errors"]:
            lines.append(f"- {error}")
        lines.append("")
    return "\n".join(lines)


def _slot_label(slot: dict[str, Any], *, short_player: bool = False) -> str:
    player = str(slot.get("player", "")).strip()
    tile = str(slot.get("candidate_tile", "")).strip() or "?"
    confidence = _float_value(slot.get("tile_confidence"))
    if short_player:
        return f"{PLAYER_PREFIX.get(player, player)} t{slot.get('turn_index')}={tile}"
    slot_id = str(slot.get("slot_id", "")).strip() or player
    return f"{slot_id}\nt{slot.get('turn_index')}={tile} conf={confidence:.3g}"


def _sum_counts(cases: list[dict[str, Any]], field: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    for case in cases:
        counts = case.get(field)
        if not isinstance(counts, dict):
            continue
        for key, value in counts.items():
            totals[str(key)] = totals.get(str(key), 0) + int(value)
    return totals


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
    for value in values or list(DEFAULT_PLAYERS):
        for item in str(value).replace(",", " ").split():
            player = item.strip()
            if player and player not in players:
                players.append(player)
    unknown = [player for player in players if player not in KNOWN_PLAYERS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown player(s): {', '.join(unknown)}")
    return players or list(DEFAULT_PLAYERS)


def _quad_points(value: Any) -> list[tuple[int, int]]:
    quad = _quad_tuple(value)
    return list(quad) if quad else []


def _quad_tuple(value: Any) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    points: list[tuple[int, int]] = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            return None
        try:
            x, y = [int(part) for part in point]
        except (TypeError, ValueError):
            return None
        points.append((x, y))
    return (points[0], points[1], points[2], points[3])


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


def _clean_filename(value: str) -> str:
    clean = str(value or "").strip().replace("_", "-")
    clean = "".join(char if char.isalnum() or char in {"-", "."} else "-" for char in clean)
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-") or "case"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare per-case crop debug sheets from discard quad review artifacts.")
    parser.add_argument("--quad-review-dir", required=True, help="Directory containing frame.quads.json files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--player",
        action="append",
        default=[],
        help="Player to include; can be repeated or comma-separated. Defaults to left/top/right opponents.",
    )
    parser.add_argument("--include-rejected", action="store_true", help="Include rejected refined/candidate slots.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
