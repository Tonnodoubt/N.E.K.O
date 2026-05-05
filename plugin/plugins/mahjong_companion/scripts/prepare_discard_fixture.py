from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

from ..perception.discard_layout import DiscardSlot, build_discard_layout
from ..perception.discard_quad_finder import refine_discard_slot_quad
from ..perception.discard_parser import crop_discard_slot
from ..storage import write_json_atomic


LABEL_SCHEMA_VERSION = "mahjong-discard-recognition-label-v1"
DEFAULT_EVAL_DIR = Path("plugin/tests/data/mahjong_companion/eval")
DEFAULT_OUTPUT_DIR = Path("plugin/plugins/mahjong_companion/plans/artifacts/discard_labeling")
PLAYER_COLORS = {
    "self": (255, 80, 80),
    "left_opponent": (255, 210, 60),
    "top_opponent": (80, 220, 255),
    "right_opponent": (120, 255, 120),
}
PLAYER_PREFIX = {
    "self": "self",
    "left_opponent": "left",
    "top_opponent": "top",
    "right_opponent": "right",
}
HONOR_ALIASES = {
    "E": "1z",
    "S": "2z",
    "W": "3z",
    "N": "4z",
    "P": "5z",
    "F": "6z",
    "C": "7z",
}
LABEL_SCOPES = {"full", "partial"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare_discard_fixture(
        image_path=Path(args.image),
        case_id=args.case_id,
        discard_items=[_parse_discard_spec(item) for item in args.discard],
        riichi_players=_parse_player_list(args.riichi_players),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        eval_dir=Path(args.eval_dir),
        write_fixture=bool(args.write_fixture),
        label_scope=args.label_scope,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


def prepare_discard_fixture(
    *,
    image_path: Path,
    case_id: str = "",
    discard_items: list[dict[str, Any]] | None = None,
    riichi_players: list[str] | None = None,
    output_dir: Path | None = None,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    write_fixture: bool = False,
    label_scope: str = "partial",
    overwrite: bool = False,
) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    resolution = (image.width, image.height)
    resolution_name = f"{resolution[0]}x{resolution[1]}"
    clean_case_id = _clean_case_id(case_id or image_path.stem)
    case_dir = (
        eval_dir / "discard_recognition" / resolution_name / clean_case_id
        if write_fixture
        else (output_dir or DEFAULT_OUTPUT_DIR) / resolution_name / clean_case_id
    )
    frame_path = case_dir / "frame.png"
    overlay_path = case_dir / "frame.overlay.png"
    sheet_path = case_dir / "frame.slots.png"
    label_path = case_dir / "frame.label.json"
    if case_dir.exists() and any(case_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"discard fixture case already exists: {case_dir}")

    layout = build_discard_layout(*resolution)
    annotated_piles = _annotated_piles(discard_items or [], layout, image=image)
    normalized_label_scope = _normalize_label_scope(label_scope)
    case_dir.mkdir(parents=True, exist_ok=True)
    image.save(frame_path, optimize=True)
    _write_overlay(image, layout, annotated_piles).save(overlay_path, optimize=True)
    _write_slot_sheet(image, layout, annotated_piles).save(sheet_path, optimize=True)
    label_payload = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "case_id": clean_case_id,
        "label_scope": normalized_label_scope,
        "image": {
            "path": "frame.png",
            "width": image.width,
            "height": image.height,
            "resolution": resolution_name,
        },
        "scene": "in_match",
        "riichi_players": list(riichi_players or []),
        "discard_piles": annotated_piles,
        "layout": {
            "discard_slots": [slot.to_dict() for slots in layout.values() for slot in slots],
        },
    }
    write_json_atomic(label_path, label_payload)
    return {
        "ok": True,
        "image_path": str(image_path),
        "resolution": list(resolution),
        "case_id": clean_case_id,
        "case_dir": str(case_dir),
        "frame_path": str(frame_path),
        "overlay_path": str(overlay_path),
        "sheet_path": str(sheet_path),
        "label_path": str(label_path),
        "write_fixture": write_fixture,
        "label_scope": normalized_label_scope,
        "discard_count": sum(len(items) for items in annotated_piles.values()),
        "riichi_players": list(riichi_players or []),
    }


def _write_overlay(
    image: Image.Image,
    layout: dict[str, list[DiscardSlot]],
    annotated_piles: dict[str, list[dict[str, Any]]],
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    annotated = {
        (player, int(item.get("turn_index", 0) or 0)): str(item.get("tile", ""))
        for player, pile in annotated_piles.items()
        for item in pile
    }
    annotated_items = {
        (player, int(item.get("turn_index", 0) or 0)): item
        for player, pile in annotated_piles.items()
        for item in pile
    }
    for player, slots in layout.items():
        color = PLAYER_COLORS.get(player, (255, 255, 255))
        prefix = PLAYER_PREFIX.get(player, player)
        for slot in slots:
            box = slot.box
            key = (player, slot.turn_index)
            width = 5 if key in annotated else 2
            item = annotated_items.get(key)
            raw_points = item.get("quad") if isinstance(item, dict) else None
            points = _quad_points(raw_points) or list(slot.corners)
            draw.line(points + [points[0]], fill=color, width=width)
            label = f"{prefix}:{slot.turn_index}"
            if key in annotated:
                label = f"{label}={annotated[key]}"
            draw.text((box.left + 2, box.top + 2), label, fill=color)
    return overlay


def _write_slot_sheet(
    image: Image.Image,
    layout: dict[str, list[DiscardSlot]],
    annotated_piles: dict[str, list[dict[str, Any]]],
) -> Image.Image:
    cell_width = 82
    cell_height = 112
    columns = 18
    rows = len(PLAYER_PREFIX)
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), color=(22, 28, 34))
    draw = ImageDraw.Draw(sheet)
    annotated = {
        (player, int(item.get("turn_index", 0) or 0)): str(item.get("tile", ""))
        for player, pile in annotated_piles.items()
        for item in pile
    }

    for row, player in enumerate(PLAYER_PREFIX):
        color = PLAYER_COLORS[player]
        prefix = PLAYER_PREFIX[player]
        for slot in layout.get(player, []):
            column = slot.turn_index - 1
            cell_left = column * cell_width
            cell_top = row * cell_height
            key = (player, slot.turn_index)
            label = f"{prefix}:{slot.turn_index}"
            if key in annotated:
                label = f"{label}={annotated[key]}"
            draw.rectangle(
                (cell_left, cell_top, cell_left + cell_width - 1, cell_top + cell_height - 1),
                outline=color,
                width=3 if key in annotated else 1,
            )
            draw.text((cell_left + 4, cell_top + 3), label, fill=color)
            crop = crop_discard_slot(image, slot)
            crop.thumbnail((cell_width - 14, cell_height - 28))
            paste_left = cell_left + (cell_width - crop.width) // 2
            paste_top = cell_top + 23 + (cell_height - 28 - crop.height) // 2
            sheet.paste(crop, (paste_left, paste_top))
    return sheet


def _annotated_piles(
    discard_items: list[dict[str, Any]],
    layout: dict[str, list[DiscardSlot]],
    *,
    image: Image.Image | None = None,
) -> dict[str, list[dict[str, Any]]]:
    slots_by_key = {
        (slot.player, slot.turn_index): slot
        for slots in layout.values()
        for slot in slots
    }
    piles: dict[str, list[dict[str, Any]]] = {}
    for item in discard_items:
        player = str(item.get("player", "")).strip()
        turn_index = int(item.get("turn_index", 0) or 0)
        tile = str(item.get("tile", "")).strip()
        slot = slots_by_key.get((player, turn_index))
        if slot is None or not tile:
            continue
        refinement = refine_discard_slot_quad(image, slot) if image is not None else None
        quad = refinement.quad if refinement is not None else slot.corners
        bbox = refinement.bbox if refinement is not None else slot.bbox
        quad_source = "refined_tile_surface" if refinement is not None else "layout_slot"
        payload = {
            "tile": tile,
            "turn_index": turn_index,
            "bbox": bbox,
            "quad": [[x, y] for x, y in quad],
            "orientation": slot.orientation,
            "quad_source": quad_source,
        }
        if refinement is not None:
            payload["quad_confidence"] = refinement.confidence
        piles.setdefault(player, []).append(
            payload
        )
    return piles


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Mahjong Soul discard recognition fixture or overlay.")
    parser.add_argument("--image", required=True, help="Source screenshot path.")
    parser.add_argument("--case-id", default="", help="Fixture/artifact case id; defaults to image stem.")
    parser.add_argument(
        "--discard",
        action="append",
        default=[],
        help="Discard label as player:turn_index:tile, e.g. self:5:5z. Can be repeated.",
    )
    parser.add_argument(
        "--riichi-players",
        default="",
        help="Comma/space separated riichi player keys, e.g. right_opponent.",
    )
    parser.add_argument("--output-dir", default="", help="Overlay output root when --write-fixture is not set.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--write-fixture", action="store_true", help="Write under eval/discard_recognition.")
    parser.add_argument("--label-scope", choices=sorted(LABEL_SCOPES), default="partial")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def _normalize_label_scope(value: str) -> str:
    scope = str(value or "partial").strip().lower().replace("-", "_")
    if scope not in LABEL_SCOPES:
        raise ValueError(f"label_scope must be one of {sorted(LABEL_SCOPES)}")
    return scope


def _parse_discard_spec(value: str) -> dict[str, Any]:
    parts = str(value).split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("discard must be player:turn_index:tile")
    player = parts[0].strip()
    if player not in PLAYER_PREFIX:
        raise argparse.ArgumentTypeError(f"unknown discard player: {player}")
    try:
        turn_index = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("discard turn_index must be an integer") from exc
    if turn_index < 1 or turn_index > 18:
        raise argparse.ArgumentTypeError("discard turn_index must be between 1 and 18")
    tile = _normalize_tile(parts[2])
    if not tile:
        raise argparse.ArgumentTypeError(f"invalid discard tile: {parts[2]}")
    return {"player": player, "turn_index": turn_index, "tile": tile}


def _parse_player_list(value: str) -> list[str]:
    players: list[str] = []
    for item in str(value).replace(",", " ").split():
        player = item.strip()
        if player in PLAYER_PREFIX and player not in players:
            players.append(player)
    return players


def _normalize_tile(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    upper = text.upper()
    if upper in HONOR_ALIASES:
        return HONOR_ALIASES[upper]
    lower = text.lower()
    if len(lower) == 2 and lower[0].isdigit() and lower[1] in {"m", "p", "s", "z"}:
        if lower[0] == "0" and lower[1] in {"m", "p", "s"}:
            return f"R5{lower[1]}"
        if lower[0] in "123456789":
            return lower
    if len(lower) == 3 and lower[0] == "r" and lower[1] == "5" and lower[2] in {"m", "p", "s"}:
        return f"R5{lower[2]}"
    return ""


def _quad_points(value: Any) -> list[tuple[int, int]]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return []
    points: list[tuple[int, int]] = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            return []
        try:
            x, y = [int(part) for part in point]
        except (TypeError, ValueError):
            return []
        points.append((x, y))
    return points


def _clean_case_id(value: str) -> str:
    clean = str(value or "").strip().lower().replace("_", "-")
    if not clean:
        raise ValueError("case_id is required")
    clean = "".join(char if char.isalnum() or char == "-" else "-" for char in clean)
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-")


if __name__ == "__main__":
    raise SystemExit(main())
