from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..perception.calibration import (
    CALIBRATION_LABEL_SCHEMA,
    CalibrationOffsets,
    build_default_calibration_profile,
    label_sidecar_path,
    iter_calibration_label_paths,
    resolve_calibration_profile,
    save_calibration_profile,
    train_calibration_profile,
    train_calibration_profile_from_paths,
    write_calibration_label,
)
from ..perception.discard_layout import build_discard_layout
from ..perception.hand_layout import build_hand_layout


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
HONOR_ALIASES = {
    "E": "1z",
    "S": "2z",
    "W": "3z",
    "N": "4z",
    "P": "5z",
    "F": "6z",
    "C": "7z",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    raw_dir = Path(args.raw_dir)
    labels_written: list[str] = []

    if args.hand_tiles and not args.image:
        raise SystemExit("--image is required when --hand-tiles is used outside --interactive")

    if args.image:
        image_path = Path(args.image)
        label_path = label_image(
            image_path,
            raw_dir=raw_dir,
            hand_tiles=_parse_tiles(args.hand_tiles),
            melds=_parse_groups(args.melds),
            dora_indicators=_parse_tiles(args.dora_indicators),
            scene=args.scene,
            client_version=args.client_version,
            calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
            hand_left=args.hand_left,
            hand_top=args.hand_top,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            tile_gap=args.tile_gap,
            draw_gap=args.draw_gap,
            discard_items=_parse_discard_specs(args.discard),
            riichi_players=_parse_csv_values(args.riichi_players),
        )
        labels_written.append(str(label_path))
    elif args.interactive:
        for image_path in _discover_unlabeled_images(raw_dir, limit=args.limit):
            label_path = _label_interactively(
                image_path,
                raw_dir=raw_dir,
                scene=args.scene,
                client_version=args.client_version,
                calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
            )
            if label_path is not None:
                labels_written.append(str(label_path))

    profile_written = ""
    if args.train_output:
        extra_roots = [Path(value) for value in args.train_extra_root]
        if extra_roots:
            label_paths = _dedupe_paths(
                [
                    *iter_calibration_label_paths(raw_dir),
                    *(path for root in extra_roots for path in iter_calibration_label_paths(root)),
                ]
            )
            profile = train_calibration_profile_from_paths(
                label_paths,
                label_root=raw_dir,
                client_version=args.client_version,
                min_samples=args.min_samples,
            )
        else:
            profile = train_calibration_profile(
                raw_dir,
                client_version=args.client_version,
                min_samples=args.min_samples,
            )
        output_path = Path(args.train_output)
        save_calibration_profile(profile, output_path)
        profile_written = str(output_path)

    summary = {
        "ok": True,
        "raw_dir": str(raw_dir),
        "train_extra_roots": [str(root) for root in args.train_extra_root],
        "labels_written": labels_written,
        "profile_written": profile_written,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


def label_image(
    image_path: Path,
    *,
    raw_dir: Path,
    hand_tiles: list[str],
    melds: list[list[str]] | None = None,
    dora_indicators: list[str] | None = None,
    scene: str = "in_match",
    client_version: str = "unknown",
    calibration_dir: Path | None = None,
    hand_left: int | None = None,
    hand_top: int | None = None,
    tile_width: int | None = None,
    tile_height: int | None = None,
    tile_gap: int | None = None,
    draw_gap: int | None = None,
    discard_items: list[dict[str, Any]] | None = None,
    riichi_players: list[str] | None = None,
) -> Path:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    with Image.open(image_path) as opened:
        width, height = opened.size

    default_profile = build_default_calibration_profile(width, height)
    default_layout = build_hand_layout(width, height, calibration=default_profile)
    default_first = default_layout["hand"][0].box

    active_profile = resolve_calibration_profile(width, height, calibration_dir=calibration_dir)
    active_layout = build_hand_layout(width, height, calibration=active_profile)
    active_first = active_layout["hand"][0].box
    active_second = active_layout["hand"][1].box
    active_thirteenth = active_layout["hand"][12].box
    active_fourteenth = active_layout["hand"][13].box

    effective_left = hand_left if hand_left is not None else active_first.left
    effective_top = hand_top if hand_top is not None else active_first.top
    effective_width = tile_width if tile_width is not None else active_first.width
    effective_height = tile_height if tile_height is not None else active_first.height
    current_gap = active_second.left - active_first.left - active_first.width
    current_draw_gap = active_fourteenth.left - active_thirteenth.left - active_thirteenth.width - current_gap
    effective_gap = tile_gap if tile_gap is not None else current_gap
    effective_draw_gap = draw_gap if draw_gap is not None else current_draw_gap
    hand_offsets = CalibrationOffsets(
        x_px=effective_left - default_first.left,
        y_px=effective_top - default_first.top,
        width_px=effective_width - default_first.width,
        height_px=effective_height - default_first.height,
        gap_px=effective_gap - int(effective_width * 0.12),
        draw_gap_px=max(0, effective_draw_gap),
    )
    label_profile = replace(
        active_profile,
        enabled=True,
        screen_width=width,
        screen_height=height,
        hand_offsets=hand_offsets,
    )
    corrected_layout = build_hand_layout(width, height, calibration=label_profile)
    discard_layout = build_discard_layout(width, height, calibration=label_profile)
    hand_slots = []
    for index, slot in enumerate(corrected_layout["hand"]):
        hand_slots.append(
            {
                "slot_id": slot.slot_id,
                "tile": hand_tiles[index] if index < len(hand_tiles) else "",
                "box": slot.box.to_dict(),
            }
        )
    discard_piles = _discard_piles_from_items(discard_items or [], discard_layout)

    payload = {
        "schema_version": CALIBRATION_LABEL_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "client": {
            "platform": "majsoul-pc",
            "version": client_version,
        },
        "image": {
            "path": _relative_or_absolute(image_path, raw_dir),
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
        },
        "scene": scene,
        "hand_tiles": list(hand_tiles),
        "melds": [list(group) for group in (melds or [])],
        "dora_indicators": list(dora_indicators or []),
        "riichi_players": list(riichi_players or []),
        "discard_piles": discard_piles,
        "layout": {
            "base_profile_id": active_profile.profile_id,
            "hand_offsets": hand_offsets.to_dict(),
            "hand_slots": hand_slots,
        },
    }
    path = label_sidecar_path(image_path)
    write_calibration_label(path, payload)
    return path


def _label_interactively(
    image_path: Path,
    *,
    raw_dir: Path,
    scene: str,
    client_version: str,
    calibration_dir: Path | None,
) -> Path | None:
    print(f"\nImage: {image_path}")
    with Image.open(image_path) as opened:
        width, height = opened.size
    active_profile = resolve_calibration_profile(width, height, calibration_dir=calibration_dir)
    active_layout = build_hand_layout(width, height, calibration=active_profile)
    first = active_layout["hand"][0].box
    print(f"Resolution: {width}x{height}")
    second = active_layout["hand"][1].box
    thirteenth = active_layout["hand"][12].box
    fourteenth = active_layout["hand"][13].box
    current_gap = second.left - first.left - first.width
    current_draw_gap = fourteenth.left - thirteenth.left - thirteenth.width - current_gap
    print(
        "Current hand origin: "
        f"left={first.left}, top={first.top}, tile={first.width}x{first.height}, "
        f"gap={current_gap}, draw_gap={current_draw_gap}",
    )
    raw_tiles = input("Hand tiles (space/comma separated, empty to skip): ").strip()
    if not raw_tiles:
        return None
    hand_left = _prompt_optional_int("Corrected hand left", first.left)
    hand_top = _prompt_optional_int("Corrected hand top", first.top)
    tile_width = _prompt_optional_int("Corrected tile width", first.width)
    tile_height = _prompt_optional_int("Corrected tile height", first.height)
    tile_gap = _prompt_optional_int("Corrected tile gap", current_gap)
    draw_gap = _prompt_optional_int("Corrected drawn-tile extra gap", current_draw_gap)
    dora = input("Dora indicators (optional): ").strip()
    melds = input("Meld groups, separated by ';' (optional): ").strip()
    return label_image(
        image_path,
        raw_dir=raw_dir,
        hand_tiles=_parse_tiles(raw_tiles),
        melds=_parse_groups(melds),
        dora_indicators=_parse_tiles(dora),
        scene=scene,
        client_version=client_version,
        calibration_dir=calibration_dir,
        hand_left=hand_left,
        hand_top=hand_top,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_gap=tile_gap,
        draw_gap=draw_gap,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label Mahjong Soul hand calibration frames.")
    parser.add_argument("--raw-dir", required=True, help="Calibration raw directory to scan or train from.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for labels for each unlabeled frame.")
    parser.add_argument("--image", help="Label one image non-interactively.")
    parser.add_argument("--hand-tiles", default="", help="Hand tiles for --image, e.g. '1m 2m 3m E R5s'.")
    parser.add_argument("--melds", default="", help="Meld groups separated by ';', e.g. '1m 2m 3m;5p 5p 5p'.")
    parser.add_argument("--dora-indicators", default="", help="Dora indicator tiles, space/comma separated.")
    parser.add_argument("--scene", default="in_match", help="Scene label to store in the sidecar.")
    parser.add_argument("--client-version", default="unknown", help="Mahjong Soul client version for trained profiles.")
    parser.add_argument("--calibration-dir", help="Existing calibration directory used to prefill slot boxes.")
    parser.add_argument("--hand-left", type=int, help="Corrected x coordinate for the first hand tile.")
    parser.add_argument("--hand-top", type=int, help="Corrected y coordinate for the first hand tile.")
    parser.add_argument("--tile-width", type=int, help="Corrected hand tile box width.")
    parser.add_argument("--tile-height", type=int, help="Corrected hand tile box height.")
    parser.add_argument("--tile-gap", type=int, help="Corrected gap between normal hand tiles.")
    parser.add_argument("--draw-gap", type=int, help="Extra gap before the drawn 14th tile.")
    parser.add_argument(
        "--discard",
        action="append",
        default=[],
        help="Discard label as player:turn_index:tile, e.g. self:1:5z. Can be repeated.",
    )
    parser.add_argument(
        "--riichi-players",
        default="",
        help="Comma/space separated riichi player keys, e.g. right_opponent.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum unlabeled frames to process in interactive mode.")
    parser.add_argument("--train-output", help="Write a trained profile JSON after labeling/training.")
    parser.add_argument(
        "--train-extra-root",
        action="append",
        default=[],
        help="Additional label root to include only when --train-output is set. Can be repeated.",
    )
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum labels required to mark a profile enabled.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON summary.")
    return parser.parse_args(argv)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def _discover_unlabeled_images(raw_dir: Path, *, limit: int = 0) -> list[Path]:
    images = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not label_sidecar_path(path).exists()
    )
    if limit > 0:
        return images[:limit]
    return images


def _parse_discard_specs(values: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for value in values:
        player, turn_index, tile = _parse_discard_spec(value)
        items.append({"player": player, "turn_index": turn_index, "tile": tile})
    return items


def _parse_discard_spec(value: str) -> tuple[str, int, str]:
    parts = str(value).split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("discard must be player:turn_index:tile")
    player = parts[0].strip()
    if player not in {"self", "left_opponent", "top_opponent", "right_opponent"}:
        raise argparse.ArgumentTypeError(f"unknown discard player: {player}")
    try:
        turn_index = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("discard turn_index must be an integer") from exc
    if turn_index < 1 or turn_index > 18:
        raise argparse.ArgumentTypeError("discard turn_index must be between 1 and 18")
    tile = _normalize_label_tile(parts[2])
    if not tile:
        raise argparse.ArgumentTypeError(f"invalid discard tile: {parts[2]}")
    return player, turn_index, tile


def _discard_piles_from_items(
    items: list[dict[str, Any]],
    layout: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    piles: dict[str, list[dict[str, Any]]] = {}
    slots_by_key = {
        (slot.player, slot.turn_index): slot
        for slots in layout.values()
        for slot in slots
    }
    for item in items:
        player = str(item.get("player", "")).strip()
        turn_index = int(item.get("turn_index", 0) or 0)
        slot = slots_by_key.get((player, turn_index))
        if slot is None:
            continue
        piles.setdefault(player, []).append(
            {
                "tile": str(item.get("tile", "")).strip(),
                "turn_index": turn_index,
                "bbox": slot.bbox,
                "quad": [[x, y] for x, y in slot.corners],
                "orientation": slot.orientation,
            }
        )
    return piles


def _parse_csv_values(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _parse_tiles(value: str) -> list[str]:
    raw = value.replace(",", " ").split()
    tiles = [_normalize_label_tile(item) for item in raw]
    return [tile for tile in tiles if tile]


def _parse_groups(value: str) -> list[list[str]]:
    groups: list[list[str]] = []
    for group in value.split(";"):
        tiles = _parse_tiles(group)
        if tiles:
            groups.append(tiles)
    return groups


def _normalize_label_tile(value: Any) -> str:
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


def _prompt_optional_int(label: str, current: int) -> int:
    raw = input(f"{label} [{current}]: ").strip()
    if not raw:
        return current
    return int(raw)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
