from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..storage import load_json_payload
from .prepare_discard_fixture import (
    DEFAULT_EVAL_DIR,
    LABEL_SCOPES,
    _parse_discard_spec,
    _parse_player_list,
    prepare_discard_fixture,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = apply_discard_confirmations(
        label_path=Path(args.label),
        confirmations=[_parse_discard_spec(item) for item in args.confirm],
        riichi_players=_parse_player_list(args.riichi_players),
        eval_dir=Path(args.eval_dir),
        case_id=args.case_id,
        label_scope=args.label_scope,
        clear_existing=bool(args.clear_existing),
        update_source_label=not bool(args.no_update_source_label),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


def apply_discard_confirmations(
    *,
    label_path: Path,
    confirmations: list[dict[str, Any]],
    riichi_players: list[str] | None = None,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    case_id: str = "",
    label_scope: str = "",
    clear_existing: bool = False,
    update_source_label: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    payload = load_json_payload(label_path, default={}, expected_type=dict)
    if not payload:
        raise ValueError(f"empty or invalid label JSON: {label_path}")

    image_path = _resolve_label_image_path(label_path, payload)
    existing_items = [] if clear_existing else _items_from_discard_piles(payload.get("discard_piles"))
    merged_items = _merge_confirmations(existing_items, confirmations)
    active_riichi_players = list(riichi_players or _text_list(payload.get("riichi_players")))
    active_label_scope = label_scope or str(payload.get("label_scope") or "partial")

    if update_source_label:
        source_update = prepare_discard_fixture(
            image_path=image_path,
            case_id=str(payload.get("case_id") or label_path.parent.name),
            discard_items=merged_items,
            riichi_players=active_riichi_players,
            output_dir=label_path.parent.parent.parent,
            label_scope=active_label_scope,
            overwrite=True,
        )
    else:
        source_update = {}

    fixture = prepare_discard_fixture(
        image_path=image_path,
        case_id=case_id or str(payload.get("case_id") or label_path.parent.name),
        discard_items=merged_items,
        riichi_players=active_riichi_players,
        eval_dir=eval_dir,
        write_fixture=True,
        label_scope=active_label_scope,
        overwrite=overwrite,
    )
    return {
        "ok": True,
        "label_path": str(label_path),
        "image_path": str(image_path),
        "confirmation_count": len(confirmations),
        "label_scope": fixture["label_scope"],
        "discard_count": len(merged_items),
        "riichi_players": active_riichi_players,
        "source_label_path": source_update.get("label_path", str(label_path)),
        "fixture_label_path": fixture["label_path"],
        "fixture_overlay_path": fixture["overlay_path"],
        "fixture_sheet_path": fixture["sheet_path"],
        "fixture_case_id": fixture["case_id"],
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply confirmed discard labels and write a v0.5 eval fixture.")
    parser.add_argument("--label", required=True, help="Label JSON from prepare_discard_fixture/batch.")
    parser.add_argument(
        "--confirm",
        action="append",
        default=[],
        help="Confirmed discard as player:turn_index:tile, e.g. self:5:5z. Can be repeated.",
    )
    parser.add_argument("--riichi-players", default="", help="Comma/space separated riichi player keys.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--case-id", default="")
    parser.add_argument("--label-scope", choices=sorted(LABEL_SCOPES), default="")
    parser.add_argument("--clear-existing", action="store_true")
    parser.add_argument("--no-update-source-label", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def _resolve_label_image_path(label_path: Path, payload: dict[str, Any]) -> Path:
    image = payload.get("image")
    if not isinstance(image, dict):
        raise ValueError(f"label JSON has no image payload: {label_path}")
    raw_path = image.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"label JSON has no image.path: {label_path}")
    candidate = Path(raw_path)
    image_path = candidate if candidate.is_absolute() else label_path.parent / candidate
    if not image_path.exists():
        raise FileNotFoundError(f"label image not found: {image_path}")
    with Image.open(image_path) as opened:
        opened.verify()
    return image_path


def _items_from_discard_piles(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    items: list[dict[str, Any]] = []
    for player, pile in value.items():
        if not isinstance(pile, list):
            continue
        for item in pile:
            if not isinstance(item, dict):
                continue
            tile = str(item.get("tile", "")).strip()
            if not tile:
                continue
            try:
                turn_index = int(item.get("turn_index", 0) or 0)
            except (TypeError, ValueError):
                continue
            items.append({"player": str(player), "turn_index": turn_index, "tile": tile})
    return items


def _merge_confirmations(
    existing_items: list[dict[str, Any]],
    confirmations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for item in existing_items + confirmations:
        player = str(item.get("player", "")).strip()
        tile = str(item.get("tile", "")).strip()
        try:
            turn_index = int(item.get("turn_index", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not player or not tile or turn_index <= 0:
            continue
        merged[(player, turn_index)] = {"player": player, "turn_index": turn_index, "tile": tile}
    return [merged[key] for key in sorted(merged, key=lambda item: (item[0], item[1]))]


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


if __name__ == "__main__":
    raise SystemExit(main())
