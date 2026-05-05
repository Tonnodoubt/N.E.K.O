from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..storage import load_json_payload, write_json_atomic


TEMPLATE_SCHEMA_VERSION = "mahjong-button-templates-v1"
LABEL_SCHEMA_VERSION = "mahjong-button-localization-label-v1"
DEFAULT_TEMPLATE_DIR = Path("plugin/plugins/mahjong_companion/perception/templates")
DEFAULT_EVAL_DIR = Path("plugin/tests/data/mahjong_companion/eval")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare_button_template(
        image_path=Path(args.image),
        button_type=args.button_type,
        bbox=_parse_bbox(args.bbox),
        template_dir=Path(args.template_dir),
        template_id=args.template_id or args.button_type,
        match_threshold=float(args.match_threshold),
        padding=int(args.padding),
        overwrite=bool(args.overwrite),
        write_fixture=bool(args.write_fixture),
        eval_dir=Path(args.eval_dir),
        fixture_case_id=args.fixture_case_id,
        fixture_buttons=[_parse_fixture_button(item) for item in args.fixture_button],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


def prepare_button_template(
    *,
    image_path: Path,
    button_type: str,
    bbox: tuple[int, int, int, int],
    template_dir: Path = DEFAULT_TEMPLATE_DIR,
    template_id: str | None = None,
    match_threshold: float = 0.90,
    padding: int = 0,
    overwrite: bool = False,
    write_fixture: bool = False,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    fixture_case_id: str = "",
    fixture_buttons: list[tuple[str, tuple[int, int, int, int]]] | None = None,
) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    clean_button_type = _clean_identifier(button_type, field_name="button_type")
    clean_template_id = _clean_identifier(template_id or clean_button_type, field_name="template_id")
    threshold = _validate_threshold(match_threshold)

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    resolution = (image.width, image.height)
    resolution_name = f"{resolution[0]}x{resolution[1]}"
    crop_box = _expand_bbox(_validate_bbox(bbox, image.size), image.size, padding=max(0, int(padding)))

    template_subdir = template_dir / resolution_name
    template_path = template_subdir / f"{clean_template_id}.png"
    if template_path.exists() and not overwrite:
        raise FileExistsError(f"template already exists: {template_path}")

    template_subdir.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box).save(template_path)
    meta_path = _update_template_meta(
        template_dir,
        template_id=clean_template_id,
        button_type=clean_button_type,
        template_path=template_path,
        resolution=resolution,
        match_threshold=threshold,
    )

    fixture_path = ""
    fixture_label_path = ""
    fixture_label_buttons: list[dict[str, Any]] = []
    if write_fixture:
        fixture_case = fixture_case_id.strip() if fixture_case_id.strip() else clean_template_id
        fixture_path, fixture_label_path, fixture_label_buttons = _write_fixture_case(
            image,
            eval_dir=eval_dir,
            resolution_name=resolution_name,
            case_id=fixture_case,
            primary_button=(clean_button_type, crop_box),
            fixture_buttons=fixture_buttons or [],
            overwrite=overwrite,
        )

    return {
        "ok": True,
        "image_path": str(image_path),
        "resolution": list(resolution),
        "button_type": clean_button_type,
        "template_id": clean_template_id,
        "template_path": str(template_path),
        "template_bbox": list(crop_box),
        "match_threshold": threshold,
        "meta_path": str(meta_path),
        "fixture_path": fixture_path,
        "fixture_label_path": fixture_label_path,
        "fixture_buttons": fixture_label_buttons,
    }


def _update_template_meta(
    template_dir: Path,
    *,
    template_id: str,
    button_type: str,
    template_path: Path,
    resolution: tuple[int, int],
    match_threshold: float,
) -> Path:
    meta_path = template_dir / "meta.json"
    payload = load_json_payload(meta_path, default={}, expected_type=dict)
    if not payload:
        payload = {
            "schema_version": TEMPLATE_SCHEMA_VERSION,
            "default_match_threshold": 0.86,
            "templates": {},
        }
    payload.setdefault("schema_version", TEMPLATE_SCHEMA_VERSION)
    payload.setdefault("default_match_threshold", 0.86)
    templates = payload.get("templates")
    if not isinstance(templates, dict):
        templates = {}
        payload["templates"] = templates

    relative_template_path = template_path.relative_to(template_dir)
    templates[template_id] = {
        "button_type": button_type,
        "file": relative_template_path.as_posix(),
        "resolution": [resolution[0], resolution[1]],
        "match_threshold": match_threshold,
    }
    write_json_atomic(meta_path, payload)
    return meta_path


def _write_fixture_case(
    image: Image.Image,
    *,
    eval_dir: Path,
    resolution_name: str,
    case_id: str,
    primary_button: tuple[str, tuple[int, int, int, int]],
    fixture_buttons: list[tuple[str, tuple[int, int, int, int]]],
    overwrite: bool,
) -> tuple[str, str, list[dict[str, Any]]]:
    clean_case_id = _clean_case_id(case_id)
    case_dir = eval_dir / "button_localization" / resolution_name / clean_case_id
    frame_path = case_dir / "frame.png"
    label_path = case_dir / "frame.label.json"
    if case_dir.exists() and any(case_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"fixture case already exists: {case_dir}")

    case_dir.mkdir(parents=True, exist_ok=True)
    image.save(frame_path)

    buttons = [_button_payload(primary_button[0], primary_button[1])]
    for button_type, bbox in fixture_buttons:
        payload = _button_payload(button_type, _validate_bbox(bbox, image.size))
        if payload not in buttons:
            buttons.append(payload)

    label_payload = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "image": {
            "path": "frame.png",
        },
        "scene": "in_match",
        "buttons": buttons,
    }
    write_json_atomic(label_path, label_payload)
    return str(frame_path), str(label_path), buttons


def _button_payload(button_type: str, bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    return {
        "button_type": _clean_identifier(button_type, field_name="fixture_button_type"),
        "bbox": list(bbox),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop and register a Mahjong Soul button template.")
    parser.add_argument("--image", required=True, help="Source screenshot path.")
    parser.add_argument("--button-type", required=True, help="Button type, e.g. chi/pon/kan/riichi/ron/tsumo/skip.")
    parser.add_argument("--bbox", required=True, help="Template bbox as left,top,right,bottom.")
    parser.add_argument("--template-id", default="", help="Template id in meta.json; defaults to button type.")
    parser.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--match-threshold", type=float, default=0.90)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--fixture-case-id", default="")
    parser.add_argument(
        "--fixture-button",
        action="append",
        default=[],
        help="Additional expected button in fixture as button_type:left,top,right,bottom.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def _parse_bbox(value: str) -> tuple[int, int, int, int]:
    parts = [item.strip() for item in str(value).replace(";", ",").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must contain four integers: left,top,right,bottom")
    try:
        return tuple(int(item) for item in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must contain four integers") from exc


def _parse_fixture_button(value: str) -> tuple[str, tuple[int, int, int, int]]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("fixture button must be button_type:left,top,right,bottom")
    button_type, raw_bbox = value.split(":", 1)
    return _clean_identifier(button_type, field_name="fixture_button_type"), _parse_bbox(raw_bbox)


def _validate_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = [int(value) for value in bbox]
    width, height = image_size
    if left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top:
        raise ValueError(f"bbox {bbox} is outside image bounds {image_size} or has non-positive size")
    return left, top, right, bottom


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    padding: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = image_size
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _validate_threshold(value: float) -> float:
    threshold = float(value)
    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError("match_threshold must be in the range (0, 1]")
    return round(threshold, 4)


def _clean_identifier(value: str, *, field_name: str) -> str:
    clean = str(value or "").strip().lower().replace("-", "_")
    if not clean:
        raise ValueError(f"{field_name} is required")
    if not all(char.isalnum() or char == "_" for char in clean):
        raise ValueError(f"{field_name} must contain only letters, numbers, and underscores")
    return clean


def _clean_case_id(value: str) -> str:
    clean = str(value or "").strip().lower().replace("_", "-")
    if not clean:
        raise ValueError("fixture_case_id is required")
    if "/" in clean or "\\" in clean or clean in {".", ".."}:
        raise ValueError("fixture_case_id must be a single path segment")
    if not all(char.isalnum() or char in {"-", "."} for char in clean):
        raise ValueError("fixture_case_id must contain only letters, numbers, dots, and hyphens")
    return clean


if __name__ == "__main__":
    raise SystemExit(main())
