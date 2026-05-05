from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

from ..perception.riichi_detector import detect_riichi_players
from ..storage import write_json_atomic


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_OUTPUT_DIR = Path("plugin/plugins/mahjong_companion/plans/artifacts/riichi_stick_review")
SCHEMA_VERSION = "mahjong-riichi-stick-review-v1"
BASE_WIDTH = 1920
BASE_HEIGHT = 1080
CENTER_CROP_BOX = (720, 270, 1165, 585)
PLAYER_COLORS = {
    "self": (255, 80, 80),
    "left_opponent": (255, 210, 60),
    "top_opponent": (80, 220, 255),
    "right_opponent": (120, 255, 120),
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare_riichi_stick_review(
        image_path=Path(args.image) if args.image else None,
        input_dir=Path(args.input_dir) if args.input_dir else None,
        output_dir=Path(args.output_dir),
        recursive=bool(args.recursive),
        limit=int(args.limit),
        columns=int(args.columns),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result["ok"] else 1


def prepare_riichi_stick_review(
    *,
    image_path: Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    recursive: bool = False,
    limit: int = 0,
    columns: int = 4,
) -> dict[str, Any]:
    image_paths = _resolve_image_paths(image_path=image_path, input_dir=input_dir, recursive=recursive, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in image_paths:
        try:
            cases.append(_prepare_case(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    sheet_path = output_dir / "riichi-stick-candidates.png"
    _write_contact_sheet(cases, sheet_path=sheet_path, columns=columns)
    summary = {
        "ok": not errors,
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "image_count": len(image_paths),
        "case_count": len(cases),
        "detected_case_count": sum(1 for case in cases if case["riichi_players"]),
        "detection_count": sum(len(case["detections"]) for case in cases),
        "counts_by_player": _counts_by_player(cases),
        "sheet_path": str(sheet_path),
        "cases": cases,
        "errors": errors,
    }
    review_json_path = output_dir / "riichi-stick-review.json"
    review_markdown_path = output_dir / "riichi-stick-review.md"
    write_json_atomic(review_json_path, summary)
    review_markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    summary["review_json_path"] = str(review_json_path)
    summary["review_markdown_path"] = str(review_markdown_path)
    return summary


def _prepare_case(image_path: Path) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    players, detections = detect_riichi_players(image)
    return {
        "case_id": _clean_case_id(image_path.stem),
        "image_path": str(image_path),
        "image": {"width": image.width, "height": image.height},
        "riichi_players": players,
        "detections": detections,
    }


def _write_contact_sheet(cases: list[dict[str, Any]], *, sheet_path: Path, columns: int) -> None:
    columns = max(1, int(columns))
    cell_width = 330
    cell_height = 250
    rows = max(1, (len(cases) + columns - 1) // columns)
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), color=(22, 28, 34))
    draw = ImageDraw.Draw(sheet)

    for index, case in enumerate(cases):
        column = index % columns
        row = index // columns
        left = column * cell_width
        top = row * cell_height
        crop = _case_center_crop(case)
        _draw_detections_on_crop(crop, case)
        crop.thumbnail((cell_width, cell_height - 28))
        sheet.paste(crop, (left + (cell_width - crop.width) // 2, top))
        label = _case_label(case)
        draw.rectangle((left, top + cell_height - 28, left + cell_width - 1, top + cell_height - 1), fill=(10, 16, 20))
        draw.text((left + 5, top + cell_height - 22), label, fill=(200, 230, 255))

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, optimize=True)


def _case_center_crop(case: dict[str, Any]) -> Image.Image:
    image_path = Path(str(case.get("image_path", "")))
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    box = _scaled_center_crop_box(image.size)
    case["review_crop_bbox"] = list(box)
    return image.crop(box)


def _draw_detections_on_crop(crop: Image.Image, case: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(crop)
    crop_bbox = case.get("review_crop_bbox")
    if not isinstance(crop_bbox, list) or len(crop_bbox) != 4:
        return
    crop_left, crop_top, _, _ = [int(part) for part in crop_bbox]
    for detection in case.get("detections", []):
        if not isinstance(detection, dict):
            continue
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        player = str(detection.get("player", "")).strip()
        color = PLAYER_COLORS.get(player, (255, 255, 255))
        local_bbox = [
            int(bbox[0]) - crop_left,
            int(bbox[1]) - crop_top,
            int(bbox[2]) - crop_left,
            int(bbox[3]) - crop_top,
        ]
        draw.rectangle(local_bbox, outline=color, width=4)
        label = f"{player}:{float(detection.get('confidence', 0.0) or 0.0):.2f}"
        label_top = max(0, local_bbox[1] - 18)
        draw.text((local_bbox[0], label_top), label, fill=color)


def _scaled_center_crop_box(image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    scale_x = width / BASE_WIDTH
    scale_y = height / BASE_HEIGHT
    left, top, right, bottom = CENTER_CROP_BOX
    return (
        _clamp_int(round(left * scale_x), 0, width - 1),
        _clamp_int(round(top * scale_y), 0, height - 1),
        _clamp_int(round(right * scale_x), 1, width),
        _clamp_int(round(bottom * scale_y), 1, height),
    )


def _resolve_image_paths(
    *,
    image_path: Path | None,
    input_dir: Path | None,
    recursive: bool,
    limit: int,
) -> list[Path]:
    if image_path is None and input_dir is None:
        raise ValueError("either image_path or input_dir is required")
    paths: list[Path] = []
    if image_path is not None:
        paths.append(image_path)
    if input_dir is not None:
        iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
        paths.extend(path for path in sorted(iterator) if path.suffix.lower() in IMAGE_EXTENSIONS)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if limit > 0:
        unique = unique[:limit]
    return unique


def _counts_by_player(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        for player in case.get("riichi_players", []):
            player_key = str(player).strip()
            if player_key:
                counts[player_key] = counts.get(player_key, 0) + 1
    return counts


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Riichi Stick Review",
        "",
        f"- image_count: `{summary['image_count']}`",
        f"- detected_case_count: `{summary['detected_case_count']}`",
        f"- detection_count: `{summary['detection_count']}`",
        f"- sheet: `{summary['sheet_path']}`",
        "",
        "## Counts By Player",
        "",
    ]
    for player, count in sorted(summary["counts_by_player"].items()):
        lines.append(f"- `{player}`: {count}")
    lines.extend(["", "## Cases", ""])
    for case in summary["cases"]:
        players = ", ".join(case["riichi_players"]) if case["riichi_players"] else "none"
        lines.append(f"- `{case['case_id']}`: {players}")
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
    return "\n".join(lines) + "\n"


def _case_label(case: dict[str, Any]) -> str:
    players = ",".join(case.get("riichi_players", [])) or "none"
    return f"{case.get('case_id', 'case')} {players}"


def _clean_case_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "case"


def _clamp_int(value: int | float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare visual review artifacts for riichi-stick detection.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", default="", help="Single image path to review.")
    source.add_argument("--input-dir", default="", help="Directory of screenshots to review.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of images to process.")
    parser.add_argument("--columns", type=int, default=4, help="Contact sheet columns.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
