from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

from ..perception.discard_layout import DiscardSlot, build_discard_layout
from ..perception.discard_parser import crop_discard_quad, crop_discard_slot
from ..perception.discard_quad_finder import DiscardQuadRefinement, refine_discard_slot_quad
from ..perception.roi import build_default_rois, collect_region_metrics
from ..perception.tile_parser import parse_tiles_from_image
from ..storage import write_json_atomic


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "data" / "calibration"
DEFAULT_OUTPUT_DIR = Path("plugin/plugins/mahjong_companion/plans/artifacts/discard_quad_review")
SCHEMA_VERSION = "mahjong-discard-quad-review-v1"
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare_discard_quad_review(
        image_path=Path(args.image) if args.image else None,
        input_dir=Path(args.input_dir) if args.input_dir else None,
        output_dir=Path(args.output_dir),
        calibration_dir=Path(args.calibration_dir) if args.calibration_dir else _default_calibration_dir(),
        recursive=bool(args.recursive),
        limit=int(args.limit),
        min_confidence=float(args.min_confidence),
        include_empty_slots=bool(args.include_empty_slots),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result["ok"] else 1


def prepare_discard_quad_review(
    *,
    image_path: Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    calibration_dir: Path | None = None,
    recursive: bool = False,
    limit: int = 0,
    min_confidence: float = 0.34,
    include_empty_slots: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    image_paths = _resolve_image_paths(image_path=image_path, input_dir=input_dir, recursive=recursive, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in image_paths:
        try:
            cases.append(
                _prepare_case(
                    path,
                    output_dir=output_dir,
                    calibration_dir=calibration_dir,
                    min_confidence=min_confidence,
                    include_empty_slots=include_empty_slots,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    summary = {
        "ok": not errors,
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "calibration_dir": str(calibration_dir) if calibration_dir is not None else "",
        "image_count": len(image_paths),
        "case_count": len(cases),
        "quad_count": sum(case["quad_count"] for case in cases),
        "accepted_count": sum(case["accepted_count"] for case in cases),
        "min_confidence": round(float(min_confidence), 3),
        "include_empty_slots": include_empty_slots,
        "cases": cases,
        "errors": errors,
    }
    summary_path = output_dir / "discard-quad-review.json"
    markdown_path = output_dir / "discard-quad-review.md"
    write_json_atomic(summary_path, summary)
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    summary["review_json_path"] = str(summary_path)
    summary["review_markdown_path"] = str(markdown_path)
    return summary


def _prepare_case(
    image_path: Path,
    *,
    output_dir: Path,
    calibration_dir: Path | None,
    min_confidence: float,
    include_empty_slots: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    case_id = _clean_case_id(image_path.stem)
    resolution_name = f"{image.width}x{image.height}"
    case_dir = output_dir / resolution_name / case_id
    if case_dir.exists() and any(case_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"quad review case already exists: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=True)

    frame_path = case_dir / "frame.png"
    overlay_path = case_dir / "frame.quad-overlay.png"
    sheet_path = case_dir / "frame.quad-slots.png"
    json_path = case_dir / "frame.quads.json"
    image.save(frame_path, optimize=True)

    layout = build_discard_layout(*image.size)
    parser_records = _parser_records_by_slot(image_path, image, calibration_dir=calibration_dir)
    records = _collect_quad_records(
        image,
        layout,
        parser_records=parser_records,
        min_confidence=min_confidence,
        include_empty_slots=include_empty_slots,
    )
    _write_overlay(image, layout, records).save(overlay_path, optimize=True)
    _write_slot_sheet(image, layout, records).save(sheet_path, optimize=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "image": {"path": "frame.png", "width": image.width, "height": image.height},
        "quad_order": "upper_left,lower_left,lower_right,upper_right",
        "min_confidence": round(float(min_confidence), 3),
        "quad_count": sum(1 for record in records if record.get("refined")),
        "accepted_count": sum(1 for record in records if record.get("accepted")),
        "slot_count": len(records),
        "slots": records,
    }
    write_json_atomic(json_path, payload)
    return {
        "case_id": case_id,
        "image_path": str(image_path),
        "frame_path": str(frame_path),
        "overlay_path": str(overlay_path),
        "sheet_path": str(sheet_path),
        "json_path": str(json_path),
        "quad_count": payload["quad_count"],
        "accepted_count": payload["accepted_count"],
        "slot_count": len(records),
        "quad_counts_by_player": _quad_counts_by_player(records),
        "accepted_counts_by_player": _accepted_counts_by_player(records),
    }


def _collect_quad_records(
    image: Image.Image,
    layout: dict[str, list[DiscardSlot]],
    *,
    parser_records: dict[str, dict[str, Any]],
    min_confidence: float,
    include_empty_slots: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for player, slots in layout.items():
        for slot in slots:
            refinement = refine_discard_slot_quad(image, slot, min_confidence=min_confidence)
            parser_record = parser_records.get(slot.slot_id, {})
            if refinement is None and not parser_record and not include_empty_slots:
                continue
            records.append(_slot_record(slot, refinement, parser_record=parser_record))
    return records


def _slot_record(
    slot: DiscardSlot,
    refinement: DiscardQuadRefinement | None,
    *,
    parser_record: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "slot_id": slot.slot_id,
        "player": slot.player,
        "turn_index": slot.turn_index,
        "orientation": slot.orientation,
        "layout_bbox": slot.bbox,
        "layout_quad": [[x, y] for x, y in slot.corners],
        "refined": refinement is not None,
    }
    if parser_record:
        base.update(
            {
                "occupied": bool(parser_record.get("occupied")),
                "candidate_tile": str(parser_record.get("candidate_tile", "")).strip(),
                "tile_confidence": _float_value(parser_record.get("confidence")),
                "template_distance": parser_record.get("template_distance"),
                "accepted": bool(parser_record.get("accepted")),
                "suppressed_duplicate": bool(parser_record.get("suppressed_duplicate")),
                "rejected_refinement_owner_slot_id": str(parser_record.get("rejected_refinement_owner_slot_id", "")).strip(),
                "parser_quad_source": str(parser_record.get("quad_source", "")).strip(),
            }
        )
    else:
        base["accepted"] = False
    if refinement is None:
        return base
    base.update(
        {
            "bbox": refinement.bbox,
            "quad": [[x, y] for x, y in refinement.quad],
            "confidence": refinement.confidence,
            "component_area": refinement.component_area,
            "search_box": refinement.search_box.to_dict(),
        }
    )
    return base


def _parser_records_by_slot(
    image_path: Path,
    image: Image.Image,
    *,
    calibration_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene="in_match",
        metrics=_frame_metrics(image),
        calibration_dir=calibration_dir,
        fixture_mode="disabled",
    )
    accepted_slot_ids = {
        str(item.get("slot_id", "")).strip()
        for pile in parsed.discard_piles.values()
        for item in pile
        if isinstance(item, dict)
    }
    records: dict[str, dict[str, Any]] = {}
    for detection in parsed.raw_detections:
        if not isinstance(detection, dict) or detection.get("group") != "discard":
            continue
        slot_id = str(detection.get("slot_id", "")).strip()
        if not slot_id:
            continue
        record = dict(detection)
        record["accepted"] = slot_id in accepted_slot_ids
        records[slot_id] = record
    return records


def _write_overlay(
    image: Image.Image,
    layout: dict[str, list[DiscardSlot]],
    records: list[dict[str, Any]],
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    records_by_slot = {str(record["slot_id"]): record for record in records}

    for player, slots in layout.items():
        color = PLAYER_COLORS.get(player, (255, 255, 255))
        prefix = PLAYER_PREFIX.get(player, player)
        for slot in slots:
            layout_points = list(slot.corners)
            draw.line(layout_points + [layout_points[0]], fill=(*color[:3],), width=1)
            record = records_by_slot.get(slot.slot_id)
            if not record or not record.get("refined"):
                continue
            search_box = record.get("search_box")
            if isinstance(search_box, dict):
                left = int(search_box.get("left", 0) or 0)
                top = int(search_box.get("top", 0) or 0)
                right = left + int(search_box.get("width", 0) or 0)
                bottom = top + int(search_box.get("height", 0) or 0)
                draw.rectangle((left, top, right, bottom), outline=(180, 180, 180), width=1)
            points = _quad_points(record.get("quad"))
            if points:
                draw.line(points + [points[0]], fill=color, width=4)
            confidence = float(record.get("confidence", 0.0) or 0.0)
            label = f"{prefix}:{slot.turn_index} {confidence:.2f}"
            label_x = points[0][0] if points else slot.box.left
            label_y = points[0][1] if points else slot.box.top
            draw.text((label_x + 2, label_y + 2), label, fill=color)
    return overlay


def _write_slot_sheet(
    image: Image.Image,
    layout: dict[str, list[DiscardSlot]],
    records: list[dict[str, Any]],
) -> Image.Image:
    cell_width = 82
    cell_height = 112
    columns = 18
    rows = len(PLAYER_PREFIX)
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), color=(22, 28, 34))
    draw = ImageDraw.Draw(sheet)
    records_by_slot = {str(record["slot_id"]): record for record in records}

    for row, player in enumerate(PLAYER_PREFIX):
        color = PLAYER_COLORS[player]
        prefix = PLAYER_PREFIX[player]
        for slot in layout.get(player, []):
            record = records_by_slot.get(slot.slot_id)
            column = slot.turn_index - 1
            cell_left = column * cell_width
            cell_top = row * cell_height
            refined = bool(record and record.get("refined"))
            accepted = bool(record and record.get("accepted"))
            tile = str(record.get("candidate_tile", "")).strip() if record else ""
            draw.rectangle(
                (cell_left, cell_top, cell_left + cell_width - 1, cell_top + cell_height - 1),
                outline=color if refined or accepted else (70, 78, 88),
                width=4 if accepted else (2 if refined else 1),
            )
            label = f"{prefix}:{slot.turn_index}"
            if tile:
                label += f"={tile}"
            elif refined:
                label += "*"
            if accepted:
                label += " ok"
            if refined:
                label += f" {float(record.get('confidence', 0.0) or 0.0):.2f}"
            elif tile:
                label += f" {float(record.get('tile_confidence', 0.0) or 0.0):.2f}"
            draw.text((cell_left + 4, cell_top + 3), label, fill=color if refined or tile else (110, 116, 124))

            crop = _record_crop(image, slot, record)
            crop.thumbnail((cell_width - 14, cell_height - 28))
            paste_left = cell_left + (cell_width - crop.width) // 2
            paste_top = cell_top + 23 + (cell_height - 28 - crop.height) // 2
            sheet.paste(crop, (paste_left, paste_top))
    return sheet


def _record_crop(image: Image.Image, slot: DiscardSlot, record: dict[str, Any] | None) -> Image.Image:
    if record and record.get("refined"):
        quad_points = _quad_points(record.get("quad"))
        bbox = record.get("bbox")
        if quad_points and isinstance(bbox, list | tuple) and len(bbox) == 4:
            width = max(1, int(bbox[2]) - int(bbox[0]))
            height = max(1, int(bbox[3]) - int(bbox[1]))
            return crop_discard_quad(
                image,
                (quad_points[0], quad_points[1], quad_points[2], quad_points[3]),
                output_size=(width, height),
                orientation=slot.orientation,
            )
    return crop_discard_slot(image, slot, refine=False)


def _resolve_image_paths(
    *,
    image_path: Path | None,
    input_dir: Path | None,
    recursive: bool,
    limit: int,
) -> list[Path]:
    if image_path is not None:
        return [image_path]
    if input_dir is None:
        raise ValueError("either --image or --input-dir is required")
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if limit > 0:
        return paths[:limit]
    return paths


def _quad_counts_by_player(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if not record.get("refined"):
            continue
        player = str(record.get("player", "")).strip()
        counts[player] = counts.get(player, 0) + 1
    return counts


def _accepted_counts_by_player(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if not record.get("accepted"):
            continue
        player = str(record.get("player", "")).strip()
        counts[player] = counts.get(player, 0) + 1
    return counts


def _frame_metrics(image: Image.Image) -> dict[str, dict[str, Any]]:
    rois = build_default_rois(*image.size)
    metrics = {name: collect_region_metrics(image, roi) for name, roi in rois.items()}
    metrics["full_frame"] = collect_region_metrics(image, None)
    return metrics


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Discard Quad Review",
        "",
        f"- image_count: {summary['image_count']}",
        f"- quad_count: {summary['quad_count']}",
        f"- accepted_count: {summary['accepted_count']}",
        f"- min_confidence: {summary['min_confidence']}",
        "",
    ]
    for case in summary["cases"]:
        lines.append(f"## {case['case_id']}")
        lines.append("")
        lines.append(f"- image: `{case['image_path']}`")
        lines.append(f"- overlay: `{case['overlay_path']}`")
        lines.append(f"- sheet: `{case['sheet_path']}`")
        lines.append(f"- json: `{case['json_path']}`")
        lines.append(f"- quads: {case['quad_count']} / {case['slot_count']}")
        lines.append(f"- accepted: {case['accepted_count']} / {case['slot_count']}")
        lines.append(f"- by_player: `{case['quad_counts_by_player']}`")
        lines.append(f"- accepted_by_player: `{case['accepted_counts_by_player']}`")
        lines.append("")
    if summary["errors"]:
        lines.append("## Errors")
        lines.append("")
        for error in summary["errors"]:
            lines.append(f"- {error}")
        lines.append("")
    return "\n".join(lines)


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
    clean = "".join(char if char.isalnum() or char == "-" else "-" for char in clean)
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-") or "discard-quad-review"


def _default_calibration_dir() -> Path | None:
    return DEFAULT_CALIBRATION_DIR if DEFAULT_CALIBRATION_DIR.exists() else None


def _float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fitted four-point discard quad review artifacts.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", default="", help="Single screenshot path.")
    source.add_argument("--input-dir", default="", help="Directory containing screenshots.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--calibration-dir", default="", help="Calibration directory for template recognizer overlay.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.34)
    parser.add_argument("--include-empty-slots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
