from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from PIL import Image

from ..perception.discard_layout import build_discard_layout
from ..perception.discard_quad_finder import refine_discard_slot_quad
from ..storage import load_json_payload, write_json_atomic


DEFAULT_OUTPUT_DIR = Path("plugin/plugins/mahjong_companion/plans/artifacts/discard_recognition_dataset")
LABEL_SCOPES = {"full", "partial"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = export_discard_recognition_dataset(
        label_root=Path(args.label_root),
        output_dir=Path(args.output_dir),
        split=args.split,
        copy_images=not bool(args.no_copy_images),
        refine_quads=bool(args.refine_quads),
        limit=int(args.limit),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result["ok"] else 1


def export_discard_recognition_dataset(
    *,
    label_root: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    split: str = "train",
    copy_images: bool = True,
    refine_quads: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    label_paths = _discover_label_paths(label_root, limit=limit)
    if not label_paths:
        raise FileNotFoundError(f"no *.label.json files found under {label_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    used_image_names: set[str] = set()
    for index, label_path in enumerate(label_paths, start=1):
        try:
            record = _record_from_label(
                label_path,
                images_dir=images_dir,
                copy_images=copy_images,
                refine_quads=refine_quads,
                used_image_names=used_image_names,
                index=index,
            )
        except Exception as exc:
            errors.append(f"{label_path}: {exc}")
            continue
        records.append(record)

    annotations_path = output_dir / f"{split}.jsonl"
    annotations_path.write_text(
        "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "ok": not errors,
        "schema_version": "mahjong-discard-recognition-dataset-v1",
        "label_root": str(label_root),
        "output_dir": str(output_dir),
        "split": split,
        "copy_images": copy_images,
        "refine_quads": refine_quads,
        "label_count": len(label_paths),
        "image_count": len(records),
        "detection_count": sum(len(record["detections"]) for record in records),
        "label_scope_counts": _label_scope_counts(records),
        "annotations_path": str(annotations_path),
        "images_dir": str(images_dir) if copy_images else "",
        "errors": errors,
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _record_from_label(
    label_path: Path,
    *,
    images_dir: Path,
    copy_images: bool,
    refine_quads: bool,
    used_image_names: set[str],
    index: int,
) -> dict[str, Any]:
    payload = load_json_payload(label_path, default={}, expected_type=dict)
    if not payload:
        raise ValueError("empty or invalid label payload")

    image_path = _resolve_label_image_path(label_path, payload)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    case_id = _clean_case_id(str(payload.get("case_id") or label_path.parent.name or label_path.stem))
    exported_image = _export_image_path(
        image_path,
        case_id=case_id,
        images_dir=images_dir,
        copy_images=copy_images,
        used_image_names=used_image_names,
        index=index,
    )
    if copy_images:
        shutil.copy2(image_path, exported_image)

    detections = _detections_from_payload(payload)
    if refine_quads:
        detections = [_with_refined_quad(image, detection) for detection in detections]

    return {
        "case_id": case_id,
        "label_scope": _label_scope_from_payload(payload),
        "label_path": str(label_path),
        "source_image_path": str(image_path),
        "image": str(exported_image.relative_to(images_dir.parent)) if copy_images else str(exported_image),
        "width": image.width,
        "height": image.height,
        "scene": str(payload.get("scene", "in_match") or "in_match"),
        "detections": detections,
    }


def _label_scope_from_payload(payload: dict[str, Any]) -> str:
    scope = str(payload.get("label_scope") or payload.get("discard_label_scope") or "full")
    scope = scope.strip().lower().replace("-", "_")
    return scope if scope in LABEL_SCOPES else "full"


def _label_scope_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {scope: 0 for scope in sorted(LABEL_SCOPES)}
    for record in records:
        scope = str(record.get("label_scope") or "full")
        counts[scope if scope in LABEL_SCOPES else "full"] += 1
    return counts


def _export_image_path(
    image_path: Path,
    *,
    case_id: str,
    images_dir: Path,
    copy_images: bool,
    used_image_names: set[str],
    index: int,
) -> Path:
    if not copy_images:
        return image_path

    suffix = image_path.suffix.lower() or ".png"
    base_name = f"{case_id}{suffix}"
    if base_name in used_image_names:
        base_name = f"{case_id}-{index:04d}{suffix}"
    used_image_names.add(base_name)
    return images_dir / base_name


def _detections_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_piles = payload.get("discard_piles")
    if not isinstance(raw_piles, dict):
        layout = payload.get("layout")
        raw_piles = layout.get("discard_piles") if isinstance(layout, dict) else None
    if not isinstance(raw_piles, dict):
        return []

    detections: list[dict[str, Any]] = []
    for player, raw_items in raw_piles.items():
        player_key = str(player).strip()
        if not player_key or not isinstance(raw_items, list):
            continue
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                continue
            tile = str(item.get("tile", "")).strip()
            if not tile:
                continue
            detection = {
                "player": str(item.get("player") or player_key),
                "turn_index": _positive_int(item.get("turn_index"), default=index),
                "tile": tile,
                "bbox": _bbox_from_payload(item),
                "quad": _quad_from_payload(item),
                "orientation": str(item.get("orientation", "")).strip() or _orientation_from_player(player_key),
                "confidence": _float_value(item.get("confidence"), default=1.0),
                "source": str(item.get("source", "")).strip() or "label",
                "quad_source": str(item.get("quad_source", "")).strip() or "label",
            }
            if detection["bbox"] is None and detection["quad"]:
                detection["bbox"] = _bbox_from_quad(detection["quad"])
            if detection["quad"] is None and detection["bbox"]:
                detection["quad"] = _quad_from_bbox(detection["bbox"])
                detection["quad_source"] = "bbox_fallback"
            if detection["bbox"] and detection["quad"]:
                detections.append(detection)
    return detections


def _with_refined_quad(image: Image.Image, detection: dict[str, Any]) -> dict[str, Any]:
    player = str(detection.get("player", "")).strip()
    turn_index = _positive_int(detection.get("turn_index"), default=0)
    layout = build_discard_layout(*image.size)
    slot = next(
        (
            candidate
            for candidate in layout.get(player, [])
            if candidate.turn_index == turn_index
        ),
        None,
    )
    if slot is None:
        return detection
    refinement = refine_discard_slot_quad(image, slot)
    if refinement is None:
        return detection

    updated = dict(detection)
    updated["bbox"] = refinement.bbox
    updated["quad"] = [[x, y] for x, y in refinement.quad]
    updated["quad_source"] = "refined_tile_surface"
    updated["quad_confidence"] = refinement.confidence
    return updated


def _discover_label_paths(label_root: Path, *, limit: int) -> list[Path]:
    paths = [label_root] if label_root.is_file() else sorted(label_root.rglob("*.label.json"))
    paths = [path for path in paths if path.is_file() and path.name.endswith(".label.json")]
    if limit > 0:
        return paths[:limit]
    return paths


def _resolve_label_image_path(label_path: Path, payload: dict[str, Any]) -> Path:
    image = payload.get("image")
    if isinstance(image, dict):
        raw_path = image.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            candidate = Path(raw_path)
            resolved = candidate if candidate.is_absolute() else label_path.parent / candidate
            if resolved.exists():
                return resolved
    stem = label_path.name.removesuffix(".label.json")
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = label_path.with_name(f"{stem}{suffix}")
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"label image not found for {label_path}")


def _bbox_from_payload(payload: dict[str, Any]) -> list[int] | None:
    bbox = payload.get("bbox")
    if isinstance(bbox, list | tuple) and len(bbox) == 4:
        try:
            left, top, right, bottom = [int(value) for value in bbox]
        except (TypeError, ValueError):
            return None
        if right > left and bottom > top:
            return [left, top, right, bottom]
    box = payload.get("box")
    if isinstance(box, dict):
        try:
            left = int(box.get("left", 0) or 0)
            top = int(box.get("top", 0) or 0)
            width = int(box.get("width", 0) or 0)
            height = int(box.get("height", 0) or 0)
        except (TypeError, ValueError):
            return None
        if width > 0 and height > 0:
            return [left, top, left + width, top + height]
    return None


def _quad_from_payload(payload: dict[str, Any]) -> list[list[int]] | None:
    quad = payload.get("quad")
    if not isinstance(quad, list | tuple) or len(quad) != 4:
        return None
    points: list[list[int]] = []
    for point in quad:
        if not isinstance(point, list | tuple) or len(point) != 2:
            return None
        try:
            x, y = [int(value) for value in point]
        except (TypeError, ValueError):
            return None
        points.append([x, y])
    return points


def _bbox_from_quad(quad: list[list[int]]) -> list[int]:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def _quad_from_bbox(bbox: list[int]) -> list[list[int]]:
    left, top, right, bottom = bbox
    return [[left, top], [left, bottom], [right, bottom], [right, top]]


def _orientation_from_player(player: str) -> str:
    if player == "self":
        return "bottom"
    if player == "top_opponent":
        return "top"
    if player == "left_opponent":
        return "left"
    if player == "right_opponent":
        return "right"
    return ""


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _float_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_case_id(value: str) -> str:
    clean = str(value or "").strip().lower().replace("_", "-")
    clean = "".join(char if char.isalnum() or char == "-" else "-" for char in clean)
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-") or "discard-case"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export discard labels for external recognizer training.")
    parser.add_argument("--label-root", required=True, help="Fixture/label directory or a single frame.label.json.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", default="train", help="JSONL split name, e.g. train/val.")
    parser.add_argument("--no-copy-images", action="store_true", help="Keep records pointing at source image paths.")
    parser.add_argument("--refine-quads", action="store_true", help="Re-fit visible tile faces before exporting quads.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
