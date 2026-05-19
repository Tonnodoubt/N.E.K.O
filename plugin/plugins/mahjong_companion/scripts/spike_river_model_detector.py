from __future__ import annotations

import argparse
import io
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from plugin.plugins.mahjong_companion.perception.river_detector_v2 import (
    MIN_RIVER_TILE_CLASSIFICATION_CONFIDENCE,
    RIVER_PLAYERS,
    RiverTileCandidate,
    build_river_rois,
    detect_river_tiles_v2,
)
from plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch import (
    classify_tiles_batch,
    onnx_backend_available,
)


TILE_LABELS = {
    *(f"{rank}{suit}" for suit in ("m", "p", "s") for rank in range(1, 10)),
    *(f"{rank}z" for rank in range(1, 8)),
    "0m",
    "0p",
    "0s",
}
PLAYER_COLORS = {
    "self": (80, 220, 120),
    "left_opponent": (255, 210, 70),
    "top_opponent": (90, 180, 255),
    "right_opponent": (255, 120, 120),
}
EMPTY_TILE_LABEL = "empty"


@dataclass(frozen=True)
class ModelDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str = ""
    source: str = "model"

    @property
    def center(self) -> tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "label": self.label,
            "source": self.source,
        }


@dataclass(frozen=True)
class AssignedDetection:
    player: str
    order_index: int
    detection: ModelDetection
    tile: str = ""
    tile_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = self.detection.to_dict()
        payload.update(
            {
                "player": self.player,
                "order_index": self.order_index,
                "tile": self.tile,
                "tile_confidence": self.tile_confidence,
            }
        )
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spike external model detection against the current river_detector_v2 output.",
    )
    parser.add_argument("image", type=Path, nargs="?", help="Input Mahjong Soul screenshot.")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Process all *-frame.png screenshots in a directory.")
    parser.add_argument("--batch-glob", default="*-frame.png", help="Glob used inside --batch-dir.")
    parser.add_argument("--batch-limit", type=int, default=0, help="Maximum screenshots to process.")
    parser.add_argument("--out-dir", type=Path, default=Path("plugin/plugins/mahjong_companion/tests/_artifacts/river_model_spike"))
    parser.add_argument("--backend", choices=("roboflow", "ultralytics", "json"), default="json")
    parser.add_argument("--detector-json", type=Path, default=None, help="Offline detector JSON payload for --backend json.")
    parser.add_argument("--detector-json-dir", type=Path, default=None, help="Directory of per-image detector JSON payloads.")
    parser.add_argument("--roboflow-model", default="mahjong-baq4s-wclnm/1", help="Roboflow model/version slug.")
    parser.add_argument("--roboflow-api-key", default="", help="Roboflow API key. Defaults to ROBOFLOW_API_KEY.")
    parser.add_argument("--roboflow-cache-dir", type=Path, default=None, help="Cache Roboflow JSON responses by image stem.")
    parser.add_argument("--roboflow-max-size", type=int, default=1280, help="Longest image side sent to Roboflow.")
    parser.add_argument("--ultralytics-model", type=Path, default=None, help="Local YOLO/Ultralytics model path.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--overlap", type=float, default=0.30, help="Detector NMS overlap threshold when supported.")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu, mps, 0.")
    parser.add_argument("--classify-crops", action="store_true", help="Classify detected boxes with the existing tile classifier.")
    parser.add_argument("--fuse-v2-gaps", action="store_true", help="Use river_detector_v2 candidates to fill model detection gaps.")
    parser.add_argument("--fusion-iou", type=float, default=0.20, help="Overlap threshold for considering model/v2 detections the same tile.")
    parser.add_argument("--fusion-max-per-player", type=int, default=18, help="Maximum per-player target count for v2 gap filling.")
    parser.add_argument("--contact-limit", type=int, default=12, help="Maximum side-by-side previews in batch mode.")
    parser.add_argument(
        "--contact-sort",
        choices=("input", "model-count", "fallback-count", "v2-count"),
        default="model-count",
        help="Which frames to show in the contact sheet.",
    )
    args = parser.parse_args()

    image_paths = _image_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    contact_items = []
    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        error = ""
        try:
            detections = _load_model_detections(image_path, args)
            assigned = assign_detections_to_rivers(detections, image.size)
        except SystemExit as exc:
            error = _redact_secrets(str(exc))
            assigned = []
        except Exception as exc:  # pragma: no cover - depends on remote detector availability.
            error = _redact_secrets(f"{type(exc).__name__}: {exc}")
            assigned = []
        current = detect_river_tiles_v2(image)
        if args.fuse_v2_gaps:
            assigned = fuse_model_with_v2_gaps(
                assigned,
                current,
                iou_threshold=args.fusion_iou,
                max_per_player=args.fusion_max_per_player,
            )
        if args.classify_crops:
            assigned = classify_assigned_detections(image, assigned)
        row = _frame_summary(image_path, current, assigned, error=error)
        rows.append(row)
        if args.contact_limit != 0:
            contact_items.append((image_path.name, image, current, assigned, row))
        print(
            f"[{index}/{len(image_paths)}] {image_path.name} "
            f"v2={row['v2_candidate_count']} model={row['model_candidate_count']}"
            f"{' error=' + error if error else ''}"
        )

    summary = {
        "source": "spike_river_model_detector",
        "backend": args.backend,
        "frame_count": len(rows),
        "v2_candidate_count": sum(row["v2_candidate_count"] for row in rows),
        "model_candidate_count": sum(row["model_candidate_count"] for row in rows),
        "fallback_candidate_count": sum(row["fallback_candidate_count"] for row in rows),
        "error_count": sum(1 for row in rows if row.get("error")),
        "frames": rows,
    }
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.md"
    contact_path = args.out_dir / "model_vs_v2.png"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_report_markdown(summary, contact_path.name if contact_items else ""), encoding="utf-8")
    contact_items = _select_contact_items(contact_items, args.contact_sort, limit=args.contact_limit)
    if contact_items:
        _save_contact_sheet(contact_items, contact_path)

    print(f"summary={summary_path}")
    print(f"report={report_path}")
    if contact_items:
        print(f"contact_sheet={contact_path}")
    return 0


def assign_detections_to_rivers(
    detections: list[ModelDetection],
    image_size: tuple[int, int],
) -> list[AssignedDetection]:
    width, height = image_size
    rois = build_river_rois(width, height)
    grouped: dict[str, list[ModelDetection]] = {player: [] for player in RIVER_PLAYERS}
    for detection in detections:
        roi = _best_roi_for_detection(detection, rois)
        if roi is not None:
            grouped[roi.player].append(detection)

    assigned = []
    for player in RIVER_PLAYERS:
        ordered = sorted(grouped[player], key=lambda item: (item.center[1], item.center[0]))
        for order_index, detection in enumerate(ordered, start=1):
            assigned.append(AssignedDetection(player=player, order_index=order_index, detection=detection))
    return assigned


def fuse_model_with_v2_gaps(
    assigned: list[AssignedDetection],
    v2_result: Any,
    *,
    iou_threshold: float = 0.20,
    max_per_player: int = 18,
) -> list[AssignedDetection]:
    fused = list(assigned)
    for player in RIVER_PLAYERS:
        model_items = [item for item in fused if item.player == player]
        v2_candidates = v2_result.by_player.get(player, [])
        target_count = max(len(model_items), len(v2_candidates))
        if max_per_player > 0:
            target_count = max(len(model_items), min(len(v2_candidates), max_per_player))
        for candidate in v2_candidates:
            if len(model_items) >= target_count:
                break
            if _candidate_overlaps_assigned(candidate, model_items, iou_threshold=iou_threshold):
                continue
            fallback = AssignedDetection(
                player=player,
                order_index=0,
                detection=ModelDetection(
                    bbox=candidate.bbox,
                    confidence=candidate.confidence,
                    label="",
                    source="river_detector_v2_fallback",
                )
            )
            fused.append(fallback)
            model_items.append(fallback)
    return _renumber_assigned_by_player(fused)


def _candidate_overlaps_assigned(
    candidate: RiverTileCandidate,
    assigned: list[AssignedDetection],
    *,
    iou_threshold: float,
) -> bool:
    return any(_box_iou(candidate.bbox, item.detection.bbox) >= iou_threshold for item in assigned)


def _renumber_assigned_by_player(assigned: list[AssignedDetection]) -> list[AssignedDetection]:
    grouped: dict[str, list[AssignedDetection]] = {player: [] for player in RIVER_PLAYERS}
    for item in assigned:
        grouped.setdefault(item.player, []).append(item)
    renumbered = []
    for player in RIVER_PLAYERS:
        ordered = sorted(grouped[player], key=lambda item: (item.detection.center[1], item.detection.center[0]))
        for order_index, item in enumerate(ordered, start=1):
            renumbered.append(
                AssignedDetection(
                    player=item.player,
                    order_index=order_index,
                    detection=item.detection,
                    tile=item.tile,
                    tile_confidence=item.tile_confidence,
                )
            )
    return renumbered


def classify_assigned_detections(
    image: Image.Image,
    assigned: list[AssignedDetection],
) -> list[AssignedDetection]:
    targets = []
    crops = []
    for item in assigned:
        label = _normalize_tile_label(item.detection.label)
        if label in TILE_LABELS:
            targets.append((item, label, item.detection.confidence))
            continue
        crops.append(image.crop(item.detection.bbox))
        targets.append((item, "", 0.0))
    crop_results = classify_tiles_batch(crops, {})
    empty_on_none = onnx_backend_available()
    crop_index = 0
    classified = []
    for item, label, confidence in targets:
        if label:
            classified.append(
                AssignedDetection(item.player, item.order_index, item.detection, tile=label, tile_confidence=confidence)
            )
            continue
        match = crop_results[crop_index] if crop_index < len(crop_results) else None
        crop_index += 1
        tile = "unknown"
        confidence = 0.0
        if match is None and empty_on_none:
            tile = EMPTY_TILE_LABEL
            confidence = 1.0
        elif match is not None:
            confidence = match.confidence
            if match.confidence >= MIN_RIVER_TILE_CLASSIFICATION_CONFIDENCE:
                tile = match.tile
        classified.append(
            AssignedDetection(
                item.player,
                item.order_index,
                item.detection,
                tile=tile,
                tile_confidence=confidence,
            )
        )
    return _cap_assigned_tile_overflow(classified)


def _cap_assigned_tile_overflow(assigned: list[AssignedDetection]) -> list[AssignedDetection]:
    tile_counts = Counter(
        item.tile
        for item in assigned
        if item.tile and item.tile not in {"unknown", EMPTY_TILE_LABEL}
    )
    overflow_tiles = {tile for tile, count in tile_counts.items() if count > 4}
    if not overflow_tiles:
        return assigned

    capped = list(assigned)
    for tile in overflow_tiles:
        indexed = [
            (index, item)
            for index, item in enumerate(capped)
            if item.tile == tile
        ]
        demote_count = max(0, len(indexed) - 4)
        demoted = sorted(
            indexed,
            key=lambda item: (item[1].tile_confidence or item[1].detection.confidence, item[1].order_index),
        )[:demote_count]
        for index, item in demoted:
            capped[index] = AssignedDetection(
                item.player,
                item.order_index,
                item.detection,
                tile="unknown",
                tile_confidence=item.tile_confidence or item.detection.confidence,
            )
    return capped


def parse_roboflow_predictions(payload: dict[str, Any]) -> list[ModelDetection]:
    scale_x = float(payload.get("_scale_x", 1.0) or 1.0)
    scale_y = float(payload.get("_scale_y", 1.0) or 1.0)
    detections = []
    for item in payload.get("predictions", []):
        x = float(item.get("x", 0.0)) * scale_x
        y = float(item.get("y", 0.0)) * scale_y
        width = float(item.get("width", 0.0)) * scale_x
        height = float(item.get("height", 0.0)) * scale_y
        if width <= 0 or height <= 0:
            continue
        detections.append(
            ModelDetection(
                bbox=(
                    int(round(x - width / 2)),
                    int(round(y - height / 2)),
                    int(round(x + width / 2)),
                    int(round(y + height / 2)),
                ),
                confidence=float(item.get("confidence", 0.0)),
                label=str(item.get("class", item.get("class_name", ""))),
                source="roboflow",
            )
        )
    return detections


def parse_ultralytics_result(result: Any) -> list[ModelDetection]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    detections = []
    xyxy = boxes.xyxy.tolist()
    confs = boxes.conf.tolist()
    classes = boxes.cls.tolist()
    for bbox, confidence, class_id in zip(xyxy, confs, classes, strict=True):
        label = str(names.get(int(class_id), int(class_id))) if isinstance(names, dict) else str(int(class_id))
        detections.append(
            ModelDetection(
                bbox=tuple(int(round(value)) for value in bbox),
                confidence=float(confidence),
                label=label,
                source="ultralytics",
            )
        )
    return detections


def _load_model_detections(image_path: Path, args: argparse.Namespace) -> list[ModelDetection]:
    if args.backend == "json":
        payload_path = _json_payload_path(image_path, args)
        if payload_path is None:
            return []
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        return parse_roboflow_predictions(payload)
    if args.backend == "roboflow":
        return _run_roboflow(image_path, args)
    if args.backend == "ultralytics":
        return _run_ultralytics(image_path, args)
    raise ValueError(f"unsupported backend: {args.backend}")


def _json_payload_path(image_path: Path, args: argparse.Namespace) -> Path | None:
    if args.detector_json_dir is not None:
        candidates = [
            args.detector_json_dir / f"{image_path.stem}.json",
            args.detector_json_dir / f"{image_path.name}.json",
            args.detector_json_dir / image_path.with_suffix(".json").name,
        ]
        return next((path for path in candidates if path.exists()), None)
    return args.detector_json


def _run_roboflow(image_path: Path, args: argparse.Namespace) -> list[ModelDetection]:
    api_key = args.roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY or --roboflow-api-key is required for --backend roboflow")
    cache_path = _roboflow_cache_path(image_path, args)
    if cache_path is not None and cache_path.exists():
        return parse_roboflow_predictions(json.loads(cache_path.read_text(encoding="utf-8")))
    import requests

    url = f"https://detect.roboflow.com/{args.roboflow_model}"
    params = {"api_key": api_key, "confidence": args.confidence, "overlap": args.overlap}
    image_bytes, scale = _roboflow_image_payload(image_path, max_size=args.roboflow_max_size)
    response = requests.post(url, params=params, files={"file": ("image.jpg", image_bytes, "image/jpeg")}, timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(_redact_secrets(str(exc))) from exc
    payload = response.json()
    payload["_scale_x"] = scale[0]
    payload["_scale_y"] = scale[1]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return parse_roboflow_predictions(payload)


def _roboflow_image_payload(image_path: Path, *, max_size: int) -> tuple[bytes, tuple[float, float]]:
    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    width, height = original.size
    longest = max(width, height)
    if max_size > 0 and longest > max_size:
        ratio = max_size / float(longest)
        resized = original.resize((int(round(width * ratio)), int(round(height * ratio))), Image.Resampling.LANCZOS)
    else:
        resized = original
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue(), (width / resized.width, height / resized.height)


def _roboflow_cache_path(image_path: Path, args: argparse.Namespace) -> Path | None:
    if args.roboflow_cache_dir is None:
        return None
    return args.roboflow_cache_dir / f"{image_path.stem}.json"


def _redact_secrets(text: str) -> str:
    if not text:
        return text
    redacted = []
    for chunk in text.split("&"):
        if "api_key=" in chunk:
            prefix, _sep, _value = chunk.partition("api_key=")
            redacted.append(f"{prefix}api_key=<redacted>")
        else:
            redacted.append(chunk)
    return "&".join(redacted)


def _run_ultralytics(image_path: Path, args: argparse.Namespace) -> list[ModelDetection]:
    if args.ultralytics_model is None:
        raise SystemExit("--ultralytics-model is required for --backend ultralytics")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed; install it or use --backend roboflow/json") from exc

    model = YOLO(str(args.ultralytics_model))
    results = model(str(image_path), conf=args.confidence, iou=args.overlap, device=args.device, verbose=False)
    if not results:
        return []
    return parse_ultralytics_result(results[0])


def _image_paths(args: argparse.Namespace) -> list[Path]:
    if args.batch_dir is not None:
        images = sorted(path for path in args.batch_dir.glob(args.batch_glob) if path.is_file())
        if args.batch_limit > 0:
            images = images[: args.batch_limit]
        return images
    if args.image is None:
        raise SystemExit("image is required unless --batch-dir is used")
    return [args.image]


def _best_roi_for_detection(detection: ModelDetection, rois: list[Any]) -> Any | None:
    cx, cy = detection.center
    center_hits = [
        roi
        for roi in rois
        if roi.left <= cx <= roi.right and roi.top <= cy <= roi.bottom
    ]
    if center_hits:
        return center_hits[0]
    best_roi = None
    best_overlap = 0.0
    area = _box_area(detection.bbox)
    for roi in rois:
        overlap = _intersection_area(detection.bbox, (roi.left, roi.top, roi.right, roi.bottom)) / max(1.0, area)
        if overlap > best_overlap:
            best_overlap = overlap
            best_roi = roi
    return best_roi if best_overlap >= 0.35 else None


def _frame_summary(
    image_path: Path,
    current: Any,
    assigned: list[AssignedDetection],
    *,
    error: str = "",
) -> dict[str, Any]:
    model_by_player = {player: 0 for player in RIVER_PLAYERS}
    v2_by_player = {player: len(current.by_player.get(player, [])) for player in RIVER_PLAYERS}
    for item in assigned:
        model_by_player[item.player] += 1
    fallback_count = sum(1 for item in assigned if item.detection.source == "river_detector_v2_fallback")
    return {
        "image": str(image_path),
        "v2_candidate_count": len(current.candidates),
        "model_candidate_count": len(assigned),
        "v2_by_player": v2_by_player,
        "model_by_player": model_by_player,
        "model_detections": [item.to_dict() for item in assigned],
        "fallback_candidate_count": fallback_count,
        "error": error,
    }


def _report_markdown(summary: dict[str, Any], contact_name: str) -> str:
    lines = [
        "# River Model Detector Spike",
        "",
        f"- Backend: {summary['backend']}",
        f"- Frames: {summary['frame_count']}",
        f"- Current v2 candidates: {summary['v2_candidate_count']}",
        f"- Model candidates: {summary['model_candidate_count']}",
        f"- v2 fallback candidates: {summary.get('fallback_candidate_count', 0)}",
        f"- Errors: {summary.get('error_count', 0)}",
        "",
    ]
    if contact_name:
        lines.extend([f"![Model vs v2]({contact_name})", ""])
    lines.extend(["## Frames", ""])
    for row in summary["frames"][:20]:
        lines.append(
            f"- `{Path(row['image']).name}`: v2={row['v2_candidate_count']}, "
            f"model={row['model_candidate_count']}"
            f"{' error=' + row['error'] if row.get('error') else ''}"
        )
    lines.append("")
    return "\n".join(lines)


def _select_contact_items(
    items: list[tuple[str, Image.Image, Any, list[AssignedDetection], dict[str, Any]]],
    sort_mode: str,
    *,
    limit: int,
) -> list[tuple[str, Image.Image, Any, list[AssignedDetection], dict[str, Any]]]:
    if limit <= 0:
        return []
    if sort_mode == "model-count":
        items = sorted(items, key=lambda item: (item[4]["model_candidate_count"], item[4]["v2_candidate_count"]), reverse=True)
    elif sort_mode == "fallback-count":
        items = sorted(
            items,
            key=lambda item: (item[4].get("fallback_candidate_count", 0), item[4]["model_candidate_count"]),
            reverse=True,
        )
    elif sort_mode == "v2-count":
        items = sorted(items, key=lambda item: (item[4]["v2_candidate_count"], item[4]["model_candidate_count"]), reverse=True)
    return items[:limit]


def _save_contact_sheet(items: list[tuple[str, Image.Image, Any, list[AssignedDetection], dict[str, Any]]], out_path: Path) -> None:
    width = 1280
    label_height = 46
    panels = []
    font = _load_font()
    for name, image, current, assigned, row in items:
        left = _draw_v2_preview(image, current)
        right = _draw_model_preview(image, assigned)
        left = ImageOps.contain(left, (width // 2, 360))
        right = ImageOps.contain(right, (width // 2, 360))
        panel = Image.new("RGB", (width, max(left.height, right.height) + label_height), (18, 20, 24))
        draw = ImageDraw.Draw(panel)
        draw.text(
            (8, 8),
            f"{name}  v2={row['v2_candidate_count']}  model={row['model_candidate_count']}",
            fill=(245, 245, 245),
            font=font,
        )
        panel.paste(left, (0, label_height))
        panel.paste(right, (width // 2, label_height))
        panels.append(panel)
    sheet = Image.new("RGB", (width, sum(panel.height for panel in panels)), (18, 20, 24))
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top))
        top += panel.height
    sheet.save(out_path)


def _draw_v2_preview(image: Image.Image, current: Any) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    font = _load_font()
    for player in RIVER_PLAYERS:
        color = PLAYER_COLORS[player]
        for candidate in current.by_player.get(player, []):
            points = list(candidate.quad)
            draw.line(points + [points[0]], fill=color, width=4)
            draw.text((candidate.bbox[0], candidate.bbox[1]), f"v2 {candidate.order_index}", fill=color, font=font)
    return preview


def _draw_model_preview(image: Image.Image, assigned: list[AssignedDetection]) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    font = _load_font()
    for item in assigned:
        color = PLAYER_COLORS[item.player]
        x0, y0, x1, y1 = item.detection.bbox
        tile = item.tile or _normalize_tile_label(item.detection.label) or "?"
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        draw.text((x0, y0), f"M {item.order_index} {tile}", fill=color, font=font)
    return preview


def _normalize_tile_label(label: str) -> str:
    text = label.strip().lower()
    text = text.replace("-", "").replace("_", "")
    if len(text) == 2 and text[0].isdigit():
        kim_suit_map = {
            "b": "s",
            "c": "m",
            "d": "p",
        }
        suit = kim_suit_map.get(text[1])
        if suit is not None:
            return f"{text[0]}{suit}"
    honor_names = {
        "east": "1z",
        "south": "2z",
        "west": "3z",
        "north": "4z",
        "white": "5z",
        "green": "6z",
        "red": "7z",
        "chun": "7z",
        "ew": "1z",
        "sw": "2z",
        "ww": "3z",
        "nw": "4z",
        "wd": "5z",
        "gd": "6z",
        "rd": "7z",
    }
    return honor_names.get(text, text)


def _box_area(bbox: tuple[int, int, int, int]) -> float:
    return float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return float(max(0, right - left) * max(0, bottom - top))


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    intersection = _intersection_area(a, b)
    union = _box_area(a) + _box_area(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 20)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
