from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .discard_parser import DiscardParseResult
from .river_detector_v2 import (
    MIN_RIVER_TILE_CLASSIFICATION_CONFIDENCE,
    RIVER_PLAYERS,
    RiverDetectionResult,
    RiverTileCandidate,
    build_river_rois,
    detect_river_tiles_v2,
)
from .tile_classifier_dispatch import classify_tiles_batch, onnx_backend_available
from .tile_templates import (
    FULL_TILE_INNER_BOUNDS,
    FULL_TILE_SIGNATURE_VERSION,
    build_hand_tile_template_payload,
    classify_tile_from_templates,
)


TILE_LABELS = {
    *(f"{rank}{suit}" for suit in ("m", "p", "s") for rank in range(1, 10)),
    *(f"{rank}z" for rank in range(1, 8)),
    "0m",
    "0p",
    "0s",
}
PLAYER_ORIENTATIONS = {
    "self": "bottom",
    "left_opponent": "left",
    "top_opponent": "top",
    "right_opponent": "right",
}
EMPTY_TILE_LABEL = "empty"
MANUAL_TEMPLATE_MIN_CONFIDENCE = 0.50
MANUAL_TEMPLATE_STRONG_CONFIDENCE = 0.95


@dataclass(frozen=True)
class ModelRiverConfig:
    detector_json: Path | None = None
    detector_json_dir: Path | None = None
    classify_crops: bool = True
    fuse_v2_gaps: bool = True
    fusion_iou: float = 0.20
    fusion_max_per_player: int = 18
    unknown_crop_dir: Path | None = None
    manual_labels_path: Path | None = None


@dataclass(frozen=True)
class ModelRiverDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str = ""
    source: str = "model"

    @property
    def center(self) -> tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)


@dataclass(frozen=True)
class ModelRiverCandidate:
    player: str
    turn_index: int
    detection: ModelRiverDetection
    tile: str = ""
    tile_confidence: float = 0.0


def parse_model_river_from_json(
    image: Image.Image,
    image_path: Path,
    *,
    config: ModelRiverConfig,
) -> DiscardParseResult:
    payload_path = model_river_json_path(image_path, config=config)
    if payload_path is None:
        return DiscardParseResult(
            analysis_hints={
                "discard_parser_source": "model_river_adapter",
                "discard_parser_available": False,
                "discard_parser_reason": "missing_detector_json",
            }
        )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    model = assign_model_detections_to_rivers(parse_roboflow_predictions(payload), image.size)
    v2 = detect_river_tiles_v2(image) if config.fuse_v2_gaps else RiverDetectionResult(image_size=image.size)
    if config.fuse_v2_gaps:
        model = fuse_model_with_v2_gaps(
            model,
            v2,
            iou_threshold=config.fusion_iou,
            max_per_player=config.fusion_max_per_player,
        )
    if config.classify_crops:
        model = classify_model_river_candidates(
            image,
            model,
            manual_template_payload=load_manual_template_payload(config.manual_labels_path),
        )
    if config.unknown_crop_dir is not None:
        save_unknown_model_river_crops(image, image_path, model, config.unknown_crop_dir)
    return model_candidates_to_parse_result(model, v2_result=v2, source_json=payload_path)


def model_river_json_path(image_path: Path, *, config: ModelRiverConfig) -> Path | None:
    if config.detector_json_dir is not None:
        candidates = [
            config.detector_json_dir / f"{image_path.stem}.json",
            config.detector_json_dir / f"{image_path.name}.json",
            config.detector_json_dir / image_path.with_suffix(".json").name,
        ]
        return next((path for path in candidates if path.exists()), None)
    return config.detector_json


def parse_roboflow_predictions(payload: dict[str, Any]) -> list[ModelRiverDetection]:
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
            ModelRiverDetection(
                bbox=(
                    int(round(x - width / 2)),
                    int(round(y - height / 2)),
                    int(round(x + width / 2)),
                    int(round(y + height / 2)),
                ),
                confidence=float(item.get("confidence", 0.0)),
                label=str(item.get("class", item.get("class_name", ""))),
                source=str(item.get("source", "roboflow")),
            )
        )
    return detections


def assign_model_detections_to_rivers(
    detections: list[ModelRiverDetection],
    image_size: tuple[int, int],
) -> list[ModelRiverCandidate]:
    width, height = image_size
    rois = build_river_rois(width, height)
    grouped: dict[str, list[ModelRiverDetection]] = {player: [] for player in RIVER_PLAYERS}
    for detection in detections:
        roi = _best_roi_for_detection(detection, rois)
        if roi is not None:
            grouped[roi.player].append(detection)

    assigned = []
    for player in RIVER_PLAYERS:
        ordered = sorted(grouped[player], key=lambda item: (item.center[1], item.center[0]))
        for turn_index, detection in enumerate(ordered, start=1):
            assigned.append(ModelRiverCandidate(player=player, turn_index=turn_index, detection=detection))
    return assigned


def fuse_model_with_v2_gaps(
    assigned: list[ModelRiverCandidate],
    v2_result: RiverDetectionResult,
    *,
    iou_threshold: float = 0.20,
    max_per_player: int = 18,
) -> list[ModelRiverCandidate]:
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
            fallback = ModelRiverCandidate(
                player=player,
                turn_index=0,
                detection=ModelRiverDetection(
                    bbox=candidate.bbox,
                    confidence=candidate.confidence,
                    source="river_detector_v2_fallback",
                ),
            )
            fused.append(fallback)
            model_items.append(fallback)
    return renumber_model_river_candidates(fused)


def classify_model_river_candidates(
    image: Image.Image,
    candidates: list[ModelRiverCandidate],
    *,
    manual_template_payload: dict[str, Any] | None = None,
) -> list[ModelRiverCandidate]:
    targets = []
    crops = []
    for candidate in candidates:
        label = normalize_tile_label(candidate.detection.label)
        if label in TILE_LABELS:
            targets.append((candidate, label, candidate.detection.confidence))
            continue
        crops.append(image.crop(candidate.detection.bbox))
        targets.append((candidate, "", 0.0))
    matches = classify_tiles_batch(crops, {})
    empty_on_none = onnx_backend_available()
    crop_index = 0
    classified = []
    for candidate, label, confidence in targets:
        if label:
            classified.append(_with_tile(candidate, label, confidence))
            continue
        match = matches[crop_index] if crop_index < len(matches) else None
        crop_index += 1
        if match is None:
            tile = EMPTY_TILE_LABEL if empty_on_none else "unknown"
            confidence = 1.0 if empty_on_none else 0.0
            classified.append(_with_tile(candidate, tile, confidence))
        elif match.confidence < MIN_RIVER_TILE_CLASSIFICATION_CONFIDENCE:
            calibrated = _manual_template_match(image.crop(candidate.detection.bbox), manual_template_payload)
            if calibrated is not None:
                classified.append(_with_tile(candidate, calibrated.tile, calibrated.confidence))
                continue
            classified.append(_with_tile(candidate, "unknown", match.confidence))
        else:
            calibrated = _manual_template_match(image.crop(candidate.detection.bbox), manual_template_payload)
            if calibrated is not None and _should_use_manual_template(match, calibrated):
                classified.append(_with_tile(candidate, calibrated.tile, calibrated.confidence))
                continue
            classified.append(_with_tile(candidate, match.tile, match.confidence))
    return cap_model_river_tile_overflow(classified)


def load_manual_template_payload(labels_path: Path | None) -> dict[str, Any] | None:
    if labels_path is None:
        return None
    rows = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return None
    samples = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label", "")).strip()
        if not label or label == "unknown":
            continue
        path = Path(str(row.get("file", "")))
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            continue
        with Image.open(path) as opened:
            samples.append((label, opened.convert("RGB")))
    if not samples:
        return None
    return build_hand_tile_template_payload(
        samples,
        max_samples_per_tile=8,
        inner_bounds=FULL_TILE_INNER_BOUNDS,
        signature_version=FULL_TILE_SIGNATURE_VERSION,
    )


def _manual_template_match(crop: Image.Image, payload: dict[str, Any] | None):
    if not payload:
        return None
    match = classify_tile_from_templates(crop, payload)
    if match is None or match.confidence < MANUAL_TEMPLATE_MIN_CONFIDENCE:
        return None
    return match


def _should_use_manual_template(classifier_match: Any, template_match: Any) -> bool:
    if template_match.confidence < MANUAL_TEMPLATE_STRONG_CONFIDENCE:
        return False
    return template_match.tile != classifier_match.tile or template_match.confidence > classifier_match.confidence


def cap_model_river_tile_overflow(candidates: list[ModelRiverCandidate]) -> list[ModelRiverCandidate]:
    tile_counts = Counter(
        candidate.tile
        for candidate in candidates
        if candidate.tile and candidate.tile not in {"unknown", EMPTY_TILE_LABEL}
    )
    overflow_tiles = {tile for tile, count in tile_counts.items() if count > 4}
    if not overflow_tiles:
        return candidates

    capped = list(candidates)
    for tile in overflow_tiles:
        indexed = [
            (index, candidate)
            for index, candidate in enumerate(capped)
            if candidate.tile == tile
        ]
        demote_count = max(0, len(indexed) - 4)
        demoted = sorted(
            indexed,
            key=lambda item: (item[1].tile_confidence or item[1].detection.confidence, item[1].turn_index),
        )[:demote_count]
        for index, candidate in demoted:
            capped[index] = _with_tile(candidate, "unknown", candidate.tile_confidence or candidate.detection.confidence)
    return capped


def save_unknown_model_river_crops(
    image: Image.Image,
    image_path: Path,
    candidates: list[ModelRiverCandidate],
    out_dir: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for candidate in candidates:
        if candidate.tile != "unknown":
            continue
        x0, y0, x1, y1 = candidate.detection.bbox
        crop = image.crop((x0, y0, x1, y1))
        if crop.width <= 0 or crop.height <= 0:
            continue
        filename = (
            f"{_safe_filename(image_path.stem)}"
            f"_{candidate.player}_{candidate.turn_index:02d}"
            f"_{_safe_filename(candidate.detection.source)}"
            f"_c{candidate.tile_confidence:.2f}.png"
        )
        crop.save(out_dir / filename)
        saved += 1
    return saved


def renumber_model_river_candidates(candidates: list[ModelRiverCandidate]) -> list[ModelRiverCandidate]:
    grouped: dict[str, list[ModelRiverCandidate]] = {player: [] for player in RIVER_PLAYERS}
    for candidate in candidates:
        grouped.setdefault(candidate.player, []).append(candidate)
    renumbered = []
    for player in RIVER_PLAYERS:
        ordered = sorted(grouped[player], key=lambda item: (item.detection.center[1], item.detection.center[0]))
        for turn_index, candidate in enumerate(ordered, start=1):
            renumbered.append(
                ModelRiverCandidate(
                    player=candidate.player,
                    turn_index=turn_index,
                    detection=candidate.detection,
                    tile=candidate.tile,
                    tile_confidence=candidate.tile_confidence,
                )
            )
    return renumbered


def model_candidates_to_parse_result(
    candidates: list[ModelRiverCandidate],
    *,
    v2_result: RiverDetectionResult,
    source_json: Path,
) -> DiscardParseResult:
    discard_piles: dict[str, list[dict[str, Any]]] = {}
    raw_detections = []
    visible_tiles = []
    unknown_by_player: Counter[str] = Counter()
    empty_by_player: Counter[str] = Counter()
    for candidate in candidates:
        item = _candidate_to_discard_item(candidate)
        raw_detections.append(item)
        if item["tile"] == EMPTY_TILE_LABEL:
            empty_by_player[candidate.player] += 1
            continue
        if item["tile"] == "unknown":
            unknown_by_player[candidate.player] += 1
            continue
        discard_piles.setdefault(candidate.player, []).append(item)
        visible_tiles.append(item["tile"])

    fallback_count = sum(1 for item in candidates if item.detection.source == "river_detector_v2_fallback")
    overflow_counts = _tile_overflow_counts(visible_tiles)
    unknown_count = sum(unknown_by_player.values())
    empty_count = sum(empty_by_player.values())
    candidate_count = len(candidates)
    return DiscardParseResult(
        discard_piles=discard_piles,
        visible_tiles=visible_tiles,
        raw_detections=raw_detections,
        analysis_hints={
            "discard_parser_source": "model_river_adapter",
            "discard_parser_available": True,
            "model_river_source_json": str(source_json),
            "model_river_candidate_count": candidate_count,
            "model_river_empty_count": empty_count,
            "model_river_empty_by_player": dict(empty_by_player),
            "model_river_known_count": len(visible_tiles),
            "model_river_unknown_count": unknown_count,
            "model_river_unknown_rate": round(unknown_count / max(1, candidate_count), 4),
            "model_river_unknown_by_player": dict(unknown_by_player),
            "model_river_v2_candidate_count": len(v2_result.candidates),
            "model_river_fallback_candidate_count": fallback_count,
            "model_river_tile_overflow_counts": overflow_counts,
        },
    )


def normalize_tile_label(label: str) -> str:
    text = label.strip().lower()
    text = text.replace("-", "").replace("_", "")
    if len(text) == 2 and text[0].isdigit():
        suit = {"b": "s", "c": "m", "d": "p"}.get(text[1])
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


def _candidate_to_discard_item(candidate: ModelRiverCandidate) -> dict[str, Any]:
    x0, y0, x1, y1 = candidate.detection.bbox
    tile = candidate.tile or normalize_tile_label(candidate.detection.label) or "unknown"
    return {
        "tile": tile,
        "player": candidate.player,
        "turn_index": candidate.turn_index,
        "bbox": [x0, y0, x1, y1],
        "quad": [[x0, y0], [x0, y1], [x1, y1], [x1, y0]],
        "confidence": candidate.tile_confidence or candidate.detection.confidence,
        "orientation": PLAYER_ORIENTATIONS[candidate.player],
        "source": candidate.detection.source,
        "model_label": candidate.detection.label,
        "model_confidence": candidate.detection.confidence,
    }


def _with_tile(candidate: ModelRiverCandidate, tile: str, confidence: float) -> ModelRiverCandidate:
    return ModelRiverCandidate(
        player=candidate.player,
        turn_index=candidate.turn_index,
        detection=candidate.detection,
        tile=tile,
        tile_confidence=confidence,
    )


def _candidate_overlaps_assigned(
    candidate: RiverTileCandidate,
    assigned: list[ModelRiverCandidate],
    *,
    iou_threshold: float,
) -> bool:
    return any(_box_iou(candidate.bbox, item.detection.bbox) >= iou_threshold for item in assigned)


def _best_roi_for_detection(detection: ModelRiverDetection, rois: list[Any]) -> Any | None:
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


def _tile_overflow_counts(tiles: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tile in tiles:
        counts[tile] = counts.get(tile, 0) + 1
    return {tile: count for tile, count in sorted(counts.items()) if count > 4}


def _safe_filename(value: str) -> str:
    cleaned = []
    for char in str(value):
        cleaned.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(cleaned).strip("_") or "unknown"


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
