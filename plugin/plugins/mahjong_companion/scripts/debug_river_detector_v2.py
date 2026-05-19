from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from plugin.plugins.mahjong_companion.perception.river_detector_v2 import (
    MIN_RIVER_TILE_CLASSIFICATION_CONFIDENCE,
    RIVER_PLAYERS,
    crop_river_candidate,
    detect_river_tiles_v2,
    expand_candidate_quad_for_classification,
    river_candidate_classification_rejection_reason,
)
from plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch import (
    classify_tiles_batch,
    onnx_backend_available,
)
from plugin.plugins.mahjong_companion.perception.tile_templates import (
    FULL_TILE_INNER_BOUNDS,
    FULL_TILE_SIGNATURE_VERSION,
    build_hand_tile_template_payload,
    classify_tile_from_templates,
)


PLAYER_COLORS = {
    "self": (80, 220, 120),
    "left_opponent": (255, 210, 70),
    "top_opponent": (90, 180, 255),
    "right_opponent": (255, 120, 120),
}
EMPTY_TILE_LABEL = "empty"
MANUAL_TEMPLATE_MIN_CONFIDENCE = 0.50
MANUAL_TEMPLATE_STRONG_CONFIDENCE = 0.95


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw class-agnostic river tile detections on a Mahjong Soul screenshot.")
    parser.add_argument("image", type=Path, nargs="?", help="Input screenshot path.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--json-out", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Process screenshots in a directory.")
    parser.add_argument("--batch-glob", default="*-frame.png", help="Glob used inside --batch-dir.")
    parser.add_argument("--batch-out", type=Path, default=None, help="Output directory for batch report artifacts.")
    parser.add_argument("--batch-limit", type=int, default=0, help="Maximum screenshots to process in batch mode.")
    parser.add_argument("--issue-limit", type=int, default=12, help="Maximum issue thumbnails in the batch contact sheet.")
    parser.add_argument("--issue-thumb-width", type=int, default=960, help="Thumbnail width for the batch issue contact sheet.")
    parser.add_argument("--issue-columns", type=int, default=1, help="Column count for the batch issue contact sheet.")
    parser.add_argument("--unknown-crop-out", type=Path, default=None, help="Directory for classified-unknown crop PNGs.")
    parser.add_argument("--manual-labels", type=Path, default=None, help="JSON labels used as a local template calibration set.")
    parser.add_argument("--draw-rois", action="store_true", help="Also draw the broad river search regions.")
    parser.add_argument("--draw-classification-quads", action="store_true", help="Draw expanded quads used for classifier crops.")
    parser.add_argument("--classify", action="store_true", help="Run tile classification and draw tile labels.")
    parser.add_argument("--only-player", choices=RIVER_PLAYERS, default=None, help="Draw candidate boxes for only one player.")
    parser.add_argument("--hide-labels", action="store_true", help="Draw boxes without tile index labels.")
    args = parser.parse_args()

    if args.batch_dir is not None:
        return _run_batch(args)
    if args.image is None:
        parser.error("image is required unless --batch-dir is used")

    image_path = args.image
    out_path = args.out or image_path.with_name(f"{image_path.stem}-river-v2.png")
    json_path = args.json_out or out_path.with_suffix(".json")

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    result = detect_river_tiles_v2(image)
    manual_template_payload = _load_manual_template_payload(args.manual_labels)
    classifications = _classify_candidates(image, result.candidates, manual_template_payload) if args.classify else {}
    if args.unknown_crop_out is not None:
        _save_unknown_crops(image_path, image, result.candidates, classifications, args.unknown_crop_out)
    preview = _draw_preview(image, result, classifications, args)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path)
    payload = result.to_dict()
    if args.classify:
        payload["classifications"] = [classifications.get(id(candidate), {}) for candidate in result.candidates]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"image={image_path}")
    print(f"output={out_path}")
    print(f"json={json_path}")
    print(f"candidate_count={len(result.candidates)}")
    for player in RIVER_PLAYERS:
        pile = result.by_player.get(player, [])
        print(f"{player}: {len(pile)}")
        for item in pile:
            classification = classifications.get(id(item), {}) if args.classify else {}
            tile_text = ""
            if classification:
                tile_text = f" tile={classification.get('tile', '')} tile_confidence={classification.get('tile_confidence', 0.0):.3f}"
            print(
                f"  {item.order_index:02d} bbox={list(item.bbox)} "
                f"quad={[[x, y] for x, y in item.quad]} confidence={item.confidence:.3f}{tile_text}"
            )
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    batch_dir = args.batch_dir
    out_dir = args.batch_out or batch_dir.with_name(f"{batch_dir.name}-river-v2-batch")
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in batch_dir.glob(args.batch_glob) if path.is_file())
    if args.batch_limit > 0:
        images = images[: args.batch_limit]
    if not images:
        raise SystemExit(f"no *-frame.png files found in {batch_dir}")
    manual_template_payload = _load_manual_template_payload(args.manual_labels)

    frame_rows = []
    issue_rows = []
    totals_by_player: Counter[str] = Counter()
    unknown_by_player: Counter[str] = Counter()
    empty_by_player: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    tile_overflow_frames = 0
    player_count_anomaly_frames = 0
    for index, image_path in enumerate(images, start=1):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        result = detect_river_tiles_v2(image)
        classifications = _classify_candidates(image, result.candidates, manual_template_payload) if args.classify else {}
        if args.unknown_crop_out is not None:
            _save_unknown_crops(image_path, image, result.candidates, classifications, args.unknown_crop_out)
        row = _batch_frame_row(image_path, result, classifications)
        frame_rows.append(row)
        totals_by_player.update(row["counts_by_player"])
        unknown_by_player.update(row["unknown_by_player"])
        empty_by_player.update(row["empty_by_player"])
        rejection_reasons.update(row["rejection_reasons"])
        if row["tile_overflow_counts"]:
            tile_overflow_frames += 1
        if row["player_count_anomalies"]:
            player_count_anomaly_frames += 1
        if row["unknown_count"] > 0:
            issue_rows.append((image_path, row))
        print(
            f"[{index}/{len(images)}] {image_path.name} "
            f"candidates={row['candidate_count']} unknown={row['unknown_count']} empty={row['empty_count']}"
        )

    summary = {
        "batch_dir": str(batch_dir),
        "frame_count": len(frame_rows),
        "candidate_count": sum(row["candidate_count"] for row in frame_rows),
        "unknown_count": sum(row["unknown_count"] for row in frame_rows),
        "empty_count": sum(row["empty_count"] for row in frame_rows),
        "counts_by_player": dict(totals_by_player),
        "unknown_by_player": dict(unknown_by_player),
        "empty_by_player": dict(empty_by_player),
        "rejection_reasons": dict(rejection_reasons),
        "tile_overflow_frame_count": tile_overflow_frames,
        "player_count_anomaly_frame_count": player_count_anomaly_frames,
        "frames": frame_rows,
    }
    summary_path = out_dir / "batch_summary.json"
    report_path = out_dir / "batch_report.md"
    contact_sheet_path = out_dir / "batch_issues.png"
    issue_previews = _build_issue_previews(issue_rows, args)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_batch_report_markdown(summary, contact_sheet_path.name if issue_previews else ""), encoding="utf-8")
    if issue_previews:
        _save_issue_contact_sheet(
            issue_previews,
            contact_sheet_path,
            thumb_width=args.issue_thumb_width,
            columns=args.issue_columns,
        )

    print(f"summary={summary_path}")
    print(f"report={report_path}")
    if issue_previews:
        print(f"issues={contact_sheet_path}")
    return 0


def _build_issue_previews(
    issue_rows: list[tuple[Path, dict[str, Any]]],
    args: argparse.Namespace,
) -> list[tuple[str, dict[str, Any], Image.Image]]:
    limit = max(0, args.issue_limit)
    if limit == 0:
        return []
    worst_rows = sorted(
        issue_rows,
        key=lambda item: (item[1]["unknown_count"], item[1]["candidate_count"]),
        reverse=True,
    )[:limit]
    previews = []
    for image_path, row in worst_rows:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        result = detect_river_tiles_v2(image)
        manual_template_payload = _load_manual_template_payload(args.manual_labels)
        classifications = _classify_candidates(image, result.candidates, manual_template_payload) if args.classify else {}
        previews.append((image_path.name, row, _draw_preview(image, result, classifications, args)))
    return previews


def _draw_preview(
    image: Image.Image,
    result: Any,
    classifications: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    font = _load_font()
    if args.draw_rois:
        for roi in result.rois:
            color = PLAYER_COLORS.get(roi.player, (255, 255, 255))
            draw.rectangle((roi.left, roi.top, roi.right, roi.bottom), outline=color, width=2)
            draw.text((roi.left + 4, roi.top + 4), roi.player, fill=color, font=font)

    players_to_draw = [args.only_player] if args.only_player else list(RIVER_PLAYERS)
    for player in players_to_draw:
        color = PLAYER_COLORS[player]
        for candidate in result.by_player.get(player, []):
            if args.draw_classification_quads:
                crop_points = list(expand_candidate_quad_for_classification(candidate))
                draw.line(crop_points + [crop_points[0]], fill=(255, 255, 255), width=2)
            points = list(candidate.quad)
            draw.line(points + [points[0]], fill=color, width=4)
            if args.hide_labels:
                continue
            label = _candidate_label(candidate, classifications) if classifications else f"{_short_player(player)}{candidate.order_index}"
            left = min(x for x, _y in points)
            top = min(y for _x, y in points)
            label_box = draw.textbbox((left, top), label, font=font)
            pad = 3
            draw.rectangle(
                (
                    label_box[0] - pad,
                    label_box[1] - pad,
                    label_box[2] + pad,
                    label_box[3] + pad,
                ),
                fill=(0, 0, 0),
            )
            draw.text((left, top), label, fill=color, font=font)
    return preview


def _batch_frame_row(
    image_path: Path,
    result: Any,
    classifications: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    counts_by_player = {player: len(result.by_player.get(player, [])) for player in RIVER_PLAYERS}
    unknown_by_player: Counter[str] = Counter()
    empty_by_player: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    tile_counts: Counter[str] = Counter()
    tiles_by_player: dict[str, list[str]] = {player: [] for player in RIVER_PLAYERS}
    for candidate in result.candidates:
        classification = classifications.get(id(candidate), {})
        tile = str(classification.get("tile", "unknown" if classifications else ""))
        if tile:
            tile_counts[tile] += 1
        if tile == EMPTY_TILE_LABEL:
            empty_by_player[candidate.player] += 1
            source = str(classification.get("classification_source", "empty"))
            rejection_reasons[source] += 1
            continue
        if tile == "unknown":
            unknown_by_player[candidate.player] += 1
            source = str(classification.get("classification_source", "unknown"))
            rejection_reasons[source] += 1
            continue
        if tile:
            tiles_by_player.setdefault(candidate.player, []).append(tile)
    tile_overflow_counts = {
        tile: count
        for tile, count in sorted(tile_counts.items())
        if tile not in {"unknown", EMPTY_TILE_LABEL} and count > 4
    }
    player_count_anomalies = {
        player: count
        for player, count in counts_by_player.items()
        if count > 18
    }
    return {
        "image": str(image_path),
        "candidate_count": len(result.candidates),
        "unknown_count": sum(unknown_by_player.values()),
        "empty_count": sum(empty_by_player.values()),
        "counts_by_player": counts_by_player,
        "unknown_by_player": dict(unknown_by_player),
        "empty_by_player": dict(empty_by_player),
        "tiles_by_player": tiles_by_player,
        "rejection_reasons": dict(rejection_reasons),
        "tile_counts": dict(tile_counts),
        "tile_overflow_counts": tile_overflow_counts,
        "player_count_anomalies": player_count_anomalies,
    }


def _batch_report_markdown(summary: dict[str, Any], contact_sheet_name: str) -> str:
    frame_count = int(summary["frame_count"])
    candidate_count = int(summary["candidate_count"])
    unknown_count = int(summary["unknown_count"])
    empty_count = int(summary.get("empty_count", 0))
    unknown_rate = unknown_count / max(1, candidate_count)
    worst_frames = sorted(
        summary["frames"],
        key=lambda row: (row["unknown_count"], row["candidate_count"]),
        reverse=True,
    )[:10]
    lines = [
        "# River Detector V2 Batch Report",
        "",
        f"- Frames: {frame_count}",
        f"- Candidates: {candidate_count}",
        f"- Unknown: {unknown_count} ({unknown_rate:.1%})",
        f"- Empty/false candidates: {empty_count}",
        f"- Counts by player: {_format_counter(summary['counts_by_player'])}",
        f"- Unknown by player: {_format_counter(summary['unknown_by_player'])}",
        f"- Empty by player: {_format_counter(summary.get('empty_by_player', {}))}",
        f"- Rejection reasons: {_format_counter(summary['rejection_reasons'])}",
        f"- Tile overflow frames: {summary.get('tile_overflow_frame_count', 0)}",
        f"- Player count anomaly frames: {summary.get('player_count_anomaly_frame_count', 0)}",
        "",
    ]
    if contact_sheet_name:
        lines.extend([f"![Issue contact sheet]({contact_sheet_name})", ""])
    lines.extend(["## Worst Frames", ""])
    for row in worst_frames:
        lines.append(
            f"- `{Path(row['image']).name}`: candidates={row['candidate_count']}, "
            f"unknown={row['unknown_count']}, players={_format_counter(row['counts_by_player'])}"
        )
    lines.append("")
    return "\n".join(lines)


def _format_counter(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={values[key]}" for key in sorted(values))


def _save_issue_contact_sheet(
    items: list[tuple[str, dict[str, Any], Image.Image]],
    out_path: Path,
    *,
    thumb_width: int = 960,
    columns: int = 1,
) -> None:
    thumb_width = max(320, thumb_width)
    label_height = 44
    columns = max(1, columns)
    rows = (len(items) + columns - 1) // columns
    thumbs = []
    font = _load_font()
    for name, row, image in items:
        thumb_height = int(round(image.height * thumb_width / image.width))
        thumb = ImageOps.contain(image, (thumb_width, thumb_height))
        tile = Image.new("RGB", (thumb_width, thumb.height + label_height), (22, 24, 28))
        tile.paste(thumb, (0, label_height))
        draw = ImageDraw.Draw(tile)
        draw.text(
            (8, 6),
            f"{name}  c={row['candidate_count']} ??={row['unknown_count']}",
            fill=(245, 245, 245),
            font=font,
        )
        thumbs.append(tile)
    cell_height = max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (thumb_width * columns, cell_height * rows), (18, 20, 24))
    for index, thumb in enumerate(thumbs):
        x = index % columns * thumb_width
        y = index // columns * cell_height
        sheet.paste(thumb, (x, y))
    sheet.save(out_path)


def _classify_candidates(
    image: Image.Image,
    candidates: list[Any],
    manual_template_payload: dict[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    payload: dict[int, dict[str, Any]] = {}
    targets = []
    crops = []
    empty_on_none = onnx_backend_available()
    for candidate in candidates:
        rejection_reason = river_candidate_classification_rejection_reason(image, candidate)
        if rejection_reason:
            payload[id(candidate)] = {
                "tile": "unknown",
                "tile_confidence": 0.0,
                "classification_source": rejection_reason,
            }
            continue
        targets.append(candidate)
        crops.append(crop_river_candidate(image, candidate))
    matches = classify_tiles_batch(crops, {})
    for candidate, match in zip(targets, matches, strict=True):
        if match is None and empty_on_none:
            payload[id(candidate)] = {
                "tile": EMPTY_TILE_LABEL,
                "tile_confidence": 1.0,
                "classification_source": "tile_classifier_empty",
            }
            continue
        if match is not None and match.confidence < MIN_RIVER_TILE_CLASSIFICATION_CONFIDENCE:
            calibrated = _manual_template_match(crop_river_candidate(image, candidate), manual_template_payload)
            if calibrated is not None:
                payload[id(candidate)] = {
                    "tile": calibrated.tile,
                    "tile_confidence": calibrated.confidence,
                    "classification_source": "manual_template_calibration",
                    "rejected_tile": match.tile,
                    "classifier_confidence": match.confidence,
                }
                continue
            payload[id(candidate)] = {
                "tile": "unknown",
                "tile_confidence": match.confidence,
                "classification_source": "low_tile_classification_confidence",
                "rejected_tile": match.tile,
            }
            continue
        calibrated = _manual_template_match(crop_river_candidate(image, candidate), manual_template_payload)
        if match is not None and calibrated is not None and _should_use_manual_template(match, calibrated):
            payload[id(candidate)] = {
                "tile": calibrated.tile,
                "tile_confidence": calibrated.confidence,
                "classification_source": "manual_template_calibration",
                "rejected_tile": match.tile,
                "classifier_confidence": match.confidence,
            }
            continue
        payload[id(candidate)] = {
            "tile": match.tile if match else "unknown",
            "tile_confidence": match.confidence if match else 0.0,
            "classification_source": "tile_classifier_dispatch" if match else "tile_classifier_no_match",
        }
    _cap_classification_tile_overflow(candidates, payload)
    return payload


def _load_manual_template_payload(labels_path: Path | None) -> dict[str, Any] | None:
    if labels_path is None:
        return None
    rows = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"manual labels must be a JSON array: {labels_path}")
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


def _cap_classification_tile_overflow(candidates: list[Any], payload: dict[int, dict[str, Any]]) -> None:
    tile_counts = Counter(
        str(item.get("tile", ""))
        for item in payload.values()
        if item.get("tile") and item.get("tile") not in {"unknown", EMPTY_TILE_LABEL}
    )
    for tile, count in tile_counts.items():
        if count <= 4:
            continue
        matched = [
            (candidate, payload[id(candidate)])
            for candidate in candidates
            if payload.get(id(candidate), {}).get("tile") == tile
        ]
        demote_count = count - 4
        demoted = sorted(
            matched,
            key=lambda item: (float(item[1].get("tile_confidence", 0.0) or 0.0), item[0].order_index),
        )[:demote_count]
        for _candidate, item in demoted:
            item["tile"] = "unknown"
            item["classification_source"] = "tile_overflow_cap"
            item["rejected_tile"] = tile


def _save_unknown_crops(
    image_path: Path,
    image: Image.Image,
    candidates: list[Any],
    classifications: dict[int, dict[str, Any]],
    out_dir: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for candidate in candidates:
        item = classifications.get(id(candidate), {})
        if item.get("tile") != "unknown":
            continue
        crop = crop_river_candidate(image, candidate)
        reason = _safe_filename(str(item.get("classification_source", "unknown")))
        rejected = _safe_filename(str(item.get("rejected_tile", "")))
        suffix = f"_{rejected}" if rejected and rejected != "unknown" else ""
        filename = (
            f"{_safe_filename(image_path.stem)}"
            f"_{candidate.player}_{candidate.order_index:02d}"
            f"_{reason}{suffix}.png"
        )
        crop.save(out_dir / filename)
        saved += 1
    return saved


def _candidate_label(candidate: Any, classifications: dict[int, dict[str, Any]]) -> str:
    prefix = f"{_short_player(candidate.player)}{candidate.order_index}"
    classification = classifications.get(id(candidate))
    if not classification:
        return prefix
    tile = classification.get("tile", "unknown")
    confidence = float(classification.get("tile_confidence", 0.0) or 0.0)
    if tile == "unknown":
        return f"{prefix} ??"
    if tile == EMPTY_TILE_LABEL:
        return f"{prefix} empty"
    return f"{prefix} {_display_tile(tile)} {confidence:.2f}"


def _display_tile(tile: str) -> str:
    honor_names = {
        "1z": "1z(E)",
        "2z": "2z(S)",
        "3z": "3z(W)",
        "4z": "4z(N)",
        "5z": "5z(Wh)",
        "6z": "6z(G)",
        "7z": "7z(R)",
    }
    return honor_names.get(tile, tile)


def _safe_filename(value: str) -> str:
    cleaned = []
    for char in str(value):
        cleaned.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(cleaned).strip("_") or "unknown"


def _short_player(player: str) -> str:
    return {
        "self": "S",
        "left_opponent": "L",
        "top_opponent": "T",
        "right_opponent": "R",
    }.get(player, "?")


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 20)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
