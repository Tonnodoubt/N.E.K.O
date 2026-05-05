from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..perception.roi import build_default_rois, collect_region_metrics
from ..perception.tile_parser import parse_tiles_from_image
from ..storage import write_json_atomic
from .prepare_discard_fixture import LABEL_SCOPES, prepare_discard_fixture


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "data" / "calibration"
DEFAULT_OUTPUT_DIR = Path("plugin/plugins/mahjong_companion/plans/artifacts/discard_labeling_batch")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare_discard_labeling_batch(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        calibration_dir=Path(args.calibration_dir) if args.calibration_dir else _default_calibration_dir(),
        recursive=bool(args.recursive),
        limit=int(args.limit),
        min_confidence=float(args.min_confidence),
        label_scope=args.label_scope,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result["ok"] else 1


def prepare_discard_labeling_batch(
    *,
    input_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    calibration_dir: Path | None = None,
    recursive: bool = False,
    limit: int = 0,
    min_confidence: float = 0.35,
    label_scope: str = "partial",
    overwrite: bool = False,
) -> dict[str, Any]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    calibration_dir = calibration_dir or _default_calibration_dir()
    image_paths = _discover_images(input_dir, recursive=recursive, limit=limit)
    cases: list[dict[str, Any]] = []
    errors: list[str] = []

    for image_path in image_paths:
        try:
            case = _prepare_case(
                image_path,
                output_dir=output_dir,
                calibration_dir=calibration_dir,
                min_confidence=min_confidence,
                label_scope=label_scope,
                overwrite=overwrite,
            )
        except Exception as exc:
            errors.append(f"{image_path}: {exc}")
            continue
        cases.append(case)

    review = {
        "ok": not errors,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "calibration_dir": str(calibration_dir) if calibration_dir is not None else "",
        "recursive": recursive,
        "image_count": len(image_paths),
        "case_count": len(cases),
        "candidate_count": sum(len(case["candidates"]) for case in cases),
        "min_confidence": round(float(min_confidence), 3),
        "label_scope": label_scope,
        "cases": cases,
        "errors": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    review_json_path = output_dir / "discard-labeling-review.json"
    review_md_path = output_dir / "discard-labeling-review.md"
    write_json_atomic(review_json_path, review)
    review_md_path.write_text(_render_review_markdown(review), encoding="utf-8")
    review["review_json_path"] = str(review_json_path)
    review["review_markdown_path"] = str(review_md_path)
    return review


def _prepare_case(
    image_path: Path,
    *,
    output_dir: Path,
    calibration_dir: Path | None,
    min_confidence: float,
    label_scope: str,
    overwrite: bool,
) -> dict[str, Any]:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene="in_match",
        metrics=_frame_metrics(image),
        calibration_dir=calibration_dir,
        fixture_mode="disabled",
    )
    candidates = _candidate_items(parsed.discard_piles, min_confidence=min_confidence)
    fixture = prepare_discard_fixture(
        image_path=image_path,
        case_id=image_path.stem,
        discard_items=candidates,
        output_dir=output_dir,
        label_scope=label_scope,
        overwrite=overwrite,
    )
    return {
        "case_id": fixture["case_id"],
        "image_path": str(image_path),
        "frame_path": fixture["frame_path"],
        "overlay_path": fixture["overlay_path"],
        "sheet_path": fixture["sheet_path"],
        "label_path": fixture["label_path"],
        "candidate_count": len(candidates),
        "label_scope": fixture["label_scope"],
        "candidates": candidates,
        "analysis_hints": {
            key: parsed.analysis_hints.get(key)
            for key in (
                "tile_parser_source",
                "discard_template_source",
                "recognized_discard_tile_count",
                "occupied_discard_slot_count",
                "discard_analysis_confidence",
            )
            if key in parsed.analysis_hints
        },
    }


def _candidate_items(
    discard_piles: dict[str, list[dict[str, Any]]],
    *,
    min_confidence: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for player, pile in discard_piles.items():
        for item in pile:
            confidence = _float_value(item.get("confidence"))
            if confidence < min_confidence:
                continue
            candidates.append(
                {
                    "player": player,
                    "turn_index": int(item.get("turn_index", 0) or 0),
                    "tile": str(item.get("tile", "")).strip(),
                    "confidence": round(confidence, 3),
                    "bbox": item.get("bbox", []),
                    "quad": item.get("quad", []),
                    "orientation": item.get("orientation", ""),
                }
            )
    candidates.sort(key=lambda item: (str(item["player"]), int(item["turn_index"])))
    return candidates


def _frame_metrics(image: Image.Image) -> dict[str, dict[str, Any]]:
    rois = build_default_rois(*image.size)
    metrics = {name: collect_region_metrics(image, roi) for name, roi in rois.items()}
    metrics["full_frame"] = collect_region_metrics(image, None)
    return metrics


def _discover_images(input_dir: Path, *, recursive: bool, limit: int) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if limit > 0:
        return images[:limit]
    return images


def _render_review_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Discard Labeling Review",
        "",
        f"- input_dir: `{review['input_dir']}`",
        f"- image_count: {review['image_count']}",
        f"- candidate_count: {review['candidate_count']}",
        "",
    ]
    for case in review["cases"]:
        lines.append(f"## {case['case_id']}")
        lines.append("")
        lines.append(f"- image: `{case['image_path']}`")
        lines.append(f"- overlay: `{case['overlay_path']}`")
        lines.append(f"- sheet: `{case['sheet_path']}`")
        if not case["candidates"]:
            lines.append("- candidates: none")
        for candidate in case["candidates"]:
            lines.append(
                "- candidate: "
                f"{candidate['player']}:{candidate['turn_index']} "
                f"= {candidate['tile']} "
                f"(confidence={candidate['confidence']})",
            )
        lines.append("")
    if review["errors"]:
        lines.append("## Errors")
        lines.append("")
        for error in review["errors"]:
            lines.append(f"- {error}")
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch prepare discard labeling overlays and review candidates.")
    parser.add_argument("--input-dir", required=True, help="Directory containing screenshots.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--calibration-dir", default="")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--label-scope", choices=sorted(LABEL_SCOPES), default="partial")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def _default_calibration_dir() -> Path | None:
    return DEFAULT_CALIBRATION_DIR if DEFAULT_CALIBRATION_DIR.exists() else None


def _float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
