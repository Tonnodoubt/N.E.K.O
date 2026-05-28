"""Run offline perception diagnostics over Mahjong Soul frame screenshots.

This tool does not change plugin state. It replays saved frames through the
same hand, meld, river, and action-window detectors used by the live coach,
then writes machine-readable summaries and optional annotated screenshots.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import types
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _ensure_package(name: str, path: Path | None = None) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    if path is not None:
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = mod


def _load_module(name: str, path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name}: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)


def _bootstrap_perception_modules(repo_root: Path) -> None:
    plugin_root = repo_root / "plugin"
    plugins_root = plugin_root / "plugins"
    mahjong_root = plugins_root / "mahjong_coach"
    perception_root = mahjong_root / "perception"
    _ensure_package("plugin", plugin_root)
    _ensure_package("plugin.plugins", plugins_root)
    _ensure_package("plugin.plugins.mahjong_coach", mahjong_root)
    _ensure_package("plugin.plugins.mahjong_coach.perception", perception_root)

    modules = [
        ("plugin.plugins.mahjong_coach.tile_labels", mahjong_root / "tile_labels.py"),
        ("plugin.plugins.mahjong_coach.perception.roi", perception_root / "roi.py"),
        ("plugin.plugins.mahjong_coach.perception.calibration", perception_root / "calibration.py"),
        ("plugin.plugins.mahjong_coach.perception.hand_layout", perception_root / "hand_layout.py"),
        ("plugin.plugins.mahjong_coach.perception.discard_layout", perception_root / "discard_layout.py"),
        ("plugin.plugins.mahjong_coach.perception.vit_tile_classifier_onnx", perception_root / "vit_tile_classifier_onnx.py"),
        ("plugin.plugins.mahjong_coach.perception.tile_templates", perception_root / "tile_templates.py"),
        ("plugin.plugins.mahjong_coach.perception.tile_classifier_dispatch", perception_root / "tile_classifier_dispatch.py"),
        ("plugin.plugins.mahjong_coach.perception.fast_hand_path", perception_root / "fast_hand_path.py"),
        ("plugin.plugins.mahjong_coach.perception.discard_parser", perception_root / "discard_parser.py"),
        ("plugin.plugins.mahjong_coach.perception.river_state", perception_root / "river_state.py"),
        ("plugin.plugins.mahjong_coach.perception.meld_state", perception_root / "meld_state.py"),
        ("plugin.plugins.mahjong_coach.perception.riichi_detector", perception_root / "riichi_detector.py"),
        ("plugin.plugins.mahjong_coach.perception.action_detector", perception_root / "action_detector.py"),
    ]
    for name, path in modules:
        _load_module(name, path)


def default_live_frames_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "N.E.K.O" / "plugins" / "mahjong_coach" / "data" / "live_frames"
    return Path.home() / "AppData" / "Local" / "N.E.K.O" / "plugins" / "mahjong_coach" / "data" / "live_frames"


def default_output_dir(repo_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return repo_root / "tmp" / f"mahjong_frame_diagnostics_{stamp}"


def iter_frames(input_dir: Path, *, limit: int = 0) -> list[Path]:
    frames = [path for path in sorted(input_dir.iterdir()) if path.suffix.lower() in IMG_EXTENSIONS and path.is_file()]
    if limit > 0:
        return frames[:limit]
    return frames


def diagnose_frame(
    frame_path: Path,
    *,
    calibration_dir: Path,
    min_hand_tiles: int,
    min_river_confidence: float,
    min_meld_confidence: float,
) -> dict[str, Any]:
    from plugin.plugins.mahjong_coach.perception.action_detector import detect_action_buttons_fast
    from plugin.plugins.mahjong_coach.perception.fast_hand_path import detect_fast_hand_path
    from plugin.plugins.mahjong_coach.perception.meld_state import detect_meld_state_path
    from plugin.plugins.mahjong_coach.perception.riichi_detector import detect_riichi_sticks
    from plugin.plugins.mahjong_coach.perception.river_state import detect_river_state_path

    started = time.perf_counter()
    with Image.open(frame_path) as opened:
        width, height = opened.size

    hand = detect_fast_hand_path(
        frame_path,
        calibration_dir=calibration_dir,
        min_hand_tiles=min_hand_tiles,
        max_hand_tiles=14,
        use_onnx_hand=False,
    )
    hand_count = len(hand.hand_tiles)
    meld = detect_meld_state_path(
        frame_path,
        min_confidence=min_meld_confidence,
        closed_hand_count=hand_count or None,
    )
    river = detect_river_state_path(
        frame_path,
        calibration_dir=calibration_dir,
        min_confidence=min_river_confidence,
    )
    buttons, action_meta = detect_action_buttons_fast(frame_path)
    riichi = detect_riichi_sticks(frame_path)

    meld_count = int(meld.open_meld_count or 0)
    expected_closed = _expected_closed_counts(meld_count)
    flags = _diagnostic_flags(hand, meld, river, buttons, expected_closed, riichi.riichi_players)

    return {
        "frame": frame_path.name,
        "path": str(frame_path),
        "width": width,
        "height": height,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "hand": hand.to_dict(),
        "hand_count": hand_count,
        "hand_tiles": list(hand.hand_tiles),
        "hand_expected_counts": expected_closed,
        "meld": meld.to_dict(),
        "meld_count": meld_count,
        "meld_tiles": list(meld.tiles),
        "river": river.to_dict(),
        "river_visible_count": len(river.visible_tiles),
        "river_visible_tiles": list(river.visible_tiles),
        "river_by_player": {player: len(items) for player, items in river.discard_piles.items()},
        "action_buttons": list(buttons),
        "action": action_meta,
        "riichi_players": list(riichi.riichi_players),
        "riichi": riichi.to_dict(),
        "flags": flags,
    }


def _expected_closed_counts(open_meld_count: int) -> list[int]:
    if open_meld_count <= 0:
        return [13, 14]
    after_discard = max(1, 13 - 3 * open_meld_count)
    on_turn = max(1, 14 - 3 * open_meld_count)
    return sorted({after_discard, on_turn})


def _diagnostic_flags(
    hand: Any,
    meld: Any,
    river: Any,
    buttons: list[str],
    expected_closed: list[int],
    riichi_players: list[str],
) -> list[str]:
    flags: list[str] = []
    if not hand.ok:
        flags.append(f"hand:{hand.reason or 'not_ok'}")
    if hand.ok and expected_closed and len(hand.hand_tiles) not in expected_closed:
        flags.append(f"hand_count_unexpected:{len(hand.hand_tiles)}_not_{'-'.join(map(str, expected_closed))}")
    if meld.ok:
        flags.append(f"meld:{meld.open_meld_count}")
    if river.ok and not river.visible_tiles:
        flags.append("river:no_visible_tiles")
    if not river.ok:
        flags.append(f"river:{river.reason or 'not_ok'}")
    if buttons:
        flags.append(f"action:{'+'.join(buttons)}")
    if riichi_players:
        flags.append(f"riichi:{'+'.join(riichi_players)}")
    return flags


def annotate_frame(frame_path: Path, result: dict[str, Any], output_path: Path) -> None:
    with Image.open(frame_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for detection in result.get("hand", {}).get("raw_detections", []):
        _draw_box_detection(draw, detection, font, accepted_color=(45, 220, 120), rejected_color=(255, 210, 80), idle_color=(120, 120, 120))
    for detection in result.get("river", {}).get("raw_detections", []):
        _draw_quad_detection(draw, detection, font, accepted_color=(50, 180, 255), rejected_color=(255, 120, 80))
    for detection in result.get("meld", {}).get("raw_detections", []):
        _draw_box_detection(draw, detection, font, accepted_color=(190, 120, 255), rejected_color=(255, 120, 80), idle_color=(120, 120, 120))

    _draw_header(draw, result, font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=88)


def _draw_header(draw: ImageDraw.ImageDraw, result: dict[str, Any], font: ImageFont.ImageFont) -> None:
    lines = [
        result["frame"],
        f"hand {result['hand_count']}: {_join_tiles(result['hand_tiles']) or '-'}",
        f"meld {result['meld_count']}: {_join_tiles(result['meld_tiles']) or '-'}",
        f"river {result['river_visible_count']}  action: {','.join(result['action_buttons']) or '-'}",
        f"riichi: {','.join(result.get('riichi_players') or []) or '-'}",
        f"flags: {', '.join(result['flags']) or '-'}",
    ]
    line_h = 14
    width = max(360, max(len(line) for line in lines) * 7 + 16)
    height = line_h * len(lines) + 12
    draw.rectangle((8, 8, 8 + width, 8 + height), fill=(0, 0, 0), outline=(255, 255, 255))
    for index, line in enumerate(lines):
        draw.text((16, 14 + index * line_h), line, fill=(255, 255, 255), font=font)


def _draw_box_detection(
    draw: ImageDraw.ImageDraw,
    detection: dict[str, Any],
    font: ImageFont.ImageFont,
    *,
    accepted_color: tuple[int, int, int],
    rejected_color: tuple[int, int, int],
    idle_color: tuple[int, int, int],
) -> None:
    box = detection.get("box")
    if not isinstance(box, dict):
        return
    left = int(box.get("left") or 0)
    top = int(box.get("top") or 0)
    right = left + int(box.get("width") or 0)
    bottom = top + int(box.get("height") or 0)
    occupied = bool(detection.get("occupied"))
    accepted = bool(detection.get("accepted"))
    color = accepted_color if accepted else rejected_color if occupied else idle_color
    draw.rectangle((left, top, right, bottom), outline=color, width=3 if occupied else 1)
    if occupied:
        _draw_label(draw, left, top, _detection_label(detection), color, font)


def _draw_quad_detection(
    draw: ImageDraw.ImageDraw,
    detection: dict[str, Any],
    font: ImageFont.ImageFont,
    *,
    accepted_color: tuple[int, int, int],
    rejected_color: tuple[int, int, int],
) -> None:
    quad = detection.get("quad")
    if not isinstance(quad, list) or len(quad) < 4:
        return
    points = [(int(point[0]), int(point[1])) for point in quad if isinstance(point, list) and len(point) >= 2]
    if len(points) < 4:
        return
    accepted = bool(detection.get("accepted"))
    color = accepted_color if accepted else rejected_color
    draw.line(points + [points[0]], fill=color, width=3)
    left = min(point[0] for point in points)
    top = min(point[1] for point in points)
    _draw_label(draw, left, top, _detection_label(detection), color, font)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    if not text:
        return
    y = max(0, top - 14)
    text_w = max(30, len(text) * 6 + 6)
    draw.rectangle((left, y, left + text_w, y + 13), fill=(0, 0, 0))
    draw.text((left + 3, y + 1), text, fill=color, font=font)


def _detection_label(detection: dict[str, Any]) -> str:
    tile = str(detection.get("candidate_tile") or "")
    confidence = float(detection.get("confidence") or 0.0)
    if tile:
        return f"{tile} {confidence:.2f}"
    if detection.get("rejection_reason"):
        return str(detection.get("rejection_reason"))
    return "occupied" if detection.get("occupied") else ""


def write_outputs(results: list[dict[str, Any]], output_dir: Path, *, annotate: bool, input_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "frames.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    _write_csv(results, output_dir / "summary.csv")
    _write_markdown(results, output_dir / "summary.md", annotate=annotate, input_dir=input_dir)


def _write_csv(results: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "frame",
        "hand_count",
        "hand_tiles",
        "hand_ok",
        "hand_reason",
        "hand_confidence",
        "meld_count",
        "meld_tiles",
        "meld_confidence",
        "river_visible_count",
        "river_confidence",
        "action_buttons",
        "riichi_players",
        "riichi_stick_count",
        "flags",
        "elapsed_ms",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "frame": result["frame"],
                    "hand_count": result["hand_count"],
                    "hand_tiles": " ".join(result["hand_tiles"]),
                    "hand_ok": result["hand"].get("ok"),
                    "hand_reason": result["hand"].get("reason"),
                    "hand_confidence": result["hand"].get("confidence"),
                    "meld_count": result["meld_count"],
                    "meld_tiles": " ".join(result["meld_tiles"]),
                    "meld_confidence": result["meld"].get("confidence"),
                    "river_visible_count": result["river_visible_count"],
                    "river_confidence": result["river"].get("confidence"),
                    "action_buttons": " ".join(result["action_buttons"]),
                    "riichi_players": " ".join(result.get("riichi_players") or []),
                    "riichi_stick_count": result.get("riichi", {}).get("stick_count"),
                    "flags": ";".join(result["flags"]),
                    "elapsed_ms": result["elapsed_ms"],
                }
            )


def _write_markdown(results: list[dict[str, Any]], path: Path, *, annotate: bool, input_dir: Path) -> None:
    flag_counts = Counter(flag.split(":", 1)[0] for result in results for flag in result["flags"])
    hand_reasons = Counter(str(result["hand"].get("reason") or "") for result in results)
    river_reasons = Counter(str(result["river"].get("reason") or "") for result in results)
    action_count = sum(1 for result in results if result["action_buttons"])
    riichi_count = sum(1 for result in results if result.get("riichi_players"))
    meld_count = sum(1 for result in results if result["meld_count"] > 0)
    unexpected = [result for result in results if any(flag.startswith("hand_count_unexpected") for flag in result["flags"])]
    lines = [
        "# Mahjong Frame Diagnostics",
        "",
        f"- input: `{input_dir}`",
        f"- frames: {len(results)}",
        f"- action frames: {action_count}",
        f"- riichi stick frames: {riichi_count}",
        f"- frames with self melds: {meld_count}",
        f"- unexpected hand count frames: {len(unexpected)}",
        f"- annotations: {'enabled' if annotate else 'disabled'}",
        "",
        "## Flag Counts",
        "",
        *_counter_lines(flag_counts),
        "",
        "## Hand Reasons",
        "",
        *_counter_lines(hand_reasons),
        "",
        "## River Reasons",
        "",
        *_counter_lines(river_reasons),
        "",
        "## Suspect Frames",
        "",
        "| frame | hand | expected | meld | river | action | riichi | flags |",
        "|---|---:|---|---:|---:|---|---|---|",
    ]
    for result in _top_suspects(results):
        lines.append(
            "| {frame} | {hand_count} | {expected} | {meld_count} | {river_visible_count} | {action} | {riichi} | {flags} |".format(
                frame=result["frame"],
                hand_count=result["hand_count"],
                expected="/".join(map(str, result["hand_expected_counts"])),
                meld_count=result["meld_count"],
                river_visible_count=result["river_visible_count"],
                action=" ".join(result["action_buttons"]) or "-",
                riichi=" ".join(result.get("riichi_players") or []) or "-",
                flags=", ".join(result["flags"]) or "-",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _counter_lines(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["- none"]
    return [f"- `{key or '-'}`: {value}" for key, value in counter.most_common()]


def _top_suspects(results: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    flagged = [result for result in results if result["flags"]]
    return flagged[:limit]


def _join_tiles(tiles: list[str]) -> str:
    return " ".join(str(tile) for tile in tiles if str(tile).strip())


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Diagnose Mahjong Coach perception over saved frames")
    parser.add_argument("--input-dir", default=str(default_live_frames_dir()), help="Directory containing saved frame images")
    parser.add_argument("--output-dir", default="", help="Output diagnostics directory")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N frames")
    parser.add_argument("--min-hand-tiles", type=int, default=1, help="Minimum hand tiles accepted for diagnostics")
    parser.add_argument("--min-river-confidence", type=float, default=0.90)
    parser.add_argument("--min-meld-confidence", type=float, default=0.72)
    parser.add_argument("--no-annotate", action="store_true", help="Skip annotated screenshots")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(repo_root)
    calibration_dir = repo_root / "plugin" / "plugins" / "mahjong_coach" / "data" / "calibration" / "profiles"
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    _bootstrap_perception_modules(repo_root)
    frames = iter_frames(input_dir, limit=args.limit)
    if not frames:
        raise SystemExit(f"No frame images found: {input_dir}")

    annotate = not args.no_annotate
    results: list[dict[str, Any]] = []
    annotated_dir = output_dir / "annotated"
    for index, frame in enumerate(frames, start=1):
        result = diagnose_frame(
            frame,
            calibration_dir=calibration_dir,
            min_hand_tiles=args.min_hand_tiles,
            min_river_confidence=args.min_river_confidence,
            min_meld_confidence=args.min_meld_confidence,
        )
        results.append(result)
        if annotate:
            annotate_frame(frame, result, annotated_dir / f"{frame.stem}-diagnostic.jpg")
        print(
            f"[{index}/{len(frames)}] {frame.name} "
            f"hand={result['hand_count']} meld={result['meld_count']} "
            f"river={result['river_visible_count']} action={','.join(result['action_buttons']) or '-'}"
            f" riichi={','.join(result.get('riichi_players') or []) or '-'}"
        )

    write_outputs(results, output_dir, annotate=annotate, input_dir=input_dir)
    print(f"\nDiagnostics written to: {output_dir}")


if __name__ == "__main__":
    main()
