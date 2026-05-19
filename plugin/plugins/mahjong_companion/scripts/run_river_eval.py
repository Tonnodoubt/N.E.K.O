from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.decision.tile_efficiency import build_mahjong_analysis
from plugin.plugins.mahjong_companion.perception.bottom_hand_detector import detect_bottom_hand_tiles
from plugin.plugins.mahjong_companion.perception.tile_parser import DISCARD_TURN_HAND_COUNTS, WAITING_HAND_COUNTS
from plugin.plugins.mahjong_companion.tile_labels import format_tile_label


ARTIFACT_ROOT = Path("plugin/plugins/mahjong_companion/tests/_artifacts/river_model_spike")
RIVER_PLAYERS = ("self", "left_opponent", "top_opponent", "right_opponent")
MILESTONE_SUMMARIES = [
    ("v2-baseline", ARTIFACT_ROOT / "material-v2-unknown-crops" / "batch_summary.json"),
    ("side-threshold-0.45", ARTIFACT_ROOT / "material-v2-side-th045" / "batch_summary.json"),
    ("empty-split", ARTIFACT_ROOT / "material-v2-empty-split" / "batch_summary.json"),
    ("confidence-0.50", ARTIFACT_ROOT / "material-v2-conf050" / "batch_summary.json"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Mahjong river recognition evaluation and write a compact report.")
    parser.add_argument("--material-dir", type=Path, required=True, help="Directory containing captured Mahjong Soul frames.")
    parser.add_argument("--batch-glob", default="**/*.png", help="Glob used under --material-dir.")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACT_ROOT / "river-eval", help="Output artifact directory.")
    parser.add_argument("--manual-labels", type=Path, default=None, help="Optional manual label JSON for calibration.")
    parser.add_argument("--batch-limit", type=int, default=0, help="Maximum frames to process; 0 means all.")
    parser.add_argument("--issue-limit", type=int, default=20, help="Maximum issue frames in the contact sheet.")
    parser.add_argument("--issue-thumb-width", type=int, default=1280, help="Issue contact sheet thumbnail width.")
    parser.add_argument("--issue-columns", type=int, default=1, help="Issue contact sheet columns.")
    parser.add_argument(
        "--skip-hand-analysis",
        action="store_true",
        help="Do not run the same-frame hand parser for the strategy report.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = args.out_dir
    batch_out = out_dir / "batch"
    unknown_crops = out_dir / "unknown-crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_cmd = [
        sys.executable,
        str(script_dir / "debug_river_detector_v2.py"),
        "--batch-dir",
        str(args.material_dir),
        "--batch-glob",
        args.batch_glob,
        "--batch-out",
        str(batch_out),
        "--unknown-crop-out",
        str(unknown_crops),
        "--issue-limit",
        str(args.issue_limit),
        "--issue-thumb-width",
        str(args.issue_thumb_width),
        "--issue-columns",
        str(args.issue_columns),
        "--classify",
        "--draw-classification-quads",
    ]
    if args.batch_limit > 0:
        debug_cmd.extend(["--batch-limit", str(args.batch_limit)])
    if args.manual_labels is not None:
        debug_cmd.extend(["--manual-labels", str(args.manual_labels)])
    _run(debug_cmd)

    manifest_path = None
    manifest_summary_path = None
    if args.manual_labels is not None:
        manifest_path = out_dir / "training_manifest.jsonl"
        manifest_summary_path = out_dir / "training_manifest.summary.json"
        _run(
            [
                sys.executable,
                str(script_dir / "export_manual_river_labels.py"),
                str(args.manual_labels),
                "--out",
                str(manifest_path),
                "--summary-out",
                str(manifest_summary_path),
            ]
        )

    summary_path = batch_out / "batch_summary.json"
    report_path = out_dir / "river_eval_report.md"
    report_path.write_text(
        _report_markdown(
            current_summary_path=summary_path,
            manual_labels=args.manual_labels,
            manifest_path=manifest_path,
            manifest_summary_path=manifest_summary_path,
        ),
        encoding="utf-8",
    )
    strategy_report_path, strategy_json_path = _write_strategy_debug_report(
        summary_path,
        out_dir,
        include_hand_analysis=not args.skip_hand_analysis,
    )

    print(f"summary={summary_path}")
    print(f"report={report_path}")
    print(f"strategy_report={strategy_report_path}")
    print(f"strategy_json={strategy_json_path}")
    if manifest_path is not None:
        print(f"manifest={manifest_path}")
    print(f"unknown_crops={unknown_crops}")
    return 0


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _write_strategy_debug_report(
    summary_path: Path,
    out_dir: Path,
    *,
    include_hand_analysis: bool = True,
) -> tuple[Path, Path]:
    summary = _read_json(summary_path)
    payload = _strategy_debug_payload(summary, include_hand_analysis=include_hand_analysis)
    json_path = out_dir / "strategy_debug_summary.json"
    report_path = out_dir / "strategy_debug_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_strategy_report_markdown(payload), encoding="utf-8")
    return report_path, json_path


def _strategy_debug_payload(
    summary: dict[str, Any],
    *,
    include_hand_analysis: bool = True,
) -> dict[str, Any]:
    frames = [
        _frame_strategy_row(row, include_hand_analysis=include_hand_analysis)
        for row in summary.get("frames", [])
        if isinstance(row, dict)
    ]
    ready_frames = [
        row
        for row in frames
        if row["known_count"] > 0 and row["unknown_count"] <= 2 and not row["overflow_tiles"]
    ]
    conservative_frames = [
        row
        for row in frames
        if row["unknown_count"] > 2 or row["overflow_tiles"]
    ]
    return {
        "source_batch": summary.get("batch_dir", ""),
        "frame_count": len(frames),
        "hand_analysis_enabled": include_hand_analysis,
        "hand_ready_frame_count": sum(1 for row in frames if _strategy_hand_count_supported(row["hand_count"])),
        "candidate_discard_frame_count": sum(1 for row in frames if row["candidate_discards"]),
        "strategy_ready_frame_count": len(ready_frames),
        "conservative_frame_count": len(conservative_frames),
        "frames": frames,
    }


def _frame_strategy_row(
    row: dict[str, Any],
    *,
    include_hand_analysis: bool = True,
) -> dict[str, Any]:
    tiles_by_player = _normalized_tiles_by_player(row.get("tiles_by_player"))
    visible_tiles = [tile for player in RIVER_PLAYERS for tile in tiles_by_player.get(player, [])]
    candidate_count = int(row.get("candidate_count", 0) or 0)
    unknown_count = int(row.get("unknown_count", 0) or 0)
    empty_count = int(row.get("empty_count", 0) or 0)
    overflow_tiles = {
        str(tile): int(count)
        for tile, count in (row.get("tile_overflow_counts") or {}).items()
        if _int_value(count) > 4
    }
    hints = {
        "tile_level_available": True,
        "discard_parser_source": "model_river_adapter",
        "model_river_candidate_count": candidate_count,
        "model_river_empty_count": empty_count,
        "model_river_known_count": len(visible_tiles),
        "model_river_unknown_count": unknown_count,
        "model_river_unknown_rate": round(unknown_count / max(1, candidate_count), 4),
        "model_river_tile_overflow_counts": overflow_tiles,
        "recognized_discard_tile_count": len(visible_tiles),
        "visible_tiles": visible_tiles,
    }
    hand_payload = _hand_analysis_payload(row) if include_hand_analysis else _empty_hand_analysis_payload("disabled")
    merged_hints = dict(hand_payload["analysis_hints"])
    merged_hints.update(hints)
    hand_count = len(hand_payload["hand_tiles"])
    strategy_hand_tiles = list(hand_payload["hand_tiles"]) if _strategy_hand_count_supported(hand_count) else []
    meld_count = _strategy_meld_count(hand_count)
    if strategy_hand_tiles and meld_count:
        merged_hints["recognized_meld_group_count"] = meld_count
        merged_hints["post_meld_hand_shape"] = _strategy_hand_shape(hand_count)
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.72,
        hand_tiles=strategy_hand_tiles,
        melds=list(hand_payload["melds"]),
        dora_indicators=list(hand_payload["dora_indicators"]),
        riichi_players=list(hand_payload["riichi_players"]),
        visible_tiles=visible_tiles,
        discard_piles=_discard_piles_from_tiles(tiles_by_player),
        analysis_hints=merged_hints,
    )
    analysis = build_mahjong_analysis(state)
    return {
        "image": row.get("image", ""),
        "hand_tiles": list(hand_payload["hand_tiles"]),
        "hand_count": len(hand_payload["hand_tiles"]),
        "hand_parser_source": hand_payload["source"],
        "hand_analysis_confidence": hand_payload["confidence"],
        "dora_indicators": list(hand_payload["dora_indicators"]),
        "riichi_players": list(hand_payload["riichi_players"]),
        "candidate_count": candidate_count,
        "known_count": len(visible_tiles),
        "unknown_count": unknown_count,
        "empty_count": empty_count,
        "overflow_tiles": overflow_tiles,
        "tiles_by_player": tiles_by_player,
        "teaching_points": list(analysis.teaching_points),
        "defense_alerts": list(analysis.defense_alerts),
        "candidate_discards": list(analysis.candidate_discards),
        "tile_level_state": analysis.tile_level_state,
        "analysis_confidence": analysis.analysis_confidence,
    }


def _strategy_report_markdown(payload: dict[str, Any]) -> str:
    frames = [row for row in payload.get("frames", []) if isinstance(row, dict)]
    interesting = sorted(
        frames,
        key=lambda row: (
            int(row.get("unknown_count", 0) or 0),
            len(row.get("overflow_tiles", {}) or {}),
            int(row.get("candidate_count", 0) or 0),
        ),
        reverse=True,
    )[:20]
    sample = [row for row in frames if int(row.get("known_count", 0) or 0) > 0][:10]
    lines = [
        "# River Strategy Debug Report",
        "",
        f"- Frames: {payload.get('frame_count', 0)}",
        f"- Hand analysis enabled: {payload.get('hand_analysis_enabled', False)}",
        f"- Hand-ready frames: {payload.get('hand_ready_frame_count', 0)}",
        f"- Candidate-discard frames: {payload.get('candidate_discard_frame_count', 0)}",
        f"- Strategy-ready frames: {payload.get('strategy_ready_frame_count', 0)}",
        f"- Conservative frames: {payload.get('conservative_frame_count', 0)}",
        "",
        "## Notes",
        "",
        "- This report uses recognized river tiles as strategy input.",
            "- Candidate discards require a valid concealed-hand shape; unsupported counts report river-only strategy signals.",
    ]
    if payload.get("hand_analysis_enabled") and int(payload.get("candidate_discard_frame_count", 0) or 0) == 0:
        lines.append(
            "- Hand parser blocker: no frame reached a supported concealed-hand count; "
            "candidate discard output is unavailable until hand calibration/detection works on this material."
        )
    lines.extend(
        [
            "",
            "## Frames Needing Review",
            "",
            "| Frame | Hand | Known | Unknown | Empty | Overflow | Candidate discard | Main strategy output |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in interesting:
        lines.append(_strategy_frame_table_row(row))
    lines.extend(["", "## Sample River Inputs", ""])
    for row in sample:
        lines.extend(_strategy_frame_detail(row))
    lines.append("")
    return "\n".join(lines)


def _strategy_frame_table_row(row: dict[str, Any]) -> str:
    points = row.get("teaching_points") if isinstance(row.get("teaching_points"), list) else []
    main_point = str(points[0]) if points else ""
    overflow = row.get("overflow_tiles") if isinstance(row.get("overflow_tiles"), dict) else {}
    candidates = row.get("candidate_discards") if isinstance(row.get("candidate_discards"), list) else []
    return (
        f"| `{Path(str(row.get('image', ''))).name}` "
        f"| {row.get('hand_count', 0)} "
        f"| {row.get('known_count', 0)} "
        f"| {row.get('unknown_count', 0)} "
        f"| {row.get('empty_count', 0)} "
        f"| {_format_overflow(overflow)} "
        f"| {_format_candidate_discards(candidates)} "
        f"| {main_point} |"
    )


def _strategy_frame_detail(row: dict[str, Any]) -> list[str]:
    lines = [
        f"### {Path(str(row.get('image', ''))).name}",
        "",
        f"- Hand: {_format_tile_list(row.get('hand_tiles'))} ({row.get('hand_parser_source', 'unknown')}, confidence={row.get('hand_analysis_confidence', 0.0)})",
        f"- Counts: known={row.get('known_count', 0)}, unknown={row.get('unknown_count', 0)}, empty={row.get('empty_count', 0)}",
    ]
    tiles_by_player = row.get("tiles_by_player") if isinstance(row.get("tiles_by_player"), dict) else {}
    for player in RIVER_PLAYERS:
        tiles = tiles_by_player.get(player, [])
        lines.append(f"- {player}: {_format_tile_list(tiles)}")
    candidate_discards = row.get("candidate_discards") if isinstance(row.get("candidate_discards"), list) else []
    if candidate_discards:
        lines.append(f"- Candidate discard: {_format_candidate_discards(candidate_discards)}")
    else:
        lines.append("- Candidate discard: unavailable without a valid hand shape")
    points = row.get("teaching_points") if isinstance(row.get("teaching_points"), list) else []
    if points:
        lines.append(f"- Strategy: {str(points[0])}")
    alerts = row.get("defense_alerts") if isinstance(row.get("defense_alerts"), list) else []
    if alerts:
        lines.append(f"- Defense: {str(alerts[0])}")
    lines.append("")
    return lines


def _hand_analysis_payload(row: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(str(row.get("image", "")))
    if not image_path.exists():
        return _empty_hand_analysis_payload("missing_image")
    try:
        from plugin.plugins.mahjong_companion.perception.pipeline import analyze_image_path

        state, _debug = analyze_image_path(image_path, fixture_mode="disabled")
    except (OSError, ValueError, RuntimeError) as exc:
        payload = _empty_hand_analysis_payload("error")
        payload["error"] = str(exc)
        return payload
    hints = dict(state.analysis_hints) if isinstance(state.analysis_hints, dict) else {}
    if not state.hand_tiles:
        fallback = _bottom_hand_detection_payload(image_path)
        if fallback["hand_tiles"]:
            return fallback
    return {
        "hand_tiles": list(state.hand_tiles),
        "melds": [list(group) for group in state.melds],
        "dora_indicators": list(state.dora_indicators),
        "riichi_players": list(state.riichi_players),
        "source": str(hints.get("tile_parser_source", "")) or "pipeline",
        "confidence": float(hints.get("analysis_confidence", 0.0) or 0.0),
        "analysis_hints": hints,
        "error": "",
    }


def _bottom_hand_detection_payload(image_path: Path) -> dict[str, Any]:
    try:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
    except OSError as exc:
        payload = _empty_hand_analysis_payload("bottom_hand_detector_error")
        payload["error"] = str(exc)
        return payload
    detection = detect_bottom_hand_tiles(image)
    if not detection.hand_tiles:
        payload = _empty_hand_analysis_payload(detection.source)
        payload["analysis_hints"].update(detection.analysis_hints())
        return payload
    return {
        "hand_tiles": list(detection.hand_tiles),
        "melds": [],
        "dora_indicators": [],
        "riichi_players": [],
        "source": detection.source,
        "confidence": detection.confidence,
        "analysis_hints": detection.analysis_hints(),
        "error": "",
    }


def _empty_hand_analysis_payload(source: str) -> dict[str, Any]:
    return {
        "hand_tiles": [],
        "melds": [],
        "dora_indicators": [],
        "riichi_players": [],
        "source": source,
        "confidence": 0.0,
        "analysis_hints": {},
        "error": "",
    }


def _strategy_hand_count_supported(hand_count: int) -> bool:
    return hand_count in DISCARD_TURN_HAND_COUNTS or hand_count in WAITING_HAND_COUNTS


def _strategy_meld_count(hand_count: int) -> int:
    return DISCARD_TURN_HAND_COUNTS.get(hand_count) or WAITING_HAND_COUNTS.get(hand_count, 0)


def _strategy_hand_shape(hand_count: int) -> str:
    if hand_count in DISCARD_TURN_HAND_COUNTS:
        return "discard_turn"
    if hand_count in WAITING_HAND_COUNTS:
        return "waiting"
    return ""


def _report_markdown(
    *,
    current_summary_path: Path,
    manual_labels: Path | None,
    manifest_path: Path | None,
    manifest_summary_path: Path | None,
) -> str:
    rows = []
    for name, path in MILESTONE_SUMMARIES:
        summary = _read_json(path)
        if summary:
            rows.append((name, path, summary))
    current = _read_json(current_summary_path)
    if current:
        rows.append(("current", current_summary_path, current))

    lines = [
        "# River Recognition Eval",
        "",
        "## Trend",
        "",
        "| Run | Frames | Candidates | Unknown | Empty | Unknown rate | Overflow frames | Count anomalies |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, path, summary in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"[{name}]({_relative(path, current_summary_path.parent.parent)})",
                    str(summary.get("frame_count", 0)),
                    str(summary.get("candidate_count", 0)),
                    str(summary.get("unknown_count", 0)),
                    str(summary.get("empty_count", 0)),
                    _unknown_rate(summary),
                    str(summary.get("tile_overflow_frame_count", 0)),
                    str(summary.get("player_count_anomaly_frame_count", 0)),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Current Output",
            "",
            f"- Summary: `{current_summary_path}`",
            f"- Batch report: `{current_summary_path.with_name('batch_report.md')}`",
            f"- Issue sheet: `{current_summary_path.with_name('batch_issues.png')}`",
            f"- Unknown crops: `{current_summary_path.parent.parent / 'unknown-crops'}`",
        ]
    )
    if manual_labels is not None:
        lines.append(f"- Manual labels: `{manual_labels}`")
    if manifest_path is not None:
        lines.append(f"- Training manifest: `{manifest_path}`")
    if manifest_summary_path is not None:
        manifest_summary = _read_json(manifest_summary_path)
        if manifest_summary:
            lines.append(f"- Manifest labels: {_format_counter(manifest_summary.get('labels', {}))}")
    lines.extend(
        [
            "",
            "![Current issue sheet](batch/batch_issues.png)",
            "",
            "## Notes",
            "",
            "- Runtime path can use `MAHJONG_COMPANION_MODEL_RIVER_MANUAL_LABELS` to enable the local manual-template calibration.",
            "- The current manual-template score is in-sample for the 21 manually labeled crops; use it to validate the direction, not as final model quality.",
            "- Long-term replacement for hosted Roboflow is still a local detector trained/exported from this growing crop manifest.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _unknown_rate(summary: dict[str, Any]) -> str:
    candidate_count = int(summary.get("candidate_count", 0) or 0)
    unknown_count = int(summary.get("unknown_count", 0) or 0)
    if candidate_count == 0:
        return "0.00%"
    return f"{unknown_count / candidate_count:.2%}"


def _format_counter(values: Any) -> str:
    if not isinstance(values, dict) or not values:
        return "none"
    return ", ".join(f"{key}={values[key]}" for key in sorted(values))


def _normalized_tiles_by_player(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {player: [] for player in RIVER_PLAYERS}
    normalized: dict[str, list[str]] = {}
    for player in RIVER_PLAYERS:
        items = value.get(player, [])
        normalized[player] = [str(tile).strip() for tile in items if str(tile).strip()]
    return normalized


def _discard_piles_from_tiles(tiles_by_player: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    piles: dict[str, list[dict[str, Any]]] = {}
    for player, tiles in tiles_by_player.items():
        piles[player] = [
            {
                "tile": tile,
                "player": player,
                "turn_index": index,
                "confidence": 1.0,
                "source": "river_eval_summary",
            }
            for index, tile in enumerate(tiles, start=1)
        ]
    return piles


def _format_tile_list(tiles: Any) -> str:
    if not isinstance(tiles, list) or not tiles:
        return "none"
    return " ".join(format_tile_label(str(tile)) for tile in tiles)


def _format_candidate_discards(candidates: list[Any]) -> str:
    parts = []
    for item in candidates[:3]:
        if not isinstance(item, dict):
            continue
        tile = str(item.get("tile", "")).strip()
        if tile:
            parts.append(format_tile_label(tile))
    return ", ".join(parts) if parts else "none"


def _format_overflow(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{format_tile_label(tile)}×{count}" for tile, count in sorted(values.items()))


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
