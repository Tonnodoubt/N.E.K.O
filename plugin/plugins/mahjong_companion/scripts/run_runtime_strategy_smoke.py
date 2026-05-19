from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from plugin.plugins.mahjong_companion.adapters import DefaultPerceptionAdapter
from plugin.plugins.mahjong_companion.contracts import DecisionResult, PerceivedGameState
from plugin.plugins.mahjong_companion.decision.generator import build_decision
from plugin.plugins.mahjong_companion.overlay.view import _advice_view
from plugin.plugins.mahjong_companion.tile_labels import format_tile_label


DEFAULT_OUT_DIR = Path("plugin/plugins/mahjong_companion/tests/_artifacts/runtime_strategy_smoke")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run runtime Mahjong strategy smoke on captured frames.")
    parser.add_argument("--material-dir", type=Path, required=True, help="Directory containing captured frames.")
    parser.add_argument("--batch-glob", default="**/*.png", help="Glob used under --material-dir.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output artifact directory.")
    parser.add_argument("--batch-limit", type=int, default=20, help="Maximum frames to process; 0 means all.")
    parser.add_argument(
        "--fixture-mode",
        default="disabled",
        choices=["auto", "disabled"],
        help="Perception fixture mode. Default disables fixtures to test runtime CV.",
    )
    parser.add_argument("--model-river-json-dir", type=Path, default=None, help="Optional model-river JSON cache dir.")
    parser.add_argument("--manual-labels", type=Path, default=None, help="Optional manual labels for model-river fallback.")
    args = parser.parse_args()

    _configure_runtime_env(args)

    frame_paths = sorted(path for path in args.material_dir.glob(args.batch_glob) if path.is_file())
    if args.batch_limit > 0:
        frame_paths = frame_paths[: args.batch_limit]

    adapter = DefaultPerceptionAdapter(fixture_mode=args.fixture_mode)
    rows = [_runtime_frame_row(adapter, path) for path in frame_paths]
    payload = _summary_payload(args.material_dir, rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "runtime_strategy_smoke.json"
    report_path = args.out_dir / "runtime_strategy_smoke.md"
    contact_sheet_path = args.out_dir / "runtime_strategy_smoke.png"
    if rows:
        _save_contact_sheet(rows, contact_sheet_path)
        payload["contact_sheet"] = contact_sheet_path.name
        frame_review_dir = args.out_dir / "frame_reviews"
        payload["frame_review_dir"] = frame_review_dir.name
        payload["frame_review_images"] = _save_frame_review_images(rows, frame_review_dir)
        audit = _build_auto_audit(payload)
        payload["auto_audit"] = audit
        audit_report_path = args.out_dir / "auto_audit_report.md"
        audit_report_path.write_text(_audit_markdown(audit), encoding="utf-8")
        _save_suspicious_frames(audit, frame_review_dir, args.out_dir / "suspicious_frames")
        payload["auto_audit_report"] = audit_report_path.name
        payload["suspicious_frame_dir"] = "suspicious_frames"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_report_markdown(payload), encoding="utf-8")

    print(f"summary={json_path}")
    print(f"report={report_path}")
    if rows:
        print(f"contact_sheet={contact_sheet_path}")
        print(f"frame_reviews={args.out_dir / 'frame_reviews'}")
        print(f"auto_audit={args.out_dir / 'auto_audit_report.md'}")
        print(f"suspicious_frames={args.out_dir / 'suspicious_frames'}")
    return 0


def _configure_runtime_env(args: argparse.Namespace) -> None:
    if args.model_river_json_dir is not None:
        os.environ["MAHJONG_COMPANION_MODEL_RIVER_JSON_DIR"] = str(args.model_river_json_dir)
    if args.manual_labels is not None:
        os.environ["MAHJONG_COMPANION_MODEL_RIVER_MANUAL_LABELS"] = str(args.manual_labels)


def _runtime_frame_row(adapter: DefaultPerceptionAdapter, image_path: Path) -> dict[str, Any]:
    state, debug_payload = adapter.analyze(image_path, live=False)
    decision = build_decision(state)
    overlay = _advice_view(_overlay_status(state, decision))
    analysis = decision.mahjong_analysis if isinstance(decision.mahjong_analysis, dict) else {}
    candidates = analysis.get("candidate_discards") if isinstance(analysis.get("candidate_discards"), list) else []
    top_candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    hints = state.analysis_hints if isinstance(state.analysis_hints, dict) else {}
    return {
        "image": str(image_path),
        "scene": state.scene,
        "effective_scene": decision.scene,
        "confidence": round(float(state.confidence or 0.0), 4),
        "is_user_turn": bool(state.is_user_turn),
        "hand_source": str(hints.get("tile_parser_source", "")),
        "hand_count": len(state.hand_tiles),
        "hand_tiles": list(state.hand_tiles),
        "visible_tile_count": len(state.visible_tiles),
        "visible_tiles": list(state.visible_tiles),
        "discard_piles": dict(state.discard_piles),
        "discard_parser_source": str(hints.get("discard_parser_source", "")),
        "candidate_discards": candidates,
        "candidate_strength": str(top_candidate.get("recommendation_strength", "")),
        "decision_type": decision.decision_type,
        "recommended_focus": decision.recommended_focus,
        "reason_codes": list(decision.reason_codes),
        "overlay_primary": overlay.get("primary", ""),
        "overlay_reason": overlay.get("reason", ""),
        "debug": {
            "button_regions_count": debug_payload.get("button_regions_count", 0),
            "recognized_hand_tile_count": hints.get("recognized_hand_tile_count"),
            "bottom_hand_unsupported_count": hints.get("bottom_hand_unsupported_count"),
        },
    }


def _overlay_status(state: PerceivedGameState, decision: DecisionResult) -> dict[str, Any]:
    return {
        "window_bound": True,
        "runtime_mode": "watching",
        "runtime_status": "runtime_strategy_smoke",
        "last_scene": state.scene,
        "last_is_user_turn": state.is_user_turn,
        "last_perception": state.to_dict(),
        "last_decision": decision.to_dict(),
    }


def _summary_payload(material_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "material_dir": str(material_dir),
        "frame_count": len(rows),
        "hand_detected_frame_count": sum(1 for row in rows if int(row.get("hand_count", 0) or 0) > 0),
        "candidate_discard_frame_count": sum(1 for row in rows if row.get("candidate_discards")),
        "tile_efficiency_hint_frame_count": sum(1 for row in rows if row.get("decision_type") == "tile_efficiency_hint"),
        "overlay_discard_frame_count": sum(1 for row in rows if _overlay_primary_looks_like_tile(row)),
        "hand_sources": dict(Counter(str(row.get("hand_source", "")) for row in rows)),
        "candidate_strengths": dict(Counter(str(row.get("candidate_strength", "")) for row in rows if row.get("candidate_discards"))),
        "decision_types": dict(Counter(str(row.get("decision_type", "")) for row in rows)),
        "rows": rows,
    }


def _overlay_primary_looks_like_tile(row: dict[str, Any]) -> bool:
    primary = str(row.get("overlay_primary", "")).strip()
    candidates = row.get("candidate_discards") if isinstance(row.get("candidate_discards"), list) else []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        tile = str(candidate.get("tile", "")).strip()
        if tile and primary == (format_tile_label(tile) or tile):
            return True
    return False


def _report_markdown(payload: dict[str, Any]) -> str:
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    lines = [
        "# Runtime Strategy Smoke",
        "",
        f"- Frames: {payload.get('frame_count', 0)}",
        f"- Hand detected frames: {payload.get('hand_detected_frame_count', 0)}",
        f"- Candidate discard frames: {payload.get('candidate_discard_frame_count', 0)}",
        f"- Tile-efficiency decision frames: {payload.get('tile_efficiency_hint_frame_count', 0)}",
        f"- Overlay discard frames: {payload.get('overlay_discard_frame_count', 0)}",
        f"- Hand sources: {_format_counter(payload.get('hand_sources', {}))}",
        f"- Candidate strengths: {_format_counter(payload.get('candidate_strengths', {}))}",
        f"- Decision types: {_format_counter(payload.get('decision_types', {}))}",
        "",
    ]
    contact_sheet = str(payload.get("contact_sheet", "")).strip()
    if contact_sheet:
        lines.extend([f"![Runtime strategy contact sheet]({contact_sheet})", ""])
    frame_review_dir = str(payload.get("frame_review_dir", "")).strip()
    frame_review_images = payload.get("frame_review_images") if isinstance(payload.get("frame_review_images"), list) else []
    if frame_review_dir and frame_review_images:
        lines.extend(["## Frame Reviews", ""])
        for name in frame_review_images:
            lines.append(f"- [{name}]({frame_review_dir}/{name})")
        lines.append("")
    audit_report = str(payload.get("auto_audit_report", "")).strip()
    if audit_report:
        lines.extend([f"- Auto audit: [{audit_report}]({audit_report})", ""])
    lines.extend(
        [
            "| Frame | Scene | Turn | Hand | Source | Visible | Candidate | Strength | Decision | Overlay |",
            "| --- | --- | --- | ---: | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{Path(str(row.get('image', ''))).name}`",
                    str(row.get("effective_scene") or row.get("scene") or ""),
                    "yes" if row.get("is_user_turn") else "no",
                    str(row.get("hand_count", 0)),
                    str(row.get("hand_source", "")),
                    str(row.get("visible_tile_count", 0)),
                    _format_candidates(row.get("candidate_discards")),
                    str(row.get("candidate_strength", "")) or "-",
                    str(row.get("decision_type", "")),
                    f"{row.get('overlay_primary', '')}: {row.get('overlay_reason', '')}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_auto_audit(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    frame_review_images = payload.get("frame_review_images") if isinstance(payload.get("frame_review_images"), list) else []
    findings = []
    for index, row in enumerate(rows):
        row_findings = _audit_row(row)
        if not row_findings:
            continue
        review_image = frame_review_images[index] if index < len(frame_review_images) else ""
        findings.append(
            {
                "frame": Path(str(row.get("image", ""))).name,
                "review_image": review_image,
                "severity": _max_finding_severity(row_findings),
                "findings": row_findings,
            }
        )
    return {
        "frame_count": len(rows),
        "suspicious_frame_count": len(findings),
        "findings": findings,
        "severity_counts": dict(Counter(item["severity"] for item in findings)),
    }


def _audit_row(row: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    hand_tiles = _string_list(row.get("hand_tiles"))
    visible_tiles = _string_list(row.get("visible_tiles"))
    candidates = row.get("candidate_discards") if isinstance(row.get("candidate_discards"), list) else []
    discard_piles = row.get("discard_piles") if isinstance(row.get("discard_piles"), dict) else {}

    hand_count = len(hand_tiles)
    if hand_count and hand_count not in {1, 2, 4, 7, 8, 10, 11, 13, 14}:
        findings.append(_finding("medium", "hand_count_unusual", f"hand_count={hand_count} is unusual for runtime strategy review"))
    if not hand_tiles and row.get("effective_scene") == "in_match":
        findings.append(_finding("high", "missing_hand", "in_match frame has no recognized hand tiles"))

    all_known_tiles = [*hand_tiles, *visible_tiles]
    overflow = {tile: count for tile, count in Counter(all_known_tiles).items() if _is_real_tile(tile) and count > 4}
    if overflow:
        detail = ", ".join(f"{format_tile_label(tile) or tile}×{count}" for tile, count in sorted(overflow.items()))
        findings.append(_finding("high", "tile_overflow", f"visible+hand contains more than four copies: {detail}"))

    hand_counts = Counter(hand_tiles)
    for candidate in candidates[:3]:
        if not isinstance(candidate, dict):
            continue
        tile = str(candidate.get("tile", "")).strip()
        if tile and hand_counts.get(tile, 0) <= 0:
            findings.append(_finding("high", "candidate_not_in_hand", f"candidate {format_tile_label(tile) or tile} is not in recognized hand"))
        strength = str(candidate.get("recommendation_strength", "")).strip()
        if strength == "strong" and (float(row.get("confidence", 0.0) or 0.0) < 0.45 or hand_count < 10):
            findings.append(_finding("medium", "strong_low_confidence", f"strong candidate on low confidence or partial hand: confidence={row.get('confidence')} hand={hand_count}"))

    pile_counts = {player: len(pile) for player, pile in discard_piles.items() if isinstance(pile, list)}
    for player, count in pile_counts.items():
        if count > 18:
            findings.append(_finding("high", "river_too_many_tiles", f"{player} river has {count} tiles"))
    known_pile_total = sum(pile_counts.values())
    if known_pile_total != len(visible_tiles):
        findings.append(_finding("medium", "river_count_mismatch", f"discard pile total={known_pile_total}, visible_tile_count={len(visible_tiles)}"))
    if row.get("candidate_discards") and not discard_piles:
        findings.append(_finding("medium", "missing_river_groups", "candidate exists but discard_piles is empty"))

    if str(row.get("candidate_strength", "")).strip() == "strong" and str(row.get("decision_type", "")) in {"danger_action", "uncertain_state"}:
        findings.append(_finding("medium", "strong_with_non_strategy_decision", f"strength=strong while decision_type={row.get('decision_type')}"))
    return findings


def _finding(severity: str, code: str, detail: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "detail": detail}


def _max_finding_severity(findings: list[dict[str, str]]) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return max((str(item.get("severity", "low")) for item in findings), key=lambda item: order.get(item, 0), default="low")


def _audit_markdown(audit: dict[str, Any]) -> str:
    findings = audit.get("findings") if isinstance(audit.get("findings"), list) else []
    lines = [
        "# Runtime Strategy Auto Audit",
        "",
        f"- Frames: {audit.get('frame_count', 0)}",
        f"- Suspicious frames: {audit.get('suspicious_frame_count', 0)}",
        f"- Severity counts: {_format_counter(audit.get('severity_counts', {}))}",
        "",
    ]
    if not findings:
        lines.append("No suspicious frames found.")
        lines.append("")
        return "\n".join(lines)
    lines.extend(["| Frame | Severity | Codes | Review Image |", "| --- | --- | --- | --- |"])
    for item in findings:
        item_findings = item.get("findings") if isinstance(item.get("findings"), list) else []
        codes = ", ".join(str(finding.get("code", "")) for finding in item_findings if isinstance(finding, dict))
        review_image = str(item.get("review_image", ""))
        review_link = f"[{review_image}](frame_reviews/{review_image})" if review_image else "-"
        lines.append(f"| `{item.get('frame', '')}` | {item.get('severity', '')} | {codes} | {review_link} |")
        for finding in item_findings:
            if isinstance(finding, dict):
                lines.append(f"|  |  | {finding.get('code', '')}: {finding.get('detail', '')} |  |")
    lines.append("")
    return "\n".join(lines)


def _save_suspicious_frames(audit: dict[str, Any], review_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = audit.get("findings") if isinstance(audit.get("findings"), list) else []
    for item in findings:
        if not isinstance(item, dict):
            continue
        name = str(item.get("review_image", "")).strip()
        if not name:
            continue
        source = review_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / name)


def _save_contact_sheet(rows: list[dict[str, Any]], out_path: Path) -> None:
    panel_width = 520
    panel_height = 430
    columns = 2
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (panel_width * columns, panel_height * sheet_rows), (245, 245, 242))
    font = _load_font(18)
    small_font = _load_font(15)

    for index, row in enumerate(rows):
        x = (index % columns) * panel_width
        y = (index // columns) * panel_height
        panel = Image.new("RGB", (panel_width, panel_height), (255, 255, 255))
        _draw_contact_panel(panel, row, font, small_font)
        sheet.paste(panel, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _save_frame_review_images(rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(str(row.get("image", "")))
        out_name = f"{index:02d}_{image_path.stem}.png"
        out_path = out_dir / out_name
        _save_frame_review_image(row, out_path)
        written.append(out_name)
    return written


def _save_frame_review_image(row: dict[str, Any], out_path: Path) -> None:
    canvas = Image.new("RGB", (1500, 980), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(26)
    font = _load_font(20)
    small_font = _load_font(17)
    image_path = Path(str(row.get("image", "")))

    try:
        original = Image.open(image_path).convert("RGB")
        screenshot = ImageOps.contain(original, (900, 510))
    except Exception:
        original = None
        screenshot = Image.new("RGB", (900, 510), (230, 230, 230))
    canvas.paste(screenshot, (24, 24))
    if original is not None:
        _draw_discard_boxes(draw, row.get("discard_piles"), original.size, screenshot.size, offset=(24, 24))
    draw.rectangle((24, 24, 924, 534), outline=(210, 210, 210), width=2)

    draw.text((950, 24), image_path.name, fill=(20, 20, 20), font=title_font)
    y = 68
    y = _draw_lines(
        draw,
        [
            f"scene: {row.get('effective_scene') or row.get('scene') or ''}",
            f"turn: {'yes' if row.get('is_user_turn') else 'no'}",
            f"confidence: {row.get('confidence', '')}",
            f"decision: {row.get('decision_type', '')} / {row.get('recommended_focus', '')}",
            f"hand source: {row.get('hand_source', '')}",
            f"river source: {row.get('discard_parser_source', '')}",
        ],
        x=950,
        y=y,
        font=small_font,
        fill=(55, 55, 55),
        step=24,
    )

    strength = str(row.get("candidate_strength", "")).strip() or "none"
    badge_color = {
        "strong": (42, 135, 76),
        "medium": (185, 125, 32),
        "weak": (112, 112, 112),
    }.get(strength, (90, 90, 90))
    draw.rectangle((950, y + 8, 1088, y + 40), fill=badge_color)
    draw.text((962, y + 13), strength, fill=(255, 255, 255), font=small_font)
    y += 58

    y = _draw_section(draw, "Hand", _format_tile_list(row.get("hand_tiles")), x=950, y=y, font=font, small_font=small_font, width=32)
    right_y = _draw_section(
        draw,
        "River By Player",
        _format_river_by_player(row.get("discard_piles")),
        x=950,
        y=y + 10,
        font=font,
        small_font=small_font,
        width=32,
    )
    _draw_section(
        draw,
        "Candidates",
        _format_candidate_details(row.get("candidate_discards")),
        x=24,
        y=570,
        font=font,
        small_font=small_font,
        width=90,
    )
    _draw_section(
        draw,
        "Overlay",
        f"{row.get('overlay_primary', '')}: {row.get('overlay_reason', '')}",
        x=950,
        y=right_y + 10,
        font=font,
        small_font=small_font,
        width=32,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _draw_discard_boxes(
    draw: ImageDraw.ImageDraw,
    discard_piles: Any,
    original_size: tuple[int, int],
    rendered_size: tuple[int, int],
    *,
    offset: tuple[int, int],
) -> None:
    if not isinstance(discard_piles, dict):
        return
    scale_x = rendered_size[0] / max(1, original_size[0])
    scale_y = rendered_size[1] / max(1, original_size[1])
    colors = {
        "self": (32, 145, 90),
        "right_opponent": (45, 119, 210),
        "top_opponent": (190, 125, 30),
        "left_opponent": (210, 75, 75),
    }
    for player, pile in discard_piles.items():
        if not isinstance(pile, list):
            continue
        color = colors.get(str(player), (80, 80, 80))
        for item in pile:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x0, y0, x1, y1 = [float(value) for value in bbox]
            except (TypeError, ValueError):
                continue
            box = (
                int(offset[0] + x0 * scale_x),
                int(offset[1] + y0 * scale_y),
                int(offset[0] + x1 * scale_x),
                int(offset[1] + y1 * scale_y),
            )
            draw.rectangle(box, outline=color, width=3)


def _draw_section(
    draw: ImageDraw.ImageDraw,
    title: str,
    body: str,
    *,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    width: int = 48,
) -> int:
    draw.text((x, y), title, fill=(20, 20, 20), font=font)
    y += 30
    return _draw_lines(draw, _wrap_text(body, limit=width), x=x, y=y, font=small_font, fill=(35, 35, 35), step=23)


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    *,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    step: int,
) -> int:
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += step
    return y


def _draw_contact_panel(
    panel: Image.Image,
    row: dict[str, Any],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(panel)
    image_path = Path(str(row.get("image", "")))
    try:
        screenshot = Image.open(image_path).convert("RGB")
        screenshot = ImageOps.contain(screenshot, (500, 285))
    except Exception:
        screenshot = Image.new("RGB", (500, 285), (230, 230, 230))
    panel.paste(screenshot, (10, 10))

    strength = str(row.get("candidate_strength", "")).strip() or "none"
    badge_color = {
        "strong": (42, 135, 76),
        "medium": (185, 125, 32),
        "weak": (112, 112, 112),
    }.get(strength, (90, 90, 90))
    draw.rectangle((10, 303, 122, 330), fill=badge_color)
    draw.text((18, 306), strength, fill=(255, 255, 255), font=small_font)

    draw.text((135, 304), image_path.name[:36], fill=(20, 20, 20), font=font)
    meta = (
        f"scene={row.get('effective_scene') or row.get('scene') or ''} "
        f"turn={'yes' if row.get('is_user_turn') else 'no'} "
        f"hand={row.get('hand_count', 0)} visible={row.get('visible_tile_count', 0)}"
    )
    draw.text((12, 336), meta, fill=(60, 60, 60), font=small_font)
    draw.text((12, 360), f"candidate: {_format_candidates(row.get('candidate_discards'))}", fill=(20, 20, 20), font=small_font)
    overlay = f"{row.get('overlay_primary', '')}: {row.get('overlay_reason', '')}"
    for offset, line in enumerate(_wrap_text(overlay, limit=44)[:2]):
        draw.text((12, 384 + offset * 20), line, fill=(20, 20, 20), font=small_font)
    draw.rectangle((0, 0, panel.width - 1, panel.height - 1), outline=(210, 210, 210))


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(value: str, *, limit: int) -> list[str]:
    raw_lines = str(value or "").splitlines() or [""]
    wrapped: list[str] = []
    for raw_line in raw_lines:
        text = " ".join(raw_line.split())
        if not text:
            wrapped.append("")
            continue
        wrapped.extend(text[index : index + limit] for index in range(0, len(text), limit))
    if not wrapped:
        return [""]
    return wrapped


def _format_tile_list(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    labels = [format_tile_label(str(tile)) or str(tile) for tile in value]
    return " ".join(labels)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _is_real_tile(value: str) -> bool:
    tile = str(value).strip()
    if len(tile) != 2:
        return False
    if tile[1] in {"m", "p", "s"}:
        return tile[0] in "123456789"
    if tile[1] == "z":
        return tile[0] in "1234567"
    return False


def _format_visible_counts(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    counts = Counter(str(tile) for tile in value if str(tile).strip())
    parts = [f"{format_tile_label(tile) or tile}×{count}" for tile, count in sorted(counts.items(), key=lambda item: _tile_sort_key(item[0]))]
    return " ".join(parts)


def _format_river_by_player(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    labels = {
        "self": "self",
        "right_opponent": "right",
        "top_opponent": "top",
        "left_opponent": "left",
    }
    lines: list[str] = []
    for player in ("self", "right_opponent", "top_opponent", "left_opponent"):
        pile = value.get(player)
        if not isinstance(pile, list) or not pile:
            lines.append(f"{labels[player]}: none")
            continue
        ordered = sorted(
            (item for item in pile if isinstance(item, dict)),
            key=lambda item: int(item.get("turn_index", 0) or 0),
        )
        tiles = [format_tile_label(str(item.get("tile", ""))) or str(item.get("tile", "")) for item in ordered]
        lines.append(f"{labels[player]}: {' '.join(tile for tile in tiles if tile)}")
    return "\n".join(lines)


def _format_candidate_details(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    lines: list[str] = []
    for index, item in enumerate(value[:3], start=1):
        if not isinstance(item, dict):
            continue
        tile = str(item.get("tile", "")).strip()
        label = format_tile_label(tile) or tile or "unknown"
        shanten = _format_shanten_pair(item)
        fields = [
            f"{index}. {label}",
            f"strength={item.get('recommendation_strength', '') or '-'}",
            f"score={item.get('score', '-')}",
            f"ukeire={item.get('ukeire_estimate', '-')}",
            f"shanten={shanten}",
            f"safety={item.get('safety_hint', '-')}",
        ]
        lines.append(" ".join(str(field) for field in fields))
        reason = str(item.get("reason", "")).strip()
        if reason:
            lines.append(f"   reason: {reason}")
    return "\n".join(lines)


def _format_shanten_pair(item: dict[str, Any]) -> str:
    current = item.get("current_shanten")
    post = item.get("post_discard_shanten")
    if current is None and post is None:
        return "-"
    return f"{current if current is not None else '?'}->{post if post is not None else '?'}"


def _tile_sort_key(tile: str) -> tuple[int, int, str]:
    tile = str(tile)
    if len(tile) < 2:
        return (9, 99, tile)
    suit_order = {"m": 0, "p": 1, "s": 2, "z": 3}
    try:
        number = int(tile[:-1])
    except ValueError:
        number = 99
    return (suit_order.get(tile[-1], 9), number, tile)


def _format_counter(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key or 'empty'}={value[key]}" for key in sorted(value))


def _format_candidates(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    labels = []
    for item in value[:3]:
        if not isinstance(item, dict):
            continue
        tile = str(item.get("tile", "")).strip()
        if tile:
            labels.append(format_tile_label(tile) or tile)
    return ", ".join(labels) if labels else "none"


if __name__ == "__main__":
    raise SystemExit(main())
