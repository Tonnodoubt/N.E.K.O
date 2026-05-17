from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from plugin.plugins.mahjong_companion.perception.river_detector_v2 import (
    RIVER_PLAYERS,
    detect_river_tiles_v2,
    expand_candidate_quad_for_classification,
)


PLAYER_COLORS = {
    "self": (80, 220, 120),
    "left_opponent": (255, 210, 70),
    "top_opponent": (90, 180, 255),
    "right_opponent": (255, 120, 120),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw class-agnostic river tile detections on a Mahjong Soul screenshot.")
    parser.add_argument("image", type=Path, help="Input screenshot path.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--json-out", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--draw-rois", action="store_true", help="Also draw the broad river search regions.")
    parser.add_argument("--draw-classification-quads", action="store_true", help="Draw expanded quads used for classifier crops.")
    parser.add_argument("--only-player", choices=RIVER_PLAYERS, default=None, help="Draw candidate boxes for only one player.")
    parser.add_argument("--hide-labels", action="store_true", help="Draw boxes without tile index labels.")
    args = parser.parse_args()

    image_path = args.image
    out_path = args.out or image_path.with_name(f"{image_path.stem}-river-v2.png")
    json_path = args.json_out or out_path.with_suffix(".json")

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    result = detect_river_tiles_v2(image)
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
            label = f"{_short_player(player)}{candidate.order_index}"
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path)
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"image={image_path}")
    print(f"output={out_path}")
    print(f"json={json_path}")
    print(f"candidate_count={len(result.candidates)}")
    for player in RIVER_PLAYERS:
        pile = result.by_player.get(player, [])
        print(f"{player}: {len(pile)}")
        for item in pile:
            print(
                f"  {item.order_index:02d} bbox={list(item.bbox)} "
                f"quad={[[x, y] for x, y in item.quad]} confidence={item.confidence:.3f}"
            )
    return 0


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
