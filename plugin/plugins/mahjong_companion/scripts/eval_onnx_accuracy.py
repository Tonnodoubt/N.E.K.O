"""Quick ONNX accuracy evaluation against multi-theme ground truth.

Uses the same crop logic as annotate_fixtures.py, classifies with ONNX,
compares against .tiles.json sidecars.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURES_ROOT = _REPO_ROOT / "plugin" / "plugins" / "mahjong_companion" / "tests" / "fixtures" / "multi_theme"

sys.path.insert(0, str(_REPO_ROOT))
from plugin.plugins.mahjong_companion.perception.vit_tile_classifier_onnx import classify_tile_crops_onnx


def _detect_hand_tiles(img: Image.Image) -> list[Image.Image]:
    w, h = img.size
    arr = np.array(img.convert("RGB"))
    band = arr[int(h * 0.78):int(h * 0.95), int(w * 0.12):int(w * 0.75), :].astype(float).mean(axis=(1, 2))
    if band.max() < 140:
        return []
    peak_local = int(np.argmax(band))
    peak_y = int(h * 0.78) + peak_local
    row = arr[peak_y, :, :].astype(float).mean(axis=1)
    bright = row > 155
    trans = np.diff(bright.astype(int))
    starts = np.where(trans == 1)[0] + 1
    ends = np.where(trans == -1)[0] + 1
    if len(bright) > 0 and bright[0]:
        starts = np.concatenate([[0], starts])
    if len(bright) > 0 and bright[-1]:
        ends = np.concatenate([ends, [len(bright)]])
    cols = [(int(s), int(e)) for s, e in zip(starts, ends) if 25 < (e - s) < 120]
    crops = []
    for xs, xe in cols[:14]:
        y_lo, y_hi = max(0, peak_y - 50), min(h, peak_y + 50)
        col = arr[y_lo:y_hi, xs:xe, :].astype(float).mean(axis=(1, 2))
        bright_rows = np.where(col > 140)[0]
        if len(bright_rows) == 0:
            yt, yb = peak_y - 30, peak_y + 30
        else:
            yt = y_lo + max(0, int(bright_rows[0]) - 3)
            yb = y_lo + min(y_hi - y_lo, int(bright_rows[-1]) + 3)
        crops.append(img.crop((xs, yt, xe, yb)))
    return crops


def _detect_discard_tiles(img: Image.Image, player: str) -> list[Image.Image]:
    w, h = img.size
    sx, sy = w / 1920, h / 1080
    specs = {
        "self": {"ox": 762, "oy": 542, "tw": 58, "th": 70, "dx": 64, "dy": 70, "cols": 6, "rows": 3},
        "left_opponent": {"ox": 624, "oy": 290, "tw": 84, "th": 58, "dx": 82, "dy": 62, "cols": 3, "rows": 6},
        "top_opponent": {"ox": 802, "oy": 242, "tw": 58, "th": 70, "dx": 64, "dy": -70, "cols": 6, "rows": 3},
        "right_opponent": {"ox": 1148, "oy": 290, "tw": 84, "th": 58, "dx": 82, "dy": 62, "cols": 3, "rows": 6},
    }
    s = specs[player]
    crops = []
    for r in range(s["rows"]):
        for c in range(s["cols"]):
            left = int((s["ox"] + c * s["dx"]) * sx)
            top = int((s["oy"] + r * s["dy"]) * sy)
            tw = max(1, int(s["tw"] * sx))
            th = max(1, int(s["th"] * sy))
            crops.append(img.crop((left, top, left + tw, top + th)))
    return crops


def _classify_batch(crops: list[Image.Image]) -> list[str]:
    if not crops:
        return []
    preds = classify_tile_crops_onnx(crops, top_k=1)
    return [p.tile if p else "" for p in preds]


def main():
    cases = []
    for theme_dir in sorted(_FIXTURES_ROOT.iterdir()):
        if not theme_dir.is_dir() or theme_dir.name.startswith("_"):
            continue
        for png in sorted(theme_dir.glob("*.png")):
            sidecar = png.with_suffix(".tiles.json")
            if not sidecar.exists():
                continue
            cases.append((theme_dir.name, png, sidecar))

    print(f"Evaluating {len(cases)} screenshots\n")

    hand_correct, hand_total = 0, 0
    discard_correct, discard_total = 0, 0
    details = []

    for theme_id, png_path, sidecar in cases:
        gt = json.loads(sidecar.read_text())
        img = Image.open(png_path)

        # --- hand tiles ---
        gt_hand = [str(t).strip() for t in gt.get("hand_tiles", []) if str(t).strip()]
        hand_crops = _detect_hand_tiles(img)
        hand_preds = _classify_batch(hand_crops)
        n = min(len(gt_hand), len(hand_preds))
        hand_ok = sum(1 for i in range(n) if hand_preds[i] == gt_hand[i])
        hand_correct += hand_ok
        hand_total += len(gt_hand)
        hand_detail = []
        for i in range(n):
            mark = "✓" if hand_preds[i] == gt_hand[i] else "✗"
            hand_detail.append(f"{hand_preds[i]}{mark}")

        # --- discard tiles ---
        discard_detail = {}
        gt_discards = gt.get("discard_piles", {})
        for player in ("self", "left_opponent", "top_opponent", "right_opponent"):
            gt_pile = gt_discards.get(player, [])
            if isinstance(gt_pile, list):
                gt_tiles = [str(t.get("tile", "") if isinstance(t, dict) else t).strip() for t in gt_pile]
            else:
                gt_tiles = []
            discard_crops = _detect_discard_tiles(img, player)
            discard_preds = _classify_batch(discard_crops)
            n_d = min(len(gt_tiles), len(discard_preds))
            d_ok = sum(1 for i in range(n_d) if discard_preds[i] == gt_tiles[i])
            discard_correct += d_ok
            discard_total += len(gt_tiles)
            if gt_tiles:
                marks = []
                for i in range(n_d):
                    mark = "✓" if discard_preds[i] == gt_tiles[i] else "✗"
                    marks.append(f"{discard_preds[i]}{mark}")
                discard_detail[player] = marks

        label = f"{theme_id}/{png_path.stem}"
        details.append({
            "label": label,
            "hand_gt": gt_hand,
            "hand_pred": hand_preds[:len(gt_hand)],
            "hand_n": len(gt_hand),
            "hand_ok": hand_ok,
            "discard_detail": discard_detail,
        })
        print(f"  {label}: hand {hand_ok}/{len(gt_hand)}  pred=[{' '.join(hand_detail)}]")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if hand_total:
        print(f"Hand tiles:   {hand_correct}/{hand_total}  ({100*hand_correct/hand_total:.1f}%)")
    if discard_total:
        print(f"Discard tiles: {discard_correct}/{discard_total}  ({100*discard_correct/discard_total:.1f}%)")
    total_correct = hand_correct + discard_correct
    total_tiles = hand_total + discard_total
    if total_tiles:
        print(f"Overall:      {total_correct}/{total_tiles}  ({100*total_correct/total_tiles:.1f}%)")


if __name__ == "__main__":
    main()
