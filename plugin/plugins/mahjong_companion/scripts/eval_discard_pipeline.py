"""Evaluate ONNX discard accuracy with the real pipeline (brightness + confidence gates).

Uses parse_discards_from_image which includes both the geometric occupancy
detector and the ONNX softmax confidence gate.
"""

import json
import sys
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_companion.perception.discard_parser import parse_discards_from_image


FIXTURES_ROOT = _REPO_ROOT / "plugin" / "plugins" / "mahjong_companion" / "tests" / "fixtures" / "multi_theme"


def main():
    cases = []
    for theme_dir in sorted(FIXTURES_ROOT.iterdir()):
        if not theme_dir.is_dir() or theme_dir.name.startswith("_"):
            continue
        for png in sorted(theme_dir.glob("*.png")):
            sidecar = png.with_suffix(".tiles.json")
            if not sidecar.exists():
                continue
            cases.append((theme_dir.name, png, sidecar))

    print(f"Evaluating {len(cases)} screenshots\n")

    total_correct, total_expected, total_predicted = 0, 0, 0
    per_theme = {}

    for theme_id, png_path, sidecar in cases:
        gt = json.loads(sidecar.read_text())
        img = Image.open(png_path).convert("RGB")

        result = parse_discards_from_image(img, {"source": "eval"})
        predicted_piles = result.discard_piles

        gt_discards = gt.get("discard_piles", {})
        case_correct, case_expected, case_predicted = 0, 0, 0

        for player in ("self", "left_opponent", "top_opponent", "right_opponent"):
            gt_pile = gt_discards.get(player, [])
            gt_tiles = [
                str(t.get("tile", "") if isinstance(t, dict) else t).strip()
                for t in gt_pile
            ]
            pred_pile = predicted_piles.get(player, [])
            pred_tiles = [str(t.get("tile", "")).strip() for t in pred_pile]

            case_expected += len(gt_tiles)
            case_predicted += len(pred_tiles)

            from collections import Counter
            gt_c = Counter(gt_tiles)
            pred_c = Counter(pred_tiles)
            overlap = sum((gt_c & pred_c).values())
            case_correct += overlap

        total_correct += case_correct
        total_expected += case_expected
        total_predicted += case_predicted

        bucket = per_theme.setdefault(theme_id, {"correct": 0, "expected": 0, "predicted": 0})
        bucket["correct"] += case_correct
        bucket["expected"] += case_expected
        bucket["predicted"] += case_predicted

        prec = case_correct / case_predicted if case_predicted else 0
        rec = case_correct / case_expected if case_expected else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f"  {theme_id}/{png_path.stem}: {case_correct}/{case_expected}  P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

    print("\n" + "=" * 60)
    print("DISCARD SUMMARY")
    print("=" * 60)
    for theme_id, b in sorted(per_theme.items()):
        p = b["correct"] / b["predicted"] if b["predicted"] else 0
        r = b["correct"] / b["expected"] if b["expected"] else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        print(f"  {theme_id:<24} {b['correct']:>3}/{b['expected']:<3}  P={p:.2f} R={r:.2f} F1={f1:.2f}")

    p = total_correct / total_predicted if total_predicted else 0
    r = total_correct / total_expected if total_expected else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"  {'OVERALL':<24} {total_correct:>3}/{total_expected:<3}  P={p:.2f} R={r:.2f} F1={f1:.2f}")


if __name__ == "__main__":
    main()
