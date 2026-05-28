"""Evaluate tile classifier on discard crops and hand crops separately.

Usage:
    # 评估 ONNX 模型
    python scripts/eval_tile_classifier.py --model-dir tmp/tile_model --test-dir data/tile_dataset/test

    # 对比弃牌和手牌精度
    python scripts/eval_tile_classifier.py --model-dir tmp/tile_model \
        --discard-dir data/tile_dataset/test --hand-dir data/hand_crops

    # 用现有模型评估
    python scripts/eval_tile_classifier.py \
        --model-dir plugin/plugins/mahjong_coach/data/models/vit_tile_classifier \
        --hand-dir data/hand_crops
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def _load_onnx_classifier_module():
    path = (
        Path(__file__).resolve().parent.parent
        / "plugin"
        / "plugins"
        / "mahjong_coach"
        / "perception"
        / "vit_tile_classifier_onnx.py"
    )
    spec = importlib.util.spec_from_file_location("mahjong_coach_vit_tile_classifier_onnx", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ONNX classifier module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_ONNX_CLASSIFIER = _load_onnx_classifier_module()
OnnxTilePrediction = _ONNX_CLASSIFIER.OnnxTilePrediction
classify_tile_crops_onnx = _ONNX_CLASSIFIER.classify_tile_crops_onnx
onnx_tile_classifier_available = _ONNX_CLASSIFIER.onnx_tile_classifier_available

TILE_CLASSES = [
    f"{rank}{suit}" for suit in ("m", "p", "s") for rank in range(1, 10)
] + [f"{rank}z" for rank in range(1, 8)]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(TILE_CLASSES)}
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    """Letterbox pad + ImageNet normalize, matching training preprocessing."""
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def collect_images(directory: Path) -> list[tuple[Path, str]]:
    """Collect (path, label) from directory/{label}/*.png structure."""
    items: list[tuple[Path, str]] = []
    if not directory.exists():
        return items
    for label_dir in sorted(directory.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("_"):
            continue
        label = label_dir.name
        for img_path in sorted(label_dir.iterdir()):
            if img_path.suffix.lower() in IMG_EXTENSIONS:
                items.append((img_path, label))
    return items


def eval_onnx(
    images: list[tuple[Path, str]],
    model_dir: Path,
    *,
    batch_size: int = 32,
    use_letterbox: bool = True,
) -> dict[str, Any]:
    """Run ONNX inference on labeled images, return per-class metrics."""
    if not images:
        return {"total": 0}

    if not onnx_tile_classifier_available(model_dir=model_dir):
        print(f"ONNX model not available at {model_dir}")
        return {"total": 0, "error": "model_unavailable"}

    # Process in batches
    all_predictions: list[str] = []
    all_labels: list[str] = []
    all_confidences: list[float] = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        crops = []
        labels = []
        for path, label in batch:
            img = Image.open(path).convert("RGB")
            if use_letterbox:
                # Apply letterbox preprocessing manually, then pass as preprocessed
                crops.append(img)
            else:
                crops.append(img)
            labels.append(label)

        predictions = classify_tile_crops_onnx(crops, model_dir=model_dir, top_k=2)
        for pred, label in zip(predictions, labels):
            if pred is not None:
                all_predictions.append(pred.tile)
                all_confidences.append(pred.confidence)
            else:
                all_predictions.append("unknown")
                all_confidences.append(0.0)
            all_labels.append(label)

        print(f"  Evaluated {min(i + batch_size, len(images))}/{len(images)}", end="\r")

    print()

    # Compute per-class metrics
    correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
    total = len(all_labels)
    accuracy = correct / max(1, total)

    per_class: dict[str, dict[str, Any]] = {}
    tp_counter: Counter = Counter()
    fp_counter: Counter = Counter()
    fn_counter: Counter = Counter()

    for pred, label in zip(all_predictions, all_labels):
        if pred == label:
            tp_counter[label] += 1
        else:
            fp_counter[pred] += 1
            fn_counter[label] += 1

    all_labels_set = set(all_labels) | set(all_predictions) - {"unknown"}
    for cls in sorted(all_labels_set):
        tp = tp_counter[cls]
        fp = fp_counter[cls]
        fn = fn_counter[cls]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
        }

    # Confusion matrix for confusion pairs
    confusion_pairs: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pred, label in zip(all_predictions, all_labels):
        if pred != label:
            confusion_pairs[label][pred] += 1

    top_confusions = []
    for true_label, pred_map in sorted(confusion_pairs.items()):
        for pred_label, count in sorted(pred_map.items(), key=lambda x: -x[1])[:3]:
            top_confusions.append((true_label, pred_label, count))
    top_confusions.sort(key=lambda x: -x[2])

    mean_confidence = sum(all_confidences) / max(1, len(all_confidences))

    # Macro F1
    f1_values = [v["f1"] for v in per_class.values() if v["support"] > 0]
    macro_f1 = sum(f1_values) / max(1, len(f1_values))

    # Weighted F1
    total_support = sum(v["support"] for v in per_class.values())
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / max(1, total_support)

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "mean_confidence": round(mean_confidence, 4),
        "per_class": per_class,
        "top_confusions": [(a, b, c) for a, b, c in top_confusions[:15]],
    }


def print_results(name: str, results: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    if results.get("total", 0) == 0:
        print("  No data.")
        return
    if "error" in results:
        print(f"  Error: {results['error']}")
        return
    print(f"  Total:        {results['total']}")
    print(f"  Accuracy:     {results['accuracy']:.4f}")
    print(f"  Macro F1:     {results['macro_f1']:.4f}")
    print(f"  Weighted F1:  {results['weighted_f1']:.4f}")
    print(f"  Mean Conf:    {results['mean_confidence']:.4f}")

    per_class = results.get("per_class", {})
    if per_class:
        print(f"\n  {'Class':<6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
        print(f"  {'-'*32}")
        for cls in sorted(per_class):
            m = per_class[cls]
            flag = " <<<" if m["f1"] < 0.90 else ""
            print(f"  {cls:<6} {m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f} {m['support']:>8}{flag}")

    confusions = results.get("top_confusions", [])
    if confusions:
        print(f"\n  Top confusion pairs:")
        for true_label, pred_label, count in confusions[:10]:
            print(f"    {true_label} -> {pred_label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tile classifier")
    parser.add_argument("--model-dir", required=True, help="ONNX model directory")
    parser.add_argument("--test-dir", default="", help="Test split directory")
    parser.add_argument("--discard-dir", default="", help="Discard-only evaluation directory")
    parser.add_argument("--hand-dir", default="", help="Hand crop evaluation directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-letterbox", action="store_true", help="Skip letterbox preprocessing")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    results_all: dict[str, Any] = {}

    if args.test_dir:
        test_dir = Path(args.test_dir)
        images = collect_images(test_dir)
        print(f"Evaluating test split: {len(images)} images from {test_dir}")
        results = eval_onnx(images, model_dir, batch_size=args.batch_size, use_letterbox=not args.no_letterbox)
        print_results("Test Split", results)
        results_all["test"] = results

    if args.discard_dir:
        discard_dir = Path(args.discard_dir)
        images = collect_images(discard_dir)
        print(f"\nEvaluating discard crops: {len(images)} images from {discard_dir}")
        results = eval_onnx(images, model_dir, batch_size=args.batch_size, use_letterbox=not args.no_letterbox)
        print_results("Discard Crops", results)
        results_all["discard"] = results

    if args.hand_dir:
        hand_dir = Path(args.hand_dir)
        images = collect_images(hand_dir)
        print(f"\nEvaluating hand crops: {len(images)} images from {hand_dir}")
        results = eval_onnx(images, model_dir, batch_size=args.batch_size, use_letterbox=not args.no_letterbox)
        print_results("Hand Crops", results)
        results_all["hand"] = results

    if results_all:
        output_file = model_dir / "eval_results.json"
        # Convert tuples to lists for JSON serialization
        serializable = json.loads(json.dumps(results_all, default=str))
        output_file.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
