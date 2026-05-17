"""Train a lightweight tile classifier and export to ONNX.

Usage (on a GPU machine)::

    pip install torch torchvision timm onnx onnxruntime pillow
    python scripts/train_tile_classifier.py --data /tmp/tile_dataset --output ./model_output

Output::

    model_output/
      model.onnx        # ONNX model for inference
      labels.txt        # class label list (one per line)
      labels.json       # class label mapping {id: name} for ONNX loader
      config.json       # model config for the ONNX loader
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

# --- Configurable backbones ---
BACKBONE_CONFIGS = {
    "mobilenetv3_small": {
        "timm_name": "mobilenetv3_small_100",
        "input_size": 224,
        "description": "MobileNetV3-Small (~2.5M params, fast CPU)",
    },
    "mobilenetv3_large": {
        "timm_name": "mobilenetv3_large_100",
        "input_size": 224,
        "description": "MobileNetV3-Large (~5.5M, balanced)",
    },
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "input_size": 224,
        "description": "EfficientNet-B0 (~5.3M, accurate)",
    },
    "efficientformer_l1": {
        "timm_name": "efficientformer_l1",
        "input_size": 224,
        "description": "EfficientFormer-L1 (~12M, ViT-like)",
    },
    "resnet18": {
        "timm_name": "resnet18",
        "input_size": 224,
        "description": "ResNet-18 (~11M, reliable baseline)",
    },
}

DEFAULT_BACKBONE = "mobilenetv3_small"
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        import torchvision.transforms as T
        import timm
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Install: pip install torch torchvision timm onnx onnxruntime", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data)
    labels_file = data_dir / "labels.txt"
    if not labels_file.exists():
        print(f"labels.txt not found in {data_dir}. Run export_tile_classifier_dataset first.", file=sys.stderr)
        return 2
    class_names = [line.strip() for line in labels_file.read_text().strip().splitlines() if line.strip()]
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    backbone = args.backbone
    cfg = BACKBONE_CONFIGS.get(backbone, BACKBONE_CONFIGS[DEFAULT_BACKBONE])
    print(f"Backbone: {backbone} ({cfg['timm_name']}), input={cfg['input_size']}")

    # --- Data ---
    input_size = cfg["input_size"]
    train_transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.RandomHorizontalFlip(p=0.3),
        T.RandomRotation(degrees=8),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists():
        print(f"train/ not found in {data_dir}", file=sys.stderr)
        return 3

    train_dataset = _ImageFolder(train_dir, class_names, train_transform)
    val_dataset = _ImageFolder(val_dir, class_names, val_transform) if val_dir.exists() else train_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # --- Model ---
    model = timm.create_model(cfg["timm_name"], pretrained=True, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Train ---
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)
        scheduler.step()

        train_acc = train_correct / max(train_total, 1)
        val_acc = _evaluate(model, val_loader, criterion, device) if val_loader else 0.0
        best_acc = max(best_acc, val_acc or train_acc)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}: train_loss={train_loss/max(len(train_loader),1):.4f} "
                  f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} lr={scheduler.get_last_lr()[0]:.6f}")

    print(f"Best accuracy: {best_acc:.3f}")

    # --- Export ONNX ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=18,
        external_data=False,
    )
    print(f"ONNX exported: {onnx_path} ({_file_size_mb(onnx_path):.1f} MB)")

    # Copy labels (two formats: labels.txt for reference, labels.json for ONNX loader)
    shutil.copy(str(labels_file), str(output_dir / "labels.txt"))
    labels_json = {str(i): name for i, name in enumerate(class_names)}
    (output_dir / "labels.json").write_text(json.dumps(labels_json, indent=2, ensure_ascii=False))

    # Write config (legacy, kept for reference)
    config = {
        "backbone": backbone,
        "timm_name": cfg["timm_name"],
        "input_size": input_size,
        "num_classes": num_classes,
        "class_names": class_names,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Config saved: {output_dir / 'config.json'}")

    # Write preprocessor.json (compatible with vit_tile_classifier_onnx.py)
    preprocessor = {
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "size": {"shortest_edge": input_size},
        "do_normalize": True,
        "do_resize": True,
        "do_rescale": True,
        "rescale_factor": 0.00392156862745098,
    }
    (output_dir / "preprocessor.json").write_text(json.dumps(preprocessor, indent=2))
    print(f"Preprocessor saved: {output_dir / 'preprocessor.json'}")
    return 0


def _evaluate(model, loader, criterion, device) -> float:
    import torch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


class _ImageFolder:
    """Minimal ImageFolder that reads from class subdirectories."""

    def __init__(self, root: Path, class_names: list[str], transform):
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        for cls_idx, cls_name in enumerate(class_names):
            cls_dir = root / cls_name
            if not cls_dir.is_dir():
                continue
            for img_file in sorted(cls_dir.glob("*.png"))[:]:
                self.samples.append((img_file, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tile classifier")
    parser.add_argument("--data", default="/tmp/tile_dataset")
    parser.add_argument("--output", default="/tmp/tile_model")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE,
                        choices=list(BACKBONE_CONFIGS.keys()))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
