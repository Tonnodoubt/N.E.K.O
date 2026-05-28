"""Fine-tune MobileNetV3-Small tile classifier with hand crop data.

Two-stage training:
  Stage 1: Freeze backbone, train classifier head only (1-2 epochs)
  Stage 2: Unfreeze last 2 backbone layers, low-lr fine-tune (5-10 epochs)

Usage:
    # 完整两阶段训练
    python scripts/train_tile_classifier.py --dataset-dir data/tile_dataset --output-dir tmp/tile_model

    # 只跑 Stage 1
    python scripts/train_tile_classifier.py --dataset-dir data/tile_dataset --output-dir tmp/tile_model --stage1-only

    # 从已有 checkpoint 继续 Stage 2
    python scripts/train_tile_classifier.py --dataset-dir data/tile_dataset --output-dir tmp/tile_model --resume tmp/tile_model/stage1_best.pt

Requires:
    pip install torch torchvision timm onnx
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TILE_CLASSES = [
    f"{rank}{suit}" for suit in ("m", "p", "s") for rank in range(1, 10)
] + [f"{rank}z" for rank in range(1, 8)]
NUM_TILE_CLASSES = len(TILE_CLASSES)
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ImageNet normalization (used by MobileNetV3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _import_torch():
    try:
        import torch
        import torchvision
        return torch, torchvision
    except ImportError:
        log.error("PyTorch required: pip install torch torchvision")
        sys.exit(1)


def _import_timm():
    try:
        import timm
        return timm
    except ImportError:
        log.error("timm required: pip install timm")
        sys.exit(1)


class TileDataset:
    """Simple image dataset that reads from split_dir/{label}/*.png."""

    def __init__(self, split_dir: Path, transform=None):
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        for idx, label in enumerate(TILE_CLASSES):
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
            for img_path in sorted(label_dir.iterdir()):
                if img_path.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((img_path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def letterbox_transform(size: int = 224):
    """Build transform that uses letterbox padding to preserve aspect ratio."""
    import torchvision.transforms as T

    return T.Compose([
        T.Lambda(lambda img: _letterbox_pad_pil(img, size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _letterbox_pad_pil(img, size: int = 224):
    """Letterbox pad PIL image to square, preserving aspect ratio."""
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def build_model(num_classes: int, pretrained: str = "mobilenetv3_small_100"):
    """Build MobileNetV3-Small model via timm."""
    import timm
    model = timm.create_model(pretrained, pretrained=True, num_classes=num_classes)
    return model


def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_last_n_layers(model, n: int = 2):
    """Unfreeze the last n blocks + classifier."""
    # timm MobileNetV3 uses blocks.X; keep features.X support for compatible backbones.
    block_prefix = "blocks."
    if not any(name.startswith(block_prefix) for name, _ in model.named_parameters()):
        block_prefix = "features."

    blocks = set()
    for name, _ in model.named_parameters():
        if not name.startswith(block_prefix):
            continue
        parts = name.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            blocks.add(int(parts[1]))

    unfreeze_blocks = sorted(blocks)[-n:] if blocks else []
    for name, param in model.named_parameters():
        should_train = "classifier" in name or name.startswith("conv_head")
        if name.startswith(block_prefix):
            parts = name.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                if int(parts[1]) in unfreeze_blocks:
                    should_train = True
        param.requires_grad = should_train


def train_one_epoch(model, loader, criterion, optimizer, device):
    import torch
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    import torch
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / max(1, total), correct / max(1, total)


def export_onnx(model, output_path: Path, size: int = 224):
    import torch
    model = model.to("cpu")
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )
    log.info(f"ONNX model exported to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def write_model_artifacts(output_dir: Path, size: int = 224):
    """Write preprocessor.json, labels.json, config.json alongside model.onnx."""
    preprocessor = {
        "image_mean": IMAGENET_MEAN,
        "image_std": IMAGENET_STD,
        "size": {"shortest_edge": size},
        "do_normalize": True,
        "do_resize": True,
        "do_rescale": True,
        "rescale_factor": 1.0 / 255.0,
        "letterbox_pad": True,
    }
    labels = {str(i): label for i, label in enumerate(TILE_CLASSES)}

    config = {
        "backbone": "mobilenetv3_small",
        "timm_name": "mobilenetv3_small_100",
        "input_size": size,
        "num_classes": NUM_TILE_CLASSES,
        "class_names": TILE_CLASSES,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }

    (output_dir / "preprocessor.json").write_text(json.dumps(preprocessor, indent=2), encoding="utf-8")
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune tile classifier")
    parser.add_argument("--dataset-dir", default="data/tile_dataset", help="Prepared dataset directory")
    parser.add_argument("--output-dir", default="tmp/tile_model", help="Training output directory")
    parser.add_argument("--resume", default="", help="Checkpoint to resume from")
    parser.add_argument("--stage1-only", action="store_true", help="Only run Stage 1")
    parser.add_argument("--stage1-epochs", type=int, default=2, help="Stage 1 epochs")
    parser.add_argument("--stage2-epochs", type=int, default=8, help="Stage 2 epochs")
    parser.add_argument("--stage1-lr", type=float, default=1e-3, help="Stage 1 learning rate")
    parser.add_argument("--stage2-lr", type=float, default=3e-5, help="Stage 2 learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch, torchvision = _import_torch()
    timm = _import_timm()

    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets with letterbox transform
    transform = letterbox_transform(args.img_size)
    train_ds = TileDataset(dataset_dir / "train", transform=transform)
    val_ds = TileDataset(dataset_dir / "val", transform=transform)
    test_ds = TileDataset(dataset_dir / "test", transform=transform)

    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Build model
    model = build_model(NUM_TILE_CLASSES).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt, strict=False)
        log.info(f"Resumed from {args.resume}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_val_acc = 0.0
    stage_info = {"stage1": {}, "stage2": {}}

    # ── Stage 1: Freeze backbone, train classifier head ──
    log.info("=" * 50)
    log.info("Stage 1: Training classifier head (backbone frozen)")
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {trainable:,}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.stage1_lr, weight_decay=0.01)

    for epoch in range(1, args.stage1_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        log.info(f"  Epoch {epoch}/{args.stage1_epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "stage1_best.pt")
            log.info(f"  -> Saved stage1_best.pt (val_acc={val_acc:.4f})")

    stage_info["stage1"]["best_val_acc"] = best_val_acc

    if args.stage1_only:
        model.load_state_dict(torch.load(output_dir / "stage1_best.pt", map_location=device, weights_only=True))
        export_onnx(model, output_dir / "model.onnx", size=args.img_size)
        write_model_artifacts(output_dir, size=args.img_size)
        log.info("Stage 1 only mode. Done.")
        return

    # ── Stage 2: Unfreeze last layers, fine-tune ──
    log.info("=" * 50)
    log.info("Stage 2: Fine-tuning last layers")

    model.load_state_dict(torch.load(output_dir / "stage1_best.pt", map_location=device, weights_only=True))
    torch.save(model.state_dict(), output_dir / "stage2_best.pt")
    unfreeze_last_n_layers(model, n=2)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {trainable:,}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.stage2_lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage2_epochs, eta_min=1e-6)

    best_val_acc_s2 = best_val_acc
    for epoch in range(1, args.stage2_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        log.info(
            f"  Epoch {epoch}/{args.stage2_epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_acc > best_val_acc_s2:
            best_val_acc_s2 = val_acc
            torch.save(model.state_dict(), output_dir / "stage2_best.pt")
            log.info(f"  -> Saved stage2_best.pt (val_acc={val_acc:.4f})")

    stage_info["stage2"]["best_val_acc"] = best_val_acc_s2

    # ── Final evaluation on test set ──
    log.info("=" * 50)
    model.load_state_dict(torch.load(output_dir / "stage2_best.pt", map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    log.info(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    # ── Export ONNX ──
    export_onnx(model, output_dir / "model.onnx", size=args.img_size)
    write_model_artifacts(output_dir, size=args.img_size)

    # Write training metadata
    metadata = {
        "model_type": "mobilenetv3_small",
        "timm_name": "mobilenetv3_small_100",
        "input_size": args.img_size,
        "preprocessing": "letterbox_pad",
        "training_stages": stage_info,
        "test_accuracy": test_acc,
        "args": vars(args),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Training complete. Artifacts in {output_dir}")


if __name__ == "__main__":
    # Patch PIL import for TileDataset
    from PIL import Image as _Image  # noqa: F401
    main()
