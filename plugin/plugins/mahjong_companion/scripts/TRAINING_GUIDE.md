# Mahjong Soul Tile Classifier Training Guide

This trains a lightweight ONNX tile classifier for the Mahjong Companion
runtime. The intended path is:

1. Prepare a local imagefolder dataset from the public Mahjong Soul crop set.
2. Optionally add `empty` negative crops from local debug frames.
3. Train MobileNetV3 and export `model.onnx`, `labels.json`, and
   `preprocessor.json`.
4. Copy the exported files into the plugin model directory.

## 1. Prepare Data

From the repository root:

```powershell
python plugin\plugins\mahjong_companion\scripts\prepare_hf_tile_dataset.py `
  --output-dir tmp\mahjong_tile_dataset `
  --empty-source-dir plugin\plugins\mahjong_companion\data\debug_samples\live `
  --empty-per-frame 4 `
  --download-workers 16 `
  --overwrite
```

Notes:

- Default dataset: `pjura/mahjong_souls_tiles`.
- The script maps HF labels such as `1n`, `1b`, and `ew` to the plugin labels
  `1m`, `1s`, and `1z`.
- If `huggingface_hub` is not installed, the script uses concurrent HTTP
  downloads and caches files under the system temp directory.
- Red fives are still handled by runtime color post-processing, so the base
  dataset does not need `0m`, `0p`, or `0s` classes.

Expected output:

```text
tmp/mahjong_tile_dataset/
  train/
  val/
  labels.txt
```

## 2. Install Training Dependencies

Use a GPU machine if available.

```powershell
pip install torch torchvision timm onnx onnxruntime onnxscript pillow
```

For NVIDIA GPUs, install a CUDA-enabled PyTorch build from the official
PyTorch wheel index before training.

## 3. Train

```powershell
$env:PYTHONIOENCODING = "utf-8"

python plugin\plugins\mahjong_companion\scripts\train_tile_classifier.py `
  --data tmp\mahjong_tile_dataset `
  --output tmp\mahjong_tile_model `
  --backbone mobilenetv3_small `
  --epochs 40 `
  --batch-size 64 `
  --lr 0.001
```

Backbones:

| Backbone | Use case |
| --- | --- |
| `mobilenetv3_small` | Default, fastest CPU inference |
| `mobilenetv3_large` | More capacity, still compact |
| `efficientnet_b0` | Accuracy-oriented small baseline |
| `resnet18` | Stable baseline, larger |

## 4. Deploy Locally

```powershell
New-Item -ItemType Directory -Force `
  plugin\plugins\mahjong_companion\data\models\vit_tile_classifier

Copy-Item tmp\mahjong_tile_model\model.onnx `
  plugin\plugins\mahjong_companion\data\models\vit_tile_classifier\model.onnx
Copy-Item tmp\mahjong_tile_model\labels.json `
  plugin\plugins\mahjong_companion\data\models\vit_tile_classifier\labels.json
Copy-Item tmp\mahjong_tile_model\preprocessor.json `
  plugin\plugins\mahjong_companion\data\models\vit_tile_classifier\preprocessor.json
```

The existing ONNX loader will pick this directory up automatically.

By default, runtime ONNX is used for discard batch classification only. Hand
tiles remain on calibrated templates because the current lightweight model is
not yet strong enough on hand crops. To experiment with ONNX hand crops:

```powershell
$env:MAHJONG_COMPANION_ONNX_HAND_ENABLED = "1"
```

## 5. Evaluate

```powershell
python -m pytest `
  plugin\plugins\mahjong_companion\tests\perception\test_vit_tile_classifier_onnx.py `
  plugin\plugins\mahjong_companion\tests\perception\test_tile_classifier_dispatch.py `
  -q

python plugin\plugins\mahjong_companion\scripts\eval_onnx_accuracy.py
python plugin\plugins\mahjong_companion\scripts\eval_discard_pipeline.py
```

`eval_discard_pipeline.py` is a gated check by default:

- precision >= 0.90
- recall >= 0.95
- F1 >= 0.94

Use `--no-gate` to print metrics without failing the command.
