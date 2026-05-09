# 雀魂牌分类器训练指南

在本地 GPU 机器上训练一个轻量级牌分类器，替代当前的 327MB ViT 模型。

---

## 前置条件

- Python 3.11+
- NVIDIA GPU + CUDA
- 数据集已在本机导出到 `/tmp/tile_dataset/`

---

## 1. 拷贝文件到 GPU 机器

本机打包数据集：

```bash
tar czf tile_dataset.tar.gz -C /tmp tile_dataset
```

把以下两个文件拷到 GPU 机器：

- `tile_dataset.tar.gz` — 数据集
- `plugin/plugins/mahjong_companion/scripts/train_tile_classifier.py` — 训练脚本

GPU 机器上解压：

```bash
tar xzf tile_dataset.tar.gz
```

## 2. 安装依赖

```bash
pip install torch torchvision timm onnx onnxruntime pillow
```

## 3. 训练

```bash
python train_tile_classifier.py \
    --data ./tile_dataset \
    --output ./model_output \
    --backbone mobilenetv3_small \
    --epochs 40
```

可选 backbone（从小到大）：

| backbone | 参数量 | 特点 |
|----------|--------|------|
| `mobilenetv3_small` (默认) | ~2.5M | 最快，CPU 推理友好 |
| `mobilenetv3_large` | ~5.5M | 均衡 |
| `efficientnet_b0` | ~5.3M | 精度更好 |
| `resnet18` | ~11M | 稳定基线 |

## 4. 产物

```
model_output/
├── model.onnx          # ONNX 模型（约 10-50MB，视 backbone 而定）
├── labels.txt          # 类别列表（一行一个）
├── labels.json         # 类别映射（ONNX loader 用）
├── preprocessor.json   # 预处理参数
└── config.json         # 模型配置
```

## 5. 部署回本机

把 `model_output/` 拷回本机，替换现有模型目录：

```bash
# 备份旧模型
cp -r plugin/plugins/mahjong_companion/data/models/vit_tile_classifier \
      plugin/plugins/mahjong_companion/data/models/vit_tile_classifier.bak

# 替换
cp model_output/model.onnx \
   plugin/plugins/mahjong_companion/data/models/vit_tile_classifier/
cp model_output/labels.json \
   plugin/plugins/mahjong_companion/data/models/vit_tile_classifier/
cp model_output/preprocessor.json \
   plugin/plugins/mahjong_companion/data/models/vit_tile_classifier/
```
