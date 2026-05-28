# 麻雀教练 ONNX 手牌分类器提升 — 操作清单

## 当前状态

- ONNX 模型弃牌 F1=0.97，手牌 F1=0.48（不可用）
- 代码层面改动已全部完成，需要数据 + 训练
- live_frames 目录已有 31 张去重帧（不够）

## 前置步骤

### 1. 重启 N.E.K.O
让新代码生效（持续存帧 + 1000帧上限）

### 2. 多打几局雀魂（目标 100+ 帧）
- 每局会自动存不同手牌的截图
- 存储位置：`C:\Users\ALEXGREENO\AppData\Local\N.E.K.O\plugins\mahjong_coach\data\live_frames\`
- 检查帧数：`ls 该目录 | wc -l`，到 100+ 即可

## 数据准备

### 3. 提取手牌 crop
```bash
cd d:/N.E.K.O
.venv/Scripts/python scripts/extract_hand_crops.py \
  --input-dir "C:\Users\ALEXGREENO\AppData\Local\N.E.K.O\plugins\mahjong_coach\data\live_frames" \
  --output-dir data/hand_crops \
  --contact-sheet
```
- 自动用 template matcher 标注
- 生成 `data/hand_crops/{牌名}/xxx.png`
- `--contact-sheet` 生成缩略图拼图，方便检查

### 4. 检查标注质量
- 看 `data/hand_crops/_contact_sheet.png` 或各子目录的缩略图
- 把标错的图片移到正确目录
- `unclassified/` 里的需要手动归类

### 5. 安装训练依赖
```bash
.venv/Scripts/pip install torch torchvision timm onnx datasets
```

### 6. 准备数据集（合并 HuggingFace + 本地手牌）
```bash
.venv/Scripts/python scripts/prepare_tile_dataset.py \
  --hand-crops-dir data/hand_crops \
  --output-dir data/tile_dataset
```

## 训练

### 7. 两阶段 fine-tune
```bash
.venv/Scripts/python scripts/train_tile_classifier.py \
  --dataset-dir data/tile_dataset \
  --output-dir tmp/tile_model
```
- Stage 1：冻结 backbone，训练分类头（2 epoch）
- Stage 2：解冻末层，低 lr fine-tune（8 epoch）
- 产出：`tmp/tile_model/model.onnx`

### 8. 评估
```bash
.venv/Scripts/python scripts/eval_tile_classifier.py \
  --model-dir tmp/tile_model \
  --hand-dir data/hand_crops
```
- 目标：弃牌 F1 >= 0.96（不退化），手牌 F1 >= 0.94

## 上线

### 9. 替换模型
```bash
cp tmp/tile_model/model.onnx plugin/plugins/mahjong_coach/data/models/vit_tile_classifier/model.onnx
cp tmp/tile_model/preprocessor.json plugin/plugins/mahjong_coach/data/models/vit_tile_classifier/preprocessor.json
```

### 10. 端到端测试
```bash
.venv/Scripts/python -m pytest plugin/plugins/mahjong_coach/tests/ -v
```
然后重启 N.E.K.O，开一局雀魂，观察教练识别是否正常。

## 已完成的代码改动

| 文件 | 改动 |
|---|---|
| `scripts/extract_hand_crops.py` | 新建：手牌 crop 提取工具 |
| `scripts/prepare_tile_dataset.py` | 新建：数据集合并 + 增强 |
| `scripts/train_tile_classifier.py` | 新建：两阶段 fine-tune + ONNX 导出 |
| `scripts/eval_tile_classifier.py` | 新建：分类评估 + 混淆矩阵 |
| `perception/vit_tile_classifier_onnx.py` | 加 letterbox 预处理支持 |
| `perception/tile_classifier_dispatch.py` | 手牌 ONNX 可选开启 + template 兜底 |
| `coach.py` | 修复防守建议只推荐手里有的牌 |
| `__init__.py` | live loop 加 fingerprint 去重存帧 |
| `models.py` | live_keep_frames 30→200 |
