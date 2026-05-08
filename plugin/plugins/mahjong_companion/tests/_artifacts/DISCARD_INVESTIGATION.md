# 牌河定位问题 — 技术简报

## 症状

自家（下方）牌河识别正常，左家/对家/右家牌河识别框严重偏移，导致：
- ONNX 分类器框不到真牌，识别失败
- quad refinement（OTSU + 连通块）搜索范围只有 18-36px padding，网格偏 50+px 时完全找不到

## 当前 pipeline（已验证的部分）

```
网格定位 (discard_layout.py, 硬编码像素坐标)
  → 占位检测 (亮度/标准差门控)
    → Quad 精修 (OTSU 阈值 + 连通块, search box = grid ± 18-36px)
      → ONNX 分类 (ViT, 224×224, softmax)
        → 置信度门控 (top-1 < 0.65 当空位)
```

### ONNX 分类器验证结果
- ONNX 导出 vs HuggingFace 原模型：100/100 top-1 一致，max logit diff = 0.017
- 牌河分类（含置信度门控）：**P=1.00 R=0.92 F1=0.96**
- 手牌分类（crop 正确时）：约 78-86% 准确率
- **结论：分类器没问题，瓶颈是定位/裁剪**

### 网格硬编码参数（1920×1080 参考）
```python
# discard_layout.py _BASE_LAYOUTS
"self":           origin=(762,542),  tile=58×70, step=(64,70),  6cols×3rows
"left_opponent":  origin=(624,290),  tile=84×58, step=(82,62),  3cols×6rows, column_major
"top_opponent":   origin=(802,242),  tile=58×70, step=(64,-70), 6cols×3rows
"right_opponent": origin=(1148,290), tile=84×58, step=(82,62),  3cols×6rows, column_major
```

### 自动测量偏移（仅自家可信，其他3家因 pipeline 检测不准不可信）
```
自家:    (+0, +0)  ← 准确
左家:    自动测量不可信（pipeline 找不到牌）
对家:    自动测量不可信
右家:    自动测量不可信
```

## 关键文件

| 文件 | 作用 |
|------|------|
| `perception/discard_layout.py` | 网格硬编码参数，改这里调位置 |
| `perception/discard_parser.py` | 牌河解析主逻辑，含亮度门控 + ONNX 置信度门控 |
| `perception/discard_quad_finder.py` | OTSU + 连通块精修，search box 基于网格位置 |
| `perception/vit_tile_classifier_onnx.py` | ONNX 推理，仅依赖 onnxruntime |
| `perception/tile_classifier_dispatch.py` | ONNX 优先，template matching 备选 |
| `data/models/vit_tile_classifier/model.onnx` | ViT 模型 (327MB) |
| `scripts/measure_anchors.py` | 手动标注工具（浏览器点击量坐标） |
| `tests/fixtures/multi_theme/` | 14 张测试截图 + .tiles.json ground truth |
| `tests/_artifacts/offset_verify/` | 14 张对比图（红框=网格，彩色框=检测） |

## 待解决：左家/对家/右家的网格偏移量

需要手动测量 3 家牌河的真实像素坐标 vs 网格坐标。工具已就绪：
```bash
uv run python -m plugin.plugins.mahjong_companion.scripts.measure_anchors
```
浏览器打开后每张图点 8 下（每家第1张+最后1张牌的左上角）。

数据进来后可以判断：
- 偏移是否系统性（UI 整体平移）→ anchor 法可解
- 偏移是否随机（各部件独立漂）→ 需要 YOLO

## 备选方案

### 方案 A：风向图标 anchor（推荐先试）
- 检测四角東南西北风向图标的 silhouette（圆角矩形 + 高光，不依赖字符）
- 4 点中心对称约束，即使 1 个没检测到也能从另外 3 个反推
- 牌河位置 = 风向图标位置 + 固定偏移
- 工程量约 1 天

### 方案 B：YOLO 牌检测器
- 标注 100 张多主题 bbox + 训练 + ONNX 集成
- 一劳永逸，自带空位过滤
- 工程量约 1.5-2 天
- 如果方案 A 只能解 80% 的 case 再上

## 本次会话改动

1. **ONNX 置信度门控** (`discard_parser.py`): 新增 `ONNX_OCCUPANCY_CONFIDENCE = 0.65`，ONNX top-1 < 0.65 时当空位。牌河 F1 从 50.6% → 96%。
2. **验证脚本**:
   - `scripts/verify_onnx_vs_hf.py` — ONNX vs HF 一致性验证
   - `scripts/eval_onnx_accuracy.py` — ONNX 分类准确率
   - `scripts/eval_discard_pipeline.py` — 牌河 pipeline 评估
   - `scripts/measure_offsets_auto.py` — 自动偏移测量（仅自家可信）
   - `scripts/verify_offsets_visual.py` — 14 张对比图
   - `scripts/measure_anchors.py` — 手动标注工具
