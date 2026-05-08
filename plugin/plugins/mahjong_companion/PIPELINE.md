# Mahjong Companion Pipeline — 技术架构文档

从截屏到出牌建议的完整数据流。

---

## 总览

```
┌──────────┐    ┌──────────────────────┐    ┌──────────┐    ┌───────────┐
│ 截屏捕获  │───▶│ 感知 (Perception)     │───▶│ 决策      │───▶│ 叙述      │
│ Capture   │    │ 场景分类 + 牌识别     │    │ Decision  │    │ Narration │
└──────────┘    └──────────────────────┘    └──────────┘    └───────────┘
     │                   │                       │
     │                   ▼                       ▼
     │            PerceivedGameState        DecisionResult
     │                   │                       │
     ▼                   ▼                       ▼
  FramePacket      MahjongAnalysis          叠加层/语音/聊天
```

三个阶段通过 Adapter 协议解耦，每个阶段可独立替换实现（如用神经网络替换模板匹配）。

---

## 0. 主循环与调度

**文件**: `orchestrator.py`

`SessionOrchestrator` 继承 8 个 Mixin，管理整个生命周期：

```
SessionOrchestrator
├── PreturnFastPathMixin      # 快速路径（摸牌即识别）
├── MeldSelectionMixin         # 副露选择
├── LifecycleControllerMixin   # 启停控制
├── OverlayControllerMixin     # 叠加层管理
├── ConfigAccessorMixin        # 配置访问
├── PipelineExecutionMixin     # 感知→决策→叙述管线
├── FrameResourceMixin         # 帧资源管理
└── StateTransitionMixin       # 状态机
```

主循环流程 (`orchestrator.py:468`)：

```
_run_loop()  [asyncio.Task]
  │
  ├─ sleep(sample_interval_ms / 1000)  ← 默认 300ms
  │
  └─ _run_runtime_cycle_locked()
       │
       ├─ mode == "off"      → skip
       ├─ mode == "standby"  → skip
       └─ mode == "active"   → _run_live_cycle_locked()
```

`_run_live_cycle_locked()` (`pipeline_execution.py:299`) 每轮执行完整管线。

### 运行模式

| 模式 | 行为 |
|------|------|
| `active` | 持续截屏、识别、生成建议 |
| `standby` | 暂停识别循环，保留状态和 UI |
| `off` | 关闭运行循环 |

### 帧去重

**文件**: `gates/frame_change.py`

在进入感知之前，通过感知哈希跳过未变化的帧：

```
DefaultFrameChangeGate.evaluate(frame_path)
  │
  ├─ _compute_hashes(frame_path)       # DCT 哈希, 9×8 灰度
  │    ├─ full_frame_hash: 64-bit
  │    └─ action_bar_hash: 64-bit (bottom_action_bar 区域)
  │
  └─ _hamming_distance(prev, curr)     # XOR + bit_count
       │
       ├─ distance < min_change_distance (默认 3) → SKIP
       └─ distance >= 3                → PROCESS
```

---

## 1. 截屏捕获

**文件**: `capture/provider.py`

`DefaultCaptureProvider` 按优先级尝试多种后端：

| 优先级 | 后端 | 平台 |
|--------|------|------|
| 1 | Win32 PrintWindow | Windows |
| 2 | ImageGrab (窗口裁剪) | Windows/macOS |
| 3 | pyautogui.screenshot | 全平台 |
| 4 | ImageGrab (全屏) | 全平台 |
| 5 | screencapture (命令行) | macOS |
| 6 | grim / gnome-screenshot | Linux |

输出: `FramePacket` (`contracts.py:7`)

```python
@dataclass
class FramePacket:
    timestamp_ms: int
    image_path: str       # 保存路径 (PNG/JPG)
    window_title: str
    width: int
    height: int
    source: str           # "pyautogui" | "screencapture" | ...
```

图片持久化 (`provider.py:332`): PNG compress_level=1, JPG quality=88。

---

## 2. 感知 (Perception)

**文件**: `perception/pipeline.py`

入口: `analyze_image_path()` (L18)

```
analyze_image_path(image_path, *, calibration_dir, template_dir)
  │
  ├─ 1. build_default_rois(width, height)          # ROI 定义
  ├─ 2. collect_region_metrics(image, roi)          # 每个 ROI 的颜色统计
  ├─ 3. classify_scene(metrics)                     # 场景分类
  ├─ 4. detect_actions(scene, metrics)              # 按钮颜色检测
  ├─ 5. detect_button_regions(image, metrics)       # 按钮模板匹配
  └─ 6. enrich_perceived_state_with_tiles(...)      # 牌识别 + 状态完善
```

输出: `PerceivedGameState` (`contracts.py:21`)

### 2.1 ROI 定义与区域度量

**文件**: `perception/roi.py`

5 个固定 ROI（百分比定义，适配任意分辨率）：

| ROI | 位置 | 用途 |
|-----|------|------|
| `top_banner` | left=5%, top=3%, w=90%, h=12% | 结果画面、标题栏 |
| `center_dialog` | left=25%, top=20%, w=50%, h=30% | 弹窗对话框 |
| `bottom_action_bar` | left=18%, top=76%, w=64%, h=16% | 操作按钮栏 |
| `bottom_hand_area` | left=12%, top=68%, w=76%, h=26% | 手牌区域 |
| `right_replay_panel` | left=78%, top=10%, w=18%, h=70% | 回放面板 |

`collect_region_metrics()` (L40) 对每个 ROI 计算（sample_step=6 隔行采样）：

| 指标 | 计算方式 | 用途 |
|------|---------|------|
| `mean_luma` | 平均亮度 | 槽位占用判断 |
| `stddev` | 亮度标准差 | 纹理丰富度 |
| `bright_ratio` | luma > 180 的像素比例 | 白色/亮面 |
| `dark_ratio` | luma < 50 的像素比例 | 暗色/空槽 |
| `white_ratio` | luma > 230 且 saturation < 25 | 纯白 |
| `colorful_ratio` | saturation > 80 | 彩色区域 |
| `gold_ratio` | hue 25-50, sat > 50, val > 100 | 金色按钮/立直棒 |
| `orange_ratio` | hue 10-25, sat > 60, val > 100 | 橙色元素 |
| `red_ratio` | R > 160, G < 95, B < 95, R > G+70 | 红色标记 |
| `green_ratio` | G > 130, R < 100, B < 100 | 绿色按钮 |

### 2.2 场景分类

**文件**: `perception/scene_classifier.py`

入口: `classify_scene(metrics)` (L6)

纯规则分类，按优先级从高到低匹配：

| 优先级 | 场景 | 关键条件 | 置信度 |
|--------|------|---------|--------|
| 1 | `replay` | 右侧面板 dark>=0.42, bright>=0.10, stddev<=78 | 0.62-0.82 |
| 2 | `result` | 顶部 gold>=0.08, 全局 dark>=0.18 | 0.60-0.78 |
| 3 | `dialog` (暗) | 中间 dark>=0.55, stddev<=48, colorful>=0.16 | 0.55-0.72 |
| 4 | `dialog` (亮) | 中间 white>=0.12, stddev>=40 | 0.50-0.68 |
| 5 | `lobby` | 顶部 orange>=0.04, bright>=0.32 | 0.52-0.72 |
| 6 | `room_setup_menu` | 特定颜色组合 | 0.42-0.60 |
| 7 | `in_match` (dark/colorful table) | action_bar + dark_ratio>=0.32, colorful>=0.45, center.dark<=0.65 | 0.61-0.78 |
| 8 | `in_match` (bright tablecloth) | full.luma>=100, full.dark<=0.25, hand+bottom bright>=0.10 | 0.70 |
| 9 | `unknown` | 兜底 | 0.22 |

`in_match` 有四条独立路径：(1) bottom_bar + match_table + player_hand，(2) match_table alone，(3) player_hand + blue_table/live_table，(4) bright_tablecloth。路径 (3) 的 `center.dark_ratio` 阈值已从 0.3 放宽到 0.65 以兼容暗色牌河（红色桌布）；路径 (4) 为亮色桌布（西瓜等）新增，不依赖 dark_ratio。Baseline 验证：14 张多主题截屏中 13/14 命中 in_match。

返回: `(scene, confidence, notes, roi_hits)`

### 2.3 操作按钮检测

**文件**: `perception/action_detector.py`

两阶段检测：

**阶段 1: 颜色比例快筛** (`detect_actions()`, L30)

| 按钮 | 条件 |
|------|------|
| chi (吃) | green >= 0.028 |
| riichi (立直) | gold >= 0.052 |
| ron (荣和) / tsumo (自摸) | red >= 0.02 |
| skip (跳过) | orange >= 0.03 |

**阶段 2: 模板匹配精筛** (`detect_button_regions()`, L96)

在 `bottom_action_bar` 和 `center_dialog` 区域内，使用模板图片匹配：

```
对每个按钮模板 PNG:
  ├─ cv2.matchTemplate(TM_CCOEFF_NORMED)    # 优先 OpenCV
  └─ numpy FFT NCC                            # 回退方案
```

模板文件位于 `perception/templates/{resolution}/`。

返回: `list[ButtonRegion]` — 每个含 button_type, bbox, confidence, template_id。

### 2.4 手牌识别

**文件**: `perception/tile_parser.py` + `perception/tile_templates.py`

这是整个感知阶段最复杂的部分。

#### 2.4.1 整体流程

```
parse_tiles_from_image(image_path, image, *, scene, ...)
  │
  ├─ resolve_calibration_profile(width, height)    # 加载校准 profile
  │
  ├─ [fixture 快捷路径]                              # 有 .tiles.json sidecar 时直接读
  │    └─ 读取预标注手牌/牌河 → 返回
  │
  └─ _from_template_profile(image, *, calibration)
       │
       ├─ build_hand_layout(width, height)          # 计算 14 个手牌槽位
       │
       ├─ _best_template_hand_result(image, ...)    # 尝试多种 draw_slot 布局
       │    ├─ draw_slot_index = 14 (无副露)
       │    ├─ draw_slot_index = 11 (1 副露)
       │    ├─ draw_slot_index = 8  (2 副露)
       │    ├─ draw_slot_index = 5  (3 副露)
       │    └─ draw_slot_index = 2  (4 副露)
       │
       ├─ _classify_hand_from_layout(image, layout) # 逐槽位模板匹配
       │
       ├─ parse_discards_from_image(image, ...)     # 牌河识别
       │
       ├─ _with_external_discard_result(...)         # 外部识别器（可选）
       │
       ├─ _with_visual_riichi_result(...)            # 立直棒检测
       │
       └─ _derive_known_genbutsu_tiles(...)          # 推导现物
```

#### 2.4.2 手牌槽位布局

**文件**: `perception/hand_layout.py`

`build_hand_layout()` (L24) 基于 calibration profile 计算槽位：

```python
# 手牌几何（基于百分比 + calibration offset）
hand_left  = width * 0.14 + offsets.x_px
hand_top   = height * 0.72 + offsets.y_px
tile_width = width * 0.036 + offsets.width_px
tile_height = height * 0.112 + offsets.height_px
gap        = tile_width * 0.12 + offsets.gap_px

# draw_slot 与前 13 张之间有额外间距
draw_gap   = tile_width * 0.28 + offsets.draw_gap_px
```

输出包含三组槽位：
- **hand**: 14 个 TileSlot（13 张 + 1 张摸牌位）
- **dora**: 5 个 TileSlot（宝牌指示）
- **meld**: 4 个 TileSlot（副露，每组 3 张）

#### 2.4.3 模板签名提取与匹配

**文件**: `perception/tile_templates.py`

核心签名算法 `extract_tile_signature()` (L81)：

```
原始裁剪 (slot.box 区域)
  │
  ├─ 内缩裁剪: inner_bounds = (0.06, 0.06, 0.94, 0.82)
  │    └─ 去掉牌边框，只保留牌面内容（上 6%, 下 18%, 左右各 6%）
  │
  ├─ resize 到 16×24 RGB
  │
  └─ 展平为 bytes → 16 × 24 × 3 = 1152 字节签名
```

匹配算法 `classify_tile_from_templates()` (L104)：

```
对模板库中每个 tile 的所有签名样本:
  │
  ├─ RMS 距离 = sqrt(mean((template - query)²))
  │
  ├─ 取最小距离的 tile 为 best_match
  ├─ 取次小距离的 tile 为 runner_up
  │
  └─ 置信度 = distance_score × 0.72 + margin_score × 0.28
       ├─ distance_score = max(0, 1 - best / max_distance)
       │   max_distance = 82.0 (默认)
       └─ margin_score = (runner_up - best) / runner_up
```

返回: `TileTemplateMatch(tile, confidence, distance, runner_up_tile, runner_up_distance)`

#### 2.4.4 槽位占用判断

`is_probably_occupied_hand_slot()` (tile_templates.py:143):

```
槽位被占用 ⟺ 同时满足:
  ├─ mean_luma >= 95
  ├─ bright_ratio >= 0.16
  ├─ dark_ratio <= 0.55
  └─ stddev >= 18
```

#### 2.4.5 布局评分

`_best_template_hand_result()` (tile_parser.py:223) 尝试 5 种 draw_slot_index，选择评分最高的布局。

`_hand_layout_score()` (tile_parser.py:321) 评分维度：

| 维度 | 权重 | 说明 |
|------|------|------|
| 手牌数量 | 高 | 优先选择 14/11/8/5/2（符合打牌轮次） |
| 平均置信度 | 中 | 手牌整体识别可信度 |
| 形状匹配 | 高 | 手牌数是否匹配已知合法形状 |

合法手牌数对照表 (`tile_parser.py:26-29`)：

| 手牌数 | 副露数 | 含义 |
|--------|--------|------|
| 14 | 0 | 正常摸牌后（需打牌） |
| 13 | 0 | 正常等待摸牌 |
| 11/10 | 1 | 1 副露 |
| 8/7 | 2 | 2 副露 |
| 5/4 | 3 | 3 副露 |
| 2/1 | 4 | 4 副露 |

#### 2.4.6 易混淆牌交叉校验

**文件**: `tile_parser.py:332`

`_AMBIGUOUS_PAIRS` 定义 6 组易混淆牌对：

| 牌对 | margin_max | conf_max |
|------|-----------|----------|
| 6s / 9s | 0.38 | 0.65 |
| 4p / 5p | 0.32 | 0.58 |
| 5p / 6p | 0.30 | 0.58 |
| 6p / 7p | 0.28 | 0.55 |
| 2m / 3m | 0.35 | 0.62 |
| 2s / 3s | 0.35 | 0.62 |

当 best_match 属于某混淆对时，取对应 partner 的签名重新比较。如果 margin 过低或置信度不够，标记为低置信。

### 2.5 牌河（弃牌）识别

**文件**: `perception/discard_parser.py` + `perception/discard_layout.py` + `perception/discard_quad_finder.py`

#### 2.5.1 牌河布局

**文件**: `perception/discard_layout.py`

4 家弃牌区基于 1920×1080 基准坐标缩放：

| 玩家 | 网格 | 朝向 |
|------|------|------|
| self (自家) | 6 列 × 3 行 | bottom (正向) |
| left_opponent | 3 列 × 6 行 (列主序) | left (旋转 90°) |
| top_opponent | 6 列 × 3 行 | top (旋转 180°) |
| right_opponent | 3 列 × 6 行 (列主序) | right (旋转 270°) |

**已知限制**: 自家网格定位准确；左家/对家/右家的硬编码 origin/step 参数与实际位置有偏移，需要从实测数据反推或换用 anchor 方案。

#### 2.5.2 四边形精修

**文件**: `perception/discard_quad_finder.py`

`refine_discard_slot_quad()` (L40) 对每个弃牌槽位做透视精修：

```
初始 slot.box
  │
  ├─ 扩展搜索区域
  ├─ _tile_face_mask(crop)               # 创建牌面遮罩
  │    ├─ OTSU 自适应亮度阈值: 自动分离牌面与背景
  │    └─ 饱和度约束: saturation <= 100 (牌面低饱和)
  │
  ├─ 连通域分析
  ├─ _best_component(mask)               # 选最佳连通域
  │    └─ score = area + overlap×1.8 - distance×5.0
  │
  └─ _quad_from_component(xs, ys)        # 百分位提取四角坐标
       ├─ 四角: (P5, P95) × (P5, P95)
       └─ 输出: Quad (4 个角点坐标)
```

#### 2.5.3 弃牌识别流程

`parse_discards_from_image()` (discard_parser.py:51) 采用三阶段批处理流水线：

```
Phase 1 — 发现 (per-slot, 无分类):
  对每个玩家的每个弃牌槽位:
    ├─ is_probably_occupied_discard_slot(metrics)    # 占用判断
    │    └─ mean_luma>=88, bright>=0.12 OR white>=0.04, dark<=0.62, stddev>=14
    ├─ refine_discard_slot_quad(image, slot)          # OTSU 四边形精修
    └─ 收集 base_crop (占用时)

Phase 2 — 批量 base 分类 (单次 ONNX forward):
  classify_tiles_batch([base_crops...], template_payload)
    └─ ONNX 可用时走神经网络，否则模板匹配

Phase 3 — 批量 refined 分类 (单次 ONNX forward, 仅 base 失败的槽):
  对 base_match.confidence < min_confidence 且有 quad refinement 的槽:
    ├─ crop_discard_quad(image, quad, orientation)   # 透视校正
    └─ classify_tiles_batch([refined_crops...])

Phase 4 — 接受/拒绝 (顺序遍历, IoU 去重):
  对每个槽位:
    ├─ 选 best match (base vs refined)
    ├─ ONNX 置信度门控: confidence >= 0.65 (ONNX_OCCUPANCY_CONFIDENCE)
    ├─ 易混淆对检测: {5p,6p}, {6p,7p}, {6s,9s}
    │    └─ confidence < 0.78 或 distance margin < 12.0 → 拒绝
    └─ IoU 去重: 重叠度 > 0.45 时保留置信度更高的
```

**关键变更 (v1.2)**: 旧版逐槽位串行分类改为三阶段批处理，一帧最多 2 次 ONNX forward；新增 ONNX 置信度门控（top-1 < 0.65 → 空位），牌河 F1 从 ~50% 提升到 0.96。

### 2.6 外部识别器

**文件**: `perception/external_discard_recognizer.py`

可通过环境变量接入外部牌河识别器（如神经网络）：

| 环境变量 | 说明 |
|---------|------|
| `MAHJONG_COMPANION_DISCARD_RECOGNIZER_CMD` | 子进程命令行 |
| `MAHJONG_COMPANION_DISCARD_RECOGNIZER_URL` | HTTP 端点 |
| `MAHJONG_COMPANION_DISCARD_RECOGNIZER_TIMEOUT_SEC` | 超时 (默认 1.5s) |

### 2.7 ONNX 神经网络识别器

ViT 神经网络通过 ONNX runtime 接管线，ONNX 模型文件存在时走神经网络（batch 推理），否则回落到模板匹配。

#### 已验证

| 验证项 | 结果 |
|--------|------|
| ONNX vs HuggingFace 一致性 | 100/100 top-1 一致，max logit diff = 0.017 |
| 牌河分类（含置信度门控 0.65） | **P=1.00 R=0.92 F1=0.96** |
| 手牌分类（crop 正确时） | 约 78-86% |
| **瓶颈** | 网格定位/裁剪，不是分类器 |

#### 已就绪

| 资源 | 状态 | 位置 |
|------|------|------|
| onnxruntime 1.25.0 | 已安装 | pyproject.toml |
| ONNX 模型 (327.6 MB) | 已导出 | `data/models/vit_tile_classifier/model.onnx` |
| ONNX 推理类 | 已就绪 | `perception/vit_tile_classifier_onnx.py` |
| Dispatch 分发层 | **已接管线** | `perception/tile_classifier_dispatch.py`（ONNX → templates 回落链） |
| 手牌识别 (`_classify_hand_from_layout`) | **已接** | `tile_parser.py` — `classify_tile()` |
| 牌河识别 (`parse_discards_from_image`) | **已接，批处理** | `discard_parser.py` — `classify_tiles_batch()`（一帧最多 2 次 forward：base + 弱-base 的 refined） |
| 快速路径 (`drawn_tile_fast_path`) | **已接** | `drawn_tile_fast_path.py` — `classify_tile()` |
| 副露选择 (`meld_selection`) | **已接** | `meld_selection.py` — `classify_tile()` |
| ViT 分类器（transformers 路径） | 已就绪，仅离线训练用 | `perception/vit_tile_classifier.py` |
| 测试 | 15 个（14 passed / 1 skipped — `test_backend_consistency_against_transformers` 在无 transformers 时 skip） | `tests/perception/test_tile_classifier_dispatch.py` + `test_vit_tile_classifier_onnx.py` |

#### 回落链

```
classify_tile(crop, template_payload)
  │
  ├─ ONNX 模型可用? ──是──→ classify_tile_crops_onnx([crop])
  │                         └─ 转换 VitTilePrediction → TileTemplateMatch
  │                            (confidence 直接映射, distance = (1-conf)×82)
  │
  └─ ONNX 不可用 ──→ classify_tile_from_templates(crop, template_payload)  [原有路径]
```

所有 4 个调用点统一通过 `classify_tile()` 分发，下游代码（拒绝逻辑、易混淆对校验）无需改动。

#### 网络受限时的替代方案

在有网环境运行导出脚本，将 `data/models/vit_tile_classifier/` 目录拷贝到离线机器即可；运行时也可设 `MAHJONG_COMPANION_VIT_ONNX_DIR` 指向其它路径。

### 2.8 积分面板锚点检测

**文件**: `perception/panel_anchor.py`

`detect_score_panel()` 检测屏幕中央的积分面板（深色矩形 UI 区域），为后续基于风向图标 anchor 的牌河定位方案提供基础：

```
积分面板检测:
  │
  ├─ 灰度转换, 暗区遮罩 (luma < 90)
  ├─ 连通域分析 (flood fill)
  ├─ 面积过滤: image_area × 0.5% ~ 8%
  ├─ 宽高比过滤: 1.5 ~ 6.0 (面板宽>高)
  ├─ 位置评分: 距屏幕中心距离
  └─ 返回 (left, top, right, bottom) 或 None
```

### 2.9 立直棒检测

**文件**: `perception/riichi_detector.py`

`detect_riichi_players()` (L61) 检测各玩家立直棒：

```
对每个玩家的立直棒 ROI:
  │
  ├─ 白色遮罩: luma >= 176 AND saturation <= 74
  ├─ 红色遮罩: R >= 160, G <= 95, B <= 95, R >= G+70
  │
  ├─ 连通域分析
  ├─ 形状评分: 长条形 (aspect >= 3.2, 长比 >= 0.42)
  │
  └─ 置信度 = min(0.98, 0.52 + score×0.26 + red_score×0.20)
```

### 2.10 快速路径：摸牌即识别

**文件**: `perception/drawn_tile_fast_path.py` + `fast_path.py`

当对手回合预计算了打牌方案（preturn plan），用户摸牌时只需识别摸到的那张牌：

```
_maybe_emit_fast_preturn_advice_locked()  [fast_path.py:220]
  │
  ├─ detect_drawn_tile_fast_path(frame_path)    # 只识别第 14 个槽位
  │    └─ 单张裁剪 + classify_tile_from_templates
  │
  └─ apply_preturn_draw_tile(state, plan, drawn_tile)  # 增量更新
       └─ build_incremental_draw_candidates()           # 不重新扫描全手牌
```

比完整管线快 3-5 倍，从 ~300ms 降到 ~60-100ms。

### 2.11 感知输出

`PerceivedGameState` (`contracts.py:21`)：

```python
@dataclass
class PerceivedGameState:
    scene: str                              # "in_match" | "lobby" | "replay" | ...
    confidence: float                       # 场景置信度 0-1
    is_user_turn: bool                      # 是否轮到自己打牌
    buttons: list[str]                      # 可见按钮: ["chi","pon","skip",...]
    notes: list[str]                        # 感知备注
    roi_hits: dict[str, bool]               # 各 ROI 是否被命中
    hand_tiles: list[str]                   # 手牌: ["1m","3p","5s","7z",...]
    melds: list[list[str]]                  # 副露: [["2m","3m","4m"], ...]
    dora_indicators: list[str]              # 宝牌指示: ["5p"]
    riichi_players: list[str]               # 立直玩家: ["left_opponent"]
    discard_piles: dict[str, list[str]]     # 牌河: {"self":["1m","3p",...], ...}
    visible_tiles: list[str]                # 所有可见牌
    known_genbutsu_tiles: list[str]         # 现物（安全牌）
    button_regions: list[ButtonRegion]      # 精确按钮位置
    raw_detections: list[dict]              # 原始检测结果
    analysis_hints: dict[str, Any]          # 扩展提示
```

---

## 3. 决策 (Decision)

**文件**: `decision/generator.py`

入口: `build_decision(state)` (L22)

```
build_decision(state)
  │
  ├─ _resolve_effective_scene(state, buttons)     # 场景提升
  │    └─ unknown → in_match (如果检测到手牌/牌河/按钮)
  │
  ├─ 按优先级分类决策类型:
  │    ├─ win_buttons (ron/tsumo)      → danger_action,  priority=96
  │    ├─ declaration (riichi)         → danger_action,  priority=88
  │    ├─ kan                          → danger_action,  priority=82
  │    ├─ call_buttons (chi/pon)       → action_available, priority=72
  │    ├─ passive_buttons (skip/confirm)→ action_available, priority=56
  │    ├─ user_turn (无按钮, 需打牌)   → scene_update,    priority=44
  │    └─ 其他                         → scene_update,    priority=20
  │
  ├─ build_mahjong_analysis(state, ...)            # 牌理分析
  │
  ├─ 根据决策类型生成文案:
  │    ├─ danger_action → _build_riichi_to_action_copy() 或直接提醒
  │    ├─ call (chi/pon/kan) → _build_call_to_action_copy()
  │    └─ tile_efficiency_hint → 打牌建议
  │
  └─ 返回 DecisionResult
```

### 3.1 牌理分析

**文件**: `decision/tile_efficiency.py`

入口: `build_mahjong_analysis()` (L33)

```
build_mahjong_analysis(state)
  │
  ├─ _estimate_shanten(hand_tiles)                 # 向听数
  ├─ _estimate_candidate_discards(hand_tiles, ...)  # 候选打牌
  ├─ _derive_attack_defense_bias(state)             # 攻防倾向
  ├─ estimate_defense_alerts(state, ...)            # 防守警告
  └─ _build_teaching_points(...)                    # 教学要点
```

### 3.2 向听数计算

**文件**: `tile_efficiency.py:532-688`

三路并行取最小值：

| 形式 | 算法 | 复杂度 |
|------|------|--------|
| 标准形 (4 面子 + 1 雀头) | 递归穷举 + `@lru_cache(131072)` | O(34^3) 级别，缓存后极快 |
| 七对子 | 直接统计对子数 | O(34) |
| 国士无双 | 统计 13 种幺九牌种数 | O(34) |

标准形递归搜索 `_standard_shanten_search(state, mentsu, taatsu, pair)` (L625):

```
向听数 = 8 - mentsu×2 - taatsu - pair

递归策略:
  对 34 种牌 (9m + 9p + 9s + 7z)，逐种决定:
  ├─ 组成面子 (刻子/顺子) → mentsu += 1, counts 减少
  ├─ 组成搭子 (对子/两面/嵌张/边张) → taatsu += 1
  ├─ 组成雀头 → pair = 1
  └─ 跳过 → 下一种牌

  剪枝: 已有的 mentsu + taatsu 超过 4 时提前返回
```

有副露时起始 mentsu = open_melds。

### 3.3 候选打牌生成

**文件**: `tile_efficiency.py:245`

`_estimate_candidate_discards()` 对手牌中每张唯一牌计算：

```
对每张候选牌:
  │
  ├─ _raw_discard_score(tile, counts, dora_tiles)         # 原始启发式评分
  │    ├─ 字牌孤立单张: +2.2
  │    ├─ 完整面子: -2.2
  │    ├─ 两面搭子: -1.8
  │    ├─ 嵌张搭子: -0.8
  │    ├─ 边张搭子: -0.7
  │    ├─ 端牌 (1/9): +1.3
  │    └─ dora 惩罚: -1.6
  │
  ├─ _estimate_post_discard_shanten_from_counts()          # 打掉后的向听数
  │
  ├─ _calculate_discard_ukeire_from_counts()               # 受入宽度
  │    └─ 遍历所有山牌，模拟摸入后向听数是否降低
  │       └─ ukeire = Σ(能使向听数降低的牌的剩余枚数)
  │
  ├─ _tenpai_wait_quality_bonus()                           # 听牌等待质量加分
  │    └─ 1 向听时，保留两面形状的 bonus
  │
  ├─ _defensive_safety_hint(tile, ...)                      # 安全性评估
  │    └─ "genbutsu" / "dead" / "suji" / "high" / "medium" / "low" / "unknown"
  │
  └─ _discard_strategy_score(...)                           # 综合策略分
       ├─ strategy_mode: defense / guarded_push / push / balanced
       └─ 加权: 攻击分 × 权重 + 防御分 × 权重
```

排序规则 (`_rank_discard_candidates()`, L796)：

```
优先级 (从高到低):
  1. 打掉后向听数最小
  2. 受入宽度最大
  3. 策略分最高
  4. 原始评分最高
  5. 安全性最高
```

取 top 3 候选。

### 3.4 攻防判断

**文件**: `tile_efficiency.py:842`

```
_derive_attack_defense_bias(state):
  │
  ├─ 有人立直 → "slightly_defensive"
  ├─ 向听 <= 1 → "slightly_attack"
  └─ 其他      → "neutral"
```

策略模式 `_discard_strategy_mode()` (L842)：

| 模式 | 条件 | 权重 |
|------|------|------|
| `defense` | 有立直压力 + 向听 >= 2 | 防御权重高 |
| `guarded_push` | 有立直压力 + 向听 <= 1 | 攻守平衡 |
| `push` | 无立直压力 + 向听 <= 1 | 攻击权重高 |
| `balanced` | 默认 | 均衡 |

### 3.5 安全性评估

**文件**: `tile_efficiency.py:1204` + `decision/risk_estimator.py`

`_defensive_safety_hint()` 分级：

| 安全性 | 条件 |
|--------|------|
| `genbutsu` (现物) | 牌在某立直玩家的牌河中出现过 |
| `dead` (死牌) | 所有 4 张都可见，对手不可能持有 |
| `suji` (筋牌) | 同花色间隔 3 的牌已出现在立直玩家牌河 |
| `high` | 字牌，已出 2+ 枚 |
| `medium` | 字牌，已出 1 枚 |
| `low` | 中张数牌，无任何安全信号 |
| `unknown` | 信息不足 |

`estimate_defense_alerts()` (risk_estimator.py:31) 额外检查：
- 立直压力警告
- 候选牌中的现物牌提示
- 立直下开杠风险警告

### 3.6 吃碰杠判断

**文件**: `decision/generator.py:475`

`_build_call_to_action_copy()` 分析吃/碰/杠建议：

```
判断逻辑:
  │
  ├─ 向听数变化: 吃碰后向听数是否降低
  ├─ 攻防偏向: 有人立直时偏保守
  ├─ 门前役评估: 开杠/碰会破坏门前
  │    ├─ 已有副露 → 可考虑断幺、混一等
  │    └─ 门前清 → 碰会断送立直/平和/一杯口等
  │
  └─ _open_hand_yaku_signal(state)  # 检测可见的副露役
       ├─ 断幺检测
       ├─ 混一色检测
       └─ 字牌对子检测
```

### 3.7 立直判断

**文件**: `decision/generator.py:396`

`_build_riichi_to_action_copy()` 评估是否应该立直：

```
立直建议条件:
  ├─ 受入宽度足够 (>= 4 种)
  ├─ 无对手立直压力 或 有好的待牌
  └─ 等待质量 (两面 > 嵌张 > 边张)
```

### 3.8 Pre-turn 预计算

**文件**: `decision/preturn_planner.py`

对手回合时预计算打牌方案：

```
build_preturn_discard_plan(state)   # 仅在 is_user_turn=False 且等待手牌数时
  │
  ├─ 保存当前手牌快照
  ├─ 计算候选打牌
  ├─ 保存 shanten/ukeire/bias/alerts
  │
  └─ 返回 PreturnDiscardPlan

用户摸牌后:
  apply_preturn_draw_tile(state, plan, drawn_tile)
    │
    ├─ _find_drawn_tile(previous_hand, current_hand)  # Counter 差值找新牌
    └─ build_incremental_draw_candidates()             # 只评估新牌 + 缓存候选
```

### 3.9 决策输出

`DecisionResult` (`contracts.py:44`)：

```python
@dataclass
class DecisionResult:
    decision_type: str               # "tile_efficiency_hint" | "danger_action" | "action_available" | "scene_update"
    priority: int                    # 20-96
    risk_level: str                  # "low" | "medium" | "high"
    action_required: str             # "discard" | "riichi" | "call" | "win" | "none"
    speakable: str                   # 语音播报文案
    summary: str                     # 摘要
    detail: str                      # 详情
    suggestion: str                  # 建议
    recommended_focus: str           # 推荐关注点
    scene: str                       # 当前场景
    buttons: list[str]               # 可见按钮
    reason_codes: list[str]          # 原因码
    review_tags: list[str]           # 审查标签
    mahjong_analysis: dict           # MahjongAnalysis 的 dict 形式
    engine_meta: dict                # 引擎元信息
```

`MahjongAnalysis` (`contracts.py:67`)：

```python
@dataclass
class MahjongAnalysis:
    analysis_version: str
    tile_level_available: bool       # 牌理分析是否可用
    tile_level_state: str            # 牌局状态描述
    analysis_confidence: float       # 分析置信度
    hand_shape_confidence: float     # 手牌形状置信度
    shanten_estimate: int | None     # 向听数 (-1=和了, 0=听牌)
    ukeire_estimate: int | None      # 受入宽度
    candidate_discards: list[dict]   # 候选打牌 top3
    attack_defense_bias: str         # 攻防倾向
    defense_alerts: list[str]        # 防守警告
    teaching_points: list[str]       # 教学要点
```

---

## 4. 校准系统

**文件**: `perception/calibration.py`

### 4.1 CalibrationProfile

```python
@dataclass
class CalibrationProfile:
    profile_id: str                  # 唯一标识
    version: str                     # "v0.3-calibration"
    source: str                      # 来源
    enabled: bool                    # 是否启用
    screen_width: int                # 分辨率宽
    screen_height: int               # 分辨率高
    confidence: float                # 置信度
    hand_offsets: CalibrationOffsets  # 手牌偏移
    meld_offsets: CalibrationOffsets  # 副露偏移
    dora_offsets: CalibrationOffsets  # 宝牌偏移
    hand_tile_templates: dict        # 手牌模板签名库
    discard_tile_templates: dict     # 牌河模板签名库
```

### 4.2 Profile 加载策略

`resolve_calibration_profile()` (L101):

```
1. 精确匹配: 找到 width×height 完全一致的 profile
2. 近似缩放: 取最近的分辨率 profile，按比例缩放偏移
   └─ 置信度惩罚: × 0.82
3. 兜底: build_default_calibration_profile()
   └─ enabled=False, confidence=0.18
```

### 4.3 模板训练

`train_calibration_profile()` (L143) 从标注样本训练 profile：

```
标注目录 (含 .tiles.json sidecar)
  │
  ├─ 收集所有标注图片
  ├─ 对每个标注的手牌槽位裁剪
  ├─ extract_tile_signature(crop) → 1152 字节签名
  ├─ 按 tile code 分组，每组最多 12 个样本
  │
  └─ 置信度公式:
       min(0.95, 0.35 + samples/min_samples×0.45 + annotated_slots/(min_samples×14)×0.15)
```

训练后的 profile 保存为 JSON 到 `data/calibration/profiles/`。

---

## 5. 牌编码规范

**文件**: `tile_labels.py`

### 编码格式

| 类型 | 格式 | 示例 |
|------|------|------|
| 万子 | `{n}m` | `1m`=一万, `9m`=九万 |
| 筒子 | `{n}p` | `1p`=一筒, `9p`=九筒 |
| 索子 | `{n}s` | `1s`=一索, `9s`=九索 |
| 字牌 | `{n}z` | `1z`=东, `2z`=南, `5z`=白, `6z`=发, `7z`=中 |

`normalize_tile()` (L100) 统一变体：
- `r5m` / `0m` → `5m` (赤牌)
- `E` / `S` / `W` / `N` → `1z` / `2z` / `3z` / `4z`
- `P` / `F` / `C` → `5z` / `6z` / `7z`

### Dora 推导

`_derive_dora_tiles()` (tile_efficiency.py:1262)：

```
指示牌 → 宝牌
  数牌: +1 (9 → 1 循环)
  风牌: 东→南→西→北→东 (1z→2z→3z→4z→1z)
  三元牌: 白→发→中→白 (5z→6z→7z→5z)
```

---

## 6. 数据结构总览

```
FramePacket
  └─ timestamp_ms, image_path, width, height, source

PerceivedGameState
  ├─ scene, confidence, is_user_turn
  ├─ buttons: list[str]
  ├─ hand_tiles: list[str]
  ├─ melds: list[list[str]]
  ├─ dora_indicators: list[str]
  ├─ riichi_players: list[str]
  ├─ discard_piles: dict[str, list[str]]
  ├─ visible_tiles: list[str]
  ├─ known_genbutsu_tiles: list[str]
  ├─ button_regions: list[ButtonRegion]
  ├─ raw_detections: list[dict]
  └─ analysis_hints: dict

MahjongAnalysis
  ├─ shanten_estimate: int | None
  ├─ ukeire_estimate: int | None
  ├─ candidate_discards: list[dict]
  │    └─ {tile, post_discard_shanten, ukeire, raw_score,
  │        strategy_score, safety_hint, confidence}
  ├─ attack_defense_bias: str
  ├─ defense_alerts: list[str]
  └─ teaching_points: list[str]

DecisionResult
  ├─ decision_type, priority, risk_level
  ├─ action_required, speakable, summary
  ├─ detail, suggestion, recommended_focus
  ├─ mahjong_analysis: dict
  └─ engine_meta: dict

TileTemplateMatch
  ├─ tile: str
  ├─ confidence: float
  ├─ distance: float
  └─ runner_up_tile, runner_up_distance

CalibrationProfile
  ├─ screen_width, screen_height, confidence
  ├─ hand/meld/dora_offsets: CalibrationOffsets
  ├─ hand_tile_templates: dict    # {tile_code: [signatures]}
  └─ discard_tile_templates: dict

PreturnDiscardPlan
  ├─ hand_tiles, hand_signature
  ├─ candidate_discards, shanten_estimate
  └─ ukeire_estimate, attack_defense_bias
```

---

## 7. 关键配置参数

**文件**: `config_defaults.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sample_interval_ms` | 300 | 主循环采样间隔 |
| `fast_poll.interval_ms` | 120 | 快速轮询间隔 |
| `fast_poll.duration_sec` | 7 | 快速轮询持续时长 |
| `frame_change_gate.min_change_distance` | 3 | 帧去重 Hamming 距离阈值 |
| `frame_change_gate.stable_skip_limit` | 300 | 连续稳定帧跳过上限 |
| `debug_samples.max_frames` | 180 | 最大调试截图数 |
| `debug_samples.max_age_sec` | 600 | 调试截图最大保留时间 |
| `speech_policy.normal_voice_cooldown_sec` | 18 | 普通语音冷却时间 |
| `speech_policy.danger_voice_cooldown_sec` | 5 | 危险语音冷却时间 |
| `speech_policy.dedupe_window_sec` | 8 | 语音去重窗口 |
| `overlay.fast_button_scan_min_interval_ms` | 60 | 快速按钮扫描最小间隔 |

---

## 8. 已知薄弱点与优化方向

1. **牌河网格定位偏移 (当前主要瓶颈)**: 自家牌河网格准确，左家/对家/右家的硬编码 origin/step 参数与实际位置有偏移（50+px）。分类器本身已验证 F1=0.96，瓶颈完全在定位。解决方案：(A) 风向图标 anchor 检测，(B) YOLO 牌检测器。14 张实测偏移数据在 `discard_offsets.json`。
2. **模板匹配精度**: 当前使用 16×24 RGB 签名 + RMS 距离，对易混淆牌（6s/9s, 5p/6p 等）区分力不足。ONNX ViT 后端已接管线作为默认 backend（见 2.7），模板匹配现在作为回落路径。
3. **Calibration 依赖**: 无对应分辨率 profile 时识别严重退化（hand_tiles 和 discard_piles 均为空）。可考虑：自适应校准、自动 profile 生成、或为常见分辨率预置模板。
4. **弃牌识别鲁棒性**: 四边形精修在牌河拥挤时精度下降，重叠牌难以分离。OTSU 自适应 mask 已替换硬编码颜色阈值，但在亮色桌布上仍需 calibration 数据量化效果。
5. **向听数缓存**: 递归穷举 + lru_cache 在极端手牌组合下可能miss率高，可考虑迭代式向听算法。
6. **场景分类硬编码阈值**: 当前 `center.dark_ratio<=0.65`、`full.dark_ratio<=0.25` 等常数基于 baseline 14 张图调参，面对新主题（绿色、紫色等）可能需要继续放宽或改用聚类。
