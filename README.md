# SAR 旋转目标检测对比实验

## 一、实验目的

本项目旨在验证我们提出的 **Full Innovation 模型** (RFR + CMLCA Plus + MSWR) 在 SAR 图像旋转目标检测任务上的有效性。

通过与主流方法进行对比实验，证明我们方法的优势和创新点。

---

## 二、为什么需要对比实验

### 2.1 学术规范要求

在学术论文中，需要与现有方法进行公平对比，以证明所提方法的先进性。对比实验是论文评审的重要考量因素。

### 2.2 验证模型泛化能力

我们使用的是**比赛提供的数据集**，这是一个**没有人跑过的新数据集**。通过对比实验可以：

- 验证不同方法在该数据集上的表现
- 建立该数据集的基准 (Baseline)
- 证明我们方法的泛化能力

### 2.3 方法类型覆盖

需要对比的方法类型包括：

| 类型 | 代表方法 | 说明 |
|------|----------|------|
| **两阶段方法** | RoI Transformer, Oriented R-CNN | 先生成候选区域，再精细化 |
| **单阶段方法** | YOLOv8-OBB, S2A-Net | 端到端直接检测 |
| **Transformer方法** | Deformable DETR | 基于 Transformer 的检测器 |

---

## 三、实验数据集

### 3.1 数据集路径

```
C:\Users\win11\Desktop\5090_1_YOLO\sar_data_backup\split
```

### 3.2 数据集说明

- **来源**: 比赛官方提供
- **类型**: SAR 图像旋转目标检测
- **格式**: YOLO-OBB 格式 (8点坐标或 xywhr)
- **特点**: 全新数据集，无公开基准

### 3.3 数据集结构

```
split/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## 四、对比方法

### 4.1 已添加的方法

#### RoI Transformer (两阶段)

- **GitHub**: https://github.com/dingjiansw101/RoITransformer_DOTA
- **论文**: Learning RoI Transformer for Oriented Object Detection in Aerial Images (CVPR 2019)
- **框架**: MXNet
- **原理**:
  1. 第一阶段：生成水平边界框建议
  2. 第二阶段：学习旋转偏移量，将水平框转换为旋转框
- **特点**: 两阶段方法精度高，但速度较慢

### 4.2 待添加的方法

| 方法 | 类型 | GitHub | 状态 |
|------|------|--------|------|
| Oriented R-CNN | 两阶段 | https://github.com/jbwang1997/OBBDetection | 待添加 |
| S2A-Net | 单阶段 | https://github.com/csuhan/s2anet | 待添加 |
| Gliding Vertex | 单阶段 | https://github.com/SJTU-Thinklab-Det/Gliding_vertex | 待添加 |
| YOLOv8-OBB | 单阶段 | Ultralytics 官方 | 待添加 |
| PP-YOLOE-R | 单阶段 | PaddleDetection | 待添加 |

---

## 五、我们的方法

### 5.1 Full Innovation 模型

```
Full Innovation = RFR (Backbone) + CMLCA Plus (Neck) + MSWR (Head)
```

### 5.2 创新点

| 模块 | 全称 | 作用 |
|------|------|------|
| **RFR** | RotaryFeatureRefiner | 旋转特征细化，增强任意方向目标特征 |
| **CMLCA Plus** | Cross-scale Multi-level Context Attention | 跨尺度多级上下文注意力 |
| **MSWR** | Multi-Scale Shared-Weight Refiner | 多尺度共享权重检测头 |

### 5.3 模型参数

- **总参数量**: 4.20M
- **模型大小**: 16.04 MB (FP32) / 8.02 MB (FP16)
- **基准对比**: 相比 YOLO11n-obb (2.70M) 增加 56%

### 5.4 项目路径

```
C:\Users\win11\Desktop\neck_david _1
```

---

## 六、实验环境

### 6.1 本地环境 (RTX 3060)

- GPU: NVIDIA RTX 3060 (6GB)
- CUDA: 12.x
- Python: 3.x
- PyTorch: 2.x

### 6.2 服务器环境 (双卡 RTX 3090)

- GPU: 2x NVIDIA RTX 3090 (24GB each)
- 用于大规模训练和对比实验

---

## 七、评估指标

### 7.1 指标说明

| 指标 | 说明 | 测试方式 |
|------|------|----------|
| **mAP@0.5** | IoU=0.5 时的平均精度 | Ultralytics 自动计算 |
| **mAP@0.5:0.95** | IoU 从 0.5 到 0.95 的平均精度 | Ultralytics 自动计算 |
| **角度误差 (°)** | 预测角度与真实角度的差异 | 自定义脚本计算 |
| **参数量** | 模型参数数量 | `sum(p.numel() for p in model.parameters())` |
| **FPS** | 每秒推理帧数 | 自定义脚本计算 |

### 7.2 角度误差测试

**脚本路径**: `C:\Users\win11\Desktop\contrast_experiments\eval_angle_error.py`

**使用方法**:
```bash
python eval_angle_error.py --model runs/obb/best.pt --data dataset.yaml
```

**参数说明**:
- `--model`: 训练好的模型权重路径
- `--data`: 数据集配置文件 (yaml)
- `--conf`: 置信度阈值 (默认 0.25)
- `--iou`: IoU 匹配阈值 (默认 0.5)
- `--imgsz`: 图像尺寸 (默认 1024)

**输出指标**:
- 平均角度误差 (Mean)
- 中位数误差 (Median)
- 标准差 (Std)
- <5°, <10°, <15° 比例

### 7.3 FPS 测试

**待补充**: 需要编写 FPS 测试脚本

---

## 八、实验记录

### 8.1 实验表格模板

| 方法 | 类型 | mAP@0.5 | mAP@0.5:0.95 | 角度误差(°) | 参数量 | FPS |
|------|------|---------|--------------|-------------|--------|-----|
| RoI Transformer | 两阶段 | - | - | - | - | - |
| Oriented R-CNN | 两阶段 | - | - | - | - | - |
| YOLOv8-OBB | 单阶段 | - | - | - | - | - |
| **Ours (Full Innovation)** | 单阶段 | 93.6% | 57.2% | 待测 | 4.20M | 待测 |

### 8.2 实验进度

- [x] Full Innovation 模型训练 (SSDD 验证)
- [ ] Full Innovation 模型训练 (比赛数据集)
- [ ] RoI Transformer 环境配置
- [ ] RoI Transformer 训练
- [ ] 其他对比方法

---

## 九、参考文献

1. Ding J, Xue N, Long Y, et al. Learning RoI Transformer for Oriented Object Detection in Aerial Images[C]//CVPR 2019.

---

## 十、更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-03-04 | 创建对比实验文件夹，添加 RoI Transformer |
| 2026-03-04 | 添加评估指标说明，创建角度误差测试脚本 |
| 2026-03-05 | 添加 YOLO11-OBB 训练脚本 (Baseline) |
| 2026-03-05 | 添加 RoI Transformer (AerialDetection) 支持 |
| 2026-03-05 | 添加 YOLO-OBB 转 DOTA 格式转换脚本 |

---

## 十一、文件说明

```
contrast_experiments/
├── README.md                           # 项目说明
├── train_yolo11_obb.py                 # YOLO11-OBB 训练脚本 (Baseline)
├── train_roitransformer.py             # RoI Transformer 训练脚本
├── convert_yolo_to_dota.py             # 数据格式转换脚本
├── eval_angle_error.py                 # 角度误差评估脚本
├── configs/
│   └── roitransformer_sar.py           # RoI Transformer SAR配置
├── AerialDetection/                    # AerialDetection 框架 (需单独克隆)
└── RoITransformer_DOTA/                # 原版 MXNet (参考)
```
