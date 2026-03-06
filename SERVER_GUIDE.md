# 服务器部署指南

## 一、环境要求

- Python 3.8+
- PyTorch 1.8+ with CUDA
- GPU: 建议 24GB+ 显存 (RTX 3090/4090)

---

## 二、快速开始

### 2.1 克隆项目

```bash
# 克隆主项目
git clone https://github.com/shibuzhuan123/contrast_experiments.git
cd contrast_experiments
```

### 2.2 创建虚拟环境 (推荐)

```bash
conda create -n sar_detection python=3.9 -y
conda activate sar_detection
```

### 2.3 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy tqdm

# AerialDetection 依赖 (用于 RoI Transformer)
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch1.13/index.html
# 如果上面安装失败，尝试:
# pip install mmcv==2.1.0 mmdet==3.2.0 mmengine
```

### 2.4 安装 AerialDetection

```bash
# 克隆 AerialDetection
git clone https://github.com/dingjiansw101/AerialDetection.git

# 编译安装
cd AerialDetection
pip install -r requirements.txt
pip install -e .
cd ..
```

---

## 三、数据准备

### 3.1 数据集位置

比赛数据集路径: `/path/to/split`

```
split/
├── images/
│   ├── train/  (71265 张)
│   └── val/    (17817 张)
└── labels/
    ├── train/
    └── val/
```

### 3.2 转换数据格式 (用于 RoI Transformer)

```bash
# YOLO-OBB 转 DOTA 格式
python convert_yolo_to_dota.py \
    --src /path/to/split \
    --dst data/dota_sar
```

转换后结构:
```
data/dota_sar/
├── train/
│   ├── images/
│   └── labelTxt/
└── val/
    ├── images/
    └── labelTxt/
```

---

## 四、训练模型

### 4.1 YOLO11-OBB (Baseline)

```bash
# YOLO11n-OBB (最小模型，快速验证)
python train_yolo11_obb.py \
    --data /path/to/split/data.yaml \
    --model n \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0,1 \
    --name yolo11n_obb_baseline

# YOLO11s-OBB (推荐)
python train_yolo11_obb.py \
    --data /path/to/split/data.yaml \
    --model s \
    --epochs 100 \
    --batch 8 \
    --imgsz 640 \
    --device 0,1 \
    --name yolo11s_obb_baseline
```

### 4.2 RoI Transformer (AerialDetection)

```bash
# 先准备 COCO 格式标注
python train_roitransformer.py --prepare-data --data-root data/dota_sar

# 单 GPU 训练
python train_roitransformer.py \
    --config configs/roitransformer_sar.py \
    --data-root data/dota_sar \
    --gpus 1

# 多 GPU 训练 (推荐)
cd AerialDetection
bash tools/dist_train.sh ../configs/roitransformer_sar.py 2
```

---

## 五、评估模型

### 5.1 YOLO11-OBB 评估

```bash
# mAP 评估 (自动计算)
yolo obb val model=runs/obb/yolo11n_obb_baseline/weights/best.pt \
    data=/path/to/split/data.yaml

# 角度误差评估
python eval_angle_error.py \
    --model runs/obb/yolo11n_obb_baseline/weights/best.pt \
    --data /path/to/split/data.yaml
```

### 5.2 RoI Transformer 评估

```bash
cd AerialDetection
python tools/test.py \
    ../configs/roitransformer_sar.py \
    work_dirs/roitransformer_sar/epoch_12.pth \
    --eval mAP
```

---

## 六、实验记录模板

| 方法 | 类型 | mAP@0.5 | mAP@0.5:0.95 | 角度误差(°) | 参数量 | FPS |
|------|------|---------|--------------|-------------|--------|-----|
| YOLO11n-OBB | 单阶段 | - | - | - | 2.7M | - |
| YOLO11s-OBB | 单阶段 | - | - | - | - | - |
| RoI Transformer | 两阶段 | - | - | - | - | - |
| **Ours (Full Innovation)** | 单阶段 | 93.6%* | 57.2%* | - | 4.20M | - |

*注: SSDD 数据集结果，比赛数据集待测

---

## 七、常见问题

### Q1: mmcv-full 安装失败

```bash
# 尝试使用预编译版本
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# 或者使用 mmcv 2.x
pip install mmcv mmdet mmengine
```

### Q2: CUDA 内存不足

```bash
# 减小 batch size
python train_yolo11_obb.py --batch 4 ...

# 或使用更小的图像尺寸
python train_yolo11_obb.py --imgsz 512 ...
```

### Q3: 数据转换太慢

数据转换涉及读取大量图像，可以在数据所在服务器上执行，或使用多进程版本。

---

## 八、文件结构

```
contrast_experiments/
├── train_yolo11_obb.py         # YOLO11-OBB 训练
├── train_roitransformer.py     # RoI Transformer 训练
├── convert_yolo_to_dota.py     # 数据格式转换
├── eval_angle_error.py         # 角度误差评估
├── configs/
│   └── roitransformer_sar.py   # RoI Transformer 配置
├── AerialDetection/            # AerialDetection 框架
├── data/                       # 数据目录
│   └── dota_sar/               # DOTA 格式数据
└── runs/                       # 训练输出
    └── obb/                    # YOLO 输出
```

---

## 九、联系方式

项目地址: https://github.com/shibuzhuan123/contrast_experiments
