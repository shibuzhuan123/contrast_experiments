#!/usr/bin/env python3
"""
YOLO11-OBB 训练脚本
用于 SAR 旋转目标检测对比实验 - Baseline
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_yolo11_obb(
    data_yaml: str,
    model_size: str = "n",
    epochs: int = 100,
    batch: int = 8,
    imgsz: int = 640,
    device: str = "0",
    project: str = "runs/obb",
    name: str = "yolo11_obb_baseline",
    resume: str = None
):
    """
    训练 YOLO11-OBB 模型 (Baseline)

    Args:
        data_yaml: 数据集配置文件路径
        model_size: 模型大小 n/s/m/l/x
        epochs: 训练轮数
        batch: batch size
        imgsz: 图像尺寸
        device: GPU设备
        project: 保存目录
        name: 实验名称
        resume: 恢复训练的模型路径
    """
    print("=" * 60)
    print("YOLO11-OBB 训练 (Baseline)")
    print("=" * 60)
    print(f"数据集: {data_yaml}")
    print(f"模型: YOLO11{model_size}-obb")
    print(f"轮数: {epochs}")
    print(f"Batch Size: {batch}")
    print(f"图像尺寸: {imgsz}")
    print(f"设备: {device}")
    print("=" * 60)

    # 加载模型
    if resume:
        print(f"恢复训练: {resume}")
        model = YOLO(resume)
    else:
        model_name = f"yolo11{model_size}-obb.pt"
        print(f"加载预训练模型: {model_name}")
        model = YOLO(model_name)

    # 开始训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        patience=20,  # 早停
        save=True,
        plots=True,
        val=True,
        # OBB 特定参数
        degrees=45.0,  # 旋转增强
        flipud=0.5,    # 上下翻转
        fliplr=0.5,    # 左右翻转
        mosaic=0.5,    # mosaic增强
    )

    print("\n训练完成!")
    print(f"最佳模型: {project}/{name}/weights/best.pt")

    return results


def main():
    parser = argparse.ArgumentParser(description="YOLO11-OBB 训练脚本 (Baseline)")
    parser.add_argument("--data", type=str, required=True, help="数据集配置文件 (yaml)")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                        help="模型大小 (n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--device", type=str, default="0", help="GPU设备")
    parser.add_argument("--project", type=str, default="runs/obb", help="保存目录")
    parser.add_argument("--name", type=str, default="yolo11_obb_baseline", help="实验名称")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的模型路径")

    args = parser.parse_args()

    train_yolo11_obb(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
