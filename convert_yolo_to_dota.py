#!/usr/bin/env python3
"""
YOLO-OBB 格式转 DOTA 格式脚本

YOLO-OBB 格式: class x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标)
DOTA 格式: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult

使用方法:
    python convert_yolo_to_dota.py --src /path/to/yolo_dataset --dst /path/to/dota_dataset
"""

import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm


# 类别映射 - 根据你的数据集修改
CLASS_NAMES = {
    0: 'ship',
    1: 'aircraft',
    2: 'car',
    3: 'tank',
    4: 'bridge',
    5: 'harbor'
}


def convert_label_file(yolo_label_path: Path, dota_label_path: Path, img_w: int, img_h: int):
    """
    转换单个标签文件

    Args:
        yolo_label_path: YOLO格式标签文件路径
        dota_label_path: DOTA格式标签文件保存路径
        img_w: 图像宽度
        img_h: 图像高度
    """
    with open(yolo_label_path, 'r') as f:
        lines = f.readlines()

    dota_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue

        cls_id = int(parts[0])
        # 归一化坐标转绝对坐标
        x1 = float(parts[1]) * img_w
        y1 = float(parts[2]) * img_h
        x2 = float(parts[3]) * img_w
        y2 = float(parts[4]) * img_h
        x3 = float(parts[5]) * img_w
        y3 = float(parts[6]) * img_h
        x4 = float(parts[7]) * img_w
        y4 = float(parts[8]) * img_h

        class_name = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
        difficult = 0  # 默认不困难

        # DOTA格式: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
        dota_line = f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x3:.2f} {y3:.2f} {x4:.2f} {y4:.2f} {class_name} {difficult}\n"
        dota_lines.append(dota_line)

    with open(dota_label_path, 'w') as f:
        f.writelines(dota_lines)


def convert_yolo_to_dota(src_dir: str, dst_dir: str):
    """
    转换整个数据集

    Args:
        src_dir: YOLO格式数据集根目录
        dst_dir: DOTA格式数据集保存目录
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    print("=" * 60)
    print("YOLO-OBB 转 DOTA 格式")
    print("=" * 60)
    print(f"源目录: {src_dir}")
    print(f"目标目录: {dst_dir}")
    print("=" * 60)

    # 创建目录结构
    for split in ['train', 'val']:
        (dst_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dst_path / split / 'labelTxt').mkdir(parents=True, exist_ok=True)

    # 转换训练集和验证集
    for split in ['train', 'val']:
        print(f"\n转换 {split} 集...")

        src_img_dir = src_path / 'images' / split
        src_label_dir = src_path / 'labels' / split
        dst_img_dir = dst_path / split / 'images'
        dst_label_dir = dst_path / split / 'labelTxt'

        if not src_img_dir.exists():
            print(f"  跳过 {split}，目录不存在")
            continue

        # 获取所有图像文件
        img_files = list(src_img_dir.glob('*.png')) + list(src_img_dir.glob('*.jpg'))

        for img_file in tqdm(img_files, desc=f"  Converting {split}"):
            # 复制图像
            dst_img = dst_img_dir / img_file.name
            if not dst_img.exists():
                shutil.copy2(img_file, dst_img)

            # 转换标签
            label_file = src_label_dir / (img_file.stem + '.txt')
            if label_file.exists():
                dst_label = dst_label_dir / (img_file.stem + '.txt')

                # 获取图像尺寸
                import cv2
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                    convert_label_file(label_file, dst_label, w, h)

    print("\n转换完成!")
    print(f"DOTA 数据集保存在: {dst_dir}")

    # 打印目录结构
    print("\n目录结构:")
    for split in ['train', 'val']:
        img_count = len(list((dst_path / split / 'images').glob('*.*')))
        label_count = len(list((dst_path / split / 'labelTxt').glob('*.txt')))
        print(f"  {split}/: {img_count} images, {label_count} labels")


def main():
    parser = argparse.ArgumentParser(description="YOLO-OBB 转 DOTA 格式")
    parser.add_argument('--src', type=str, required=True, help='YOLO格式数据集目录')
    parser.add_argument('--dst', type=str, required=True, help='DOTA格式保存目录')

    args = parser.parse_args()
    convert_yolo_to_dota(args.src, args.dst)


if __name__ == '__main__':
    main()
