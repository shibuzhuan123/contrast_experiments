#!/usr/bin/env python3
"""
RoI Transformer (AerialDetection) 训练脚本

使用方法:
    # 单GPU训练
    python train_roitransformer.py --config configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_sar.py

    # 多GPU训练
    bash tools/dist_train.sh configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_sar.py 2
"""

import argparse
import os
import sys
from pathlib import Path

# 添加 AerialDetection 到路径
AERIAL_DET_PATH = Path(__file__).parent / 'AerialDetection'
sys.path.insert(0, str(AERIAL_DET_PATH))


def prepare_data(data_root: str):
    """
    准备数据：生成 COCO 格式的标注文件

    Args:
        data_root: DOTA 格式数据集根目录
    """
    import json
    from PIL import Image
    import numpy as np

    print("准备数据标注文件...")

    for split in ['train', 'val']:
        img_dir = Path(data_root) / split / 'images'
        label_dir = Path(data_root) / split / 'labelTxt'

        if not img_dir.exists():
            print(f"跳过 {split}，目录不存在")
            continue

        # COCO 格式结构
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 0, 'name': 'ship'},
                {'id': 1, 'name': 'aircraft'},
                {'id': 2, 'name': 'car'},
                {'id': 3, 'name': 'tank'},
                {'id': 4, 'name': 'bridge'},
                {'id': 5, 'name': 'harbor'},
            ]
        }

        ann_id = 1
        img_id = 1

        img_files = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))

        for img_file in img_files:
            label_file = label_dir / (img_file.stem + '.txt')

            # 获取图像尺寸
            with Image.open(img_file) as img:
                width, height = img.size

            coco_data['images'].append({
                'id': img_id,
                'file_name': img_file.name,
                'width': width,
                'height': height
            })

            # 解析标签
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 9:
                            coords = [float(x) for x in parts[:8]]
                            class_name = parts[8]

                            # 找到类别ID
                            cat_id = None
                            for cat in coco_data['categories']:
                                if cat['name'] == class_name:
                                    cat_id = cat['id']
                                    break

                            if cat_id is None:
                                continue

                            # 计算边界框 (用于COCO格式)
                            xs = coords[0::2]
                            ys = coords[1::2]
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                            # 计算分割区域
                            segmentation = [coords]

                            coco_data['annotations'].append({
                                'id': ann_id,
                                'image_id': img_id,
                                'category_id': cat_id,
                                'bbox': bbox,
                                'area': bbox[2] * bbox[3],
                                'segmentation': segmentation,
                                'iscrowd': 0
                            })
                            ann_id += 1

            img_id += 1

        # 保存COCO格式标注
        output_file = Path(data_root) / split / f'DOTA_{split}.json'
        with open(output_file, 'w') as f:
            json.dump(coco_data, f)

        print(f"  {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")


def main():
    parser = argparse.ArgumentParser(description="RoI Transformer 训练脚本")
    parser.add_argument('--config', type=str,
                        default='AerialDetection/configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_sar.py',
                        help='配置文件路径')
    parser.add_argument('--data-root', type=str, default='data/dota_sar',
                        help='DOTA格式数据集路径')
    parser.add_argument('--prepare-data', action='store_true',
                        help='只准备数据，不训练')
    parser.add_argument('--gpus', type=int, default=1, help='GPU数量')

    args = parser.parse_args()

    if args.prepare_data:
        prepare_data(args.data_root)
        return

    # 检查数据是否准备好
    train_json = Path(args.data_root) / 'train' / 'DOTA_train.json'
    if not train_json.exists():
        print("数据标注文件不存在，正在生成...")
        prepare_data(args.data_root)

    # 开始训练
    print("=" * 60)
    print("RoI Transformer 训练")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"数据目录: {args.data_root}")
    print(f"GPU数量: {args.gpus}")
    print("=" * 60)

    # 使用 mmdet 的训练工具
    from mmdet.apis import set_random_seed, train_detector
    from mmdet.models import build_detector
    from mmdet.datasets import build_dataset
    from mmcv import Config

    # 加载配置
    cfg = Config.fromfile(args.config)

    # 设置随机种子
    set_random_seed(0, deterministic=True)

    # 构建模型
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]

    # 开始训练
    train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == '__main__':
    main()
