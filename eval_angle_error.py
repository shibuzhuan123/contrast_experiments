#!/usr/bin/env python3
"""
角度误差评估脚本

计算旋转目标检测模型的角度预测误差
支持 YOLO-OBB 格式

使用方法:
    python eval_angle_error.py --model runs/obb/ssdd_full_innovation6/weights/best.pt --data ssdd_dataset.yaml
"""

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='评估旋转目标检测的角度误差')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径 (best.pt)')
    parser.add_argument('--data', type=str, required=True, help='数据集配置文件 (yaml)')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU 匹配阈值')
    parser.add_argument('--imgsz', type=int, default=1024, help='图像尺寸')
    return parser.parse_args()


def xywhr_to_corners(cx, cy, w, h, angle_deg):
    """
    将旋转框 (cx, cy, w, h, angle) 转换为四个角点

    Args:
        cx, cy: 中心点坐标
        w, h: 宽度和高度
        angle_deg: 角度 (度数, OpenCV 格式)

    Returns:
        corners: 四个角点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # 四个角相对于中心的偏移
    w_half, h_half = w / 2, h / 2
    corners = np.array([
        [-w_half, -h_half],
        [w_half, -h_half],
        [w_half, h_half],
        [-w_half, h_half]
    ])

    # 旋转
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = corners @ rotation_matrix.T

    # 平移到中心点
    rotated_corners[:, 0] += cx
    rotated_corners[:, 1] += cy

    return rotated_corners


def compute_rotated_iou(box1, box2):
    """
    计算两个旋转框的 IoU

    Args:
        box1, box2: (cx, cy, w, h, angle) 格式

    Returns:
        iou: IoU 值
    """
    cx1, cy1, w1, h1, a1 = box1
    cx2, cy2, w2, h2, a2 = box2

    corners1 = xywhr_to_corners(cx1, cy1, w1, h1, a1)
    corners2 = xywhr_to_corners(cx2, cy2, w2, h2, a2)

    # 使用 OpenCV 计算多边形面积
    rect1 = np.array(corners1, dtype=np.float32)
    rect2 = np.array(corners2, dtype=np.float32)

    # 计算交集
    ret, intersection = cv2.intersectConvexConvex(rect1, rect2)

    if ret > 0 and intersection is not None and len(intersection) > 0:
        intersection_area = cv2.contourArea(intersection)
    else:
        intersection_area = 0

    # 计算并集
    area1 = cv2.contourArea(rect1)
    area2 = cv2.contourArea(rect2)
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0

    return intersection_area / union_area


def angle_error_deg(pred_angle, gt_angle):
    """
    计算角度误差 (考虑周期性)

    角度范围 [0, 180)，0° 和 180° 实际相同

    Args:
        pred_angle: 预测角度 (度)
        gt_angle: 真实角度 (度)

    Returns:
        error: 角度误差 (度)
    """
    # 归一化到 [0, 180)
    pred_angle = pred_angle % 180
    gt_angle = gt_angle % 180

    # 计算差异
    diff = abs(pred_angle - gt_angle)

    # 考虑周期性: 10° 和 170° 差异是 20°，不是 160°
    error = min(diff, 180 - diff)

    return error


def load_gt_labels(label_path, img_w, img_h):
    """
    加载真实标签

    Args:
        label_path: 标签文件路径
        img_w, img_h: 图像宽高

    Returns:
        list of dict: [{'cx', 'cy', 'w', 'h', 'angle', 'class'}, ...]
    """
    gt_boxes = []

    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:  # class x1 y1 x2 y2 x3 y3 x4 y4
                cls_id = int(parts[0])
                # 8点坐标
                coords = [float(x) for x in parts[1:9]]
                x1, y1, x2, y2, x3, y3, x4, y4 = coords

                # 计算旋转框参数
                cx = (x1 + x2 + x3 + x4) / 4 * img_w
                cy = (y1 + y2 + y3 + y4) / 4 * img_h

                # 使用 OpenCV 拟合旋转框
                points = np.array([
                    [x1 * img_w, y1 * img_h],
                    [x2 * img_w, y2 * img_h],
                    [x3 * img_w, y3 * img_h],
                    [x4 * img_w, y4 * img_h]
                ], dtype=np.float32)

                rect = cv2.minAreaRect(points)
                (cx, cy), (w, h), angle = rect

                # 确保宽度 >= 高度，角度在 [0, 180) 范围
                if w < h:
                    w, h = h, w
                    angle = angle + 90

                if angle < 0:
                    angle += 180
                if angle >= 180:
                    angle -= 180

                gt_boxes.append({
                    'cx': cx, 'cy': cy, 'w': w, 'h': h,
                    'angle': angle, 'class': cls_id
                })

    return gt_boxes


def evaluate_angle_error(model_path, data_yaml, conf_thres=0.25, iou_thres=0.5, imgsz=1024):
    """
    评估模型的角度误差

    Args:
        model_path: 模型权重路径
        data_yaml: 数据集配置文件
        conf_thres: 置信度阈值
        iou_thres: IoU 匹配阈值
        imgsz: 图像尺寸
    """
    print('='*60)
    print('角度误差评估')
    print('='*60)
    print(f'模型: {model_path}')
    print(f'数据集: {data_yaml}')
    print(f'置信度阈值: {conf_thres}')
    print(f'IoU 匹配阈值: {iou_thres}')
    print('='*60)

    # 加载模型
    model = YOLO(model_path)

    # 获取验证集图像路径
    import yaml
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    val_path = data_cfg.get('val', data_cfg.get('valid', ''))
    if isinstance(val_path, str):
        # 获取图像目录
        if '/' in val_path or '\\' in val_path:
            img_dir = Path(val_path).parent / 'images' / Path(val_path).name
        else:
            img_dir = Path(val_path)

    # 如果是目录名，查找图像
    data_root = Path(data_yaml).parent
    val_img_dir = data_root / 'images' / 'val'
    val_label_dir = data_root / 'labels' / 'val'

    if not val_img_dir.exists():
        print(f'错误: 验证集图像目录不存在: {val_img_dir}')
        return

    # 收集所有角度误差
    all_angle_errors = []
    matched_count = 0
    total_gt = 0
    total_pred = 0

    # 遍历验证集图像
    img_files = list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png'))
    print(f'\n找到 {len(img_files)} 张验证图像')

    for img_file in img_files:
        # 读取图像获取尺寸
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # 加载真实标签
        label_file = val_label_dir / (img_file.stem + '.txt')
        gt_boxes = load_gt_labels(label_file, img_w, img_h)
        total_gt += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        # 模型推理
        results = model.predict(str(img_file), conf=conf_thres, iou=0.7, imgsz=imgsz, verbose=False)

        pred_boxes = []
        if len(results) > 0 and results[0].obb is not None:
            obb = results[0].obb
            if len(obb) > 0:
                # 获取预测框
                for i in range(len(obb)):
                    data = obb.data[i]
                    if hasattr(data, 'cpu'):
                        data = data.cpu().numpy()
                    else:
                        data = np.array(data)

                    cx, cy, w, h, angle = data[:5]
                    conf = data[5] if len(data) > 5 else 1.0
                    cls_id = int(data[6]) if len(data) > 6 else 0

                    pred_boxes.append({
                        'cx': cx, 'cy': cy, 'w': w, 'h': h,
                        'angle': angle * 180 / math.pi,  # 转换为度
                        'conf': conf, 'class': cls_id
                    })

        total_pred += len(pred_boxes)

        # 匹配预测框和真实框
        for gt in gt_boxes:
            best_iou = 0
            best_pred = None

            for pred in pred_boxes:
                # 计算旋转 IoU
                iou = compute_rotated_iou(
                    (gt['cx'], gt['cy'], gt['w'], gt['h'], gt['angle']),
                    (pred['cx'], pred['cy'], pred['w'], pred['h'], pred['angle'])
                )
                if iou > best_iou and iou >= iou_thres:
                    best_iou = iou
                    best_pred = pred

            if best_pred is not None:
                # 计算角度误差
                error = angle_error_deg(best_pred['angle'], gt['angle'])
                all_angle_errors.append(error)
                matched_count += 1

    # 统计结果
    print('\n' + '='*60)
    print('评估结果')
    print('='*60)
    print(f'总 GT 框数: {total_gt}')
    print(f'总预测框数: {total_pred}')
    print(f'匹配框数: {matched_count}')
    print(f'召回率: {matched_count/total_gt*100:.1f}%' if total_gt > 0 else 'N/A')

    if len(all_angle_errors) > 0:
        errors = np.array(all_angle_errors)
        print(f'\n角度误差统计:')
        print(f'  平均误差 (Mean): {errors.mean():.2f}°')
        print(f'  中位数 (Median): {np.median(errors):.2f}°')
        print(f'  标准差 (Std): {errors.std():.2f}°')
        print(f'  最大误差 (Max): {errors.max():.2f}°')
        print(f'  最小误差 (Min): {errors.min():.2f}°')
        print(f'  <5° 比例: {(errors < 5).sum() / len(errors) * 100:.1f}%')
        print(f'  <10° 比例: {(errors < 10).sum() / len(errors) * 100:.1f}%')
        print(f'  <15° 比例: {(errors < 15).sum() / len(errors) * 100:.1f}%')
    else:
        print('没有匹配的框，无法计算角度误差')

    print('='*60)

    return all_angle_errors


if __name__ == '__main__':
    args = parse_args()
    evaluate_angle_error(
        model_path=args.model,
        data_yaml=args.data,
        conf_thres=args.conf,
        iou_thres=args.iou,
        imgsz=args.imgsz
    )
