#!/usr/bin/env python3
"""
YOLO标注文件验证工具
检查标注格式的正确性和完整性
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path


class AnnotationValidator:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.errors = []
        self.warnings = []

    def validate_coordinates(self, coords):
        """验证坐标值是否在有效范围内"""
        if len(coords) != 5:
            return False, "坐标数量不正确"

        class_id, x, y, w, h = coords

        # 检查数据类型
        try:
            class_id = int(class_id)
            x, y, w, h = float(x), float(y), float(w), float(h)
        except ValueError:
            return False, "坐标格式错误"

        # 检查范围
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            return False, f"坐标超出范围: x={x}, y={y}, w={w}, h={h}"

        return True, "坐标有效"

    def validate_file(self, label_path):
        """验证单个标注文件"""
        if not os.path.exists(label_path):
            self.errors.append(f"标注文件不存在: {label_path}")
            return

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            if not lines:
                self.warnings.append(f"空标注文件: {label_path}")
                return

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                coords = line.split()
                is_valid, message = self.validate_coordinates(coords)

                if not is_valid:
                    self.errors.append(f"{label_path}:{line_num} - {message}")

        except Exception as e:
            self.errors.append(f"读取文件错误 {label_path}: {str(e)}")

    def check_image_label_pairs(self):
        """检查图像和标注文件的配对"""
        images_dir = self.dataset_path / "images" / "train2017"
        labels_dir = self.dataset_path / "labels" / "train2017"

        if not images_dir.exists() or not labels_dir.exists():
            self.errors.append("图像或标注目录不存在")
            return

        # 获取所有图像文件
        image_files = set(f.stem for f in images_dir.glob("*.jpg"))
        label_files = set(f.stem for f in labels_dir.glob("*.txt"))

        # 检查缺失的标注文件
        missing_labels = image_files - label_files
        if missing_labels:
            self.warnings.append(f"缺失标注文件: {len(missing_labels)}个")
            for missing in list(missing_labels)[:5]:  # 只显示前5个
                self.warnings.append(f"  - {missing}.txt")

        # 检查多余的标注文件
        extra_labels = label_files - image_files
        if extra_labels:
            self.warnings.append(f"多余标注文件: {len(extra_labels)}个")
            for extra in list(extra_labels)[:5]:  # 只显示前5个
                self.warnings.append(f"  - {extra}.txt")

    def visualize_annotations(self, image_path, label_path, output_path=None):
        """可视化标注结果"""
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            return None

        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]

        # 读取标注
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            coords = line.split()
            if len(coords) != 5:
                continue

            class_id, x, y, width, height = coords
            class_id = int(class_id)
            x, y, width, height = float(x), float(y), float(width), float(height)

            # 转换为像素坐标
            center_x = int(x * w)
            center_y = int(y * h)
            box_w = int(width * w)
            box_h = int(height * h)

            # 计算边界框坐标
            x1 = center_x - box_w // 2
            y1 = center_y - box_h // 2
            x2 = center_x + box_w // 2
            y2 = center_y + box_h // 2

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Class {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if output_path:
            cv2.imwrite(output_path, img)

        return img

    def run_validation(self):
        """运行完整验证"""
        print("开始验证YOLO标注数据...")

        # 验证所有标注文件
        labels_dir = self.dataset_path / "labels" / "train2017"
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            print(f"找到 {len(label_files)} 个标注文件")

            for label_file in label_files:
                self.validate_file(label_file)

        # 检查图像标注配对
        self.check_image_label_pairs()

        # 输出结果
        print(f"\n验证完成!")
        print(f"错误: {len(self.errors)} 个")
        print(f"警告: {len(self.warnings)} 个")

        if self.errors:
            print("\n错误列表:")
            for error in self.errors[:10]:  # 只显示前10个错误
                print(f"  ❌ {error}")

        if self.warnings:
            print("\n警告列表:")
            for warning in self.warnings[:10]:  # 只显示前10个警告
                print(f"  ⚠️  {warning}")

        return len(self.errors) == 0


def main():
    """主函数"""
    dataset_path = "D://AIGC/yolo/datasets/coco128"
    validator = AnnotationValidator(dataset_path)

    # 运行验证
    is_valid = validator.run_validation()

    if is_valid:
        print("\n✅ 标注数据验证通过!")
    else:
        print("\n❌ 标注数据存在问题，请检查并修复!")

    # 可视化示例（可选）
    print("\n生成可视化示例...")
    images_dir = Path(dataset_path) / "images" / "train2017"
    labels_dir = Path(dataset_path) / "labels" / "train2017"

    if images_dir.exists() and labels_dir.exists():
        # 找到第一个图像和对应的标注文件
        image_files = list(images_dir.glob("*.jpg"))
        if image_files:
            sample_image = image_files[0]
            sample_label = labels_dir / f"{sample_image.stem}.txt"

            if sample_label.exists():
                output_path = "annotation_sample.jpg"
                validator.visualize_annotations(sample_image, sample_label, output_path)
                print(f"可视化示例已保存到: {output_path}")


if __name__ == "__main__":
    main()

