# -*- coding: utf-8 -*-
"""
人脸表情识别工具函数
包含数据预处理、标注转换、数据增强等功能
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
import shutil
from typing import List, Tuple, Dict
import random

class FaceExpressionUtils:
    """人脸表情识别工具类"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.classes = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def create_sample_dataset(self):
        """创建示例数据集结构"""
        print("创建示例数据集...")
        
        # 创建示例图片和标签
        sample_data = [
            ("happy_sample.jpg", "happy", [0.5, 0.5, 0.3, 0.4]),  # x_center, y_center, width, height
            ("sad_sample.jpg", "sad", [0.5, 0.5, 0.3, 0.4]),
            ("neutral_sample.jpg", "neutral", [0.5, 0.5, 0.3, 0.4]),
        ]
        
        for img_name, expression, bbox in sample_data:
            # 创建示例图片（白色背景）
            img = np.ones((640, 640, 3), dtype=np.uint8) * 255
            cv2.putText(img, f"Sample {expression}", (200, 320), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            
            # 保存图片
            img_path = self.dataset_path / "images" / "train" / img_name
            cv2.imwrite(str(img_path), img)
            
            # 创建对应的标签文件
            label_path = self.dataset_path / "labels" / "train" / f"{Path(img_name).stem}.txt"
            class_id = self.class_to_id[expression]
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
        
        print(f"示例数据集已创建在: {self.dataset_path}")
    
    def convert_bbox_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        将边界框坐标转换为YOLO格式
        bbox: [x_min, y_min, x_max, y_max]
        返回: [x_center, y_center, width, height] (归一化坐标)
        """
        x_min, y_min, x_max, y_max = bbox
        
        # 计算中心点和宽高
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        # 归一化
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]
    
    def convert_yolo_to_bbox(self, yolo_bbox: List[float], img_width: int, img_height: int) -> List[int]:
        """
        将YOLO格式边界框转换为像素坐标
        yolo_bbox: [x_center, y_center, width, height] (归一化坐标)
        返回: [x_min, y_min, x_max, y_max] (像素坐标)
        """
        x_center, y_center, width, height = yolo_bbox
        
        # 反归一化
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # 计算边界框坐标
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        
        return [x_min, y_min, x_max, y_max]
    
    def visualize_annotations(self, img_path: str, label_path: str, output_path: str = None):
        """
        可视化标注结果
        """
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return
        
        img_height, img_width = img.shape[:2]
        
        # 读取标签
        if not os.path.exists(label_path):
            print(f"标签文件不存在: {label_path}")
            return
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 绘制边界框
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            yolo_bbox = [float(x) for x in parts[1:5]]
            
            # 转换为像素坐标
            bbox = self.convert_yolo_to_bbox(yolo_bbox, img_width, img_height)
            x_min, y_min, x_max, y_max = bbox
            
            # 绘制边界框
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # 添加类别标签
            class_name = self.classes[class_id]
            cv2.putText(img, class_name, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"可视化结果已保存到: {output_path}")
        else:
            cv2.imshow("标注可视化", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """
        分割数据集为训练集、验证集和测试集
        """
        # 获取所有图片文件
        all_images = list(self.dataset_path.glob("images/train/*.jpg")) + \
                    list(self.dataset_path.glob("images/train/*.png"))
        
        # 随机打乱
        random.shuffle(all_images)
        
        # 计算分割点
        train_count = int(len(all_images) * train_ratio)
        val_count = int(len(all_images) * val_ratio)
        
        # 分割文件
        train_files = all_images[:train_count]
        val_files = all_images[train_count:train_count + val_count]
        test_files = all_images[train_count + val_count:]
        
        # 移动文件到对应目录
        for file_list, split_name in [(train_files, "train"), (val_files, "val"), (test_files, "test")]:
            for img_path in file_list:
                # 移动图片
                new_img_path = self.dataset_path / "images" / split_name / img_path.name
                shutil.move(str(img_path), str(new_img_path))
                
                # 移动对应的标签文件
                label_name = img_path.stem + ".txt"
                old_label_path = self.dataset_path / "labels" / "train" / label_name
                new_label_path = self.dataset_path / "labels" / split_name / label_name
                
                if old_label_path.exists():
                    shutil.move(str(old_label_path), str(new_label_path))
        
        print(f"数据集分割完成:")
        print(f"训练集: {len(train_files)} 张图片")
        print(f"验证集: {len(val_files)} 张图片")
        print(f"测试集: {len(test_files)} 张图片")
    
    def create_data_yaml(self, output_path: str = None):
        """
        创建数据集配置文件
        """
        if output_path is None:
            output_path = self.dataset_path / "data.yaml"
        
        yaml_content = f"""# 人脸表情识别数据集配置
path: {self.dataset_path.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(self.classes)}
names: {self.classes}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"数据集配置文件已创建: {output_path}")

def main():
    """主函数 - 演示工具使用"""
    # 创建工具实例
    utils = FaceExpressionUtils("datasets/face_expression")
    
    # 创建示例数据集
    utils.create_sample_dataset()
    
    # 创建数据集配置文件
    utils.create_data_yaml()
    
    print("人脸表情识别工具初始化完成！")

if __name__ == "__main__":
    main()
