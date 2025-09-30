# -*- coding: utf-8 -*-
"""
人脸表情识别模型训练脚本
使用YOLO进行人脸表情检测的训练
"""

import os
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path

class FaceExpressionTrainer:
    """人脸表情识别训练器"""
    
    def __init__(self, config_path: str = "face_expression_config.yaml"):
        self.config_path = config_path
        self.model = None
        self.results = None
        
    def load_config(self):
        """加载训练配置"""
        # 检查配置文件是否存在
        if not os.path.exists(self.config_path):
            # 尝试使用数据集目录中的配置文件
            dataset_config = "datasets/face_expression/data.yaml"
            if os.path.exists(dataset_config):
                self.config_path = dataset_config
                print(f"使用数据集配置文件: {self.config_path}")
            else:
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        print(f"配置已加载: {self.config_path}")
        return self.config
    
    def prepare_model(self, model_size: str = "n"):
        """
        准备YOLO模型
        model_size: 'n', 's', 'm', 'l', 'x' 分别对应不同大小的模型
        """
        # 选择预训练模型
        model_name = f"yolo11{model_size}.pt"
        
        # 检查是否有预训练模型文件
        if os.path.exists(model_name):
            print(f"使用本地预训练模型: {model_name}")
            self.model = YOLO(model_name)
        else:
            print(f"下载预训练模型: {model_name}")
            self.model = YOLO(model_name)
        
        print(f"模型准备完成: {model_name}")
        return self.model
    
    def train(self, 
              epochs: int = 100,
              imgsz: int = 640,
              batch: int = 16,
              lr0: float = 0.01,
              patience: int = 50,
              save_period: int = 10,
              device: str = None):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            imgsz: 输入图片尺寸
            batch: 批次大小
            lr0: 初始学习率
            patience: 早停耐心值
            save_period: 保存周期
            device: 设备 ('cpu', 'cuda', '0', '1', etc.)
        """
        if self.model is None:
            raise ValueError("请先调用 prepare_model() 准备模型")
        
        # 设置设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"开始训练，使用设备: {device}")
        print(f"训练参数: epochs={epochs}, imgsz={imgsz}, batch={batch}, lr0={lr0}")
        
        # 开始训练
        self.results = self.model.train(
            data=self.config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            patience=patience,
            save_period=save_period,
            device=device,
            workers=4,
            project="runs/detect",
            name="face_expression_train",
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            cos_lr=True,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            freeze=None,
            multi_scale=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            split='val',
            save_json=False,
            conf=None,
            iou=0.7,
            max_det=300,
            half=False,
            dnn=False,
            plots=True,
            source=None,
            show=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            show_labels=True,
            show_conf=True,
            vid_stride=1,
            line_width=None,
            visualize=False,
            augment=False,
            agnostic_nms=False,
            classes=None,
            retina_masks=False,
            show_boxes=True,
            format='torchscript',
            keras=False,
            optimize=False,
            int8=False,
            dynamic=False,
            simplify=False,
            opset=None,
            workspace=4,
            nms=False,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=2.0,
            nbs=64,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            auto_augment='randaugment',
            erasing=0.4
        )
        
        print("训练完成！")
        return self.results
    
    def validate(self, model_path: str = None):
        """
        验证模型性能
        """
        if model_path is None:
            # 使用最新训练的模型
            model_path = "runs/detect/face_expression_train/weights/best.pt"
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None
        
        # 加载训练好的模型
        model = YOLO(model_path)
        
        # 在验证集上评估
        results = model.val(
            data=self.config_path,
            split='val',
            imgsz=640,
            batch=16,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=False,
            device=None,
            workers=8,
            project="runs/val",
            name="face_expression_val",
            exist_ok=True,
            save_json=False,
            save_hybrid=False,
            plots=True
        )
        
        print("验证完成！")
        return results
    
    def test_single_image(self, image_path: str, model_path: str = None):
        """
        测试单张图片
        """
        if model_path is None:
            model_path = "runs/detect/face_expression_train/weights/best.pt"
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None
        
        # 加载模型
        model = YOLO(model_path)
        
        # 预测
        results = model(image_path)
        
        # 显示结果
        results[0].show()
        
        # 保存结果
        results[0].save("face_expression_result.jpg")
        print(f"结果已保存为: face_expression_result.jpg")
        
        return results
    
    def export_model(self, model_path: str = None, formats: list = ['onnx', 'torchscript']):
        """
        导出模型为不同格式
        """
        if model_path is None:
            model_path = "runs/detect/face_expression_train/weights/best.pt"
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None
        
        # 加载模型
        model = YOLO(model_path)
        
        # 导出模型
        for format_name in formats:
            try:
                exported_path = model.export(format=format_name)
                print(f"模型已导出为 {format_name} 格式: {exported_path}")
            except Exception as e:
                print(f"导出 {format_name} 格式失败: {e}")
        
        return True

def main():
    """主函数 - 演示训练流程"""
    # 创建训练器，优先使用数据集配置文件
    config_path = "datasets/face_expression/data.yaml"
    if not os.path.exists(config_path):
        config_path = "face_expression_config.yaml"
    
    trainer = FaceExpressionTrainer(config_path)
    
    # 加载配置
    trainer.load_config()
    
    # 准备模型
    trainer.prepare_model(model_size="n")  # 使用nano模型，训练更快
    
    # 检查数据集
    dataset_path = Path("datasets/face_expression")
    if not (dataset_path / "images" / "train").exists():
        print("数据集不存在，请先运行 face_expression_utils.py 创建示例数据集")
        return
    
    # 开始训练（使用较小的参数进行快速训练）
    print("开始训练人脸表情识别模型...")
    trainer.train(
        epochs=50,      # 减少训练轮数
        imgsz=640,      # 图片尺寸
        batch=8,        # 减少批次大小
        lr0=0.01,       # 学习率
        patience=20,    # 早停耐心值
        device='cpu'    # 使用CPU训练（如果有GPU可以改为'cuda'）
    )
    
    # 验证模型
    print("验证模型性能...")
    trainer.validate()
    
    # 测试单张图片
    print("测试模型...")
    test_image = "img/12.jpg"  # 使用现有图片测试
    if os.path.exists(test_image):
        trainer.test_single_image(test_image)
    
    print("训练流程完成！")

if __name__ == "__main__":
    main()
