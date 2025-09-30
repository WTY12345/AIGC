# -*- coding: utf-8 -*-
"""
人脸表情实时检测脚本
使用训练好的YOLO模型进行实时人脸表情检测
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import argparse

class FaceExpressionDetector:
    """人脸表情检测器"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.classes = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']
        self.colors = [
            (0, 255, 0),    # happy - 绿色
            (255, 0, 0),    # sad - 蓝色
            (0, 0, 255),    # angry - 红色
            (255, 255, 0),  # surprised - 青色
            (128, 0, 128),  # fearful - 紫色
            (0, 255, 255),  # disgusted - 黄色
            (128, 128, 128) # neutral - 灰色
        ]
        
    def load_model(self, model_path: str = None):
        """加载训练好的模型"""
        if model_path is None:
            # 尝试找到最新的训练模型
            possible_paths = [
                "runs/detect/face_expression_train/weights/best.pt",
                "yolo11n.pt",  # 使用预训练模型作为备选
                "3times.pt"    # 使用现有模型作为备选
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError("未找到可用的模型文件")
        
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.model_path = model_path
        print("模型加载完成！")
        
    def detect_expression(self, frame: np.ndarray, conf_threshold: float = 0.5) -> tuple:
        """
        检测单帧图片中的表情
        
        Args:
            frame: 输入图片帧
            conf_threshold: 置信度阈值
            
        Returns:
            (annotated_frame, detections): 标注后的图片和检测结果
        """
        if self.model is None:
            raise ValueError("请先加载模型")
        
        # 进行预测
        results = self.model(frame, conf=conf_threshold)
        
        # 获取检测结果
        detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
                confidences = result.boxes.conf.cpu().numpy()  # 置信度
                class_ids = result.boxes.cls.cpu().numpy().astype(int)  # 类别ID
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = map(int, box)
                    expression = self.classes[class_id]
                    color = self.colors[class_id]
                    
                    # 绘制边界框
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签
                    label = f"{expression}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 记录检测结果
                    detections.append({
                        'expression': expression,
                        'confidence': float(conf),
                        'bbox': [x1, y1, x2, y2],
                        'class_id': int(class_id)
                    })
        
        return annotated_frame, detections
    
    def detect_video(self, video_path: str = 0, output_path: str = None, 
                    conf_threshold: float = 0.5, show_fps: bool = True):
        """
        检测视频文件或摄像头中的表情
        
        Args:
            video_path: 视频文件路径或摄像头索引（0为默认摄像头）
            output_path: 输出视频路径（可选）
            conf_threshold: 置信度阈值
            show_fps: 是否显示FPS
        """
        # 打开视频源
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频源: {video_path}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频源: {video_path}")
        print(f"分辨率: {width}x{height}, FPS: {fps}")
        
        # 设置输出视频（如果需要）
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"输出视频: {output_path}")
        
        # 性能统计
        frame_count = 0
        start_time = time.time()
        
        print("开始检测，按 'q' 键退出...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测表情
                annotated_frame, detections = self.detect_expression(frame, conf_threshold)
                
                # 显示FPS
                if show_fps:
                    frame_count += 1
                    if frame_count % 30 == 0:  # 每30帧计算一次FPS
                        elapsed_time = time.time() - start_time
                        current_fps = frame_count / elapsed_time
                        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示检测统计
                if detections:
                    expression_counts = {}
                    for det in detections:
                        expr = det['expression']
                        expression_counts[expr] = expression_counts.get(expr, 0) + 1
                    
                    stats_text = "检测到: " + ", ".join([f"{expr}({count})" 
                                                       for expr, count in expression_counts.items()])
                    cv2.putText(annotated_frame, stats_text, (10, height - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 显示结果
                cv2.imshow('人脸表情检测', annotated_frame)
                
                # 保存视频
                if out:
                    out.write(annotated_frame)
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("检测被用户中断")
        
        finally:
            # 清理资源
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # 显示性能统计
            if frame_count > 0:
                total_time = time.time() - start_time
                avg_fps = frame_count / total_time
                print(f"检测完成！总帧数: {frame_count}, 平均FPS: {avg_fps:.2f}")
    
    def detect_image(self, image_path: str, output_path: str = None, 
                    conf_threshold: float = 0.5):
        """
        检测单张图片中的表情
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径（可选）
            conf_threshold: 置信度阈值
        """
        # 读取图片
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图片: {image_path}")
            return
        
        print(f"检测图片: {image_path}")
        
        # 检测表情
        annotated_frame, detections = self.detect_expression(frame, conf_threshold)
        
        # 显示检测结果
        if detections:
            print(f"检测到 {len(detections)} 个表情:")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['expression']} (置信度: {det['confidence']:.3f})")
        else:
            print("未检测到任何表情")
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            print(f"结果已保存到: {output_path}")
        else:
            # 显示结果
            cv2.imshow('人脸表情检测', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='人脸表情检测')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--source', type=str, default='0', help='视频源（文件路径或摄像头索引）')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--mode', type=str, choices=['video', 'image'], default='video', 
                       help='检测模式：video或image')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = FaceExpressionDetector()
    
    try:
        # 加载模型
        detector.load_model(args.model)
        
        if args.mode == 'video':
            # 视频检测模式
            source = int(args.source) if args.source.isdigit() else args.source
            detector.detect_video(
                video_path=source,
                output_path=args.output,
                conf_threshold=args.conf
            )
        else:
            # 图片检测模式
            detector.detect_image(
                image_path=args.source,
                output_path=args.output,
                conf_threshold=args.conf
            )
            
    except Exception as e:
        print(f"错误: {e}")
        print("请确保模型文件存在，或先运行训练脚本")

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认设置
    import sys
    if len(sys.argv) == 1:
        # 默认使用摄像头进行实时检测
        detector = FaceExpressionDetector()
        try:
            detector.load_model()
            print("开始实时表情检测（按 'q' 键退出）...")
            detector.detect_video(0)  # 使用默认摄像头
        except Exception as e:
            print(f"错误: {e}")
            print("请确保有可用的摄像头或提供正确的模型路径")
    else:
        main()
