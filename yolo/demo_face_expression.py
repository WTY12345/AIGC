# -*- coding: utf-8 -*-
"""
人脸表情识别演示脚本
展示如何使用YOLO进行人脸表情检测
"""

import cv2
import os
from ultralytics import YOLO
import numpy as np

def demo_with_existing_model():
    """使用现有模型进行演示"""
    print("=== 人脸表情识别演示 ===")
    
    # 使用现有的YOLO模型（虽然它不是专门训练的表情识别模型）
    model_path = "yolo11n.pt"
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("正在下载预训练模型...")
    
    # 加载模型
    model = YOLO(model_path)
    print(f"已加载模型: {model_path}")
    
    # 表情类别映射（基于COCO数据集的person类别进行演示）
    expression_classes = {
        0: "person",  # 这里我们用人脸检测来演示
    }
    
    # 检测图片
    test_images = ["img/12.jpg", "img/2.jpg", "img/bus.jpg"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n检测图片: {img_path}")
            
            # 进行检测
            results = model(img_path, conf=0.5)
            
            # 显示结果
            for result in results:
                if result.boxes is not None:
                    print(f"检测到 {len(result.boxes)} 个目标")
                    
                    for i, box in enumerate(result.boxes):
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id]
                        
                        print(f"  目标 {i+1}: {class_name} (置信度: {confidence:.3f})")
                
                # 保存标注结果
                result.save(f"demo_result_{os.path.basename(img_path)}")
                print(f"结果已保存为: demo_result_{os.path.basename(img_path)}")
        else:
            print(f"图片不存在: {img_path}")

def demo_camera_detection():
    """摄像头实时检测演示"""
    print("\n=== 摄像头实时检测演示 ===")
    print("按 'q' 键退出")
    
    # 加载模型
    model = YOLO("yolo11n.pt")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("开始实时检测...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 进行检测
            results = model(frame, conf=0.5)
            
            # 绘制结果
            annotated_frame = results[0].plot()
            
            # 添加说明文字
            cv2.putText(annotated_frame, "Face Detection Demo - Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('人脸检测演示', annotated_frame)
            
            # 检查退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("检测被用户中断")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头检测结束")

def create_expression_demo_data():
    """创建表情识别演示数据"""
    print("\n=== 创建表情识别演示数据 ===")
    
    # 创建演示图片
    expressions = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (128, 0, 128), (0, 255, 255), (128, 128, 128)]
    
    # 确保输出目录存在
    os.makedirs("demo_expressions", exist_ok=True)
    
    for i, (expr, color) in enumerate(zip(expressions, colors)):
        # 创建演示图片
        img = np.ones((640, 640, 3), dtype=np.uint8) * 255
        
        # 绘制表情文字
        cv2.putText(img, f"Expression: {expr.upper()}", (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        cv2.putText(img, f"Class ID: {i}", (250, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        
        # 绘制简单的表情符号
        if expr == 'happy':
            cv2.circle(img, (320, 400), 50, color, 3)  # 脸
            cv2.ellipse(img, (300, 380), (20, 10), 0, 0, 180, color, 3)  # 眼睛
            cv2.ellipse(img, (340, 380), (20, 10), 0, 0, 180, color, 3)  # 眼睛
            cv2.ellipse(img, (320, 420), (30, 20), 0, 0, 180, color, 3)  # 嘴巴
        elif expr == 'sad':
            cv2.circle(img, (320, 400), 50, color, 3)  # 脸
            cv2.ellipse(img, (300, 380), (20, 10), 0, 0, 180, color, 3)  # 眼睛
            cv2.ellipse(img, (340, 380), (20, 10), 0, 0, 180, color, 3)  # 眼睛
            cv2.ellipse(img, (320, 430), (30, 20), 0, 180, 360, color, 3)  # 嘴巴
        elif expr == 'angry':
            cv2.circle(img, (320, 400), 50, color, 3)  # 脸
            cv2.rectangle(img, (280, 370), (320, 390), color, 3)  # 眉毛
            cv2.rectangle(img, (320, 370), (360, 390), color, 3)  # 眉毛
            cv2.ellipse(img, (320, 420), (30, 20), 0, 0, 180, color, 3)  # 嘴巴
        
        # 保存图片
        output_path = f"demo_expressions/{expr}_demo.jpg"
        cv2.imwrite(output_path, img)
        print(f"创建演示图片: {output_path}")
    
    print("表情演示数据创建完成！")

def main():
    """主函数"""
    print("人脸表情识别YOLO演示")
    print("=" * 50)
    
    # 1. 创建演示数据
    create_expression_demo_data()
    
    # 2. 使用现有模型检测图片
    demo_with_existing_model()
    
    # 3. 询问是否进行摄像头检测
    choice = input("\n是否进行摄像头实时检测？(y/n): ").lower().strip()
    if choice == 'y':
        demo_camera_detection()
    
    print("\n演示完成！")
    print("\n要使用专门的表情识别功能，请运行：")
    print("1. python face_expression_utils.py  # 创建数据集")
    print("2. python face_expression_train.py  # 训练模型")
    print("3. python face_expression_detect.py # 实时检测")

if __name__ == "__main__":
    main()
