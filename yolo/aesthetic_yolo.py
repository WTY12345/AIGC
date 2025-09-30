#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + 美学评估集成程序
结合目标检测和美学评估的完整系统
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from aesthetic_evaluator import AestheticEvaluator
import matplotlib.pyplot as plt
import time


class AestheticYOLO:
    """YOLO + 美学评估集成系统"""

    def __init__(self, model_path="3times.pt"):
        """
        初始化系统

        Args:
            model_path (str): YOLO模型路径
        """
        print("🚀 初始化AestheticYOLO系统...")

        # 加载YOLO模型
        self.yolo_model = YOLO(model_path)
        print(f"✅ YOLO模型加载成功: {model_path}")

        # 初始化美学评估器
        self.aesthetic_evaluator = AestheticEvaluator()
        print("✅ 美学评估器初始化成功")

        print("🎉 系统初始化完成!")

    def analyze_image(self, image_path, show_result=True):
        """
        分析单张图像

        Args:
            image_path (str): 图像路径
            show_result (bool): 是否显示结果

        Returns:
            dict: 分析结果
        """
        print(f"\n🔍 开始分析图像: {image_path}")

        # 1. YOLO目标检测
        print("📡 进行目标检测...")
        detection_results = self.yolo_model(image_path)

        # 2. 美学评估
        print("🎨 进行美学评估...")
        aesthetic_scores = self.aesthetic_evaluator.evaluate_image(image_path)

        # 3. 整合结果
        analysis_result = {
            "image_path": image_path,
            "detection": self._process_detection_results(detection_results),
            "aesthetic": aesthetic_scores,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 4. 显示结果
        if show_result:
            self._display_results(analysis_result)

        return analysis_result

    def _process_detection_results(self, detection_results):
        """处理YOLO检测结果"""
        detections = []

        for result in detection_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # 获取置信度
                    confidence = box.conf[0].cpu().numpy()

                    # 获取类别
                    class_id = int(box.cls[0].cpu().numpy())

                    detection = {
                        "class_id": class_id,
                        "class_name": self.yolo_model.names[class_id],
                        "confidence": round(float(confidence), 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    }
                    detections.append(detection)

        return detections

    def _display_results(self, result):
        """显示分析结果"""
        print("\n" + "=" * 60)
        print("📊 图像分析结果")
        print("=" * 60)

        # 显示基本信息
        print(f"📁 文件: {os.path.basename(result['image_path'])}")
        print(f"⏰ 时间: {result['timestamp']}")

        # 显示检测结果
        detections = result["detection"]
        print(f"\n🎯 目标检测结果 (检测到 {len(detections)} 个目标):")
        if detections:
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class_name']} (置信度: {det['confidence']:.3f})")
        else:
            print("  未检测到目标")

        # 显示美学评估结果
        aesthetic = result["aesthetic"]
        print(f"\n🎨 美学评估结果:")
        print(f"  📊 总分: {aesthetic['total']}/100")
        print(f"  🔍 清晰度: {aesthetic['clarity']}/100")
        print(f"  🌈 色彩: {aesthetic['color']}/100")
        print(f"  📐 构图: {aesthetic['composition']}/100")
        print(f"  ⚖️ 对比度: {aesthetic['contrast']}/100")
        print(f"  💡 亮度: {aesthetic['brightness']}/100")

        # 显示综合评级
        total_score = aesthetic["total"]
        if total_score >= 80:
            grade = "🌟 优秀"
            comment = "这是一张非常出色的图像！"
        elif total_score >= 60:
            grade = "👍 良好"
            comment = "这是一张不错的图像。"
        elif total_score >= 40:
            grade = "👌 一般"
            comment = "图像质量中等。"
        else:
            grade = "👎 较差"
            comment = "图像质量需要改进。"

        print(f"\n🏆 综合评级: {grade}")
        print(f"💬 评价: {comment}")

        # 显示改进建议
        self._show_improvement_suggestions(aesthetic)

        print("=" * 60)

    def _show_improvement_suggestions(self, aesthetic_scores):
        """显示改进建议"""
        suggestions = []

        if aesthetic_scores["clarity"] < 60:
            suggestions.append("🔍 建议提高图像清晰度")

        if aesthetic_scores["color"] < 60:
            suggestions.append("🌈 建议调整色彩饱和度和对比度")

        if aesthetic_scores["composition"] < 60:
            suggestions.append("📐 建议改善构图，遵循三分法规则")

        if aesthetic_scores["contrast"] < 60:
            suggestions.append("⚖️ 建议增加图像对比度")

        if aesthetic_scores["brightness"] < 60:
            suggestions.append("💡 建议调整图像亮度")

        if suggestions:
            print(f"\n💡 改进建议:")
            for suggestion in suggestions:
                print(f"  {suggestion}")

    def batch_analyze(self, image_dir, output_file="batch_analysis_results.txt"):
        """
        批量分析图像

        Args:
            image_dir (str): 图像目录
            output_file (str): 输出文件路径
        """
        print(f"\n🔄 开始批量分析目录: {image_dir}")

        # 支持的图像格式
        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        results = []
        total_images = 0

        # 遍历目录
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(supported_formats):
                total_images += 1
                image_path = os.path.join(image_dir, filename)

                try:
                    result = self.analyze_image(image_path, show_result=False)
                    results.append(result)
                    print(f"✅ 完成分析: {filename}")
                except Exception as e:
                    print(f"❌ 分析失败: {filename} - {e}")

        # 保存结果
        self._save_batch_results(results, output_file)

        # 显示统计信息
        if results:
            self._show_batch_statistics(results)

        print(f"\n🎉 批量分析完成! 共处理 {total_images} 张图像")

    def _save_batch_results(self, results, output_file):
        """保存批量分析结果"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("YOLO + 美学评估批量分析结果\n")
            f.write("=" * 60 + "\n\n")

            for result in results:
                filename = os.path.basename(result["image_path"])
                f.write(f"文件: {filename}\n")
                f.write(f"时间: {result['timestamp']}\n")

                # 检测结果
                detections = result["detection"]
                f.write(f"检测目标数: {len(detections)}\n")
                for det in detections:
                    f.write(f"  - {det['class_name']} (置信度: {det['confidence']:.3f})\n")

                # 美学评估
                aesthetic = result["aesthetic"]
                f.write(f"美学总分: {aesthetic['total']}\n")
                f.write(f"清晰度: {aesthetic['clarity']}\n")
                f.write(f"色彩: {aesthetic['color']}\n")
                f.write(f"构图: {aesthetic['composition']}\n")
                f.write(f"对比度: {aesthetic['contrast']}\n")
                f.write(f"亮度: {aesthetic['brightness']}\n")

                f.write("-" * 40 + "\n\n")

        print(f"📊 批量分析结果已保存到: {output_file}")

    def _show_batch_statistics(self, results):
        """显示批量分析统计信息"""
        print(f"\n📈 批量分析统计:")

        # 美学评分统计
        total_scores = [r["aesthetic"]["total"] for r in results]
        avg_score = np.mean(total_scores)
        max_score = max(total_scores)
        min_score = min(total_scores)

        print(f"  📊 平均美学评分: {avg_score:.2f}")
        print(f"  📊 最高评分: {max_score:.2f}")
        print(f"  📊 最低评分: {min_score:.2f}")

        # 检测统计
        total_detections = sum(len(r["detection"]) for r in results)
        avg_detections = total_detections / len(results)

        print(f"  🎯 平均检测目标数: {avg_detections:.2f}")
        print(f"  🎯 总检测目标数: {total_detections}")

        # 评级分布
        excellent = sum(1 for score in total_scores if score >= 80)
        good = sum(1 for score in total_scores if 60 <= score < 80)
        fair = sum(1 for score in total_scores if 40 <= score < 60)
        poor = sum(1 for score in total_scores if score < 40)

        print(f"  🏆 评级分布:")
        print(f"    优秀 (≥80): {excellent} 张")
        print(f"    良好 (60-79): {good} 张")
        print(f"    一般 (40-59): {fair} 张")
        print(f"    较差 (<40): {poor} 张")


def main():
    """主函数"""
    print("🎨 YOLO + 美学评估系统")
    print("=" * 50)

    try:
        # 创建系统实例
        system = AestheticYOLO()

        # 测试单张图像
        test_image = "img/12.jpg"
        if os.path.exists(test_image):
            result = system.analyze_image(test_image)

            # 批量分析
            print(f"\n{'='*50}")
            system.batch_analyze("img")

        else:
            print(f"❌ 测试图像不存在: {test_image}")
            print("请确保img目录中有图像文件")

    except Exception as e:
        print(f"❌ 系统运行错误: {e}")
        print("请检查模型文件和依赖包是否正确安装")


if __name__ == "__main__":
    main()
