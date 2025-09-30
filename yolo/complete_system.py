#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的YOLO + 美学评估系统
结合目标检测和美学评估的集成系统
"""

import os
import time
from ultralytics import YOLO
from simple_evaluator import SimpleAestheticEvaluator


class CompleteAestheticSystem:
    """完整的YOLO + 美学评估系统"""

    def __init__(self, model_path="3times.pt"):
        """
        初始化系统

        Args:
            model_path (str): YOLO模型路径
        """
        print("🚀 初始化完整美学评估系统...")

        # 加载YOLO模型
        try:
            self.yolo_model = YOLO(model_path)
            print(f"✅ YOLO模型加载成功: {model_path}")
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")
            self.yolo_model = None

        # 初始化美学评估器
        self.aesthetic_evaluator = SimpleAestheticEvaluator()
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
        print(f"\n🔍 开始分析图像: {os.path.basename(image_path)}")

        analysis_result = {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detection": {},
            "aesthetic": {},
            "recommendations": [],
        }

        # 1. YOLO目标检测
        if self.yolo_model:
            print("📡 进行目标检测...")
            try:
                detection_results = self.yolo_model(image_path)
                analysis_result["detection"] = self._process_detection_results(detection_results)
                print(f"✅ 检测完成，发现 {len(analysis_result['detection']['objects'])} 个目标")
            except Exception as e:
                print(f"❌ 目标检测失败: {e}")
                analysis_result["detection"] = {"objects": [], "error": str(e)}
        else:
            print("⚠️ YOLO模型未加载，跳过目标检测")
            analysis_result["detection"] = {"objects": [], "error": "YOLO模型未加载"}

        # 2. 美学评估
        print("🎨 进行美学评估...")
        try:
            analysis_result["aesthetic"] = self.aesthetic_evaluator.evaluate_image(image_path)
            print(f"✅ 美学评估完成，总分: {analysis_result['aesthetic']['total']}")
        except Exception as e:
            print(f"❌ 美学评估失败: {e}")
            analysis_result["aesthetic"] = {"error": str(e)}

        # 3. 生成改进建议
        if "total" in analysis_result["aesthetic"]:
            analysis_result["recommendations"] = self._generate_recommendations(analysis_result["aesthetic"])

        # 4. 显示结果
        if show_result:
            self._display_results(analysis_result)

        return analysis_result

    def _process_detection_results(self, detection_results):
        """处理YOLO检测结果"""
        objects = []

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

                    obj = {
                        "class_id": class_id,
                        "class_name": self.yolo_model.names[class_id],
                        "confidence": round(float(confidence), 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    }
                    objects.append(obj)

        return {"objects": objects, "count": len(objects)}

    def _generate_recommendations(self, aesthetic_scores):
        """生成改进建议"""
        recommendations = []

        if aesthetic_scores["brightness"] < 60:
            recommendations.append("💡 建议调整图像亮度，使其更明亮或更暗")

        if aesthetic_scores["contrast"] < 60:
            recommendations.append("⚖️ 建议增加图像对比度，使图像更有层次感")

        if aesthetic_scores["color_balance"] < 60:
            recommendations.append("🌈 建议调整色彩平衡，减少色偏")

        if aesthetic_scores["composition"] < 60:
            recommendations.append("📐 建议改善构图，尝试不同的拍摄角度或裁剪")

        # 总分建议
        total_score = aesthetic_scores["total"]
        if total_score < 40:
            recommendations.append("🎯 整体图像质量较低，建议重新拍摄或进行大幅调整")
        elif total_score < 60:
            recommendations.append("📈 图像质量中等，可以通过细节调整来提升")
        elif total_score < 80:
            recommendations.append("👍 图像质量良好，可以进行微调优化")
        else:
            recommendations.append("🌟 图像质量优秀，保持当前水平")

        return recommendations

    def _display_results(self, result):
        """显示分析结果"""
        print("\n" + "=" * 70)
        print("📊 完整图像分析结果")
        print("=" * 70)

        # 显示基本信息
        print(f"📁 文件: {result['filename']}")
        print(f"⏰ 时间: {result['timestamp']}")

        # 显示检测结果
        detection = result["detection"]
        print(f"\n🎯 目标检测结果:")
        if "objects" in detection and detection["objects"]:
            print(f"  检测到 {len(detection['objects'])} 个目标:")
            for i, obj in enumerate(detection["objects"], 1):
                print(f"    {i}. {obj['class_name']} (置信度: {obj['confidence']:.3f})")
        else:
            print("  未检测到目标")

        # 显示美学评估结果
        aesthetic = result["aesthetic"]
        if "total" in aesthetic:
            print(f"\n🎨 美学评估结果:")
            print(f"  📊 总分: {aesthetic['total']}/100")
            print(f"  💡 亮度: {aesthetic['brightness']}/100")
            print(f"  ⚖️ 对比度: {aesthetic['contrast']}/100")
            print(f"  🌈 色彩平衡: {aesthetic['color_balance']}/100")
            print(f"  📐 构图: {aesthetic['composition']}/100")

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
        if result["recommendations"]:
            print(f"\n💡 改进建议:")
            for rec in result["recommendations"]:
                print(f"  {rec}")

        print("=" * 70)

    def batch_analyze(self, image_dir, output_file="complete_analysis_results.txt"):
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
        if results:
            self._save_batch_results(results, output_file)
            self._show_batch_statistics(results)

        print(f"\n🎉 批量分析完成! 共处理 {total_images} 张图像")
        return results

    def _save_batch_results(self, results, output_file):
        """保存批量分析结果"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("YOLO + 美学评估完整分析结果\n")
            f.write("=" * 70 + "\n")
            f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析图像数: {len(results)}\n\n")

            for result in results:
                f.write(f"文件: {result['filename']}\n")
                f.write(f"时间: {result['timestamp']}\n")

                # 检测结果
                detection = result["detection"]
                if "objects" in detection:
                    f.write(f"检测目标数: {len(detection['objects'])}\n")
                    for obj in detection["objects"]:
                        f.write(f"  - {obj['class_name']} (置信度: {obj['confidence']:.3f})\n")

                # 美学评估
                aesthetic = result["aesthetic"]
                if "total" in aesthetic:
                    f.write(f"美学总分: {aesthetic['total']}\n")
                    f.write(f"亮度: {aesthetic['brightness']}\n")
                    f.write(f"对比度: {aesthetic['contrast']}\n")
                    f.write(f"色彩平衡: {aesthetic['color_balance']}\n")
                    f.write(f"构图: {aesthetic['composition']}\n")

                    # 评级
                    total_score = aesthetic["total"]
                    if total_score >= 80:
                        grade = "优秀"
                    elif total_score >= 60:
                        grade = "良好"
                    elif total_score >= 40:
                        grade = "一般"
                    else:
                        grade = "较差"
                    f.write(f"评级: {grade}\n")

                # 改进建议
                if result["recommendations"]:
                    f.write("改进建议:\n")
                    for rec in result["recommendations"]:
                        f.write(f"  {rec}\n")

                f.write("-" * 50 + "\n\n")

        print(f"📊 完整分析结果已保存到: {output_file}")

    def _show_batch_statistics(self, results):
        """显示批量分析统计信息"""
        print(f"\n📈 批量分析统计:")

        # 美学评分统计
        aesthetic_results = [r for r in results if "total" in r["aesthetic"]]
        if aesthetic_results:
            total_scores = [r["aesthetic"]["total"] for r in aesthetic_results]
            avg_score = sum(total_scores) / len(total_scores)
            max_score = max(total_scores)
            min_score = min(total_scores)

            print(f"  📊 平均美学评分: {avg_score:.2f}")
            print(f"  📊 最高评分: {max_score:.2f}")
            print(f"  📊 最低评分: {min_score:.2f}")

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

        # 检测统计
        total_detections = 0
        for result in results:
            if "objects" in result["detection"]:
                total_detections += len(result["detection"]["objects"])

        avg_detections = total_detections / len(results) if results else 0

        print(f"  🎯 平均检测目标数: {avg_detections:.2f}")
        print(f"  🎯 总检测目标数: {total_detections}")


def main():
    """主函数"""
    print("🎨 完整YOLO + 美学评估系统")
    print("=" * 60)

    try:
        # 创建系统实例
        system = CompleteAestheticSystem()

        # 测试单张图像
        test_image = "img/12.jpg"
        if os.path.exists(test_image):
            result = system.analyze_image(test_image)

            # 批量分析
            print(f"\n{'='*60}")
            system.batch_analyze("img")

        else:
            print(f"❌ 测试图像不存在: {test_image}")
            print("请确保img目录中有图像文件")

    except Exception as e:
        print(f"❌ 系统运行错误: {e}")
        print("请检查模型文件和依赖包是否正确安装")


if __name__ == "__main__":
    main()
