#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版美学评估器
不依赖OpenCV，使用PIL进行基础图像分析
"""

import os
from PIL import Image, ImageStat
import numpy as np
import time


class SimpleAestheticEvaluator:
    """简化版美学评估器"""

    def __init__(self):
        """初始化评估器"""
        self.weights = {
            "brightness": 0.25,  # 亮度权重
            "contrast": 0.25,  # 对比度权重
            "color_balance": 0.25,  # 色彩平衡权重
            "composition": 0.25,  # 构图权重
        }

    def evaluate_image(self, image_path):
        """
        评估单张图像的美学质量

        Args:
            image_path (str): 图像路径

        Returns:
            dict: 包含各项评分和总分的字典
        """
        try:
            # 读取图像
            img = Image.open(image_path)

            # 转换为RGB格式
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 计算各项指标
            scores = {}

            # 1. 亮度评估
            scores["brightness"] = self._evaluate_brightness(img)

            # 2. 对比度评估
            scores["contrast"] = self._evaluate_contrast(img)

            # 3. 色彩平衡评估
            scores["color_balance"] = self._evaluate_color_balance(img)

            # 4. 构图评估
            scores["composition"] = self._evaluate_composition(img)

            # 计算总分
            total_score = sum(scores[key] * self.weights[key] for key in scores)
            scores["total"] = round(total_score, 2)

            return scores

        except Exception as e:
            raise ValueError(f"无法处理图像 {image_path}: {e}")

    def _evaluate_brightness(self, img):
        """评估图像亮度"""
        # 转换为灰度图
        gray = img.convert("L")

        # 计算平均亮度
        stat = ImageStat.Stat(gray)
        avg_brightness = stat.mean[0]

        # 理想亮度范围是100-150
        if 100 <= avg_brightness <= 150:
            brightness_score = 100
        else:
            # 距离理想范围越远，分数越低
            distance = min(abs(avg_brightness - 100), abs(avg_brightness - 150))
            brightness_score = max(0, 100 - distance / 2)

        return round(brightness_score, 2)

    def _evaluate_contrast(self, img):
        """评估图像对比度"""
        # 转换为灰度图
        gray = img.convert("L")

        # 转换为numpy数组
        img_array = np.array(gray)

        # 计算标准差作为对比度指标
        contrast = np.std(img_array)

        # 归一化到0-100分
        normalized_contrast = min(100, contrast / 2.55)

        return round(normalized_contrast, 2)

    def _evaluate_color_balance(self, img):
        """评估色彩平衡"""
        # 获取RGB通道
        r, g, b = img.split()

        # 计算各通道的平均值
        stat_r = ImageStat.Stat(r)
        stat_g = ImageStat.Stat(g)
        stat_b = ImageStat.Stat(b)

        avg_r = stat_r.mean[0]
        avg_g = stat_g.mean[0]
        avg_b = stat_b.mean[0]

        # 计算色彩平衡（理想情况下RGB值应该相近）
        color_balance = 100 - (abs(avg_r - avg_g) + abs(avg_g - avg_b) + abs(avg_r - avg_b)) / 10

        return round(max(0, color_balance), 2)

    def _evaluate_composition(self, img):
        """评估图像构图"""
        width, height = img.size

        # 计算宽高比
        aspect_ratio = width / height

        # 理想宽高比评分
        if 1.2 <= aspect_ratio <= 1.8:  # 接近黄金比例
            ratio_score = 100
        elif 1.0 <= aspect_ratio <= 2.0:  # 可接受范围
            ratio_score = 80
        else:
            ratio_score = 60

        # 计算图像复杂度（基于颜色变化）
        img_array = np.array(img)
        complexity = np.std(img_array)
        complexity_score = min(100, complexity / 50)

        # 综合构图评分
        composition_score = ratio_score * 0.6 + complexity_score * 0.4

        return round(composition_score, 2)

    def batch_evaluate(self, image_dir, output_file=None):
        """
        批量评估图像

        Args:
            image_dir (str): 图像目录路径
            output_file (str): 输出文件路径（可选）

        Returns:
            list: 评估结果列表
        """
        results = []

        # 支持的图像格式
        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        print(f"🔍 开始批量评估目录: {image_dir}")

        # 遍历目录中的图像文件
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(image_dir, filename)
                try:
                    scores = self.evaluate_image(image_path)
                    scores["filename"] = filename
                    results.append(scores)
                    print(f"✅ {filename} - 总分: {scores['total']}")
                except Exception as e:
                    print(f"❌ {filename} - 错误: {e}")

        # 保存结果到文件
        if output_file and results:
            self._save_results(results, output_file)

        return results

    def _save_results(self, results, output_file):
        """保存评估结果到文件"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("图像美学评估结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评估图像数: {len(results)}\n\n")

            for result in results:
                f.write(f"文件名: {result['filename']}\n")
                f.write(f"总分: {result['total']}\n")
                f.write(f"亮度: {result['brightness']}\n")
                f.write(f"对比度: {result['contrast']}\n")
                f.write(f"色彩平衡: {result['color_balance']}\n")
                f.write(f"构图: {result['composition']}\n")

                # 评级
                if result["total"] >= 80:
                    grade = "优秀"
                elif result["total"] >= 60:
                    grade = "良好"
                elif result["total"] >= 40:
                    grade = "一般"
                else:
                    grade = "较差"

                f.write(f"评级: {grade}\n")
                f.write("-" * 30 + "\n")

        print(f"📊 评估结果已保存到: {output_file}")


def main():
    """主函数 - 测试美学评估功能"""
    print("🎨 简化版美学评估系统")
    print("=" * 40)

    # 创建评估器
    evaluator = SimpleAestheticEvaluator()

    # 测试单张图像
    test_image = "img/12.jpg"
    if os.path.exists(test_image):
        print(f"🔍 开始评估图像: {test_image}")
        scores = evaluator.evaluate_image(test_image)

        print("\n📊 评估结果:")
        print(f"总分: {scores['total']}")
        print(f"亮度: {scores['brightness']}")
        print(f"对比度: {scores['contrast']}")
        print(f"色彩平衡: {scores['color_balance']}")
        print(f"构图: {scores['composition']}")

        # 评估等级
        if scores["total"] >= 80:
            grade = "优秀"
        elif scores["total"] >= 60:
            grade = "良好"
        elif scores["total"] >= 40:
            grade = "一般"
        else:
            grade = "较差"

        print(f"\n🎯 美学等级: {grade}")

        # 批量评估
        print(f"\n🔄 开始批量评估...")
        results = evaluator.batch_evaluate("img", "simple_aesthetic_results.txt")

        if results:
            avg_score = sum(r["total"] for r in results) / len(results)
            print(f"\n📈 平均美学评分: {avg_score:.2f}")

            # 显示统计信息
            excellent = sum(1 for r in results if r["total"] >= 80)
            good = sum(1 for r in results if 60 <= r["total"] < 80)
            fair = sum(1 for r in results if 40 <= r["total"] < 60)
            poor = sum(1 for r in results if r["total"] < 40)

            print(f"\n🏆 评级分布:")
            print(f"优秀 (≥80): {excellent} 张")
            print(f"良好 (60-79): {good} 张")
            print(f"一般 (40-59): {fair} 张")
            print(f"较差 (<40): {poor} 张")

    else:
        print(f"❌ 测试图像不存在: {test_image}")
        print("请确保img目录中有图像文件")


if __name__ == "__main__":
    main()
