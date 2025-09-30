#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美学评估模块
用于量化评估图像的美学质量
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


class AestheticEvaluator:
    """图像美学评估器"""

    def __init__(self):
        """初始化评估器"""
        self.weights = {
            "clarity": 0.25,  # 清晰度权重
            "color": 0.20,  # 色彩权重
            "composition": 0.25,  # 构图权重
            "contrast": 0.15,  # 对比度权重
            "brightness": 0.15,  # 亮度权重
        }

    def evaluate_image(self, image_path):
        """
        评估单张图像的美学质量

        Args:
            image_path (str): 图像路径

        Returns:
            dict: 包含各项评分和总分的字典
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 计算各项指标
        scores = {}

        # 1. 清晰度评估
        scores["clarity"] = self._evaluate_clarity(img)

        # 2. 色彩评估
        scores["color"] = self._evaluate_color(img_rgb)

        # 3. 构图评估
        scores["composition"] = self._evaluate_composition(img_rgb)

        # 4. 对比度评估
        scores["contrast"] = self._evaluate_contrast(img)

        # 5. 亮度评估
        scores["brightness"] = self._evaluate_brightness(img)

        # 计算总分
        total_score = sum(scores[key] * self.weights[key] for key in scores)
        scores["total"] = round(total_score, 2)

        return scores

    def _evaluate_clarity(self, img):
        """评估图像清晰度"""
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用拉普拉斯算子计算清晰度
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        clarity_score = np.var(laplacian)

        # 归一化到0-100分
        normalized_score = min(100, max(0, clarity_score / 1000 * 100))
        return round(normalized_score, 2)

    def _evaluate_color(self, img_rgb):
        """评估图像色彩"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # 计算饱和度
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)

        # 计算色彩分布均匀性
        hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])

        # 计算色彩丰富度
        color_richness = np.std(hist_r) + np.std(hist_g) + np.std(hist_b)

        # 综合评分
        color_score = (avg_saturation * 0.6 + color_richness * 0.4) / 10
        normalized_score = min(100, max(0, color_score))

        return round(normalized_score, 2)

    def _evaluate_composition(self, img_rgb):
        """评估图像构图"""
        height, width = img_rgb.shape[:2]

        # 计算三分法规则
        rule_of_thirds = self._calculate_rule_of_thirds(img_rgb)

        # 计算对称性
        symmetry = self._calculate_symmetry(img_rgb)

        # 计算平衡性
        balance = self._calculate_balance(img_rgb)

        # 综合构图评分
        composition_score = rule_of_thirds * 0.4 + symmetry * 0.3 + balance * 0.3

        return round(composition_score, 2)

    def _calculate_rule_of_thirds(self, img):
        """计算三分法规则评分"""
        height, width = img.shape[:2]

        # 定义三分法网格线位置
        h1, h2 = height // 3, 2 * height // 3
        w1, w2 = width // 3, 2 * width // 3

        # 计算网格线附近的亮度
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 计算水平线和垂直线附近的对比度
        h_line_contrast = np.std(gray[h1 - 10 : h1 + 10, :]) + np.std(gray[h2 - 10 : h2 + 10, :])
        v_line_contrast = np.std(gray[:, w1 - 10 : w1 + 10]) + np.std(gray[:, w2 - 10 : w2 + 10])

        # 归一化评分
        thirds_score = min(100, (h_line_contrast + v_line_contrast) / 20)

        return thirds_score

    def _calculate_symmetry(self, img):
        """计算对称性评分"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape

        # 计算水平对称性
        top_half = gray[: height // 2, :]
        bottom_half = cv2.flip(gray[height // 2 :, :], 0)

        # 计算垂直对称性
        left_half = gray[:, : width // 2]
        right_half = cv2.flip(gray[:, width // 2 :], 1)

        # 计算对称性得分
        h_symmetry = 100 - np.mean(np.abs(top_half - bottom_half)) / 2.55
        v_symmetry = 100 - np.mean(np.abs(left_half - right_half)) / 2.55

        # 取平均值
        symmetry_score = (h_symmetry + v_symmetry) / 2

        return max(0, symmetry_score)

    def _calculate_balance(self, img):
        """计算平衡性评分"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape

        # 计算四个象限的平均亮度
        q1 = np.mean(gray[: height // 2, : width // 2])  # 左上
        q2 = np.mean(gray[: height // 2, width // 2 :])  # 右上
        q3 = np.mean(gray[height // 2 :, : width // 2])  # 左下
        q4 = np.mean(gray[height // 2 :, width // 2 :])  # 右下

        # 计算对角线平衡
        diag1_balance = abs(q1 - q4)
        diag2_balance = abs(q2 - q3)

        # 计算总体平衡
        balance_score = 100 - (diag1_balance + diag2_balance) / 10

        return max(0, balance_score)

    def _evaluate_contrast(self, img):
        """评估图像对比度"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算对比度
        contrast = np.std(gray)

        # 归一化到0-100分
        normalized_contrast = min(100, contrast / 2.55)

        return round(normalized_contrast, 2)

    def _evaluate_brightness(self, img):
        """评估图像亮度"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算平均亮度
        avg_brightness = np.mean(gray)

        # 理想亮度范围是100-150
        if 100 <= avg_brightness <= 150:
            brightness_score = 100
        else:
            # 距离理想范围越远，分数越低
            distance = min(abs(avg_brightness - 100), abs(avg_brightness - 150))
            brightness_score = max(0, 100 - distance / 2)

        return round(brightness_score, 2)

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

        # 遍历目录中的图像文件
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(image_dir, filename)
                try:
                    scores = self.evaluate_image(image_path)
                    scores["filename"] = filename
                    results.append(scores)
                    print(f"✅ 评估完成: {filename} - 总分: {scores['total']}")
                except Exception as e:
                    print(f"❌ 评估失败: {filename} - 错误: {e}")

        # 保存结果到文件
        if output_file:
            self._save_results(results, output_file)

        return results

    def _save_results(self, results, output_file):
        """保存评估结果到文件"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("图像美学评估结果\n")
            f.write("=" * 50 + "\n\n")

            for result in results:
                f.write(f"文件名: {result['filename']}\n")
                f.write(f"总分: {result['total']}\n")
                f.write(f"清晰度: {result['clarity']}\n")
                f.write(f"色彩: {result['color']}\n")
                f.write(f"构图: {result['composition']}\n")
                f.write(f"对比度: {result['contrast']}\n")
                f.write(f"亮度: {result['brightness']}\n")
                f.write("-" * 30 + "\n")

        print(f"📊 评估结果已保存到: {output_file}")


def main():
    """主函数 - 测试美学评估功能"""
    # 创建评估器
    evaluator = AestheticEvaluator()

    # 测试单张图像
    test_image = "img/12.jpg"
    if os.path.exists(test_image):
        print(f"🔍 开始评估图像: {test_image}")
        scores = evaluator.evaluate_image(test_image)

        print("\n📊 评估结果:")
        print(f"总分: {scores['total']}")
        print(f"清晰度: {scores['clarity']}")
        print(f"色彩: {scores['color']}")
        print(f"构图: {scores['composition']}")
        print(f"对比度: {scores['contrast']}")
        print(f"亮度: {scores['brightness']}")

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
        results = evaluator.batch_evaluate("img", "aesthetic_results.txt")

        if results:
            avg_score = sum(r["total"] for r in results) / len(results)
            print(f"\n📈 平均美学评分: {avg_score:.2f}")

    else:
        print(f"❌ 测试图像不存在: {test_image}")


if __name__ == "__main__":
    main()
