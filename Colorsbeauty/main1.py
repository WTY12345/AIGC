import cv2
import numpy as np
from sklearn.cluster import KMeans
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 修复 NumPy 2.0 兼容性问题
if not hasattr(np, "asscalar"):

    def asscalar(a):
        return a.item()

    np.asscalar = asscalar


class ColorPsychologyAnalyzer:
    def __init__(self, dominant_colors_k=5):
        self.dominant_colors_k = dominant_colors_k

    def get_dominant_colors(self, img_rgb, k=5):
        """
        使用K-means聚类提取图像的主色调
        """
        # 重塑图像为像素列表
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)

        # 使用K-means聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.astype(np.uint8)

        # 计算每个颜色的比例
        counts = np.bincount(labels)
        proportions = counts / counts.sum()

        return centers, proportions

    def analyze_color_harmony(self, dominant_colors, proportions):
        """
        基于CIE Lab颜色空间分析色彩和谐度（加权计算）
        """
        # 1. 过滤占比<1%的颜色
        significant_indices = [i for i, p in enumerate(proportions) if p >= 0.01]
        if len(significant_indices) < 2:
            return 0.5  # 如果过滤后颜色少于2种，返回中等和谐度

        significant_colors = [dominant_colors[i] for i in significant_indices]
        significant_proportions = [proportions[i] for i in significant_indices]

        # 2. 转换到Lab颜色空间
        lab_colors = []
        for color in significant_colors:
            # 将RGB归一化到0-1范围
            rgb = sRGBColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
            lab = convert_color(rgb, LabColor)
            lab_colors.append(lab)

        # 3. 加权和谐度计算
        weighted_harmony_sum = 0
        total_weight = 0

        for i in range(len(lab_colors)):
            for j in range(i + 1, len(lab_colors)):
                # 计算两个颜色的和谐度
                delta_e = delta_e_cie2000(lab_colors[i], lab_colors[j])
                # 色彩和谐理论：适中的色差（30-60）最和谐
                if delta_e < 15:  # 颜色太相似
                    harmony_score = 0.3
                elif delta_e > 80:  # 颜色对比太强烈
                    harmony_score = 0.4
                else:  # 适中的对比度
                    harmony_score = 1.0 - abs(delta_e - 45) / 45  # 45为理想色差

                # 计算权重：两个颜色占比的乘积
                weight = significant_proportions[i] * significant_proportions[j]

                # 累加加权和谐度
                weighted_harmony_sum += harmony_score * weight
                total_weight += weight

        # 4. 返回加权平均和谐度
        return weighted_harmony_sum / total_weight if total_weight > 0 else 0.5

    def analyze_color_temperature(self, img_hsv):
        """
        分析图像的色调一致性
        """
        hue = img_hsv[:, :, 0]  # 色相通道 (0-180)

        # 定义冷暖色调范围 (OpenCV中Hue范围是0-180)
        # 冷色调: 蓝色、青色 (90-150)
        cool_pixels = np.sum((hue >= 90) & (hue <= 150))

        # 暖色调: 红色、黄色 (0-30, 150-180)
        warm_pixels = np.sum((hue <= 30) | (hue >= 150))

        # 中性色调: 绿色等 (30-90)
        neutral_pixels = np.sum((hue > 30) & (hue < 90))

        total_pixels = img_hsv.shape[0] * img_hsv.shape[1]

        if total_pixels == 0:
            return 0.5

        cool_ratio = cool_pixels / total_pixels
        warm_ratio = warm_pixels / total_pixels
        neutral_ratio = neutral_pixels / total_pixels

        # 色调一致性分析：评估主导色温的占比
        max_ratio = max(cool_ratio, warm_ratio, neutral_ratio)

        # 评分标准：
        # - 如果主导色温占比很高（>70%），给予高分
        # - 如果有明确的主导色温（>50%），给予中等偏高分
        # - 如果色温分布过于分散，给予低分
        if max_ratio >= 0.7:
            # 纯色调或接近纯色调，给予高分
            consistency_score = 0.8 + 0.2 * (max_ratio - 0.7) / 0.3
        elif max_ratio >= 0.5:
            # 有明确主导色温，给予中等偏高分
            consistency_score = 0.6 + 0.2 * (max_ratio - 0.5) / 0.2
        else:
            # 色温分布分散，给予低分
            consistency_score = 0.3 + 0.3 * (max_ratio - 0.33) / 0.17

        return min(1.0, max(0.0, consistency_score))

    def analyze_saturation_distribution(self, img_hsv):
        """
        分析饱和度分布的合理性
        """
        saturation = img_hsv[:, :, 1] / 255.0  # 归一化到0-1

        # 核心思路：饱和度突变越大，分数越低
        saturation_std = np.std(saturation)

        # 使用指数递减赋分，突变越大分数越低
        saturation_score = np.exp(-saturation_std * 3)

        return saturation_score

    def analyze_brightness_balance(self, img_hsv):
        """
        分析亮度平衡（避免过曝或过暗）
        """
        value = img_hsv[:, :, 2] / 255.0  # 亮度通道

        if value.size == 0:
            return 0.5

        # 核心思路：亮度突变越大，分数越低
        brightness_std = np.std(value)

        # 使用指数递减赋分，突变越大分数越低
        brightness_score = np.exp(-brightness_std * 3)

        return brightness_score

    def calculate_comprehensive_score(self, image_path):
        """
        计算综合色彩评分
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 提取主色调
        dominant_colors, proportions = self.get_dominant_colors(img_rgb, self.dominant_colors_k)

        # 计算各项指标
        harmony_score = self.analyze_color_harmony(dominant_colors, proportions)
        temperature_score = self.analyze_color_temperature(img_hsv)
        saturation_score = self.analyze_saturation_distribution(img_hsv)
        brightness_score = self.analyze_brightness_balance(img_hsv)

        # 权重分配（可根据需求调整）
        weights = {
            "harmony": 0.25,  # 色彩和谐度
            "temperature": 0.25,  # 冷暖平衡
            "saturation": 0.25,  # 饱和度分布
            "brightness": 0.25,  # 亮度平衡
        }

        # 计算综合评分（0-10分）
        weighted_score = (
            harmony_score * weights["harmony"]
            + temperature_score * weights["temperature"]
            + saturation_score * weights["saturation"]
            + brightness_score * weights["brightness"]
        )

        final_score = weighted_score * 10  # 转换为0-10分制

        return {
            "final_score": round(final_score, 2),
            "detailed_scores": {
                "color_harmony": round(harmony_score, 3),
                "temperature_balance": round(temperature_score, 3),
                "saturation_distribution": round(saturation_score, 3),
                "brightness_balance": round(brightness_score, 3),
            },
            "dominant_colors": dominant_colors.tolist(),
            "color_proportions": proportions.tolist(),
            "recommendations": self.generate_recommendations(
                harmony_score, temperature_score, saturation_score, brightness_score
            ),
        }

    def generate_recommendations(self, harmony, temperature, saturation, brightness):
        """
        根据各项指标生成改进建议
        """
        recommendations = []

        if harmony < 0.6:
            recommendations.append("建议调整颜色搭配，使主色调更加和谐")

        if temperature < 0.6:
            recommendations.append("建议平衡冷暖色调比例")

        if saturation < 0.6:
            recommendations.append("建议减少饱和度变化，使色彩更加平滑")

        if brightness < 0.6:
            recommendations.append("建议减少亮度变化，使明暗过渡更加自然")

        if not recommendations:
            recommendations.append("色彩搭配良好，继续保持！")

        return recommendations

    def visualize_analysis(self, image_path, save_path=None):
        """
        可视化分析结果
        """
        result = self.calculate_comprehensive_score(image_path)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 原始图像
        ax1.imshow(img_rgb)
        ax1.set_title(f'原始图像 - 综合评分: {result["final_score"]}/10', fontsize=14)
        ax1.axis("off")

        # 主色调展示
        dominant_colors = np.array(result["dominant_colors"]).reshape(-1, 1, 3)
        ax2.imshow(dominant_colors, aspect="auto")
        ax2.set_title("主色调分布", fontsize=14)
        ax2.set_yticks(range(len(result["dominant_colors"])))
        ax2.set_yticklabels([f"{p*100:.1f}%" for p in result["color_proportions"]])
        ax2.axis("on")

        # 评分雷达图
        categories = list(result["detailed_scores"].keys())
        scores = list(result["detailed_scores"].values())

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # 闭合雷达图
        angles += angles[:1]
        categories = [cat.replace("_", "\n") for cat in categories]

        ax3 = plt.subplot(2, 2, 3, polar=True)
        ax3.plot(angles, scores, "o-", linewidth=2, label="色彩指标")
        ax3.fill(angles, scores, alpha=0.25)
        ax3.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax3.set_ylim(0, 1)
        ax3.set_title("色彩分析雷达图", size=14, y=1.1)
        ax3.grid(True)

        # 建议文本
        ax4.text(0.1, 0.9, "改进建议:", fontsize=14, fontweight="bold")
        for i, recommendation in enumerate(result["recommendations"]):
            ax4.text(
                0.1, 0.8 - i * 0.1, f"• {recommendation}", fontsize=11, verticalalignment="top", transform=ax4.transAxes
            )
        ax4.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return result


# 使用示例
def main():
    # 创建分析器实例
    analyzer = ColorPsychologyAnalyzer(dominant_colors_k=5)

    # 分析单张图像
    image_path = "D://AIGC/Colorsbeauty/picture/china.png"  # 替换为你的图像路径

    result = analyzer.calculate_comprehensive_score(image_path)

    print("=== 色彩心理学分析结果 ===")
    print(f"综合评分: {result['final_score']}/10")
    print("\n详细评分:")
    for category, score in result["detailed_scores"].items():
        print(f"  {category}: {score:.3f}")

        # 可视化结果已移除
        # analyzer.visualize_analysis(image_path, "color_analysis_result.png")


if __name__ == "__main__":
    main()
