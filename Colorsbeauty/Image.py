# -*- coding: gbk -*-
import cv2
import numpy as np

class Image:
    def __init__(self, file):
        self.image_path = file
        self.image = cv2.imread(file)
        self.main_colors = None
        self.color_percentages = None
        self._extract_main_colors_once()
    
    def _extract_main_colors_once(self):
        """预处理主色，只计算一次"""
        if self.image is not None:
            self.main_colors, self.color_percentages = self.extract_main_colors_octree(max_colors=8, show=False)

    def extract_main_colors_octree(self, max_colors=8, show=False):
        """
        使用八叉树算法提取主要颜色
        :param max_colors: 最大颜色数量
        :param show: 是否显示主色块
        :return: 主色RGB列表及其占比
        """
        img = self.image.copy()
        
        if img is None:
            print("Error: Image loading failed!")
            return [], []
        
        data = img.reshape((-1, 3))
        
        class OctreeNode:
            def __init__(self, level=0):
                self.level = level
                self.colors = []
                self.children = [None] * 8
                self.is_leaf = True
                self.pixel_count = 0
                self.red_sum = 0.0
                self.green_sum = 0.0
                self.blue_sum = 0.0
            
            def add_color(self, color):
                self.colors.append(color)
                self.pixel_count += 1
                self.blue_sum += float(color[0])
                self.green_sum += float(color[1])
                self.red_sum += float(color[2])
            
            def get_average_color(self):
                if self.pixel_count == 0:
                    return [0, 0, 0]
                return [
                    int(self.blue_sum / self.pixel_count),
                    int(self.green_sum / self.pixel_count),
                    int(self.red_sum / self.pixel_count)
                ]
            
            def split(self):
                if self.level >= 7:
                    return
                self.is_leaf = False
                for color in self.colors:
                    child_index = self.get_child_index(color)
                    if self.children[child_index] is None:
                        self.children[child_index] = OctreeNode(self.level + 1)
                    self.children[child_index].add_color(color)
                self.colors = []
            
            def get_child_index(self, color):
                b, g, r = color
                index = 0
                if r >= 128:
                    index |= 4
                if g >= 128:
                    index |= 2
                if b >= 128:
                    index |= 1
                return index
        
        root = OctreeNode()
        for color in data:
            root.add_color(color)
        
        def build_octree(node, max_colors):
            if node.pixel_count <= max_colors or node.level >= 7:
                return [node] if node.pixel_count > 0 else []
            
            if len(node.colors) > 0:
                colors_array = np.array(node.colors)
                color_std = np.std(colors_array, axis=0)
                if np.all(color_std < 10):
                    return [node] if node.pixel_count > 0 else []
            
            node.split()
            leaves = []
            for child in node.children:
                if child is not None:
                    leaves.extend(build_octree(child, max_colors))
            return leaves
        
        leaves = build_octree(root, max_colors)
        leaves = [leaf for leaf in leaves if leaf.pixel_count > 0]
        
        # 使用numpy排序提高速度
        if leaves:
            pixel_counts = np.array([leaf.pixel_count for leaf in leaves])
            sorted_indices = np.argsort(pixel_counts)[::-1]
            leaves = [leaves[i] for i in sorted_indices[:max_colors]]
        
        main_colors = []
        total_pixels = sum(leaf.pixel_count for leaf in leaves)
        
        for leaf in leaves:
            color = leaf.get_average_color()
            percentage = leaf.pixel_count / total_pixels
            main_colors.append((color, percentage))

        return [color for color, _ in main_colors], [percent for _, percent in main_colors]

    def analyze_color_harmony(self):
        """
        基于色相环分析色彩和谐度（优化版本）
        """
        main_colors = self.main_colors[:5] if self.main_colors else []
        
        if len(main_colors) < 2:
            return 0.5
        
        # 批量转换颜色空间
        colors_array = np.array(main_colors, dtype=np.uint8).reshape(-1, 1, 3)
        hsv_colors = cv2.cvtColor(colors_array, cv2.COLOR_BGR2HSV)[:, 0, 0]
        
        # 预计算所有色相对
        n_colors = len(hsv_colors)
        if n_colors < 2:
            return 0.5
        
        # 使用numpy向量化计算
        hue_diffs = np.abs(hsv_colors[:, None] - hsv_colors[None, :])
        # 处理色相环循环性
        hue_diffs = np.minimum(hue_diffs, 180 - hue_diffs)
        
        # 只取上三角矩阵（避免重复计算）
        upper_triangle = np.triu(hue_diffs, k=1)
        valid_pairs = upper_triangle > 0
        
        if not np.any(valid_pairs):
            return 0.5
        
        # 向量化和谐度计算
        harmony_scores = np.zeros_like(upper_triangle, dtype=np.float32)
        
        # 单色和谐 (0° ± 15°)
        mask = (upper_triangle <= 15) & valid_pairs
        harmony_scores[mask] = 1.0 - (upper_triangle[mask] / 15.0)
        
        # 类似色和谐 (30° ± 15°)
        mask = (upper_triangle > 15) & (upper_triangle <= 45) & valid_pairs
        harmony_scores[mask] = 1.0 - np.abs(upper_triangle[mask] - 30) / 15.0
        
        # 分裂互补色和谐 (60° ± 15°)
        mask = (upper_triangle > 45) & (upper_triangle <= 75) & valid_pairs
        harmony_scores[mask] = 1.0 - np.abs(upper_triangle[mask] - 60) / 15.0
        
        # 三角色和谐 (120° ± 20°)
        mask = (upper_triangle >= 100) & (upper_triangle <= 140) & valid_pairs
        harmony_scores[mask] = 1.0 - np.abs(upper_triangle[mask] - 120) / 20.0
        
        # 互补色和谐 (180° ± 20°)
        mask = (upper_triangle >= 160) & (upper_triangle <= 180) & valid_pairs
        harmony_scores[mask] = 1.0 - np.abs(upper_triangle[mask] - 180) / 20.0
        
        # 其他情况的基础分数
        other_mask = valid_pairs & (harmony_scores == 0)
        if np.any(other_mask):
            other_diffs = upper_triangle[other_mask]
            other_scores = np.where(other_diffs <= 30, 0.3,np.where(other_diffs <= 60, 0.2,np.where(other_diffs <= 120, 0.1, 0.05)))
            harmony_scores[other_mask] = other_scores
        
        # 计算平均和谐度
        valid_scores = harmony_scores[valid_pairs]
        avg_harmony = np.mean(valid_scores) if len(valid_scores) > 0 else 0.5
        
        return min(1.0, max(0.0, avg_harmony))

    def analyze_saturation_harmony(self):
        """
        分析饱和度和谐度（优化版本）
        """
        img = self.image
        
        # 如果图片太大，先降采样
        height, width = img.shape[:2]
        if height * width > 500000:  # 50万像素以上降采样
            scale = min(1.0, np.sqrt(500000 / (height * width)))
            new_height = int(height * scale)
            new_width = int(width * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].flatten()
        
        # 使用更快的标准差计算
        saturation_std = np.std(s, dtype=np.float32)
        harmony_score = 1.0 / (1.0 + saturation_std / 50.0)
        
        return harmony_score

    def analyze_color_temperature_harmony(self):
        """
        分析色温和谐度（优化版本）
        """
        main_colors = self.main_colors if self.main_colors else []
        color_percentages = self.color_percentages if self.color_percentages else []
        
        if len(main_colors) < 2:
            return 0.5
        
        # 批量转换颜色空间
        colors_array = np.array(main_colors, dtype=np.uint8).reshape(-1, 1, 3)
        hsv_colors = cv2.cvtColor(colors_array, cv2.COLOR_BGR2HSV)[:, 0, 0]
        percentages = np.array(color_percentages)
        
        # 向量化色温分类
        warm_mask = ((hsv_colors >= 0) & (hsv_colors <= 60)) | ((hsv_colors >= 150) & (hsv_colors <= 180))
        cool_mask = (hsv_colors >= 90) & (hsv_colors <= 150)
        neutral_mask = ~(warm_mask | cool_mask)
        
        # 计算权重
        warm_weight = np.sum(percentages[warm_mask])
        cool_weight = np.sum(percentages[cool_mask])
        neutral_weight = np.sum(percentages[neutral_mask])
        
        if warm_weight + cool_weight > 0:
            balance_score = 1.0 - abs(warm_weight - cool_weight) / (warm_weight + cool_weight)
        else:
            balance_score = 0.5
        
        total_weight = warm_weight + cool_weight + neutral_weight
        if total_weight > 0:
            purity_score = (warm_weight + cool_weight) / total_weight
        else:
            purity_score = 0.5
        
        consistency_score = 0.5
        
        warm_hues = hsv_colors[warm_mask]
        cool_hues = hsv_colors[cool_mask]
        
        if len(warm_hues) > 1:
            warm_diff = np.max(warm_hues) - np.min(warm_hues)
            if warm_diff > 0 and warm_diff < 180:
                warm_consistency = 1.0 - min(warm_diff / 60.0, 1.0)
                consistency_score = max(consistency_score, warm_consistency)
        
        if len(cool_hues) > 1:
            cool_diff = np.max(cool_hues) - np.min(cool_hues)
            if cool_diff > 0 and cool_diff < 180:
                cool_consistency = 1.0 - min(cool_diff / 60.0, 1.0)
                consistency_score = max(consistency_score, cool_consistency)
        
        temperature_harmony = (balance_score * 0.4 + purity_score * 0.3 + consistency_score * 0.3)
        
        return temperature_harmony

    def get_color_harmony_score(self):
        """
        综合色彩和谐度评分（0-1，越高越和谐）
        """
        hue_harmony = self.analyze_color_harmony()
        # print("a")
        sat_harmony = self.analyze_saturation_harmony()
        # print("b")
        temp_harmony = self.analyze_color_temperature_harmony()
        # print("c")
        total_score = (hue_harmony * 0.2 + sat_harmony * 0.4 + temp_harmony * 0.4)
        
        print(f"色相和谐度: {hue_harmony:.3f}")
        print(f"饱和度和谐度: {sat_harmony:.3f}")
        print(f"色温和谐度: {temp_harmony:.3f}")
        print(f"总体和谐度: {total_score:.3f}")
        
        return total_score