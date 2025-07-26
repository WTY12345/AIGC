import cv2
import numpy as np
from collections import deque

class Image:
    def __init__(self, file):
        self.image_path = file
        self.image = cv2.imread(file)

    def show_image(self):
        cv2.imshow("ShowImage", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_gray_hist(self):
        gray = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # 绘制直方图
        import matplotlib.pyplot as plt
        plt.plot(hist)
        plt.title("Gray Histogram")
        plt.show()

    def show_rgb_gray_hist(self):
        img = self.image
        color = ('b', 'g', 'r')
        import matplotlib.pyplot as plt
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        plt.plot(gray_hist, color='k')
        plt.title("RGB and Gray Histogram")
        plt.show()

    def show_hs_hist(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        import matplotlib.pyplot as plt
        plt.hist2d(h.flatten(), s.flatten(), bins=[16, 8], range=[[0, 180], [0, 256]])
        plt.xlabel('H')
        plt.ylabel('S')
        plt.title('HS Histogram')
        plt.colorbar()
        plt.show()

    def show_color_moment(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        means = np.mean(hsv, axis=(0, 1))
        stds = np.std(hsv, axis=(0, 1))
        skews = np.mean((hsv - means) ** 3, axis=(0, 1)) ** (1/3)
        print("Mean (H, S, V):", means)
        print("Std (H, S, V):", stds)
        print("Skewness (H, S, V):", skews)

    def extract_main_colors_octree(self, max_colors=8, show=True):
        """
        使用八叉树算法提取主要颜色
        :param max_colors: 最大颜色数量
        :param show: 是否显示主色块
        :return: 主色RGB列表及其占比
        """
        img = self.image.copy()
        
        # 检查图片是否正确加载
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
                self.red_sum = 0.0    # 改为float避免溢出
                self.green_sum = 0.0  # 改为float避免溢出
                self.blue_sum = 0.0   # 改为float避免溢出
            
            def add_color(self, color):
                self.colors.append(color)
                self.pixel_count += 1
                # 修正：OpenCV是BGR格式，所以color[0]是B，color[1]是G，color[2]是R
                self.blue_sum += float(color[0])   # B
                self.green_sum += float(color[1])  # G
                self.red_sum += float(color[2])    # R
            
            def get_average_color(self):
                if self.pixel_count == 0:
                    return [0, 0, 0]
                return [
                    int(self.blue_sum / self.pixel_count),   # B
                    int(self.green_sum / self.pixel_count),  # G
                    int(self.red_sum / self.pixel_count)     # R
                ]
            
            def split(self):
                if self.level >= 7:  # 最大深度限制
                    return
                
                self.is_leaf = False
                for color in self.colors:
                    child_index = self.get_child_index(color)
                    if self.children[child_index] is None:
                        self.children[child_index] = OctreeNode(self.level + 1)
                    self.children[child_index].add_color(color)
                
                self.colors = []  # 清空当前节点的颜色
            
            def get_child_index(self, color):
                # 修正：OpenCV是BGR格式
                b, g, r = color  # B, G, R
                index = 0
                if r >= 128:  # R分量
                    index |= 4
                if g >= 128:  # G分量
                    index |= 2
                if b >= 128:  # B分量
                    index |= 1
                return index
        
        # 构建八叉树
        root = OctreeNode()
        for color in data:
            root.add_color(color)
        
        # 简化的分割逻辑
        def build_octree(node, max_colors):
            # 如果节点像素数很少或达到最大深度，停止分割
            if node.pixel_count <= max_colors or node.level >= 7:
                return [node] if node.pixel_count > 0 else []
            
            # 检查颜色是否足够分散，如果颜色太相似就不分割
            if len(node.colors) > 0:
                colors_array = np.array(node.colors)
                color_std = np.std(colors_array, axis=0)
                if np.all(color_std < 10):  # 如果所有通道的标准差都很小，说明颜色很相似
                    return [node] if node.pixel_count > 0 else []
            
            node.split()
            leaves = []
            for child in node.children:
                if child is not None:
                    leaves.extend(build_octree(child, max_colors))
            return leaves
        
        # 获取所有叶子节点
        leaves = build_octree(root, max_colors)
        leaves = [leaf for leaf in leaves if leaf.pixel_count > 0]
        
        # 按像素数量排序
        leaves.sort(key=lambda x: x.pixel_count, reverse=True)
        
        # 提取主色
        main_colors = []
        total_pixels = sum(leaf.pixel_count for leaf in leaves)
        
        for leaf in leaves[:max_colors]:
            color = leaf.get_average_color()
            percentage = leaf.pixel_count / total_pixels
            main_colors.append((color, percentage))
        
        if show:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 2))
            for i, (color, percent) in enumerate(main_colors):
                plt.subplot(1, len(main_colors), i+1)
                plt.axis('off')
                # 将BGR格式转换为RGB格式用于显示
                color_rgb = [color[2], color[1], color[0]]  # BGR -> RGB
                plt.imshow(np.ones((50, 50, 3), dtype=np.uint8) * color_rgb)
                # plt.title(f"BGR{color}\nRGB{color_rgb}\n{percent:.1%}")
            plt.suptitle("Main Colors (Octree Method)")
            plt.show()
        
        return [color for color, _ in main_colors], [percent for _, percent in main_colors]

    # 颜色量化、CCV、颜色相关图等可继续补充