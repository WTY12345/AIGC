#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的GUI界面
用于展示YOLO + 美学评估结果
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from complete_system import CompleteAestheticSystem
from simple_evaluator import SimpleAestheticEvaluator


class AestheticGUI:
    """美学评估GUI界面"""

    def __init__(self, root):
        """初始化GUI"""
        self.root = root
        self.root.title("YOLO + 美学评估系统")
        self.root.geometry("800x600")

        # 初始化系统
        self.system = None
        self.aesthetic_evaluator = SimpleAestheticEvaluator()

        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 标题
        title_label = ttk.Label(main_frame, text="🎨 YOLO + 美学评估系统", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="📁 选择图像", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_entry.grid(row=0, column=0, padx=(0, 10))

        browse_btn = ttk.Button(file_frame, text="浏览", command=self.browse_file)
        browse_btn.grid(row=0, column=1)

        # 分析按钮
        analyze_btn = ttk.Button(file_frame, text="🔍 分析图像", command=self.analyze_image)
        analyze_btn.grid(row=0, column=2, padx=(10, 0))

        # 批量分析按钮
        batch_btn = ttk.Button(file_frame, text="📊 批量分析", command=self.batch_analyze)
        batch_btn.grid(row=0, column=3, padx=(10, 0))

        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="📊 分析结果", padding="10")
        result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # 创建滚动文本框
        self.result_text = tk.Text(result_frame, height=20, width=80, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

    def browse_file(self):
        """浏览文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件", filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def analyze_image(self):
        """分析单张图像"""
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("错误", "请选择有效的图像文件")
            return

        # 在新线程中运行分析
        thread = threading.Thread(target=self._analyze_image_thread, args=(file_path,))
        thread.daemon = True
        thread.start()

    def _analyze_image_thread(self, file_path):
        """在后台线程中分析图像"""
        try:
            self.status_var.set("正在分析图像...")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"正在分析: {os.path.basename(file_path)}\n")
            self.result_text.insert(tk.END, "请稍候...\n\n")
            self.root.update()

            # 执行分析
            if self.system is None:
                self.system = CompleteAestheticSystem()

            result = self.system.analyze_image(file_path, show_result=False)

            # 显示结果
            self._display_result(result)
            self.status_var.set("分析完成")

        except Exception as e:
            self.result_text.insert(tk.END, f"❌ 分析失败: {e}\n")
            self.status_var.set("分析失败")

    def _display_result(self, result):
        """显示分析结果"""
        self.result_text.delete(1.0, tk.END)

        # 基本信息
        self.result_text.insert(tk.END, f"📁 文件: {result['filename']}\n")
        self.result_text.insert(tk.END, f"⏰ 时间: {result['timestamp']}\n\n")

        # 检测结果
        detection = result["detection"]
        self.result_text.insert(tk.END, "🎯 目标检测结果:\n")
        if "objects" in detection and detection["objects"]:
            self.result_text.insert(tk.END, f"  检测到 {len(detection['objects'])} 个目标:\n")
            for i, obj in enumerate(detection["objects"], 1):
                self.result_text.insert(tk.END, f"    {i}. {obj['class_name']} (置信度: {obj['confidence']:.3f})\n")
        else:
            self.result_text.insert(tk.END, "  未检测到目标\n")

        self.result_text.insert(tk.END, "\n")

        # 美学评估结果
        aesthetic = result["aesthetic"]
        if "total" in aesthetic:
            self.result_text.insert(tk.END, "🎨 美学评估结果:\n")
            self.result_text.insert(tk.END, f"  📊 总分: {aesthetic['total']}/100\n")
            self.result_text.insert(tk.END, f"  💡 亮度: {aesthetic['brightness']}/100\n")
            self.result_text.insert(tk.END, f"  ⚖️ 对比度: {aesthetic['contrast']}/100\n")
            self.result_text.insert(tk.END, f"  🌈 色彩平衡: {aesthetic['color_balance']}/100\n")
            self.result_text.insert(tk.END, f"  📐 构图: {aesthetic['composition']}/100\n\n")

            # 综合评级
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

            self.result_text.insert(tk.END, f"🏆 综合评级: {grade}\n")
            self.result_text.insert(tk.END, f"💬 评价: {comment}\n\n")

        # 改进建议
        if result["recommendations"]:
            self.result_text.insert(tk.END, "💡 改进建议:\n")
            for rec in result["recommendations"]:
                self.result_text.insert(tk.END, f"  {rec}\n")

    def batch_analyze(self):
        """批量分析"""
        dir_path = filedialog.askdirectory(title="选择图像目录")
        if not dir_path:
            return

        # 在新线程中运行批量分析
        thread = threading.Thread(target=self._batch_analyze_thread, args=(dir_path,))
        thread.daemon = True
        thread.start()

    def _batch_analyze_thread(self, dir_path):
        """在后台线程中进行批量分析"""
        try:
            self.status_var.set("正在进行批量分析...")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"正在批量分析目录: {dir_path}\n")
            self.result_text.insert(tk.END, "请稍候...\n\n")
            self.root.update()

            # 执行批量分析
            if self.system is None:
                self.system = CompleteAestheticSystem()

            results = self.system.batch_analyze(dir_path, "gui_batch_results.txt")

            # 显示统计结果
            self._display_batch_results(results)
            self.status_var.set("批量分析完成")

        except Exception as e:
            self.result_text.insert(tk.END, f"❌ 批量分析失败: {e}\n")
            self.status_var.set("批量分析失败")

    def _display_batch_results(self, results):
        """显示批量分析结果"""
        self.result_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END, "📊 批量分析结果统计\n")
        self.result_text.insert(tk.END, "=" * 50 + "\n\n")

        # 基本统计
        self.result_text.insert(
            tk.END, f"📁 分析目录: {os.path.basename(results[0]['image_path']) if results else 'N/A'}\n"
        )
        self.result_text.insert(tk.END, f"📊 分析图像数: {len(results)}\n\n")

        # 美学评分统计
        aesthetic_results = [r for r in results if "total" in r["aesthetic"]]
        if aesthetic_results:
            total_scores = [r["aesthetic"]["total"] for r in aesthetic_results]
            avg_score = sum(total_scores) / len(total_scores)
            max_score = max(total_scores)
            min_score = min(total_scores)

            self.result_text.insert(tk.END, "🎨 美学评分统计:\n")
            self.result_text.insert(tk.END, f"  平均评分: {avg_score:.2f}\n")
            self.result_text.insert(tk.END, f"  最高评分: {max_score:.2f}\n")
            self.result_text.insert(tk.END, f"  最低评分: {min_score:.2f}\n\n")

            # 评级分布
            excellent = sum(1 for score in total_scores if score >= 80)
            good = sum(1 for score in total_scores if 60 <= score < 80)
            fair = sum(1 for score in total_scores if 40 <= score < 60)
            poor = sum(1 for score in total_scores if score < 40)

            self.result_text.insert(tk.END, "🏆 评级分布:\n")
            self.result_text.insert(tk.END, f"  优秀 (≥80): {excellent} 张\n")
            self.result_text.insert(tk.END, f"  良好 (60-79): {good} 张\n")
            self.result_text.insert(tk.END, f"  一般 (40-59): {fair} 张\n")
            self.result_text.insert(tk.END, f"  较差 (<40): {poor} 张\n\n")

        # 检测统计
        total_detections = 0
        for result in results:
            if "objects" in result["detection"]:
                total_detections += len(result["detection"]["objects"])

        avg_detections = total_detections / len(results) if results else 0

        self.result_text.insert(tk.END, "🎯 目标检测统计:\n")
        self.result_text.insert(tk.END, f"  平均检测目标数: {avg_detections:.2f}\n")
        self.result_text.insert(tk.END, f"  总检测目标数: {total_detections}\n\n")

        # 详细结果
        self.result_text.insert(tk.END, "📋 详细结果:\n")
        for result in results:
            filename = result["filename"]
            aesthetic = result["aesthetic"]
            detection = result["detection"]

            self.result_text.insert(tk.END, f"  {filename}:\n")
            if "total" in aesthetic:
                self.result_text.insert(tk.END, f"    美学评分: {aesthetic['total']}\n")
            if "objects" in detection:
                self.result_text.insert(tk.END, f"    检测目标: {len(detection['objects'])}\n")
            self.result_text.insert(tk.END, "\n")

        self.result_text.insert(tk.END, "📄 详细结果已保存到: gui_batch_results.txt\n")


def main():
    """主函数"""
    root = tk.Tk()
    app = AestheticGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
