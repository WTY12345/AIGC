#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO环境测试脚本
用于验证环境是否正确配置
"""

import sys
import os


def test_environment():
    """测试YOLO环境配置"""
    print("=" * 50)
    print("YOLO环境测试")
    print("=" * 50)

    # 检查Python版本
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")

    # 检查工作目录
    print(f"当前工作目录: {os.getcwd()}")

    # 检查模型文件
    model_files = ["3times.pt", "yolo11n.pt"]
    print("\n模型文件检查:")
    for model in model_files:
        if os.path.exists(model):
            print(f"✅ {model} - 存在")
        else:
            print(f"❌ {model} - 不存在")

    # 检查图片文件
    img_dir = "img"
    print(f"\n图片目录检查:")
    if os.path.exists(img_dir):
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"✅ {img_dir} - 存在，包含 {len(img_files)} 个图片文件")
        for img in img_files:
            print(f"  - {img}")
    else:
        print(f"❌ {img_dir} - 不存在")

    # 检查依赖包
    print(f"\n依赖包检查:")
    required_packages = ["ultralytics", "torch", "torchvision", "opencv-python", "numpy", "pillow", "matplotlib"]

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} - 已安装")
        except ImportError:
            print(f"❌ {package} - 未安装")

    print("\n" + "=" * 50)
    print("环境测试完成")
    print("=" * 50)


if __name__ == "__main__":
    test_environment()



