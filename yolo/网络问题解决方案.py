#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO环境网络问题解决方案
提供多种安装方法和下载链接
"""

import os
import sys
import webbrowser
from pathlib import Path


def print_solution():
    """打印解决方案"""
    print("=" * 60)
    print("🔧 YOLO环境网络问题解决方案")
    print("=" * 60)

    print("\n📋 当前问题分析:")
    print("❌ pip安装失败 - 网络连接问题")
    print("❌ conda安装失败 - 权限问题")
    print("✅ 环境已创建 - yolo-env")
    print("✅ 代码已修复 - 路径问题")
    print("✅ 配置文件已创建 - 完整文档")

    print("\n🚀 推荐解决方案 (按优先级排序):")

    print("\n1️⃣ 方案一：使用国内镜像源 (最推荐)")
    print("   执行以下命令:")
    print("   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple")
    print("   pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn")
    print("   pip install ultralytics torch torchvision opencv-python")

    print("\n2️⃣ 方案二：使用conda-forge (备选)")
    print("   执行以下命令:")
    print("   conda install -c conda-forge ultralytics pytorch torchvision opencv -y")

    print("\n3️⃣ 方案三：离线安装包 (网络问题严重时)")
    print("   手动下载以下文件:")
    print("   - PyTorch CPU: https://download.pytorch.org/whl/torch_stable.html")
    print("   - ultralytics: https://pypi.org/project/ultralytics/#files")
    print("   - opencv: https://pypi.org/project/opencv-python/#files")

    print("\n4️⃣ 方案四：使用代理或更换网络")
    print("   - 尝试使用手机热点")
    print("   - 配置代理设置")
    print("   - 联系网络管理员")

    print("\n🧪 验证安装:")
    print("   python test_environment.py")
    print("   python -c \"from ultralytics import YOLO; print('成功!')\"")

    print("\n📞 技术支持:")
    print("   如果问题持续，请联系: 2918932790@qq.com")

    print("\n" + "=" * 60)


def create_install_script():
    """创建安装脚本"""
    script_content = """@echo off
echo 正在配置YOLO环境...
echo.

echo 方案1: 配置国内镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

echo.
echo 方案2: 安装依赖包
pip install ultralytics torch torchvision opencv-python numpy pillow matplotlib

if %errorlevel% neq 0 (
    echo.
    echo 网络问题，尝试conda安装...
    conda install -c conda-forge ultralytics pytorch torchvision opencv -y
)

echo.
echo 测试安装结果...
python test_environment.py

echo.
echo 安装完成！按任意键退出...
pause
"""

    with open("install_with_mirror.bat", "w", encoding="utf-8") as f:
        f.write(script_content)

    print("✅ 已创建 install_with_mirror.bat 脚本")


def open_download_links():
    """打开下载链接"""
    links = [
        "https://pypi.tuna.tsinghua.edu.cn/simple/",
        "https://download.pytorch.org/whl/torch_stable.html",
        "https://pypi.org/project/ultralytics/#files",
    ]

    print("\n🌐 正在打开下载链接...")
    for link in links:
        try:
            webbrowser.open(link)
        except:
            print(f"无法打开链接: {link}")

    print("✅ 下载链接已在浏览器中打开")


def main():
    """主函数"""
    print_solution()
    create_install_script()

    # 询问是否打开下载链接
    response = input("\n是否打开下载链接? (y/n): ").lower().strip()
    if response in ["y", "yes", "是"]:
        open_download_links()

    print("\n🎯 下一步操作:")
    print("1. 双击运行 install_with_mirror.bat")
    print("2. 或者手动执行上面的命令")
    print("3. 运行 python test_environment.py 验证")


if __name__ == "__main__":
    main()



