#!/usr/bin/env python3
"""
表情识别项目快速开始脚本
提供完整的项目使用指南和示例
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse


class ExpressionDetectionQuickStart:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tools_dir = self.project_root / "tools"

        # 表情类别
        self.expressions = {
            0: "anger",
            1: "contempt",
            2: "disgust",
            3: "fear",
            4: "happiness",
            5: "neutral",
            6: "sadness",
            7: "surprise",
        }

    def print_banner(self):
        """打印项目横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    Expression Detection                      ║
║                    表情识别项目                              ║
║                                                              ║
║  支持7种基本表情: anger, contempt, disgust, fear,           ║
║  happiness, neutral, sadness, surprise                       ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def check_environment(self):
        """检查环境依赖"""
        print("检查环境依赖...")

        required_packages = [
            "ultralytics",
            "opencv-python",
            "numpy",
            "matplotlib",
            "pillow",
            "pyyaml",
            "torch",
            "torchvision",
        ]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✓ {package}")
            except ImportError:
                print(f"❌ {package}")
                missing_packages.append(package)

        if missing_packages:
            print(f"\n缺少依赖包: {', '.join(missing_packages)}")
            print("请运行以下命令安装:")
            print(f"pip install {' '.join(missing_packages)}")
            return False

        print("✓ 环境检查通过")
        return True

    def show_project_structure(self):
        """显示项目结构"""
        print("\n项目结构:")
        structure = """
Expression Detection/
├── images/                    # 图像数据
│   ├── train/                # 训练图像
│   ├── val/                  # 验证图像
│   └── test/                 # 测试图像
├── labels/                   # 标注数据
│   ├── train/                # 训练标注
│   ├── val/                  # 验证标注
│   └── test/                 # 测试标注
├── tools/                    # 工具脚本
│   ├── annotation_tool.py    # 标注工具
│   ├── dataset_validator.py  # 数据集验证
│   ├── training_script.py    # 训练脚本
│   └── inference_demo.py     # 推理演示
├── dataset.yaml              # 数据集配置
├── README.md                 # 项目说明
└── quick_start.py            # 快速开始脚本
        """
        print(structure)

    def show_usage_guide(self):
        """显示使用指南"""
        print("\n使用指南:")
        print("=" * 50)

        print("\n1. 数据准备:")
        print("   - 将图像放入 images/train, images/val, images/test")
        print("   - 使用标注工具创建对应的标注文件")
        print("   - 运行数据集验证确保数据质量")

        print("\n2. 模型训练:")
        print("   python tools/training_script.py --dataset . --epochs 100")

        print("\n3. 模型推理:")
        print("   python tools/inference_demo.py --model best.pt --input image.jpg")
        print("   python tools/inference_demo.py --model best.pt --webcam")

        print("\n4. 工具使用:")
        print("   python tools/annotation_tool.py      # 标注工具")
        print("   python tools/dataset_validator.py    # 数据验证")

    def create_sample_data(self):
        """创建示例数据"""
        print("\n创建示例数据...")

        # 创建示例图像目录
        sample_dir = self.project_root / "sample_data"
        sample_dir.mkdir(exist_ok=True)

        # 创建示例标注文件
        sample_annotations = [
            "4 0.5 0.5 0.3 0.4  # happiness",
            "5 0.5 0.5 0.3 0.4  # neutral",
            "6 0.5 0.5 0.3 0.4  # sadness",
        ]

        for i, annotation in enumerate(sample_annotations):
            sample_file = sample_dir / f"sample_{i+1}.txt"
            with open(sample_file, "w") as f:
                f.write(annotation.split("#")[0].strip() + "\n")

        print(f"✓ 示例数据已创建: {sample_dir}")

    def run_annotation_tool(self):
        """运行标注工具"""
        print("\n启动标注工具...")
        try:
            subprocess.run([sys.executable, str(self.tools_dir / "annotation_tool.py")])
        except Exception as e:
            print(f"❌ 启动标注工具失败: {e}")

    def run_dataset_validator(self):
        """运行数据集验证"""
        print("\n运行数据集验证...")
        try:
            subprocess.run([sys.executable, str(self.tools_dir / "dataset_validator.py")])
        except Exception as e:
            print(f"❌ 运行数据集验证失败: {e}")

    def run_training_demo(self):
        """运行训练演示"""
        print("\n运行训练演示...")
        print("注意: 这需要有效的训练数据")

        try:
            cmd = [
                sys.executable,
                str(self.tools_dir / "training_script.py"),
                "--dataset",
                str(self.project_root),
                "--epochs",
                "10",
                "--batch",
                "8",
            ]
            subprocess.run(cmd)
        except Exception as e:
            print(f"❌ 运行训练演示失败: {e}")

    def show_expression_classes(self):
        """显示表情类别"""
        print("\n表情类别:")
        print("=" * 30)
        for class_id, expression in self.expressions.items():
            print(f"{class_id}: {expression}")

    def interactive_menu(self):
        """交互式菜单"""
        while True:
            print("\n" + "=" * 50)
            print("表情识别项目 - 交互式菜单")
            print("=" * 50)
            print("1. 显示项目结构")
            print("2. 显示使用指南")
            print("3. 显示表情类别")
            print("4. 检查环境依赖")
            print("5. 创建示例数据")
            print("6. 启动标注工具")
            print("7. 运行数据集验证")
            print("8. 运行训练演示")
            print("9. 退出")

            choice = input("\n请选择操作 (1-9): ").strip()

            if choice == "1":
                self.show_project_structure()
            elif choice == "2":
                self.show_usage_guide()
            elif choice == "3":
                self.show_expression_classes()
            elif choice == "4":
                self.check_environment()
            elif choice == "5":
                self.create_sample_data()
            elif choice == "6":
                self.run_annotation_tool()
            elif choice == "7":
                self.run_dataset_validator()
            elif choice == "8":
                self.run_training_demo()
            elif choice == "9":
                print("退出程序")
                break
            else:
                print("无效选择，请重新输入")

    def run_quick_setup(self):
        """快速设置"""
        print("开始快速设置...")

        # 1. 检查环境
        if not self.check_environment():
            print("请先安装依赖包")
            return False

        # 2. 创建示例数据
        self.create_sample_data()

        # 3. 显示使用指南
        self.show_usage_guide()

        print("\n✓ 快速设置完成!")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="表情识别项目快速开始")
    parser.add_argument("--setup", action="store_true", help="运行快速设置")
    parser.add_argument("--menu", action="store_true", help="启动交互式菜单")
    parser.add_argument("--check-env", action="store_true", help="检查环境依赖")
    parser.add_argument("--show-guide", action="store_true", help="显示使用指南")

    args = parser.parse_args()

    # 创建快速开始实例
    quick_start = ExpressionDetectionQuickStart()
    quick_start.print_banner()

    if args.setup:
        quick_start.run_quick_setup()
    elif args.menu:
        quick_start.interactive_menu()
    elif args.check_env:
        quick_start.check_environment()
    elif args.show_guide:
        quick_start.show_usage_guide()
    else:
        # 默认显示使用指南
        quick_start.show_usage_guide()
        print("\n使用 --help 查看所有选项")
        print("使用 --menu 启动交互式菜单")


if __name__ == "__main__":
    main()
