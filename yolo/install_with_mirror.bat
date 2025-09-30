@echo off
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
