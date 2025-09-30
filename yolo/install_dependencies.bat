@echo off
echo 正在安装YOLO依赖包...
echo.

echo 方案1: 使用pip安装
call conda activate yolo-env
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo pip安装失败，尝试使用国内镜像源...
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
    pip install -r requirements.txt
)

if %errorlevel% neq 0 (
    echo.
    echo 网络问题，请手动下载安装包或检查网络连接
    pause
    exit /b 1
)

echo.
echo 依赖包安装完成！
echo 运行 "python main.py" 测试YOLO环境
pause



