@echo off
echo 正在激活YOLO环境...
call conda activate yolo-env
echo YOLO环境已激活！
echo.
echo 可用命令:
echo   python main.py          - 运行YOLO推理
echo   python video.py         - 运行视频处理
echo   conda deactivate        - 退出环境
echo.
cmd /k



