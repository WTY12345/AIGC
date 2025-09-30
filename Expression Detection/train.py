import os
from ultralytics import YOLO


def train_yolo():
    """
    YOLO模型推理函数
    """
    # 加载预训练模型
    model = YOLO("yolo11n.yaml")

    # 训练代码（已注释）
    result = model.train(
        data="D://AIGC/Expression Detection/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="cpu",
        patience=30,
        save=True,
        plots=True,
    )
    # 保存当前模型
    model.save("D://AIGC/Expression Detection/model001.pt")


if __name__ == "__main__":

    train_yolo()
