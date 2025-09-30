import os
from ultralytics import YOLO


def train_yolo():
    """
    YOLO模型推理函数
    """
    # 加载预训练模型
    model = YOLO("D://AIGC/yolo/3times.pt")

    # 训练代码（已注释）
    # result = model.train(
    #     data="coco128.yaml",
    #     epochs=3,
    #     imgsz=640,
    #     batch=16,
    #     workers=0
    # )

    # 使用相对路径进行推理
    img_path = os.path.join("D://AIGC/yolo", "001.jpg")
    results = model(img_path)
    results[0].show()
    return results


if __name__ == "__main__":

    train_yolo()
