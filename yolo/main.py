
from ultralytics import YOLO

def train_yolo():


    model = YOLO('3times.pt')
    

    # result = model.train(
    #     data="coco128.yaml", 
    #     epochs=3, 
    #     imgsz=640, 
    #     batch=16,
    #     workers=0
    # )
    results=model("D://AIGC/yolo/img/12.jpg")
    results[0].show()
    return results

if __name__ == "__main__":

    train_yolo()