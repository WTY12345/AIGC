# 人脸表情识别项目

这是一个基于YOLO的人脸表情识别项目，可以检测和识别7种基本表情：快乐、悲伤、愤怒、惊讶、恐惧、厌恶和中性。

## 项目结构

```
yolo/
├── face_expression_config.yaml    # 训练配置文件
├── face_expression_utils.py       # 数据预处理工具
├── face_expression_train.py       # 模型训练脚本
├── face_expression_detect.py      # 实时检测脚本
├── datasets/
│   └── face_expression/           # 数据集目录
│       ├── images/
│       │   ├── train/             # 训练图片
│       │   ├── val/               # 验证图片
│       │   └── test/              # 测试图片
│       └── labels/
│           ├── train/             # 训练标签
│           ├── val/               # 验证标签
│           └── test/              # 测试标签
└── runs/                          # 训练结果目录
    └── detect/
        └── face_expression_train/
            └── weights/
                ├── best.pt        # 最佳模型
                └── last.pt        # 最新模型
```

## 快速开始

### 1. 环境准备

确保已安装必要的依赖：

```bash
pip install ultralytics opencv-python torch torchvision
```

### 2. 创建示例数据集

```bash
python face_expression_utils.py
```

这将创建示例数据集和必要的目录结构。

### 3. 训练模型

```bash
python face_expression_train.py
```

训练过程会：
- 加载预训练模型
- 在示例数据集上训练
- 保存最佳模型到 `runs/detect/face_expression_train/weights/best.pt`

### 4. 实时检测

使用摄像头进行实时表情检测：

```bash
python face_expression_detect.py
```

检测图片：

```bash
python face_expression_detect.py --mode image --source path/to/image.jpg --output result.jpg
```

检测视频：

```bash
python face_expression_detect.py --mode video --source path/to/video.mp4 --output result.mp4
```

## 详细使用说明

### 数据准备

1. **收集表情图片**：
   - 将不同表情的人脸图片放入 `datasets/face_expression/images/train/` 目录
   - 建议每个表情至少100张图片

2. **标注数据**：
   - 使用标注工具（如LabelImg）标注人脸边界框
   - 标签格式为YOLO格式：`class_id x_center y_center width height`
   - 表情类别ID：
     - 0: happy (快乐)
     - 1: sad (悲伤)
     - 2: angry (愤怒)
     - 3: surprised (惊讶)
     - 4: fearful (恐惧)
     - 5: disgusted (厌恶)
     - 6: neutral (中性)

3. **数据分割**：
   ```python
   from face_expression_utils import FaceExpressionUtils
   
   utils = FaceExpressionUtils("datasets/face_expression")
   utils.split_dataset(train_ratio=0.8, val_ratio=0.1)
   ```

### 模型训练

#### 基础训练

```python
from face_expression_train import FaceExpressionTrainer

trainer = FaceExpressionTrainer()
trainer.load_config()
trainer.prepare_model(model_size="n")  # 使用nano模型
trainer.train(epochs=100, batch=16, device="cuda")
```

#### 高级训练参数

```python
trainer.train(
    epochs=200,           # 训练轮数
    imgsz=640,           # 输入图片尺寸
    batch=32,            # 批次大小
    lr0=0.01,            # 初始学习率
    patience=50,         # 早停耐心值
    device="cuda"        # 使用GPU
)
```

### 模型评估

```python
# 验证模型性能
trainer.validate()

# 测试单张图片
trainer.test_single_image("test_image.jpg")

# 导出模型
trainer.export_model(formats=['onnx', 'torchscript'])
```

### 实时检测

#### 摄像头检测

```python
from face_expression_detect import FaceExpressionDetector

detector = FaceExpressionDetector()
detector.load_model("runs/detect/face_expression_train/weights/best.pt")
detector.detect_video(0)  # 使用默认摄像头
```

#### 图片检测

```python
detector.detect_image(
    image_path="test.jpg",
    output_path="result.jpg",
    conf_threshold=0.5
)
```

#### 视频检测

```python
detector.detect_video(
    video_path="input.mp4",
    output_path="output.mp4",
    conf_threshold=0.5
)
```

## 配置说明

### 训练配置 (face_expression_config.yaml)

```yaml
# 数据集路径
path: D:/AIGC/yolo/datasets/face_expression
train: images/train
val: images/val
test: images/test

# 类别配置
nc: 7  # 表情类别数量
names: ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']

# 数据增强参数
augment: true
hsv_h: 0.015    # 色调变化
hsv_s: 0.7      # 饱和度变化
hsv_v: 0.4      # 明度变化
fliplr: 0.5     # 左右翻转概率
mosaic: 1.0     # 马赛克增强
```

### 检测参数

- `conf_threshold`: 置信度阈值，默认0.5
- `model_path`: 模型文件路径
- `source`: 输入源（摄像头索引或文件路径）
- `output`: 输出文件路径

## 性能优化

### 1. 模型选择

- **YOLO11n**: 最快，精度较低，适合实时检测
- **YOLO11s**: 平衡速度和精度
- **YOLO11m**: 精度较高，速度较慢
- **YOLO11l**: 高精度，适合离线处理
- **YOLO11x**: 最高精度，速度最慢

### 2. 输入尺寸

- 640x640: 标准尺寸，平衡速度和精度
- 416x416: 更快，精度略低
- 832x832: 更慢，精度更高

### 3. 批次大小

根据GPU内存调整：
- 4GB GPU: batch=8
- 8GB GPU: batch=16
- 16GB+ GPU: batch=32

## 常见问题

### Q: 训练时出现内存不足错误
A: 减少批次大小或输入图片尺寸

### Q: 检测精度不高
A: 增加训练数据，调整数据增强参数，或使用更大的模型

### Q: 实时检测速度慢
A: 使用YOLO11n模型，减少输入尺寸，或使用GPU加速

### Q: 无法检测到人脸
A: 检查图片质量，调整置信度阈值，或重新训练模型

## 扩展功能

### 1. 添加新的表情类别

1. 修改 `face_expression_config.yaml` 中的 `nc` 和 `names`
2. 更新 `face_expression_utils.py` 中的类别列表
3. 重新标注数据
4. 重新训练模型

### 2. 集成到其他应用

```python
# 在您的应用中使用
detector = FaceExpressionDetector()
detector.load_model("path/to/model.pt")

# 检测单帧
frame = cv2.imread("image.jpg")
annotated_frame, detections = detector.detect_expression(frame)

# 处理检测结果
for det in detections:
    expression = det['expression']
    confidence = det['confidence']
    bbox = det['bbox']
    print(f"检测到表情: {expression}, 置信度: {confidence}")
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！
