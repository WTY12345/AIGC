# Expression Detection Dataset

## 数据集概述

这是一个用于面部表情识别的 YOLO 训练数据集，包含 7 种基本表情类别。

## 表情类别

| ID  | 类别名称 | 英文名称  | 描述                         |
| --- | -------- | --------- | ---------------------------- |
| 0   | 愤怒     | anger     | 眉毛紧皱，嘴角向下，眼神严厉 |
| 1   | 轻蔑     | contempt  | 单侧嘴角上扬，眼神不屑       |
| 2   | 厌恶     | disgust   | 鼻子皱起，嘴角向下，眼神回避 |
| 3   | 恐惧     | fear      | 眉毛上扬，眼睛睁大，嘴巴微张 |
| 4   | 快乐     | happiness | 嘴角上扬，眼睛眯起，脸颊鼓起 |
| 5   | 中性     | neutral   | 面部肌肉放松，无明显表情     |
| 6   | 悲伤     | sadness   | 眉毛下垂，嘴角向下，眼神黯淡 |
| 7   | 惊讶     | surprise  | 眉毛上扬，眼睛睁大，嘴巴张开 |

## 数据集结构

```
Expression Detection/
├── images/
│   ├── train/          # 训练图像
│   ├── val/            # 验证图像
│   └── test/           # 测试图像
├── labels/
│   ├── train/          # 训练标注
│   ├── val/            # 验证标注
│   └── test/           # 测试标注
├── dataset.yaml        # 数据集配置
├── README.md           # 说明文档
└── tools/              # 工具脚本
    ├── annotation_tool.py
    ├── dataset_validator.py
    └── training_script.py
```

## 标注格式

使用 YOLO 格式，每行包含：

```
class_id center_x center_y width height
```

- `class_id`: 表情类别 ID (0-7)
- `center_x`: 边界框中心点 x 坐标 (归一化 0-1)
- `center_y`: 边界框中心点 y 坐标 (归一化 0-1)
- `width`: 边界框宽度 (归一化 0-1)
- `height`: 边界框高度 (归一化 0-1)

## 标注要求

1. **边界框精度**

   - 边界框应紧贴面部轮廓
   - 确保面部完整可见
   - 避免包含过多背景

2. **特殊情况处理**

   - 部分遮挡：标注可见部分
   - 多个人脸：分别标注
   - 小目标：确保至少 10x10 像素
   - 边界人脸：允许边界框超出图像

3. **质量要求**
   - 图像清晰，避免模糊
   - 表情明显，易于识别
   - 光照条件良好
   - 避免极端角度

## 数据集分割建议

- **训练集**: 70% (用于模型训练)
- **验证集**: 20% (用于模型验证和调参)
- **测试集**: 10% (用于最终性能评估)

## 使用工具

### 1. 标注工具

```bash
python tools/annotation_tool.py
```

### 2. 数据集验证

```bash
python tools/dataset_validator.py
```

### 3. 训练模型

```bash
python tools/training_script.py
```

## 数据增强建议

- 水平翻转 (flip_left_right)
- 亮度调整 (brightness)
- 对比度调整 (contrast)
- 小角度旋转 (±15 度)
- 噪声添加 (gaussian_noise)

## 性能指标

目标性能指标：

- mAP@0.5: > 0.85
- mAP@0.5:0.95: > 0.70
- 推理速度: < 50ms (GPU)
- 模型大小: < 50MB

## 注意事项

1. 确保各类别样本数量相对均衡
2. 定期检查标注质量
3. 使用验证脚本检查数据完整性
4. 保持数据集版本控制

## 更新日志

- v1.0 (2024): 初始版本，包含 7 种基本表情类别
