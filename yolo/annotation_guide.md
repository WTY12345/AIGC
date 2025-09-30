# YOLO 数据标注完整指南

## 1. 标注格式说明

### YOLO 格式特点

- **归一化坐标**：所有坐标值都在 0-1 之间
- **中心点表示**：使用边界框的中心点和宽高
- **类别 ID**：从 0 开始的整数

### 标注文件格式

```
class_id center_x center_y width height
```

### 坐标转换公式

```python
# 归一化坐标 → 像素坐标
center_x_pixel = center_x_norm * image_width
center_y_pixel = center_y_norm * image_height
width_pixel = width_norm * image_width
height_pixel = height_norm * image_height

# 边界框坐标
x1 = center_x_pixel - width_pixel / 2
y1 = center_y_pixel - height_pixel / 2
x2 = center_x_pixel + width_pixel / 2
y2 = center_y_pixel + height_pixel / 2
```

## 2. 标注工具推荐

### 2.1 LabelImg (推荐新手)

```bash
# 安装
pip install labelimg

# 启动
labelimg
```

**使用步骤：**

1. 打开图像文件夹
2. 选择 YOLO 格式
3. 开始标注
4. 保存标注文件

**快捷键：**

- `W`：创建边界框
- `D`：下一张图像
- `A`：上一张图像
- `Del`：删除选中标注

### 2.2 Roboflow (推荐专业用户)

**网址：** https://roboflow.com

**优势：**

- 自动标注功能
- 数据增强
- 团队协作
- 直接导出 YOLO 格式

### 2.3 CVAT (企业级)

```bash
# Docker安装
docker run -it -p 8080:8080 openvino/cvat

# 访问
http://localhost:8080
```

## 3. 标注质量标准

### 3.1 边界框要求

- ✅ 紧贴目标边缘
- ✅ 包含完整目标
- ✅ 避免包含背景
- ❌ 不要过大或过小

### 3.2 类别标注

- ✅ 类别定义清晰
- ✅ 标注一致性
- ✅ 处理遮挡情况
- ❌ 避免类别混淆

### 3.3 特殊情况处理

- **部分遮挡**：标注可见部分
- **多目标重叠**：分别标注
- **小目标**：确保至少 10x10 像素
- **边界目标**：允许边界框超出图像

## 4. 数据集组织

### 4.1 目录结构

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

### 4.2 数据集分割建议

- **训练集**：70-80%
- **验证集**：15-20%
- **测试集**：5-10%

### 4.3 数据增强

```python
# 常见增强技术
- 水平翻转
- 旋转（小角度）
- 亮度调整
- 对比度调整
- 噪声添加
```

## 5. 标注验证

### 5.1 使用验证脚本

```bash
python annotation_validator.py
```

### 5.2 手动检查要点

- 坐标范围是否正确（0-1）
- 边界框是否合理
- 类别 ID 是否正确
- 文件配对是否完整

## 6. 常见问题解决

### 6.1 坐标超出范围

**问题：** 坐标值大于 1 或小于 0
**解决：** 检查图像尺寸和标注工具设置

### 6.2 边界框异常

**问题：** 边界框过大或过小
**解决：** 重新标注，确保紧贴目标

### 6.3 类别 ID 错误

**问题：** 类别 ID 不连续或超出范围
**解决：** 统一类别定义，从 0 开始编号

### 6.4 文件配对问题

**问题：** 图像和标注文件不匹配
**解决：** 检查文件名一致性

## 7. 最佳实践

### 7.1 标注流程

1. **准备数据**：整理图像，定义类别
2. **选择工具**：根据需求选择标注工具
3. **开始标注**：按照质量标准标注
4. **质量检查**：使用验证脚本检查
5. **数据集分割**：合理分配训练/验证/测试集
6. **配置 YAML**：设置数据集配置文件

### 7.2 效率提升技巧

- 使用快捷键操作
- 批量处理相似图像
- 利用自动标注功能
- 建立标注规范文档

### 7.3 团队协作

- 统一标注标准
- 定期质量检查
- 使用版本控制
- 建立反馈机制

## 8. 数据集配置文件示例

```yaml
# dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 3 # 类别数量
names: ["person", "car", "bicycle"] # 类别名称
```

## 9. 验证和测试

### 9.1 数据完整性检查

```bash
# 检查文件数量
find images/train -name "*.jpg" | wc -l
find labels/train -name "*.txt" | wc -l
```

### 9.2 标注质量检查

```bash
# 运行验证脚本
python annotation_validator.py
```

### 9.3 可视化检查

```python
# 可视化标注结果
validator.visualize_annotations(image_path, label_path, output_path)
```

## 10. 性能优化建议

### 10.1 数据平衡

- 确保各类别样本数量相对均衡
- 避免某个类别样本过少

### 10.2 图像质量

- 使用高分辨率图像
- 避免模糊、过暗的图像
- 保持图像多样性

### 10.3 标注精度

- 标注越精确，模型性能越好
- 定期检查标注质量
- 建立标注审核机制

