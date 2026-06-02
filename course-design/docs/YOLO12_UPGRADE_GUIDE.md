# YOLO12 升级指南

**文档版本**: 1.0
**更新日期**: 2026-04-25
**适用版本**: YOLO12 及更高版本

---

## 一、YOLO12 简介

### 1.1 发布信息
- **发布日期**: 2025年2月18日
- **开发团队**: 纽约州立大学布法罗分校 + 中国科学院
- **架构创新**: 以注意力机制为中心（Attention-Centric Architecture）

### 1.2 与 YOLOv8 的主要区别

| 特性 | YOLOv8 | YOLO12 | 改进幅度 |
|------|--------|--------|----------|
| 架构基础 | CNN | Attention + CNN | +42% 效率提升 |
| mAP@50 | ~37.3 | ~48.0 (s模型) | +28.7% |
| 召回率 | ~89% | ~95% | +6.7% |
| F1分数 | ~94% | ~96.03% | +2.2% |
| 参数量 | 3.2M (n) | 9.3M (s) | 更强表征 |
| 推理延迟 | ~2.8ms | ~2.61ms | -6.8% |

### 1.3 可用模型变体

```
YOLO12n  - Nano      (最小,最快,精度较低)
YOLO12s  - Small     (推荐,平衡)
YOLO12m  - Medium    (较高精度,较慢)
YOLO12l  - Large     (高精度,慢)
YOLO12x  - Extra-Large (最高精度,最慢)
```

---

## 二、升级步骤

### 2.1 检查当前环境

```bash
# 检查 ultralytics 版本
pip show ultralytics

# 更新 ultralytics 到最新版本
pip install --upgrade ultralytics

# 验证安装
python -c "from ultralytics import YOLO; print(YOLO('yolo12n.pt'))"
```

### 2.2 下载 YOLO12 模型

#### 方法一：自动下载（推荐）
```bash
python scripts\upgrade_model.py --model yolo12s
```

#### 方法二：使用 Python 代码
```python
from ultralytics import YOLO

# 自动下载并加载模型
model = YOLO("yolo12s.pt")  # 自动从 GitHub 下载

# 或指定本地路径
model = YOLO("models/yolo12s.pt")
```

#### 方法三：手动下载
访问 Ultralytics GitHub releases 页面下载：
```
https://github.com/ultralytics/assets/releases
```

### 2.3 配置新模型

#### 更新配置文件
编辑 `configs/default.yaml`:

```yaml
model:
  path: "models/yolo12s.pt"  # 更新为 YOLO12 模型
  imgsz: 640                    # 输入图像大小
  conf: 0.35                    # 置信度阈值
  iou: 0.5                      # NMS IoU 阈值
  device: "auto"                # auto/cpu/cuda:0
```

---

## 三、兼容性检查

### 3.1 API 兼容性

YOLO12 与 YOLOv8 API 完全兼容，Ultralytics 保持了向后兼容性：

```python
from ultralytics import YOLO

# ✅ 相同的加载方式
model = YOLO("yolo12s.pt")

# ✅ 相同的推理方式
results = model.predict(source="image.jpg", conf=0.5)

# ✅ 相同的结果处理
for result in results:
    boxes = result.boxes  # 边界框
    masks = result.masks  # 分割掩码
    probs = result.probs  # 分类概率
```

### 3.2 代码适配检查清单

| 检查项 | YOLOv8 | YOLO12 | 状态 |
|--------|--------|--------|------|
| `model.predict()` | ✅ | ✅ | 兼容 |
| `model.train()` | ✅ | ✅ | 兼容 |
| `result.boxes` | ✅ | ✅ | 兼容 |
| `result.masks` | ✅ | ✅ | 兼容 |
| `result.probs` | ✅ | ✅ | 兼容 |
| `device` 参数 | ✅ | ✅ | 兼容 |
| `half()` 精度 | ✅ | ✅ | 兼容 |
| TensorRT 导出 | ✅ | ✅ | 兼容 |
| ONNX 导出 | ✅ | ✅ | 兼容 |

### 3.3 依赖要求

```
# 基础依赖
ultralytics >= 8.3.0  # YOLO12 需要更新版本

# 可选 GPU 加速
torch >= 2.0.0
torchvision >= 0.15.0
```

---

## 四、性能测试

### 4.1 基准测试命令

```bash
# 比较所有模型
python scripts/benchmark_models.py

# 只测试 YOLO12
python scripts/benchmark_models.py yolo12s.pt

# 与当前模型比较
python scripts/benchmark_models.py yolov8n.pt yolo12s.pt
```

---

## 五、升级后的改进

### 5.1 精度提升

- **更高的召回率**: 减少漏检，特别是遮挡场景
- **更好的小目标检测**: YOLO12 对小物体检测有显著提升
- **更强的泛化能力**: 注意力机制提供更好的特征学习

### 5.2 效率优化

- **更好的速度-精度平衡**: 相比 YOLOv8，在相同精度下更快
- **改进的 CUDA 优化**: 更好的 GPU 利用率
- **更高效的内存使用**: 优化的内存访问模式

---

## 六、潜在挑战

### 6.1 计算资源

| 模型 | 参数量 | 内存需求 (FP32) |
|------|--------|----------------|
| YOLOv8n | 3.2M | ~300MB |
| YOLO12n | ~5.4M | ~500MB |
| YOLOv8s | 11.2M | ~600MB |
| YOLO12s | 9.3M | ~550MB |

**建议**: 使用 YOLO12s 而非 YOLO12n 以获得最佳性价比

### 6.2 迁移注意事项

1. **首次下载较慢**: YOLO12 模型文件较大（~20-50MB）
2. **CUDA 版本**: 确保 CUDA >= 11.8 以获得最佳性能
3. **内存限制**: 边缘设备可能需要使用 INT8 量化

---

## 七、推荐配置

### 7.1 实时监控（推荐）

```yaml
model:
  path: "models/yolo12s.pt"  # 最佳平衡
  imgsz: 640
  conf: 0.35
  iou: 0.5
  device: "cuda:0"  # GPU
```

### 7.2 CPU 部署

```yaml
model:
  path: "models/yolo12n.pt"  # CPU 优化
  imgsz: 480                 # 降低分辨率
  conf: 0.4                 # 提高阈值减少误检
  device: "cpu"
```

---

## 八、下一步行动

### 立即执行

1. 更新 ultralytics: `pip install --upgrade ultralytics`
2. 下载 YOLO12s: `python scripts/upgrade_model.py --model yolo12s`
3. 更新配置: `configs/default.yaml`
4. 运行测试套件: `python -m pytest tests/ -v`
5. 启动新系统: `python scripts\launcher.py`

---

**文档编写**: AI Assistant
**最后更新**: 2026-04-25
**版本**: 1.0
