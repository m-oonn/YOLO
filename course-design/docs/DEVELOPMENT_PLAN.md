# 14 周开发计划回顾

## 项目背景

本项目按照项目二《基于 YOLO 与多模态的校园安全智能监控系统》的 14 周计划执行，以下是各阶段的完成情况回顾。

## 计划 vs 实际对照

### 第 1-2 周：调研 YOLO 架构，准备数据集

**计划**：调研 YOLOv11/v12 架构，准备校园场景数据集标注

**实际完成**：✅

- 调研了 YOLOv8、v11、v12 三代模型，产出 [YOLO12 升级指南](YOLO12_UPGRADE_GUIDE.md)
- 从 COCO 2017 提取 person 类别作为基础数据集（[prepare_coco_yolo.py](../scripts/prepare_coco_yolo.py)）
- 整合 4 个 Kaggle 数据集：fire+smoke、weapon、fall、fight（[merge_datasets.py](../scripts/merge_datasets.py)）
- 下载并转换 FPDS 跌倒检测数据集（[convert_fpds.py](../scripts/convert_fpds.py)）

### 第 3-5 周：训练检测模型

**计划**：训练目标检测模型，完成人员/车辆/物品检测模块

**实际完成**：✅ 人员检测完成，车辆检测本次补充完成

- 训练 campus_safety 系列模型：person 单类、person+fire+smoke+fallen 四类
- 训练 COCO 2017 完整模型（[train_coco2017.py](../scripts/train_coco2017.py)）
- YOLO12s 作为生产模型，支持 80 类 COCO 检测
- **车辆检测**：本阶段补充完成，利用 COCO 预训练权重（car/bus/truck/motorcycle/bicycle）实现车辆入侵检测
- 物品检测：依赖 COCO 预训练的 80 类覆盖

### 第 6-8 周：集成跟踪 + 行为分析

**计划**：集成多目标跟踪算法，实现行为分析模块

**实际完成**：✅ 超出预期

- ByteTrack 多目标跟踪（[pipeline.py](../core/pipeline.py)）
- **BoT-SORT**：本阶段补充完成，支持 ByteTrack / BoT-SORT 双跟踪器切换
- 5 种行为 BBox 规则引擎（[rules.py](../core/rules.py)）：
  - 奔跑（速度分析，px/s → km/h 校准）
  - 摔倒（宽高比突变检测 + 连续帧确认）
  - 人群聚集（BFS 空间聚类）
  - 区域入侵（射线法多边形检测）
  - 打架（距离 + 速度阈值）
- 骨架行为分析引擎（[behavior_analyzer.py](../core/behavior_analyzer.py)）：
  - 骨架奔跑（速度 + 步态分析）
  - 骨架摔倒（5 信号融合）
  - 骨架打架（时序序列 + 手腕速度 + 肢体频率）
  - 骨架人群（三维密度估计）
  - 骨架入侵（髋部定位）
- LSTM 序列分类器（影子模式）（[sequence_classifier.py](../core/sequence_classifier.py)）
- 自适应阈值系统（[behavior_analyzer.py](../core/behavior_analyzer.py)）

### 第 9-11 周：MLLM + Web 仪表板

**计划**：接入多模态大模型实现场景描述，开发 Web 监控仪表板

**实际完成**：✅

- Qwen2-VL-2B 视觉语言模型集成（[mllm/inference_engine.py](../core/mllm/inference_engine.py)）
- 支持 SmolVLM-500M、Florence-2 作为备选模型
- TensorRT 加速推理（[export_mllm_tensorrt.py](../scripts/export_mllm_tensorrt.py)）
- 场景描述（scene_describer）和报警增强（alarm_enhancer）
- Vue 3 + Element Plus 深色主题专业仪表盘（[frontend/](../frontend/)）
- 5 个页面：仪表盘、实时监控、事件记录、报警管理、系统配置
- WebSocket 实时更新 + MJPEG 视频流

### 第 12-13 周：TensorRT 部署 + 摄像头联调

**计划**：TensorRT 边缘部署优化，实际摄像头联调测试

**实际完成**：✅

- YOLO → ONNX → TensorRT 完整导出流水线（[export_yolo_trt.py](../scripts/export_yolo_trt.py)）
- NVIDIA Jetson 部署脚本（[deploy_jetson.sh](../scripts/deploy_jetson.sh)）
- GPU Manager 自适应（[gpu_manager.py](../core/gpu_manager.py)）
- Docker + docker-compose 容器化部署
- 生产部署文档（[DEPLOYMENT.md](DEPLOYMENT.md)）

### 第 14 周：系统评估 + 文档

**计划**：系统整体评估、文档撰写与演示

**实际完成**：✅ 本次补充完成

- 项目评估报告（[EVALUATION_REPORT.md](EVALUATION_REPORT.md)）
- 14 周开发计划回顾（本文档）
- 补充完成车辆检测、BoT-SORT 跟踪器、CLIP 特征检索

## 关键技术决策

| 决策 | 原因 |
|------|------|
| FastAPI 替代 Django | 更轻量、异步支持更好、AI 生态集成更方便 |
| YOLO12s 替代 YOLO11x | 更小的模型更好的速度-精度平衡 |
| 深色主题仪表盘 | 24/7 监控场景，减少视觉疲劳 |
| 骨架检测默认关闭 | 需要 GPU 推理，仅在性能充裕时启用 |
| MLLM 影子模式 | 安全评估阶段，不影响生产告警 |
| CLIP 特征检索 | 零样本跨模态检索，无需训练专用模型 |

## 最终完成度

| 计划书要求 | 状态 |
|-----------|------|
| YOLOv11/v12 目标检测 | ✅ |
| 人员/车辆/物品检测 | ✅（本阶段补充车辆检测） |
| ByteTrack / BoT-SORT 跟踪 | ✅（本阶段补充 BoT-SORT） |
| 骨架动作识别 | ✅ 超出预期（5 信号融合） |
| MLLM 场景描述 | ✅ |
| TensorRT + Jetson 部署 | ✅ |
| Web 监控仪表板 | ✅ |
| 按特征检索 | ✅（本阶段补充 CLIP） |
| 14 周计划 | ✅（本文档） |

**总体完成度：100%**
