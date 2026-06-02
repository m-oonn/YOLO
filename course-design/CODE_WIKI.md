# YOLO 课程设计项目 — Code Wiki

> 基于 YOLOv11/v12 的实时目标检测与行为分析系统
> 版本: 1.1.0 | 许可证: MIT

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈](#2-技术栈)
3. [项目结构](#3-项目结构)
4. [核心模块详解](#4-核心模块详解)
   - 4.1 [core/ — 核心检测引擎](#41-core--核心检测引擎)
   - 4.2 [backend/ — FastAPI 后端服务](#42-backend--fastapi-后端服务)
   - 4.3 [frontend/ — Vue.js 前端](#43-frontend--vuejs-前端)
5. [关键类与函数](#5-关键类与函数)
6. [数据流与处理流程](#6-数据流与处理流程)
7. [配置系统](#7-配置系统)
8. [依赖关系](#8-依赖关系)
9. [项目运行方式](#9-项目运行方式)
10. [API 端点汇总](#10-api-端点汇总)
11. [数据库 Schema](#11-数据库-schema)
12. [性能优化策略](#12-性能优化策略)
13. [TensorRT 加速部署](#13-tensorrt-加速部署)
14. [Jetson 边缘部署](#14-jetson-边缘部署)
15. [模型微调训练](#15-模型微调训练)

---

## 1. 项目概述

本项目是一个面向校园安全的实时目标检测与行为分析系统，基于 Ultralytics YOLO 框架实现。系统能够：

- **实时目标检测**：基于 YOLO 的 80 类 COCO 目标检测
- **多目标跟踪**：集成 ByteTrack/BoT-SORT 跟踪算法
- **行为检测规则**：奔跑、摔倒、人群聚集、禁区入侵、打架检测、车辆入侵
- **骨架分析**：基于人体关键点的增强行为检测
- **MLLM 智能分析**：集成多模态大语言模型进行场景理解
- **TensorRT 加速**：ONNX → TensorRT 引擎推理优化
- **Jetson 边缘部署**：NVIDIA Jetson 平台适配与 DLA 加速
- **REST API + WebSocket**：完整的后端服务与实时流推送
- **事件存储与检索**：SQLite 持久化与高效查询
- **视频存档**：事件触发的视频片段录制

---

## 2. 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| **核心引擎** | Python 3.10+, Ultralytics YOLO, OpenCV, PyTorch | 目标检测与图像处理 |
| **推理加速** | TensorRT, ONNX, CUDA | GPU 推理优化 |
| **边缘部署** | NVIDIA Jetson, DLA | 边缘设备适配 |
| **后端** | FastAPI, Uvicorn, WebSocket | REST API + 实时流 |
| **前端** | Vue.js 3, Element Plus, Vite | 监控仪表板 |
| **存储** | SQLite | 事件与告警持久化 |
| **部署** | Docker, docker-compose | 容器化部署 |
| **MLLM** | Transformers, Accelerate | 多模态语言模型推理 |

---

## 3. 项目结构

```
course-design/
├── backend/                    # FastAPI 后端服务
│   ├── api/                    # API 路由模块
│   │   ├── __init__.py
│   │   ├── alarms.py           # 告警管理 API
│   │   ├── archives.py         # 视频存档 API
│   │   ├── cameras.py          # 摄像头管理 API
│   │   ├── detection.py        # 检测控制 + MJPEG/WebSocket 流
│   │   ├── events.py           # 事件查询 API
│   │   └── mllm.py             # MLLM 配置 API
│   ├── __init__.py
│   ├── alarm_singleton.py      # 告警引擎单例
│   ├── camera_utils.py         # 摄像头工具（超时打开、诊断）
│   ├── detection_manager.py    # 检测线程管理器（核心）
│   ├── exceptions.py           # 自定义异常类
│   ├── limiter.py              # API 限流器
│   ├── logging_utils.py        # 结构化日志工具
│   ├── main.py                 # FastAPI 应用入口
│   ├── security.py             # 安全验证工具
│   └── store.py                # EventsStore 单例管理
│
├── core/                       # 核心检测引擎
│   ├── mllm/                   # 多模态大语言模型模块
│   │   ├── __init__.py
│   │   ├── alarm_enhancer.py   # 告警增强器
│   │   ├── export_utils.py     # 模型导出工具
│   │   ├── inference_engine.py # MLLM 推理引擎
│   │   ├── mllm_config.py      # MLLM 配置
│   │   ├── mllm_sidecar.py     # MLLM 异步协调器
│   │   ├── scene_context_buffer.py # 场景上下文缓冲
│   │   └── scene_describer.py  # 场景描述生成
│   ├── action_analyzer.py      # 动作序列分析器
│   ├── alarm_engine.py         # 告警引擎（分级/抑制/聚合/升级）
│   ├── behavior_analyzer.py    # 骨架行为分析 orchestrator
│   ├── config.py               # 配置加载与数据类
│   ├── constants.py            # 常量定义（COCO 类别、颜色等）
│   ├── db_base.py              # SQLite 基础类
│   ├── events_store.py         # 事件存储（SQLite）
│   ├── feature_indexer.py      # CLIP 特征索引
│   ├── geometry.py             # 几何工具函数
│   ├── gpu_manager.py          # GPU 资源管理器
│   ├── jetson_utils.py         # Jetson 平台工具
│   ├── model_registry.py       # 模型注册表
│   ├── models.py               # 共享数据模型
│   ├── notifiers.py            # 通知渠道
│   ├── pipeline.py             # 检测流水线（核心）
│   ├── pose_features.py        # 姿态特征提取
│   ├── rules.py                # 规则引擎（bbox 基础检测）
│   ├── sequence_classifier.py  # 序列分类器（LSTM）
│   ├── skeleton.py             # 骨架提取与渲染
│   ├── tensorrt_utils.py       # TensorRT 推理工具
│   └── video_archiver.py       # 视频片段录制器
│
├── frontend/                   # Vue.js 3 前端
│   ├── src/
│   │   ├── views/              # 页面组件
│   │   │   ├── DashboardView.vue    # 仪表盘首页
│   │   │   ├── MonitorView.vue      # 实时监控
│   │   │   ├── EventsView.vue       # 事件记录
│   │   │   ├── AlarmsView.vue       # 报警管理
│   │   │   └── ConfigView.vue       # 系统配置
│   │   ├── components/         # 可复用组件
│   │   │   ├── ConfidenceGauge.vue
│   │   │   ├── PageHeader.vue
│   │   │   ├── SearchFilterBar.vue
│   │   │   ├── SkeletonLoader.vue
│   │   │   ├── StatCard.vue
│   │   │   ├── StatusBadge.vue
│   │   │   └── StreamPlayer.vue
│   │   ├── api/                # API 客户端
│   │   ├── router/             # 路由配置
│   │   └── ...
│   ├── package.json
│   └── vite.config.js
│
├── configs/
│   └── default.yaml            # 默认配置文件
│
├── datasets/                   # 数据集
│   └── campus_safety_v5/       # 校园安全数据集
│
├── models/                     # YOLO 模型文件存放目录
│   └── training/               # 训练输出目录
├── outputs/                    # 检测输出（快照、日志、数据库）
├── scripts/                    # 工具脚本
│   ├── convert_to_tensorrt.py  # TensorRT 模型转换
│   ├── setup_jetson.py         # Jetson 环境配置
│   └── train_campus_model.py   # 校园数据集训练
├── requirements.txt            # Python 依赖
├── Dockerfile                  # 后端 Docker 镜像
├── Dockerfile.frontend         # 前端 Docker 镜像
└── README.md
```

---

## 4. 核心模块详解

### 4.1 core/ — 核心检测引擎

#### 4.1.1 DetectionPipeline (`core/pipeline.py`)

**职责**：整个系统的核心处理流水线，负责视频帧的捕获、YOLO 推理、跟踪、规则应用、事件存储和结果输出。

**TensorRT 支持**：
- 当 `cfg.tensorrt.enabled=True` 且 TensorRT 可用时，自动加载 TensorRT 引擎
- 回退机制：TensorRT 初始化失败时自动回退到 PyTorch
- 统一的 `_wrap_trt_results()` 方法将 TRT 输出包装为 Ultralytics 兼容格式

**主要成员**：

| 成员 | 类型 | 说明 |
|------|------|------|
| `cfg` | `AppConfig` | 应用配置（frozen dataclass） |
| `model` | `YOLO` | 主检测模型（PyTorch 模式） |
| `_trt_model` | `TRTModelWrapper` | TensorRT 推理包装器 |
| `_use_tensorrt` | `bool` | 是否使用 TensorRT |
| `rules` | `RulesEngine` | BBox 基础规则引擎 |
| `_behavior_analyzer` | `BehaviorAnalyzer` | 骨架行为分析器 |
| `store` | `EventsStore` | 事件存储（SQLite） |
| `_alarm_engine` | `AlarmEngine` | 告警引擎 |
| `_mllm_sidecar` | `MLLMSidecar` | MLLM 异步协调器 |
| `_clip_recorder` | `VideoClipRecorder` | 视频片段录制器 |

---

#### 4.1.2 TensorRT 模块 (`core/tensorrt_utils.py`)

**职责**：TensorRT 模型转换和推理加速。

**核心类**：

```python
class TensorRTInferenceSession:
    """TensorRT 推理会话，管理引擎加载和 CUDA 内存"""
    def __init__(self, engine_path: str, input_shape: tuple, device_id: int = 0)
    def infer(self, image: np.ndarray) -> list[np.ndarray]

class TRTModelWrapper:
    """Ultralytics 兼容的 TensorRT 包装器"""
    def __init__(self, engine_path: str, imgsz: int = 640, ...)
    def predict(self, source, conf=None, iou=None, ...) -> list[dict]
    def track(self, source, tracker="bytetrack", ...) -> list[dict]

class TensorRTConverter:
    """YOLO .pt → ONNX → TensorRT 转换流水线"""
    def convert(self, model_path: str, imgsz: int = 640, force_rebuild: bool = False) -> str

class YOLOInt8Calibrator:
    """INT8 熵校准器"""
```

**关键函数**：

```python
def export_yolo_to_onnx(model_path, output_path, imgsz=640, simplify=True) -> str
def build_tensorrt_engine(onnx_path, engine_path, precision="fp16", max_batch_size=1, workspace_gb=2.0, dla_core=-1) -> str
def is_tensorrt_available() -> bool
def get_tensorrt_version() -> str
```

**精度模式**：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `fp32` | 单精度浮点 | 最高精度，最大显存 |
| `fp16` | 半精度浮点（默认） | 速度/精度平衡，推荐 |
| `int8` | 8位整数量化 | 最高速度，需校准数据 |

---

#### 4.1.3 Jetson 工具 (`core/jetson_utils.py`)

**职责**：NVIDIA Jetson 平台检测、配置和优化。

**核心类**：

```python
class JetsonManager:
    """Jetson 设备管理和优化"""
    def __init__(self)
    def _detect_jetson(self) -> JetsonInfo          # 自动检测设备型号
    def get_power_mode(self) -> str                 # 获取电源模式
    def set_max_performance(self) -> bool           # 设置最大性能
    def get_tensorrt_config(self) -> dict           # 获取优化配置
    def get_temperature(self) -> dict[str, float]   # 温度监控
    def is_throttling(self) -> bool                 # 是否过热降频
    def get_status_dict(self) -> dict               # 完整状态信息
```

**支持的 Jetson 平台**：

| 平台 | SoC | CUDA | DLA | 推荐精度 |
|------|-----|------|-----|----------|
| Jetson Nano | Tegra X1 | 5.3 | 无 | FP16 |
| Jetson TX2 | Tegra186 | 6.2 | 无 | FP16 |
| Jetson Xavier NX | Tegra194 | 7.2 | 2核 | FP16 |
| Jetson AGX Xavier | Tegra194 | 7.2 | 2核 | FP16 |
| Jetson Orin NX/Nano | Tegra234 | 8.7 | 2核 | FP16 |
| Jetson AGX Orin | Tegra234 | 8.7 | 2核 | FP16 |

---

#### 4.1.4 RulesEngine (`core/rules.py`)

**职责**：基于边界框（BBox）的行为检测规则引擎，支持 6 种检测类型。

**支持规则**：

| 规则 | 检测逻辑 | 关键参数 |
|------|----------|----------|
| **Running** | 像素速度 + 透视校正（px/s → km/h） | `speed_px_s`, `min_duration_s` |
| **Fall** | 宽高比变化 + 连续帧确认 + 快速过渡检测 | `upright_aspect_min`, `fallen_aspect_max`, `confirm_frames` |
| **Crowd** | 空间聚类（DFS）+ 透视补偿 | `min_people`, `proximity_px` |
| **Intrusion** | 多边形区域入侵（BBox 角点 + 中心检测） | `zones[]` |
| **Fight** | 多因子评分（距离/IoU/接近/混乱/移动） | `distance_threshold`, `required_score` |
| **Vehicle Intrusion** | 车辆类别入侵检测 | `zones[]` |

---

#### 4.1.5 BehaviorAnalyzer (`core/behavior_analyzer.py`)

**职责**：骨架（Skeleton）行为分析的 orchestrator，协调多个基于人体关键点的检测规则。

**包含规则类**：

| 类 | 检测类型 | 核心算法 |
|----|----------|----------|
| `SkeletonRunningRule` | 奔跑 | 线性回归速度 + 步态分析 + 自适应阈值 |
| `SkeletonFallRule` | 摔倒 | 多信号融合（躯干角度/头部速度/宽高比/髋部位移） |
| `SkeletonFightRule` | 打架 | 时序分析 + 肢体交互 + 腕部速度 |
| `CrowdDensityAnalyzer` | 人群密度 | Voronoi/凸包/社交密度多指标融合 |
| `SkeletonIntrusionRule` | 入侵 | 骨架中心点（髋中点）多边形检测 |

---

#### 4.1.6 Skeleton (`core/skeleton.py`)

**职责**：人体骨架提取、后处理、渲染和特征计算。

**数据类**：

```python
@dataclass
class SkeletonKeypoint:
    x: float
    y: float
    confidence: float
    visible: bool = True

@dataclass
class PersonSkeleton:
    track_id: int | None
    keypoints: list[SkeletonKeypoint]      # 17-point COCO format
    bbox: dict[str, float]
    body_angle: float = 0.0
    aspect_ratio: float = 0.0
    limb_lengths: dict[str, float] = field(default_factory=dict)
    head_height: float = 0.0
```

---

#### 4.1.7 AlarmEngine (`core/alarm_engine.py`)

**职责**：事件到告警的转换，包含分级、抑制、聚合和自动升级机制。

**告警级别**：

| 级别 | 数值 | 中文标签 | 默认触发事件 |
|------|------|----------|-------------|
| `INFO` | 1 | 提示 | running |
| `WARNING` | 2 | 警告 | intrusion, crowd, vehicle_intrusion |
| `CRITICAL` | 3 | 严重 | fight, fall |

---

#### 4.1.8 MLLMSidecar (`core/mllm/mllm_sidecar.py`)

**职责**：多模态大语言模型的异步协调器，非阻塞地处理场景描述和告警增强。

---

#### 4.1.9 VideoClipRecorder (`core/video_archiver.py`)

**职责**：事件触发的视频片段录制，使用 JPEG 环形缓冲区实现低内存占用。

---

#### 4.1.10 GPUManager (`core/gpu_manager.py`)

**职责**：GPU 资源检测、管理和监控。单例模式。

---

#### 4.1.11 EventsStore (`core/events_store.py`)

**职责**：SQLite 事件持久化存储，线程安全。

---

### 4.2 backend/ — FastAPI 后端服务

#### 4.2.1 main.py

FastAPI 应用入口，负责：
- 应用生命周期管理 (`lifespan`)
- CORS 中间件配置
- 全局异常处理 (`YOLOException`, `HTTPException`)
- 请求追踪中间件 (`X-Request-ID`)
- 请求日志中间件
- 路由注册

**注册路由**：
- `/api/cameras` → cameras.router
- `/api/events` → events.router
- `/api/detection` → detection.router
- `/api/alarms` → alarms_router
- `/api/archives` → archives_router
- `/api/mllm` → mllm_router

---

#### 4.2.2 DetectionManager (`backend/detection_manager.py`)

**职责**：检测流水线的生命周期管理、线程安全和 MJPEG 流状态管理。**模块级单例**。

---

### 4.3 frontend/ — Vue.js 前端

基于 Vue.js 3 + Element Plus + Vite 构建的监控仪表板。

**页面结构**：

| 页面 | 路径 | 功能 |
|------|------|------|
| **仪表盘** | `/` | 系统概览、实时画面、统计卡片 |
| **实时监控** | `/monitor` | 检测控制、MJPEG 流、MLLM 面板、实时事件 |
| **事件记录** | `/events` | 事件列表、过滤搜索、视频回放、AI 场景分析 |
| **报警管理** | `/alarms` | 告警列表、确认/解决、统计卡片 |
| **系统配置** | `/config` | 参数配置、模型切换 |

**依赖**：
- `vue@^3.4.0`
- `vue-router@^4.3.0`
- `element-plus@^2.7.0`
- `@element-plus/icons-vue@^2.3.1`
- `axios@^1.6.0`

---

## 5. 关键类与函数

### 5.1 数据模型

#### Detection (`core/models.py`)

```python
@dataclass
class Detection:
    track_id: int | None      # 跟踪 ID（ByteTrack）
    class_id: int             # COCO 类别 ID
    conf: float               # 置信度
    x1, y1, x2, y2: float     # 边界框坐标
```

#### Event (`core/models.py`)

```python
@dataclass
class Event:
    event_type: str           # 事件类型
    timestamp_s: float        # 时间戳
    frame_index: int          # 帧序号
    track_id: int | None = None
    zone_name: str | None = None
    confidence: float | None = None
    bbox: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    source: str | None = None
```

### 5.2 配置数据类

#### AppConfig (`core/config.py`)

```python
@dataclass(frozen=True)
class AppConfig:
    model_path: str = "models/yolov11x.pt"
    device: str = "auto"              # auto | cuda | cpu | mps
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.5
    tracker: str = "bytetrack"        # bytetrack | botsort
    inference_scale: float = 1.0      # 推理缩放（性能优化）
    jpeg_quality: int = 80            # MJPEG 编码质量
    tensorrt: TensorRTConfig = field(default_factory=TensorRTConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    mllm: MLLMConfig = field(default_factory=MLLMConfig)
```

#### TensorRTConfig (`core/mllm/mllm_config.py`)

```python
@dataclass
class TensorRTConfig:
    enabled: bool = False
    precision: str = "fp16"           # fp32 | fp16 | int8
    max_batch_size: int = 1
    workspace_gb: float = 2.0
    engine_dir: str = "models/tensorrt_engines"
    onnx_dir: str = "models/onnx"
    dla_core: int = -1                # -1=GPU, 0/1=DLA
    optimization_level: int = 3       # 0-5
```

---

## 6. 数据流与处理流程

```
摄像头/视频文件/RTSP
    ↓
FrameCapture (独立线程)
    ↓
DetectionThread (独立线程)
    ↓
┌──────────────────────────────────────────┐
│  TensorRT / PyTorch YOLO 推理 + 跟踪      │
└────────────────┬─────────────────────────┘
                 │
        ┌────────┼────────┐
        ↓        ↓        ↓
   RulesEngine Skeleton Secondary
   (BBox规则)  Analysis   Models
        │        (Pose)    (CPU)
        ↓        ↓
    EventMerge ←─── 骨架事件
        ↓
    EventsStore (SQLite)
        ↓
    ┌───────┼───────┐
    ↓       ↓       ↓
AlarmEngine  ClipRecorder  MLLMSidecar
(分级/抑制)  (视频存档)    (场景描述)
    ↓                       ↓
REST API ←─────────────── WebSocket
    ↓
Vue.js Frontend
```

---

## 7. 配置系统

配置文件位于 `configs/default.yaml`，通过 `core/config.py` 中的 `load_config()` 加载为不可变的 `AppConfig` dataclass。

### 7.1 配置结构

```yaml
camera:
  fps: 30
  url: ''

model:
  path: models/yolo12s.pt
  device: cuda
  imgsz: 640
  conf: 0.35
  iou: 0.5
  tracker: bytetrack
  inference_scale: 1.0
  jpeg_quality: 80

tensorrt:
  enabled: false
  precision: fp16
  max_batch_size: 1
  workspace_gb: 2.0
  engine_dir: models/tensorrt_engines
  onnx_dir: models/onnx
  dla_core: -1
  optimization_level: 3

rules:
  running:
    enabled: true
    speed_px_s: 600.0
    min_duration_s: 0.5
    debounce_s: 5.0
  fall:
    enabled: true
    upright_aspect_min: 1.3
    fallen_aspect_max: 0.80
    confirm_frames: 10
  crowd:
    enabled: true
    min_people: 5
    proximity_px: 150.0
  intrusion:
    enabled: true
    zones: []
  fight:
    enabled: true
    distance_threshold: 120
    required_score: 4

pose:
  enabled: false
  model_path: models/yolo11n-pose.pt
  kp_threshold: 0.5
  process_interval: 2

mllm:
  enabled: true
  model_type: qwen2-vl-2b
  model_path: models/mllm/qwen2-vl-2b
  scene_description_enabled: true
  alarm_enhance_enabled: true

output:
  directory: outputs
  save_snapshots: true
```

---

## 8. 依赖关系

### 8.1 Python 依赖

```
ultralytics>=8.4.0      # YOLO 检测框架
opencv-python>=4.8.0    # 图像/视频处理
numpy>=1.24.0           # 数值计算
torch>=2.0.0            # 深度学习框架

fastapi>=0.110.0        # Web 框架
uvicorn[standard]>=0.27.0  # ASGI 服务器
websockets>=12.0        # WebSocket 支持
python-multipart>=0.0.9 # 文件上传
pyjwt>=2.8.0            # JWT 认证

python-dotenv>=1.0.0    # 环境变量
pyyaml>=6.0             # YAML 配置
pydantic>=2.0.0         # 数据验证
filetype>=1.2.0         # 文件类型检测
slowapi>=0.1.9          # API 限流

transformers>=4.40.0    # MLLM 推理
accelerate>=0.28.0      # 模型加速

# TensorRT (optional, for GPU acceleration)
# tensorrt>=8.6.0
# onnx>=1.14.0
# onnxsim>=0.4.0
# pycuda>=2022.1

pytest>=8.0             # 测试框架
```

### 8.2 前端依赖

```
vue@^3.4.0
vue-router@^4.3.0
element-plus@^2.7.0
axios@^1.6.0
vite@^5.2.0
```

---

## 9. 项目运行方式

### 9.1 一键启动（推荐）

**Windows:**
```bash
双击 start.bat
```

**macOS/Linux:**
```bash
./start.sh
```

### 9.2 手动运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型到 models/ 目录
# 从 https://github.com/ultralytics/assets/releases 下载

# 3. 启动后端
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. 启动前端（新终端）
cd frontend
npm install
npm run dev
```

### 9.3 Docker 部署

```bash
docker-compose up
```

### 9.4 命令行检测

```bash
# 摄像头检测
python scripts/run_detection.py --source 0

# 视频文件检测
python scripts/run_detection.py --source video.mp4 --save-video output.mp4
```

---

## 10. API 端点汇总

### 10.1 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| GET | `/api/detection/health` | 组件健康状态 |

### 10.2 检测控制

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/detection/start` | 启动检测 |
| POST | `/api/detection/stop` | 停止检测 |
| GET | `/api/detection/status` | 检测状态 |
| POST | `/api/detection/config` | 更新运行时配置 |

### 10.3 流与监控

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/detection/stream.mjpg` | MJPEG 实时视频流 |
| WS | `/api/detection/stream` | WebSocket 实时状态 |
| GET | `/api/detection/performance` | 性能指标 |
| GET | `/api/detection/monitoring` | 综合监控数据 |
| GET | `/api/detection/gpu` | GPU 状态 |

### 10.4 模型管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/detection/models` | 列出可用模型 |
| POST | `/api/detection/models/switch` | 切换模型 |

### 10.5 配置

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/detection/config` | 读取配置 |
| POST | `/api/detection/save-config` | 保存配置 |
| POST | `/api/detection/quality` | 调整 MJPEG 质量 |

### 10.6 事件

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/events/` | 查询事件（分页/过滤） |
| GET | `/api/events/stats` | 事件统计 |
| GET | `/api/events/types` | 事件类型列表 |
| DELETE | `/api/events/` | 条件删除 |
| DELETE | `/api/events/all` | 清空事件 |
| GET | `/api/events/{id}/snapshot` | 事件快照 |

### 10.7 摄像头

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/cameras/` | 列出摄像头 |
| GET | `/api/cameras/{id}` | 摄像头信息 |

### 10.8 告警

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/alarms/` | 查询告警 |
| POST | `/api/alarms/{id}/acknowledge` | 确认告警 |
| POST | `/api/alarms/{id}/resolve` | 解决告警 |
| GET | `/api/alarms/stats` | 告警统计 |

---

## 11. 数据库 Schema

### 11.1 events 表

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    timestamp_s REAL NOT NULL,
    frame_index INTEGER NOT NULL,
    track_id INTEGER,
    zone_name TEXT,
    confidence REAL,
    bbox_json TEXT,
    snapshot_path TEXT,
    extra_json TEXT,
    description TEXT,
    keypoints_json TEXT,
    skeleton_count INTEGER DEFAULT 0,
    priority TEXT DEFAULT 'INFO',
    source TEXT,
    feature_blob BLOB
);

CREATE INDEX idx_events_time ON events(timestamp_s);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_type_time ON events(event_type, timestamp_s);
```

### 11.2 alarms 表

```sql
CREATE TABLE alarms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alarm_key TEXT NOT NULL,
    event_type TEXT NOT NULL,
    level INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    count INTEGER NOT NULL DEFAULT 1,
    first_event_time REAL NOT NULL,
    last_event_time REAL NOT NULL,
    description TEXT,
    extra_json TEXT,
    acknowledged_at REAL,
    resolved_at REAL,
    escalated_at REAL,
    notified_channels_json TEXT
);

CREATE INDEX idx_alarms_status ON alarms(status);
CREATE INDEX idx_alarms_level ON alarms(level);
CREATE INDEX idx_alarms_key_time ON alarms(alarm_key, last_event_time);
CREATE INDEX idx_alarms_event_type ON alarms(event_type);
```

### 11.3 clips 表

```sql
CREATE TABLE clips (
    clip_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    duration_s REAL NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER DEFAULT 0,
    event_count INTEGER DEFAULT 1,
    tags TEXT DEFAULT '[]'
);
```

---

## 12. 性能优化策略

### 12.1 推理优化

| 策略 | 实现位置 | 说明 |
|------|----------|------|
| TensorRT 引擎 | `core/tensorrt_utils.py` | ONNX → TensorRT，FP16/INT8 精度 |
| DLA 加速 | `TensorRTConfig.dla_core` | Jetson Deep Learning Accelerator |
| 推理缩放 | `DetectionPipeline._inference_scale` | 降低输入分辨率加速推理 |
| 半精度推理 | `GPUManager.should_use_half()` | FP16 减少显存占用 |
| GPU 预处理 | `DetectionPipeline._use_gpu_preprocess` | CUDA 预处理加速 |
| cuDNN Benchmark | `DetectionPipeline._warmup_model()` | 固定尺寸输入优化 |
| CUDA 缓存清理 | 每 300 帧 | 防止 OOM |

### 12.2 线程优化

| 策略 | 实现位置 | 说明 |
|------|----------|------|
| 分离捕获线程 | `DetectionManager._capture_loop()` | 避免 I/O 阻塞推理 |
| 后台快照写入 | `ThreadPoolExecutor(max_workers=1)` | 异步 JPEG 编码 |
| 后台特征索引 | `_index_snapshot_features()` | CLIP 编码不阻塞 |
| 后台 DB VACUUM | 独立线程 | 避免数据库阻塞 |

### 12.3 内存优化

| 策略 | 实现位置 | 说明 |
|------|----------|------|
| JPEG 环形缓冲 | `VideoClipRecorder._buffer` | 预事件帧压缩存储 |
| 动态质量调整 | `DetectionManager.calculate_dynamic_quality()` | 根据客户端数调整 |
| GPU 内存管理 | `GPUManager.get_memory_pressure()` | 高压力时释放模型 |
| 模型预热 | `_warmup_model()` | 2 次 dummy 推理 |

---

## 13. TensorRT 加速部署

### 13.1 安装 TensorRT 依赖

```bash
pip install tensorrt>=8.6.0 onnx>=1.14.0 onnxsim>=0.4.0 pycuda>=2022.1
```

### 13.2 转换模型

```bash
# 基本转换（FP16）
python scripts/convert_to_tensorrt.py --model models/yolov11x.pt

# INT8 量化（需校准数据）
python scripts/convert_to_tensorrt.py --model models/yolov11x.pt \
    --precision int8 --calib-data datasets/calibration

# 强制重建
python scripts/convert_to_tensorrt.py --model models/yolov11x.pt --force

# 检查 TensorRT 可用性
python scripts/convert_to_tensorrt.py --check
```

### 13.3 配置启用 TensorRT

编辑 `configs/default.yaml`：

```yaml
tensorrt:
  enabled: true
  precision: fp16
  max_batch_size: 1
  workspace_gb: 2.0
  engine_dir: models/tensorrt_engines
  onnx_dir: models/onnx
  dla_core: -1
  optimization_level: 3
```

### 13.4 TensorRT 性能对比

| 指标 | PyTorch | TensorRT FP16 | TensorRT INT8 |
|------|---------|---------------|---------------|
| 推理延迟 | ~30ms | ~15ms | ~10ms |
| 显存占用 | 高 | 中 | 低 |
| 精度损失 | 基准 | <1% | ~2-3% |
| 首次启动 | 快 | 慢（需构建引擎） | 慢（需校准） |

---

## 14. Jetson 边缘部署

### 14.1 环境准备

```bash
# 在 Jetson 设备上运行
sudo python scripts/setup_jetson.py --model models/yolov11x.pt
```

### 14.2 Jetson 优化配置

```bash
# 设置最大性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 使用 DLA 核心
python scripts/setup_jetson.py --model models/yolov11x.pt --dla-core 0
```

### 14.3 Jetson Docker 部署

```dockerfile
# Dockerfile.jetson
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# 安装 TensorRT
RUN apt-get update && apt-get install -y tensorrt python3-libnvinfer-dev

# 复制项目代码
COPY . /app
WORKDIR /app

# 安装 Python 依赖
RUN pip install -r requirements.txt

# 转换模型为 TensorRT（带 DLA）
RUN python scripts/convert_to_tensorrt.py --model models/yolov11x.pt \
    --precision fp16 --dla-core 0

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 14.4 Jetson 性能预期

| 平台 | 输入尺寸 | 精度 | 推理延迟 | FPS |
|------|----------|------|----------|-----|
| Jetson Nano | 640x640 | FP16 | ~120ms | ~8 |
| Jetson Xavier NX | 640x640 | FP16 | ~35ms | ~28 |
| Jetson Xavier NX | 640x640 | DLA FP16 | ~45ms | ~22 |
| Jetson AGX Orin | 640x640 | FP16 | ~15ms | ~66 |
| Jetson AGX Orin | 640x640 | DLA FP16 | ~20ms | ~50 |

---

## 15. 模型微调训练

### 15.1 准备数据集

确保 `datasets/campus_safety.yaml` 配置正确：

```yaml
train: datasets/campus_safety_v5/images/train
val: datasets/campus_safety_v5/images/val
nc: 1
names: ['person']
```

### 15.2 启动训练

```bash
# 基础训练
python scripts/train_campus_model.py --model yolo11n.pt --epochs 100

# 高级配置
python scripts/train_campus_model.py \
    --model yolo11s.pt \
    --epochs 50 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --freeze 10 \
    --lr0 0.01

# 恢复训练
python scripts/train_campus_model.py --model yolo11n.pt --resume
```

### 15.3 训练后导出

训练完成后自动导出 ONNX，然后转换为 TensorRT：

```bash
python scripts/convert_to_tensorrt.py \
    --model models/training/campus_safety/weights/best.pt \
    --precision fp16
```

---

## 附录 A: 事件类型与优先级

| 事件类型 | 优先级 | 检测方法 |
|----------|--------|----------|
| `fight` | CRITICAL | BBox 多因子 + Skeleton 时序分析 |
| `fall` | CRITICAL | BBox 宽高比 + Skeleton 多信号融合 |
| `intrusion` | WARNING | 多边形区域检测 |
| `crowd` | WARNING | 空间聚类 + Voronoi 密度 |
| `running` | INFO | 像素速度 + 步态分析 |
| `vehicle_intrusion` | WARNING | 车辆类别 + 区域检测 |

## 附录 B: 骨架关键点定义（COCO 17-point）

```
 0: nose           9: left_wrist
 1: left_eye      10: right_wrist
 2: right_eye     11: left_hip
 3: left_ear      12: right_hip
 4: right_ear     13: left_knee
 5: left_shoulder 14: right_knee
 6: right_shoulder 15: left_ankle
 7: left_elbow    16: right_ankle
 8: right_elbow
```

## 附录 C: 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2025-05 | 初始版本 |
| 1.1.0 | 2025-05 | 新增 TensorRT 支持、Jetson 适配、模型训练脚本 |

---

*文档生成时间: 2026-05-22*
*项目版本: 1.1.0*
