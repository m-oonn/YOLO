# YOLOv12 校园安全智能监控系统

基于 YOLOv12 + 骨架分析 + Qwen2-VL 的实时校园安全监控平台。支持摄像头/视频文件检测、行为规则报警、MLLM 语义场景理解，以及 Web 可视化控制台。

## 功能特性

- **实时检测**：YOLOv12 目标检测 + ByteTrack 多目标跟踪
- **行为分析**：奔跑、跌倒、人群聚集、打架、区域入侵、车辆违停
- **骨架融合**：YOLO Pose 17 关键点 + 多信号融合，降低误报
- **MLLM 场景理解**：Qwen2-VL-2B-Instruct 按需加载，识别复杂校园场景
- **Web 控制台**：Vue 3 实时监控、事件管理、报警中心、配置面板
- **快速启动**：模型预加载，检测启动 < 3 秒（GPU）

## 技术栈

| 层级 | 技术 |
|------|------|
| 检测引擎 | Ultralytics YOLOv12, PyTorch, OpenCV |
| 后端 | FastAPI, Uvicorn, WebSocket |
| 前端 | Vue 3, Vite, Element Plus |
| MLLM | Transformers, Qwen2-VL-2B-Instruct |
| 存储 | SQLite |

## 环境要求

- Python 3.10+
- Node.js 18+（前端开发）
- NVIDIA GPU + CUDA 11.8+（推荐，CPU 可运行但较慢）
- Windows / Linux / macOS

## 快速开始（约 15 分钟）

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/campus-safety.git
cd campus-safety
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> **PyTorch GPU**：若需 CUDA 版本，请按 [pytorch.org](https://pytorch.org) 说明安装对应 `torch` 后再安装其余依赖。

### 3. 配置环境变量（可选）

```bash
cp .env.example .env
# 编辑 .env：YOLO_DEVICE=cuda, API_KEY=...
```

### 4. 下载模型权重

```bash
python scripts/download_models.py
```

主检测模型 `yolo12s.pt` 也会在首次运行时由 Ultralytics 自动下载到 `models/`。

### 5. 复制配置文件

```bash
cp config.example.yaml configs/default.yaml
```

### 6. 一键启动

```bash
# Windows
start.bat

# Linux / macOS
./start.sh
```

或手动启动：

```bash
# 终端 1 — 后端
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 终端 2 — 前端
cd frontend && npm install && npm run dev
```

浏览器访问：**http://localhost:8080**（前端） / **http://localhost:8000/docs**（API 文档）

## 使用方法

1. 打开 **实时监控** 页面，选择摄像头或上传视频
2. 点击 **启动** 开始检测
3. 在 **事件** / **报警** 页面查看检测结果
4. 可选：开启 **MLLM 场景理解**（监控页开关，按需加载模型）

### 命令行检测

```bash
python scripts/run_detection.py --source 0
python scripts/run_detection.py --source path/to/video.mp4
```

### 验证脚本

```bash
python scripts/verify_startup.py --quick   # 启动性能
python scripts/verify_mllm.py                # MLLM 懒加载
python scripts/check_api.py                  # API 健康检查
```

## 项目结构

```
campus-safety/
├── backend/           # FastAPI 服务（检测 API、WebSocket）
├── core/              # 检测流水线、规则引擎、MLLM
├── frontend/          # Vue 3 Web UI
├── configs/           # 运行时配置（不提交敏感信息）
├── scripts/           # 启动、下载、训练、验证脚本
├── tests/             # 单元测试
├── docs/              # 详细文档
├── models/            # 模型权重目录（需自行下载）
├── outputs/           # 运行时输出（事件、快照）
├── config.example.yaml
├── requirements.txt
└── start.bat / start.sh
```

## 需自行下载的文件

| 资源 | 大小（约） | 获取方式 |
|------|-----------|----------|
| `models/yolo12s.pt` | 18 MB | `python scripts/download_models.py` 或首次运行自动下载 |
| `models/yolo11n-pose.pt` | 6 MB | `python scripts/download_models.py --only yolo-pose` |
| Qwen2-VL-2B-Instruct | ~4 GB | 开启 MLLM 时从 HuggingFace 自动下载 |

国内用户可设置：`HF_ENDPOINT=https://hf-mirror.com`（见 `.env.example`）

## 配置说明

完整参数见 [`config.example.yaml`](config.example.yaml)。主要配置项：

- `model.path` — YOLO 权重路径
- `model.device` — `cuda` / `cpu`（或用环境变量 `YOLO_DEVICE`）
- `mllm.enabled` — 默认 `false`，避免占用显存
- `rules.*` — 各行为检测规则阈值

## 运行测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -q --no-cov
```

## 文档

- [API 参考](docs/API.md)
- [部署指南](docs/DEPLOYMENT.md)
- [开发指南](docs/DEVELOPMENT.md)
- [架构概览](docs/architecture/overview.md)

## 许可证

[Apache License 2.0](LICENSE)

## 贡献

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。
