# YOLO 课程设计 — 实时目标检测与行为分析系统

[![CI](https://github.com/m-oonn/YOLO/actions/workflows/ci.yml/badge.svg)](https://github.com/m-oonn/YOLO/actions)
[![codecov](https://codecov.io/gh/m-oonn/YOLO/branch/main/graph/badge.svg)](https://codecov.io/gh/m-oonn/YOLO)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Vue.js](https://img.shields.io/badge/Vue.js-3.4+-4FC08D?logo=vue.js&logoColor=white)](https://vuejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

基于 **YOLOv11/v12** 的实时目标检测与行为分析系统，包含完整的 **FastAPI** 后端和 **Vue.js 3** 前端。

## 项目概述

本项目是一个面向课程设计的计算机视觉应用，实现了：

- **实时目标检测**：基于 Ultralytics YOLO 的 80 类 COCO 目标检测
- **多目标跟踪**：集成 ByteTrack 跟踪算法，支持跨帧身份保持
- **行为检测规则**：奔跑、摔倒、人群聚集、禁区入侵、打架检测
- **REST API**：基于 FastAPI 的完整后端服务
- **实时 Web 界面**：基于 Vue.js 3 + Element Plus 的监控仪表板
- **事件存储与检索**：SQLite 数据库存储事件记录

## 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| **核心引擎** | Python 3.10+, Ultralytics YOLO, OpenCV | 目标检测与图像处理 |
| **后端** | FastAPI, Uvicorn, WebSocket | REST API + 实时流 |
| **前端** | Vue.js 3, Element Plus, Vite | 监控仪表板 |
| **存储** | SQLite | 事件持久化 |
| **部署** | Docker, docker-compose | 容器化部署 |

## 快速开始

### 方式一：直接运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型
# 将 YOLO 模型文件放入 models/ 目录
# 可从 https://github.com/ultralytics/assets/releases 下载

# 3. 启动后端
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. 启动前端 (新终端)
cd frontend
npm install
npm run dev
```

### 方式二：Docker 部署

```bash
docker-compose up
```

### 方式三：命令行检测

```bash
# 摄像头检测
python scripts/run_detection.py --source 0

# 视频文件检测
python scripts/run_detection.py --source video.mp4 --save-video output.mp4
```

## 项目结构

```
course-design/
├── core/                    # 核心检测引擎
│   ├── config.py           # 配置加载
│   ├── pipeline.py         # 检测流水线
│   ├── rules.py            # 规则引擎
│   ├── events_store.py     # 事件存储
│   ├── geometry.py         # 几何工具
│   └── constants.py        # 常量定义
├── backend/                 # FastAPI 后端
│   ├── main.py             # 应用入口
│   └── api/
│       ├── cameras.py      # 摄像头 API
│       ├── events.py       # 事件 API
│       └── detection.py    # 检测 API + WebSocket
├── frontend/                # Vue.js 3 前端
│   └── src/
│       ├── views/           # 页面组件
│       ├── api/             # API 客户端
│       └── router/          # 路由配置
├── configs/                 # 配置文件
├── models/                  # YOLO 模型文件
├── scripts/                 # 工具脚本
└── outputs/                 # 检测输出

```

## API 文档

启动后端后访问 `http://localhost:8000/docs` 查看交互式 API 文档。

### 主要端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/cameras/` | 列出摄像头 |
| POST | `/api/detection/start` | 启动检测 |
| POST | `/api/detection/stop` | 停止检测 |
| GET | `/api/detection/status` | 检测状态 |
| GET | `/api/events/` | 查询事件 |
| GET | `/api/events/stats` | 事件统计 |
| GET | `/api/detection/stream.mjpg` | MJPEG 实时视频流 |
| WS | `/api/detection/stream` | WebSocket 实时状态 |

## 功能演示

### 实时监控
- 选择摄像头或视频文件作为检测源
- 实时显示检测结果（目标框、类别、置信度）
- 显示 FPS、帧数、事件数等运行指标

### 事件记录
- 按类型筛选事件（奔跑、摔倒、人群、入侵、打架）
- 事件统计概览
- 分页查询

### 系统配置
- 模型参数配置（路径、输入尺寸、阈值）
- 检测规则开关（独立启用/禁用每项规则）
- 输出配置

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

### 第三方依赖许可证声明

- **Ultralytics YOLO**：AGPL-3.0 许可证，详见 [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **YOLOv8 模型权重**：遵循 Ultralytics 许可证条款
- 其余依赖均为 MIT/BSD 兼容许可证
