# YOLO Campus Safety Intelligent Monitoring System

# 基于 YOLO 与多模态的校园安全智能监控系统

[![CI](https://github.com/m-oonn/YOLO/actions/workflows/ci.yml/badge.svg)](https://github.com/m-oonn/YOLO/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Vue.js](https://img.shields.io/badge/Vue.js-3.4+-4FC08D?logo=vue.js&logoColor=white)](https://vuejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://m-oonn.github.io/YOLO/)

## 项目简介 / Overview

**YOLO 校园安全智能监控系统**是一个基于深度学习的实时目标检测与行为分析平台，专为校园安全场景设计。

核心能力：
- **实时目标检测** — YOLOv12 检测 80 类 COCO 目标（人、车、物品等）
- **多目标跟踪** — ByteTrack / BoT-SORT 跨帧身份保持
- **6 种行为识别** — 奔跑、摔倒、人群聚集、禁区入侵、打架斗殴、车辆闯入
- **骨架行为分析** — 基于人体关键点的精细姿态识别
- **MLLM 场景理解** — 多模态大模型辅助场景描述与报警验证
- **实时 Web 仪表板** — Vue 3 + Element Plus 暗色主题监控界面

> 主要代码位于 `course-design/` 目录下。本仓库同时托管前端演示页面于 GitHub Pages。

## 在线演示 / Live Demo

**[https://m-oonn.github.io/YOLO/](https://m-oonn.github.io/YOLO/)**

> 注意：GitHub Pages 演示为前端静态展示。完整功能需连接运行中的后端服务。

## 快速开始 / Quick Start

> 详细文档请参阅 [course-design/README.md](course-design/README.md)

### 一键启动（推荐）

**Windows:**
```bash
cd course-design
start.bat
```

**macOS / Linux:**
```bash
cd course-design
./start.sh
```

### 手动启动

```bash
# 1. 安装 Python 依赖
cd course-design
pip install -r requirements.txt

# 2. 启动后端（终端 1）
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 3. 启动前端（终端 2）
cd frontend
npm install
npm run dev
```

访问 `http://localhost:8080` 打开监控界面。

## 项目结构 / Project Structure

```
YOLO/
├── course-design/             # 主要项目代码
│   ├── core/                  # 核心检测引擎（YOLO + 规则 + 骨架）
│   ├── backend/               # FastAPI 后端 API 服务
│   ├── frontend/              # Vue 3 前端 SPA
│   ├── configs/               # YAML 配置文件
│   ├── models/                # 预训练模型文件
│   ├── docs/                  # 详细文档
│   ├── scripts/               # 工具脚本
│   └── tests/                 # 单元测试
├── reports/                   # 项目报告与文档
├── assets/                    # 图表、截图、幻灯片
│   ├── screenshots/           # 系统截图
│   ├── charts/                # 分析图表
│   └── slides/                # PPT 幻灯片图片
├── evaluation-screenshots/    # 评估截图
├── ppt_assets/                # PPT 素材
└── ppt_charts/                # PPT 图表
```

## 文档索引 / Documentation

| 文档 | 说明 |
|------|------|
| [course-design/README.md](course-design/README.md) | 项目详细说明 |
| [course-design/CODE_WIKI.md](course-design/CODE_WIKI.md) | 代码开发 Wiki |
| [course-design/docs/API.md](course-design/docs/API.md) | REST API 文档 |
| [course-design/docs/DEPLOYMENT.md](course-design/docs/DEPLOYMENT.md) | 部署指南 |
| [course-design/docs/DEVELOPMENT.md](course-design/docs/DEVELOPMENT.md) | 开发调试指南 |
| [course-design/CONTRIBUTING.md](course-design/CONTRIBUTING.md) | 贡献指南 |
| [course-design/CHANGELOG.md](course-design/CHANGELOG.md) | 变更日志 |
| [SECURITY.md](SECURITY.md) | 安全策略 |
| `http://localhost:8000/docs` | Swagger API 交互文档 |

## 技术栈 / Tech Stack

| 层级 | 技术 | 说明 |
|------|------|------|
| **核心引擎** | Python 3.10+, Ultralytics YOLOv12, OpenCV, PyTorch | 目标检测与图像处理 |
| **跟踪** | ByteTrack / BoT-SORT | 多目标实时跟踪 |
| **后端** | FastAPI, Uvicorn, WebSocket | REST API + 实时流推送 |
| **前端** | Vue.js 3, Element Plus, Vite 5 | 响应式监控仪表板 |
| **存储** | SQLite (WAL 模式) | 事件与报警持久化 |
| **AI 增强** | Qwen2-VL, Chinese CLIP | 场景描述 / 特征检索 |
| **部署** | Docker Compose, GitHub Pages | 容器化 + 静态演示 |

## 许可证 / License

本项目采用 [Apache License 2.0](LICENSE)。

Copyright 2025 YOLO Campus Safety Contributors

## 致谢 / Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — 目标检测引擎
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — 多目标跟踪算法
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) — 多模态视觉语言模型
- [Chinese CLIP](https://github.com/OFA-Sys/chinese-clip) — 中文图文检索
- [Element Plus](https://element-plus.org/) — Vue 3 UI 组件库
