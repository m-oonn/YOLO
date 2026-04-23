# 课程设计总结

## 项目概述

本项目基于 **YOLOv11** 目标检测算法，构建了一个完整的实时目标检测与行为分析系统。系统包含核心检测引擎、REST API 后端、Vue.js 前端和 Docker 部署方案，是一个典型的企业级计算机视觉应用架构。

## 核心技术点

### 1. YOLO 目标检测算法
- 使用 Ultralytics YOLOv11x 模型（COCO 80 类）
- 支持多目标跟踪（ByteTrack）
- 可配置的置信度、IoU 阈值
- 支持不同 YOLO 版本切换（v8/v11/v12）

### 2. 行为分析规则引擎
- 基于目标轨迹的行为检测算法
- 5 种安全场景检测：奔跑、摔倒、人群、入侵、打架
- 独立可配置的检测参数
- 防重复触发机制（debounce）

### 3. 后端架构（FastAPI）
- RESTful API 设计
- WebSocket 实时通信
- 交互式 API 文档（Swagger/OpenAPI）
- 模块化路由组织

### 4. 前端架构（Vue.js 3）
- 响应式 SPA 设计
- Element Plus 组件库
- 实时监控仪表板
- 事件查询与统计

### 5. DevOps
- Docker 容器化部署
- docker-compose 多服务编排
- GPU 加速支持

## 简历亮点

在简历中可突出以下技能：

| 技能 | 体现 |
|------|------|
| **计算机视觉** | YOLO 目标检测、多目标跟踪、行为识别 |
| **Python 开发** | FastAPI、OpenCV、NumPy、PyTorch |
| **前端开发** | Vue.js 3、TypeScript、Element Plus |
| **后端开发** | REST API、WebSocket、SQLite |
| **DevOps** | Docker、docker-compose、CI/CD |
| **系统设计** | 模块化架构、配置驱动、事件驱动 |

## 与原始项目计划的对应关系

| 项目计划要求 | 实现情况 |
|-------------|---------|
| YOLO 目标检测 | ✅ YOLOv11x, 80类 COCO |
| 多目标跟踪 | ✅ ByteTrack 集成 |
| 行为检测（奔跑/摔倒/人群/入侵/打架） | ✅ 5种规则引擎 |
| 实时监控 | ✅ Vue.js 前端仪表板 |
| 事件存储与检索 | ✅ SQLite + REST API |
| REST API | ✅ FastAPI + Swagger |
| Docker 部署 | ✅ Docker + docker-compose |
| RAG 知识库 | ⏳ 可扩展（预留接口） |
| AI Agent | ⏳ 可扩展 |
| Django 企业后端 | ❌ 使用 FastAPI 简化 |

## 学习建议

1. **算法理解**：阅读 `core/pipeline.py` 了解检测流程，阅读 `core/rules.py` 了解行为分析逻辑
2. **工程实践**：阅读 `backend/api/` 了解 API 设计，阅读 `frontend/` 了解前端架构
3. **实验改进**：尝试更换不同 YOLO 模型，调整检测参数，添加新的检测规则
4. **扩展方向**：集成音频检测、VLM 场景理解、RAG 知识库、AI Agent

## 参考资料

- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Vue.js 3 文档](https://vuejs.org/)
- [Element Plus 文档](https://element-plus.org/)
