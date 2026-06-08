# YOLO 校园安全智能监控系统 — 代码学习版

本文件夹是从完整项目中提取的**纯源代码**，每个文件顶部都有解析注释，说明该文件在系统中的角色、依赖关系和核心职责。

## 代码分层（从上到下阅读）

```
campus-safety/
│
├── configs/           ← 第1层：一切从这里开始
│   └── default.yaml       所有参数入口（模型/规则/摄像头）
│
├── core/              ← 第2层：核心检测引擎
│   ├── pipeline.py         ★ 主检测流水线（系统中枢）
│   ├── rules.py            ★ 6种行为检测规则引擎
│   ├── alarm_engine.py     ★ 报警引擎（事件→报警）
│   ├── config.py           配置中枢（YAML→数据类）
│   ├── models.py           共享数据模型（Detection/Event）
│   ├── skeleton.py         人体骨架提取（17关键点）
│   ├── behavior_analyzer   骨架行为分析（5种行为）
│   ├── events_store.py     事件持久化（SQLite）
│   ├── mllm/               多模态大模型子系统
│   └── ...（共22个文件）
│
├── backend/           ← 第3层：FastAPI后端服务
│   ├── main.py             ★ 应用入口（路由/中间件/生命周期）
│   ├── detection_manager   检测线程管理器（最复杂，42KB）
│   ├── api/
│   │   ├── detection.py    检测控制端点（启动/停止/视频流）
│   │   ├── events.py       事件查询端点
│   │   ├── alarms.py       报警管理端点
│   │   └── ...（8个路由）
│   └── ...（共18个文件）
│
├── frontend/          ← 第4层：Vue 3 前端界面
│   └── src/
│       ├── views/          5个页面（仪表盘/监控/事件/报警/配置）
│       ├── components/     8个通用组件
│       ├── composables/    3个可复用逻辑
│       └── api/client.js   ★ 后端API客户端
│
├── tests/             ← 第5层：单元测试
│   ├── test_rules.py       ★ 规则引擎测试（最重要）
│   └── ...（共26个测试文件）
│
├── scripts/           ← 第6层：工具脚本
│   ├── run_detection.py    命令行检测
│   ├── download_models.py  模型下载
│   └── ...（18个脚本）
│
└── docs/              ← 第7层：项目文档
    ├── API.md              REST API 参考
    ├── DEPLOYMENT.md       部署指南
    ├── SUMMARY.md          课程设计总结
    └── architecture/       架构文档
```

## 数据流总览

```
摄像头/视频 → pipeline.py → YOLO推理 → ByteTrack跟踪
                                    ↓
                              rules.py（6种规则检测）
                                    ↓
                         Event → alarm_engine.py → Alarm
                                    ↓                   ↓
                           events_store.py        WebSocket推送
                           (SQLite存储)            前端实时更新
                                    ↓
                           backend/main.py
                           (FastAPI REST API)
                                    ↓
                           frontend/ (Vue 3 SPA)
```

## 阅读建议

| 天数 | 阅读内容 | 目标 |
|------|---------|------|
| 第1天 | `configs/default.yaml` → `core/config.py` → `core/models.py` → `core/constants.py` | 理解系统的"配置→数据"基础 |
| 第2天 | `core/rules.py` → `core/geometry.py` → `core/skeleton.py` | 理解6种行为是怎么检测的 |
| 第3天 | `core/pipeline.py` → `core/alarm_engine.py` | 理解一帧画面到报警的完整流程 |
| 第4天 | `backend/main.py` → `backend/api/detection.py` → `backend/detection_manager.py` | 理解后端服务如何响应前端请求 |
| 第5天 | `frontend/src/router/index.js` → `MonitorView.vue` → `client.js` | 理解前端如何展示数据和控制检测 |

每个文件顶部都有 `─── 【层级】文件名 — 职责 ───` 格式的注释，阅读前先看那些注释即可快速理解。
