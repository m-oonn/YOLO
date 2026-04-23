# 开发指南

## 开发环境配置

### 系统要求

- Python 3.10+
- Node.js 18+
- NVIDIA GPU（可选，用于 GPU 加速）

### 安装

```bash
# 1. 创建 Python 虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# 2. 安装后端依赖
pip install -r requirements.txt

# 3. 安装前端依赖
cd frontend
npm install
cd ..
```

### 开发模式运行

**终端 1 - 后端：**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**终端 2 - 前端：**
```bash
cd frontend
npm run dev
```

访问 `http://localhost:8080`

## 项目架构

### 核心引擎 (`core/`)

```
core/
├── config.py        配置加载 (YAML → AppConfig dataclass)
├── pipeline.py      检测流水线 (YOLO → Rules → Events)
├── rules.py         规则引擎 (5种行为检测)
├── events_store.py  事件存储 (SQLite)
├── geometry.py      几何工具 (点/多边形检测)
└── constants.py     常量 (COCO类别, 颜色)
```

**处理流程：**
1. 视频帧输入 → YOLO 检测 → 提取检测结果
2. 检测结果 → 规则引擎 → 行为事件
3. 行为事件 → SQLite 存储 + WebSocket 广播

### 后端 (`backend/`)

```
backend/
├── main.py          FastAPI 应用入口
└── api/
    ├── cameras.py   摄像头管理 API
    ├── events.py    事件查询 API
    └── detection.py 检测控制 API + WebSocket
```

### 前端 (`frontend/`)

```
frontend/src/
├── App.vue          主应用组件
├── main.js          入口文件
├── api/client.js    API 客户端 (axios)
├── router/index.js  路由配置
└── views/
    ├── MonitorView.vue   实时监控
    ├── EventsView.vue    事件记录
    └── ConfigView.vue    系统配置
```

## 添加新检测规则

在 `core/rules.py` 中添加新规则：

1. 在 `RulesConfig` 中添加规则配置类
2. 在 `config.py` 中解析新配置
3. 在 `RulesEngine` 中添加检测方法
4. 在 `configs/default.yaml` 中添加默认配置
5. 在前端 `ConfigView.vue` 中添加配置界面

示例 - 添加"静止检测"规则：

```python
# core/rules.py
def _detect_stationary(self, persons, t, frame_idx):
    events = []
    for d in persons:
        tid = self._resolve_tid(d, t)
        hist = self._pos_hist.get(tid)
        if not hist or len(hist) < 30:
            continue
        # 检测位置是否长时间不变
        positions = [(cx, cy) for _, cx, cy in hist[-30:]]
        variance = np.var([p[0] for p in positions]) + np.var([p[1] for p in positions])
        if variance < 5.0:  # 几乎不动
            events.append(self._make_event("stationary", t, frame_idx, tid))
    return events
```

## 测试

```bash
# 运行核心引擎测试
python -m pytest tests/

# 测试特定模块
python -m pytest tests/test_rules.py -v
```

## 代码规范

- Python: 遵循 PEP 8，使用 Black 格式化
- Vue: 使用 ESLint + Prettier
- 提交信息: 遵循 Conventional Commits

## 性能优化

### CPU 优化
- 使用 `yolov11n.pt`（nano 版本）提高速度
- 降低 `imgsz`（如 480）减少计算量
- 减少 `camera_fps` 降低处理频率

### GPU 优化
- 确保 CUDA 和 cuDNN 正确安装
- 使用 TensorRT 导出优化模型
- 开启批处理模式

### 内存优化
- 定期清理事件历史
- 限制帧缓冲区大小
- 使用 `maxlen` 参数控制 deque 大小
