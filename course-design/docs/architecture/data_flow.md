# YOLO 实时检测系统 - 数据流文档

## 1. 系统启动流程

```
用户运行 start.bat/start.sh
    │
    ├── 系统预检检查
    │   ├── Python 版本检查
    │   ├── 依赖包检查
    │   ├── GPU 可用性检查
    │   ├── 端口可用性检查
    │   └── 模型文件检查
    │
    ├── 启动后端 (FastAPI)
    │   ├── 加载配置文件 (configs/default.yaml)
    │   ├── 初始化 EventsStore (SQLite)
    │   ├── 初始化 AlarmEngine
    │   ├── 注册 API 路由
    │   └── 启动 Uvicorn 服务器
    │
    └── 启动前端 (Vite Dev Server)
        ├── 加载 Vite 配置
        ├── 配置代理到后端 (localhost:8000)
        └── 启动开发服务器
```

## 2. 检测流水线数据流

### 2.1 视频帧处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    DetectionPipeline.run()                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 视频源初始化                                                 │
│     └── cv2.VideoCapture(0) 或 cv2.VideoCapture(video_path)    │
│                                                                 │
│  2. 帧读取循环                                                   │
│     └── while running: ret, frame = cap.read()                 │
│                                                                 │
│  3. YOLO 推理                                                   │
│     ├── frame → model(frame) → ultralytics results              │
│     ├── 设备：GPU (cuda:0) + 半精度 (FP16)                      │
│     └── 输出：boxes, confs, classes                            │
│                                                                 │
│  4. 后处理                                                      │
│     ├── NMS 过滤重复检测                                        │
│     ├── ByteTrack 目标跟踪 → track_ids                          │
│     ├── 骨架提取 (17 COCO keypoints)                            │
│     └── 绘制检测结果 (bbox + labels + skeletons)               │
│                                                                 │
│  5. 行为分析                                                    │
│     ├── RulesEngine.update(detections, frame_idx, timestamp)   │
│     │   ├── 跌倒检测 (bbox aspect ratio + transition)          │
│     │   ├── 拥挤检测 (ROI count threshold)                     │
│     │   └── 打架检测 (interaction analysis)                    │
│     ├── BehaviorAnalyzer.analyze(skeletons, timestamp)          │
│     │   ├── 跌倒检测 (torso angle + head velocity)             │
│     │   ├── 奔跑检测 (speed + gait analysis)                   │
│     │   └── 入侵检测 (zone violation)                          │
│     └── Event 合并去重                                         │
│                                                                 │
│  6. 事件处理                                                    │
│     ├── store.add(event) → SQLite                              │
│     ├── alarm_engine.process(event) → 告警/通知                │
│     ├── WebSocket broadcast → 前端实时推送                      │
│     └── MLLM sidecar → 场景理解 (可选)                         │
│                                                                 │
│  7. 帧输出                                                     │
│     ├── 编码为 JPEG (动态质量调整)                             │
│     └── MJPEG 流推送 / WebSocket 帧推送                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 事件生命周期

```
事件生成
    │
    ├── RulesEngine / BehaviorAnalyzer 检测到行为
    │
    ▼
事件创建
    │
    ├── Event 对象实例化
    │   ├── event_type: "fall" / "running" / "crowd" / "fight" / "intrusion"
    │   ├── timestamp: 当前时间
    │   ├── frame_idx: 当前帧号
    │   ├── track_id: 关联目标 ID
    │   ├── confidence: 检测置信度
    │   ├── bbox: 目标边界框
    │   └── extra: 附加信息 (速度、角度、计数等)
    │
    ▼
事件存储
    │
    ├── events_store.add(event)
    │   ├── 序列化到 SQLite
    │   └── 索引优化查询
    │
    ▼
事件处理
    │
    ├── alarm_engine.process(event)
    │   ├── 冷却期检查
    │   ├── 优先级评估
    │   ├── 通知发送 (可选)
    │   └── 记录到数据库
    │
    ▼
事件推送
    │
    ├── WebSocket.broadcast(event)
    │   ├── 序列化 JSON
    │   └── 推送所有连接客户端
    │
    ▼
事件展示
    │
    └── 前端 EventsView 显示
        ├── 事件列表
        ├── 时间筛选
        └── 类型过滤
```

## 3. API 请求响应流

### 3.1 检测控制

```
POST /api/detection/start
    │
    ├── 请求体: { source: "0", config: "configs/default.yaml" }
    │
    ▼
API 端点 (api/detection.py)
    │
    ├── 验证请求参数
    ├── 加载配置文件
    ├── 初始化 DetectionPipeline
    │   ├── 加载 YOLO 模型 (yolo12s.pt)
    │   ├── 配置 GPU/FP16
    │   ├── 初始化 ByteTrack
    │   ├── 初始化骨架提取
    │   └── 初始化行为分析
    │
    ▼
启动后台线程
    │
    ├── threading.Thread(target=pipeline.run)
    └── pipeline.run() 在后台运行
```

### 3.2 MJPEG 视频流

```
GET /api/detection/stream
    │
    ├── 客户端连接
    │
    ▼
MJPEG 生成器
    │
    ├── while client_connected:
    │   ├── frame = pipeline.get_latest_frame()
    │   ├── jpeg = cv2.imencode('.jpg', frame, [quality])
    │   ├── yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg
    │   └── asyncio.sleep(0)
    │
    ▼
HTTP 响应
    │
    └── Content-Type: multipart/x-mixed-replace; boundary=--frame
        ├── 浏览器持续接收帧
        └── <img src="/api/detection/stream"> 显示实时视频
```

### 3.3 WebSocket 实时推送

```
WebSocket 连接 (ws://localhost:8000/ws)
    │
    ├── 客户端连接
    │
    ▼
消息循环
    │
    ├── while connected:
    │   ├── 接收客户端消息
    │   │   ├── subscribe: 订阅事件
    │   │   ├── unsubscribe: 取消订阅
    │   │   └── get_status: 获取状态
    │   │
    │   └── 推送服务器消息
    │       ├── event: 新事件推送
    │       ├── status: 状态更新
    │       └── alarm: 告警通知
    │
    ▼
事件广播
    │
    ├── DetectionPipeline 产生新事件
    ├── WebSocket 管理器广播给所有订阅者
    └── 前端实时更新
```

## 4. MLLM 场景理解流

```
事件触发 (fall / running / etc.)
    │
    ▼
场景上下文收集
    │
    ├── SceneContextBuffer 收集最近帧
    │   ├── 最近 N 帧图像
    │   ├── 关联事件列表
    │   └── 检测结果元数据
    │
    ▼
MLLM 推理请求
    │
    ├── MLLMSidecar.on_frame(detections, events)
    │   ├── 构建提示词
    │   ├── 附加关键帧图像
    │   └── 发送到 MLLM 后端
    │
    ▼
MLLM 响应
    │
    ├── 场景描述文本
    │   ├── "A person is running across the hallway..."
    │   └── 置信度评分
    │
    ▼
结果处理
    │
    ├── AlarmEnhancer 增强告警
    │   ├── 添加场景描述到事件
    │   └── 提升告警优先级
    │
    ▼
前端展示
    │
    └── MLLM View 显示场景描述
```

## 5. 错误处理流

```
异常发生
    │
    ├── 检测层异常 (pipeline.py)
    │   ├── 视频源断开 → 重试/停止
    │   ├── GPU OOM → 降级到 CPU
    │   └── 模型推理失败 → 跳过帧
    │
    ├── API 层异常 (api/*.py)
    │   ├── 参数验证失败 → 400 Bad Request
    │   ├── 资源未找到 → 404 Not Found
    │   └── 内部服务器错误 → 500 Internal Error
    │
    └── 安全层异常 (security.py, limiter.py)
        ├── 路径遍历攻击 → 403 Forbidden
        ├── 速率限制触发 → 429 Too Many Requests
        └── 认证失败 → 401 Unauthorized
```
