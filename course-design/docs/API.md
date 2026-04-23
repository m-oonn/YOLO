# API 文档

## 基础信息

- **Base URL**: `http://localhost:8000`
- **交互式文档**: `http://localhost:8000/docs`

## 端点

### 健康检查

```
GET /health
```

响应：
```json
{ "status": "healthy" }
```

### 摄像头管理

```
GET /api/cameras/
```

列出所有可用摄像头设备。

响应：
```json
{
  "devices": [
    { "id": 0, "name": "Camera 0", "available": true, "resolution": "640x480", "fps": 30 }
  ]
}
```

### 检测控制

#### 启动检测

```
POST /api/detection/start
Content-Type: application/json

{
  "source": "0",
  "config": "configs/default.yaml"
}
```

参数：
- `source`: 摄像头索引（字符串数字）或视频文件路径
- `config`: 配置文件路径

响应：
```json
{ "status": "started", "source": "0" }
```

#### 停止检测

```
POST /api/detection/stop
```

响应：
```json
{ "status": "stopped" }
```

#### 检测状态

```
GET /api/detection/status
```

响应：
```json
{
  "running": true,
  "fps": 28.5,
  "frame_count": 1234,
  "elapsed_s": 43.2,
  "events_count": 5
}
```

### 事件查询

#### 查询事件列表

```
GET /api/events/?event_type=running&limit=20&offset=0
```

参数：
- `event_type`: 事件类型（可选）
- `limit`: 每页数量（默认 50）
- `offset`: 偏移量
- `start_time`: 开始时间戳
- `end_time`: 结束时间戳

响应：
```json
{
  "events": [
    {
      "id": 1,
      "event_type": "running",
      "timestamp_s": 1234.56,
      "frame_index": 100,
      "track_id": 1,
      "zone_name": null,
      "confidence": 0.85,
      "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
      "description": "running (track 1)",
      "extra": {"speed_px_s": 520.3}
    }
  ],
  "total": 50,
  "limit": 20,
  "offset": 0
}
```

#### 事件统计

```
GET /api/events/stats
```

响应：
```json
{
  "total_events": 100,
  "by_type": { "running": 30, "fall": 10, "crowd": 20, "intrusion": 25, "fight": 15 },
  "first_event": 1234.0,
  "last_event": 5678.0
}
```

#### 事件类型列表

```
GET /api/events/types
```

响应：
```json
{
  "event_types": ["running", "fall", "crowd", "intrusion", "fight"]
}
```

#### 删除事件

```
DELETE /api/events/?event_type=running&before=1234567890
```

参数：
- `event_type`: 按类型过滤（可选）
- `before`: 删除此时间戳之前的事件（可选）

响应：
```json
{ "status": "deleted", "count": 10 }
```

#### 清空所有事件

```
DELETE /api/events/all
```

响应：
```json
{ "status": "cleared", "count": 100 }
```

### WebSocket 实时状态

```
WS /api/detection/stream
```

发送任意文本消息后接收 JSON 格式的状态更新。

### MJPEG 视频流

```
GET /api/detection/stream.mjpg
```

返回多部分 MJPEG 流，可直接在浏览器 `<img>` 标签中显示。

响应：
- Content-Type: `multipart/x-mixed-replace; boundary=frame`
- 每帧为 JPEG 格式，包含检测标注（目标框、标签、区域）

#### 保存配置

```
POST /api/detection/save-config
Content-Type: application/json

{
  "model": { "path": "models/yolov11x.pt", "imgsz": 640, "conf": 0.35, "iou": 0.5 },
  "rules": {
    "running": { "enabled": true, "speed_px_s": 50 },
    "fall": { "enabled": true, "upright_aspect_min": 1.2 },
    "crowd": { "enabled": true, "min_people": 3 },
    "intrusion": { "enabled": true },
    "fight": { "enabled": true }
  },
  "output": { "save_snapshots": true }
}
```

将配置写入 YAML 文件并更新运行时配置。重启后保持。

响应：
```json
{ "status": "saved", "path": "configs/default.yaml" }
```

#### 运行时更新配置

```
POST /api/detection/config
Content-Type: application/json

{
  "config": "configs/default.yaml"
}
```

从指定配置文件重新加载配置到运行中的检测管线（不写回 YAML 文件）。

响应：
```json
{ "status": "updated" }
```

## 数据模型

### 事件类型

| 类型 | 说明 | 触发条件 |
|------|------|---------|
| `running` | 奔跑检测 | 移动速度超过阈值 |
| `fall` | 摔倒检测 | 高宽比从竖直变为横向 |
| `crowd` | 人群聚集 | 同一画面人数超过阈值 |
| `intrusion` | 禁区入侵 | 目标进入多边形区域 |
| `fight` | 打架检测 | 两人近距离快速移动 |

### 检测结果

| 字段 | 类型 | 说明 |
|------|------|------|
| `track_id` | int | 跟踪ID |
| `class_id` | int | COCO类别ID |
| `class_name` | str | 类别名称 |
| `confidence` | float | 置信度 |
| `bbox` | object | 边界框 {x1, y1, x2, y2} |
