# Docker 部署指南

本文档说明如何使用 Docker 一键部署校园安全监控系统（GPU 加速）。

## 前置要求

- Docker 20.10+
- Docker Compose v2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA 驱动 + CUDA 12.x 兼容 GPU
- 至少 8 GB 显存（仅 YOLO）、16 GB+（开启 MLLM）

### 安装 NVIDIA Container Toolkit（Ubuntu）

```bash
# 添加仓库
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证 GPU 可见
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## 快速部署

### 1. 准备配置与模型

```bash
cd campus-safety
cp .env.example .env
cp config.example.yaml configs/default.yaml

# 下载模型到 models/ 目录（挂载进容器）
python scripts/download_models.py
```

### 2. 构建前端（可选，用于 Web UI）

```bash
cd frontend
npm ci
npm run build
cd ..
```

### 3. 构建并启动

```bash
# 构建镜像
docker compose build

# 启动后端 + 前端（后台运行）
docker compose up -d

# 仅启动后端 API
docker compose up -d backend
```

- API：http://localhost:8000/docs
- 前端（需已 build）：http://localhost:8080

### 4. 查看日志

```bash
docker compose logs -f backend
docker compose logs -f frontend
```

### 5. 停止服务

```bash
docker compose down
```

## 环境变量

在 `.env` 中配置（`docker compose` 自动读取）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BACKEND_PORT` | API 端口映射 | 8000 |
| `FRONTEND_PORT` | 前端端口映射 | 8080 |
| `API_KEY` | API 认证密钥（生产环境建议设置） | 空 |
| `YOLO_DEVICE` | 推理设备 | cuda |
| `HF_ENDPOINT` | HuggingFace 镜像 | 空 |

## 卷挂载

| 宿主机路径 | 容器路径 | 用途 |
|-----------|---------|------|
| `./models` | `/app/models` | YOLO / MLLM 权重 |
| `./outputs` | `/app/outputs` | 事件数据库、快照 |
| `./configs` | `/app/configs` | 运行时配置 |

## 常见问题

### GPU 不可用

```bash
# 检查 toolkit
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# 若失败，确认 docker daemon.json 含 nvidia runtime
```

### 模型未找到

确保 `models/yolo12s.pt` 存在于宿主机 `models/` 目录，或首次启动后进入容器下载：

```bash
docker compose exec backend python scripts/download_models.py
```

### 显存不足

- 保持 `mllm.enabled: false`（默认）
- 降低 `model.imgsz` 或 `model.inference_scale`

### 前端 404

需先执行 `cd frontend && npm run build` 生成 `frontend/dist/`。

## 仅 API 模式（无前端容器）

```bash
docker compose up -d backend
curl http://localhost:8000/health
```
