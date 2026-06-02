#!/bin/bash
# ============================================================
# Jetson Deployment Script — Campus Safety Detection System
# Run ON the Jetson device (Orin AGX/NX, JetPack 6.x)
# ============================================================
set -e

echo "=========================================="
echo " Campus Safety — Jetson Deployment"
echo "=========================================="

# --- 1. System Check ---
echo "[1/5] Checking Jetson hardware..."
if [ -f /proc/device-tree/model ]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model)
    echo "  Device: $MODEL"
else
    echo "  [ERROR] Not running on a Jetson device!"
    exit 1
fi

# Memory
RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
RAM_GB=$((RAM_KB / 1024 / 1024))
echo "  RAM: ${RAM_GB}GB"

# GPU
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# --- 2. Install Dependencies ---
echo "[2/5] Installing dependencies..."
pip install --no-cache-dir \
    ultralytics \
    torch==2.5.0 \
    torchvision==0.20.0 \
    onnx==1.16.1 \
    onnxruntime-gpu==1.19.2 \
    fastapi==0.115.0 \
    uvicorn==0.30.0 \
    opencv-python==4.10.0 \
    numpy==1.26.4 \
    2>&1 | tail -3

# --- 3. Export Models to TensorRT ---
echo "[3/5] Exporting detection model to TensorRT..."

# Already have yolo12s.pt in models/
cd "$(dirname "$0")/.."

if [ ! -f models/trt_engines/yolo12s_fp16.engine ]; then
    python scripts/export_yolo_trt.py \
        --model models/yolo12s.pt \
        --precision fp16 \
        --imgsz 640
else
    echo "  Engine already exists: models/trt_engines/yolo12s_fp16.engine"
fi

# --- 4. Configure for Jetson ---
echo "[4/5] Configuring for Jetson..."

cat > configs/jetson.yaml << 'YAMLEOF'
# Jetson Orin Production Config
model:
  path: models/trt_engines/yolo12s_fp16.engine
  device: cuda
  imgsz: 640
  conf: 0.35
  iou: 0.5

pose:
  enabled: false

mllm:
  enabled: false

rules:
  running:
    enabled: true
    speed_kmh: 12.0
    min_duration_s: 0.5
    debounce_s: 5.0
  fall:
    enabled: true
    upright_aspect_min: 1.3
    fallen_aspect_max: 0.75
    transition_window_s: 1.5
    debounce_s: 5.0
  crowd:
    enabled: true
    min_people: 5
    proximity_px: 150.0
    debounce_s: 10.0
  intrusion:
    enabled: true
    debounce_s: 5.0
    zones: []
  fight:
    enabled: true
    distance_threshold: 80
    movement_threshold: 80
    min_duration_s: 0.8
    debounce_s: 5.0

camera:
  buffer_size: 30
  fps: 30

alarm:
  enabled: true
  cooldown_s: 30.0

output:
  directory: outputs
  save_snapshots: true
YAMLEOF

echo "  Config written: configs/jetson.yaml"

# --- 5. Start System ---
echo "[5/5] Starting detection system..."
echo ""
echo "  Backend:  uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo "  Frontend: (serve built frontend, or run npm run dev)"
echo ""
echo "  Start with:"
echo "    python scripts/launcher.py --config configs/jetson.yaml"
echo ""
echo "=========================================="
echo " Deployment Complete!"
echo "=========================================="
