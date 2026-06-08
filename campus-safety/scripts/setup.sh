#!/bin/bash
# Setup script for YOLO Course Design Project

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_DIR/models"

echo "=== YOLO Course Design Setup ==="
echo ""

# Check Python version
echo "[1/4] Checking Python version..."
python3 --version || python --version

# Create virtual environment
echo "[2/4] Setting up virtual environment..."
if [ ! -d "$PROJECT_DIR/venv" ]; then
    python3 -m venv "$PROJECT_DIR/venv" 2>/dev/null || python -m venv "$PROJECT_DIR/venv"
fi
source "$PROJECT_DIR/venv/bin/activate" || source "$PROJECT_DIR/venv/Scripts/activate"

# Install dependencies
echo "[3/4] Installing Python dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Link model files
echo "[4/4] Linking model files..."
mkdir -p "$MODELS_DIR"
if [ -f "$PARENT_DIR/models/yolo11x.pt" ]; then
    ln -sf "$PARENT_DIR/models/yolo11x.pt" "$MODELS_DIR/yolo11x.pt"
    echo "  -> Linked yolo11x.pt"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the backend:"
echo "  cd $PROJECT_DIR && uvicorn backend.main:app --reload"
echo ""
echo "To start the frontend:"
echo "  cd $PROJECT_DIR/frontend && npm install && npm run dev"
