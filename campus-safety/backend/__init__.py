# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【后端API】模块入口
# 本目录是FastAPI后端服务的完整实现
# main.py — 应用入口 | api/ — 8个路由模块
# detection_manager.py — 检测线程管理（最复杂，42KB）
# 服务启动：uvicorn backend.main:app --host 0.0.0.0 --port 8000
# ──────────────────────────────────────────────────────────

"""
YOLO Course Design - Backend Application.

FastAPI-based REST API and WebSocket server for YOLO detection system.
"""

__version__ = "1.0.0"
