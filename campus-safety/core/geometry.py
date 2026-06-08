# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【核心引擎】geometry.py — 几何计算工具
# 依赖：无
# 被调用：rules.py（跌倒宽高比判断、入侵多边形判断）
# 核心职责：
#   ① point_in_polygon — 射线法判断点是否在多边形内
#   ② bbox_aspect_h_over_w — bbox宽高比计算（跌倒检测核心）
# 纯函数，无副作用，易于测试
# ──────────────────────────────────────────────────────────

"""Geometry utilities for detection and zone checking."""


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        if intersects:
            inside = not inside
    return inside


def bbox_aspect_h_over_w(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute height/width aspect ratio of a bounding box."""
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return h / w
