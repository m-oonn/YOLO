# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【后端API】limiter.py — API限流器
# 依赖：slowapi（可选，基于 Flask-Limiter 的 FastAPI 适配）
# 被调用：backend/main.py（全局应用限流）
# 核心职责：
#   ① 默认限制：每IP每分钟60次请求
#   ② 敏感端点可配置更严格限制（如 /api/detection/start）
#   ③ 限流超限返回 HTTP 429 Too Many Requests
# ──────────────────────────────────────────────────────────

"""Shared rate limiter for the application.

Provides a global SlowAPI rate limiter instance that limits API requests
to 60 per minute per client IP address by default. Individual routes can
override the default limit via the ``@app_limiter.limit()`` decorator.
"""

# Try to import slowapi, otherwise provide a dummy limiter
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    app_limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
except ImportError:
    # Dummy limiter decorator that does nothing
    def dummy_limiter(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            return dummy_limiter
    
    app_limiter = DummyLimiter()
