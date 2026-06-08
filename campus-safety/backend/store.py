# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【后端API】store.py — EventsStore 单例
# 依赖：core/events_store.py
# 被调用：所有 API 路由（统一使用同一个存储实例）
# 核心职责：
#   ① 懒初始化 EventsStore（首次访问时创建，之后复用）
#   ② 进程级单例（整个FastAPI应用共享一个数据库连接）
#   ③ 确保 pipeline 和 API 路由读写同一个事件数据库
# ──────────────────────────────────────────────────────────

"""Shared EventsStore singleton for cross-module access.

Provides a process-wide EventsStore instance that is lazily initialised
on first access and reused for the lifetime of the application. All API
routers and the detection pipeline share the same store so that events
written by the pipeline are immediately visible to the API layer.
"""

from __future__ import annotations

import os

from core.events_store import EventsStore

_store: EventsStore | None = None


def get_store(db_path: str | None = None) -> EventsStore:
    """Get or create the shared EventsStore singleton."""
    global _store
    if _store is None:
        path = db_path or os.path.join(os.getcwd(), "outputs", "events.db")
        _store = EventsStore(db_path=path)
    return _store


def close_store() -> None:
    """Close the shared store if open."""
    global _store
    if _store:
        _store.close()
        _store = None
