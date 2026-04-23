# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""Shared EventsStore singleton for cross-module access."""

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
