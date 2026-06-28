# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared SQLite base class for event and alarm storage.

Provides connection management with WAL mode, thread safety, and schema
migration helpers used by both EventsStore and AlarmStore.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from contextlib import suppress

logger = logging.getLogger(__name__)


class SQLiteBase:
    """Shared SQLite connection management with WAL mode, thread safety, and schema helpers."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-8000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=268435456")
        self.conn.execute("PRAGMA page_size=4096")
        self.conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        self._lock = threading.Lock()

    def ensure_column_exists(
        self, cur: sqlite3.Cursor, table: str, col_name: str, col_type: str
    ) -> None:
        """Add column to table if it doesn't already exist."""
        cur.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cur.fetchall()}
        if col_name not in columns:
            try:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                logger.info("Added missing column '%s' to %s", col_name, table)
            except sqlite3.OperationalError as e:
                logger.warning(
                    "Could not add column '%s' to %s: %s", col_name, table, e
                )

    def close(self) -> None:
        with suppress(sqlite3.Error):
            self.conn.close()
