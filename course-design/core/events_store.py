# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

"""Event storage using SQLite for persistence and querying."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import suppress
from typing import Any

from .rules import Event

logger = logging.getLogger(__name__)


class EventsStore:
    """Stores detection events in SQLite with indexing for efficient querying."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read/write performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Reduce fsync calls for better throughput
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._vacuum_counter = 0
        self._create_schema()

    def _create_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                timestamp_s REAL NOT NULL,
                frame_index INTEGER NOT NULL,
                track_id INTEGER,
                zone_name TEXT,
                confidence REAL,
                bbox_json TEXT,
                snapshot_path TEXT,
                extra_json TEXT,
                description TEXT
            );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp_s);"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, timestamp_s);"
        )
        self.conn.commit()

    def record_batch(self, events_with_paths: list[tuple[Event, str | None]]) -> int:
        """Insert multiple events in a single transaction. Returns count of successful inserts."""
        if not events_with_paths:
            return 0
        success = 0
        try:
            with self._lock:
                cur = self.conn.cursor()
                for event, snapshot_path in events_with_paths:
                    bbox_json = json.dumps(event.bbox) if event.bbox else None
                    extra_json = json.dumps(event.extra or {})
                    cur.execute(
                        """INSERT INTO events
                           (event_type, timestamp_s, frame_index, track_id, zone_name,
                            confidence, bbox_json, snapshot_path, extra_json, description)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            event.event_type,
                            event.timestamp_s,
                            event.frame_index,
                            event.track_id,
                            event.zone_name,
                            event.confidence,
                            bbox_json,
                            snapshot_path,
                            extra_json,
                            event.description,
                        ),
                    )
                    success += 1
                self.conn.commit()
                self._vacuum_counter += success
                if self._vacuum_counter >= 1000:
                    self._vacuum_counter = 0
                    self.conn.execute("VACUUM")
                    logger.debug("Database VACUUM completed")
            return success
        except sqlite3.Error:
            logger.exception("Database error recording event batch")
            return success

    def record(self, event: Event, snapshot_path: str | None = None) -> bool:
        """Store a single event. Thread-safe via internal lock. Returns True on success."""
        try:
            bbox_json = json.dumps(event.bbox) if event.bbox else None
            extra_json = json.dumps(event.extra or {})
            with self._lock:
                cur = self.conn.cursor()
                cur.execute(
                    """INSERT INTO events
                       (event_type, timestamp_s, frame_index, track_id, zone_name,
                        confidence, bbox_json, snapshot_path, extra_json, description)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event.event_type,
                        event.timestamp_s,
                        event.frame_index,
                        event.track_id,
                        event.zone_name,
                        event.confidence,
                        bbox_json,
                        snapshot_path,
                        extra_json,
                        event.description,
                    ),
                )
                self.conn.commit()
                # Periodic VACUUM to prevent unbounded DB growth
                self._vacuum_counter += 1
                if self._vacuum_counter >= 1000:
                    self._vacuum_counter = 0
                    self.conn.execute("VACUUM")
                    logger.debug("Database VACUUM completed")
            return True
        except sqlite3.Error:
            logger.exception("Database error recording event")
            return False

    def _build_where(self, event_type, start_time, end_time):
        """Build WHERE clause and params from filters."""
        conditions = []
        params: list[Any] = []
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if start_time is not None:
            conditions.append("timestamp_s >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("timestamp_s <= ?")
            params.append(end_time)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        return where, params

    def _rows_to_dicts(self, rows) -> list[dict[str, Any]]:
        """Convert SQLite rows to dicts with parsed JSON fields."""
        results = []
        for row in rows:
            d = dict(row)
            if d["bbox_json"]:
                d["bbox"] = json.loads(d["bbox_json"])
            else:
                d["bbox"] = None
            if d["extra_json"]:
                d["extra"] = json.loads(d["extra_json"])
            else:
                d["extra"] = {}
            del d["bbox_json"]
            del d["extra_json"]
            results.append(d)
        return results

    def query(
        self,
        event_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """Query events with optional filters. Thread-safe via internal lock."""
        where, params = self._build_where(event_type, start_time, end_time)
        with self._lock:
            cur = self.conn.cursor()
            # NOTE: `where` is constructed from controlled string literals only.
            # All user-provided values are passed as ? placeholders below.
            sql = "SELECT * FROM events{w} ORDER BY timestamp_s DESC LIMIT ? OFFSET ?"
            cur.execute(sql.format(w=where), params + [limit, offset])
            return self._rows_to_dicts(cur.fetchall())

    def query_with_total(
        self,
        event_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query events with total count in a single query using window function."""
        where, params = self._build_where(event_type, start_time, end_time)
        with self._lock:
            cur = self.conn.cursor()
            # NOTE: `where` is constructed from controlled string literals only.
            sql = (
                "SELECT *, COUNT(*) OVER() as _total FROM events{w} "
                "ORDER BY timestamp_s DESC LIMIT ? OFFSET ?"
            )
            cur.execute(sql.format(w=where), params + [limit, offset])
            rows = cur.fetchall()
            total = rows[0]["_total"] if rows else 0
            results = self._rows_to_dicts(rows)
            for r in results:
                r.pop("_total", None)
            return results, total

    def count(self, event_type: str | None = None) -> int:
        """Get total event count with optional type filter. Thread-safe."""
        with self._lock:
            cur = self.conn.cursor()
            if event_type:
                cur.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = ?", (event_type,)
                )
            else:
                cur.execute("SELECT COUNT(*) FROM events")
            return cur.fetchone()[0]

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics about stored events. Thread-safe."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT event_type, COUNT(*) as count FROM events GROUP BY event_type"
            )
            type_counts = {row["event_type"]: row["count"] for row in cur.fetchall()}
            cur.execute(
                "SELECT MIN(timestamp_s) as first, MAX(timestamp_s) as last FROM events"
            )
            time_range = cur.fetchone()
            return {
                "total_events": sum(type_counts.values()),
                "by_type": type_counts,
                "first_event": time_range["first"] if time_range else None,
                "last_event": time_range["last"] if time_range else None,
            }

    def delete_events(
        self, event_type: str | None = None, before_timestamp: float | None = None
    ) -> int:
        """Delete events matching filters. Returns count of deleted rows. Thread-safe."""
        conditions = []
        params: list[Any] = []
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if before_timestamp is not None:
            conditions.append("timestamp_s < ?")
            params.append(before_timestamp)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._lock:
            cur = self.conn.cursor()
            # NOTE: `where` is constructed from controlled string literals only.
            cur.execute(f"DELETE FROM events{where}", params)
            deleted = cur.rowcount
            self.conn.commit()
            return deleted

    def clear_all(self) -> int:
        """Delete ALL events. Returns count of deleted rows. Thread-safe."""
        return self.delete_events()

    def close(self) -> None:
        with suppress(Exception):
            self.conn.close()
