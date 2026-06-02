# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Event storage using SQLite for persistence and querying."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from typing import Any

from core.constants import EVENT_FLUSH_BATCH, VACUUM_EVENT_COUNT

from .constants import DEFAULT_PRIORITY, EVENT_PRIORITIES
from .db_base import SQLiteBase
from .rules import Event

logger = logging.getLogger(__name__)


class EventsStore(SQLiteBase):
    """Stores detection events in SQLite with indexing for efficient querying."""

    def __init__(self, db_path: str):
        super().__init__(db_path)
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
                description TEXT,
                keypoints_json TEXT,
                skeleton_count INTEGER DEFAULT 0,
                priority TEXT DEFAULT 'INFO',
                source TEXT,
                feature_blob BLOB
            );
        """)
        # Ensure all required columns exist (for databases created before schema updates)
        self.ensure_column_exists(cur, "events", "keypoints_json", "TEXT")
        self.ensure_column_exists(cur, "events", "skeleton_count", "INTEGER DEFAULT 0")
        self.ensure_column_exists(cur, "events", "priority", "TEXT DEFAULT 'INFO'")
        self.ensure_column_exists(cur, "events", "source", "TEXT")
        self.ensure_column_exists(cur, "events", "feature_blob", "BLOB")
        
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
                    keypoints_json = None
                    if event.extra and "skeleton_kps" in event.extra:
                        keypoints_json = json.dumps(event.extra.get("skeleton_kps"))
                    skel_count = 1 if keypoints_json else 0
                    priority = EVENT_PRIORITIES.get(event.event_type, DEFAULT_PRIORITY)
                    cur.execute(
                        """INSERT INTO events
                           (event_type, timestamp_s, frame_index, track_id, zone_name,
                            confidence, bbox_json, snapshot_path, extra_json, description,
                            keypoints_json, skeleton_count, priority, source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                            keypoints_json,
                            skel_count,
                            priority,
                            event.source,
                        ),
                    )
                    success += 1
                self.conn.commit()
                self._vacuum_counter += success
                if self._vacuum_counter >= VACUUM_EVENT_COUNT:
                    self._vacuum_counter = 0
                    # VACUUM 是重量级操作（重建整个数据库），放在后台线程执行
                    # 避免阻塞检测线程的热路径
                    threading.Thread(
                        target=self._do_vacuum,
                        daemon=True,
                        name="DB-VACUUM",
                    ).start()
            return success
        except sqlite3.Error:
            logger.exception("Database error recording event batch")
            return success

    def _do_vacuum(self) -> None:
        """Run incremental_vacuum in background — non-blocking, unlike VACUUM."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("PRAGMA incremental_vacuum(100)")
                logger.debug("Database incremental_vacuum completed (background)")
            finally:
                conn.close()
        except Exception:
            logger.debug("Background incremental_vacuum skipped (non-critical)")

    def record(self, event: Event, snapshot_path: str | None = None) -> bool:
        """Store a single event. Delegates to record_batch()."""
        return self.record_batch([(event, snapshot_path)]) > 0

    def _build_where(self, event_type, start_time, end_time, search_text=None):
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
        if search_text:
            conditions.append(
                "(description LIKE ? OR extra_json LIKE ? OR snapshot_path LIKE ?)")
            like = f"%{search_text}%"
            params.extend([like, like, like])
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
            if d["keypoints_json"]:
                d["keypoints"] = json.loads(d["keypoints_json"])
            else:
                d["keypoints"] = None
            del d["bbox_json"]
            del d["extra_json"]
            del d["keypoints_json"]
            results.append(d)
        return results

    def query(
        self,
        event_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        start_time: float | None = None,
        end_time: float | None = None,
        search_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query events with optional filters. Thread-safe via internal lock."""
        where, params = self._build_where(event_type, start_time, end_time, search_text)
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
        search_text: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query events with total count in a single query using window function."""
        where, params = self._build_where(event_type, start_time, end_time, search_text)
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
        super().close()
