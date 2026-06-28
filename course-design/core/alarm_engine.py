# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Alarm engine with level classification, suppression, aggregation, and escalation.

Processes events from the rules engine and generates alarms with:
- Level classification (critical/warning/info)
- Suppression (debounce per alarm key)
- Aggregation (merge similar alarms within time window)
- Escalation (auto-escalate unresolved critical alarms)
- Multi-channel notification dispatch
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from .db_base import SQLiteBase
from .rules import Event

logger = logging.getLogger(__name__)


class AlarmLevel(IntEnum):
    INFO = 1
    WARNING = 2
    CRITICAL = 3


class AlarmStatus(str):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    # Two-stage cascade states: a rule-engine alarm starts as SUSPECTED, then
    # the MLLM second stage reviews it and moves it to CONFIRMED or DISMISSED.
    SUSPECTED = "suspected"  # rule engine fired, awaiting MLLM review
    CONFIRMED = "confirmed"  # MLLM validated (or escalated) the alarm
    DISMISSED = "dismissed"  # MLLM judged it a false positive


# Statuses that represent a "live" alarm still needing attention. Includes the
# two-stage states so confirmed/suspected alarms keep showing on the dashboard.
# Excludes dismissed (false positive) and resolved (handled).
LIVE_STATUSES = ("active", "escalated", "confirmed", "suspected")
_LIVE_STATUSES_SQL = "('active','escalated','confirmed','suspected')"


EVENT_LEVEL_MAP: dict[str, AlarmLevel] = {
    "fight": AlarmLevel.CRITICAL,
    "fall": AlarmLevel.CRITICAL,
    "intrusion": AlarmLevel.WARNING,
    "running": AlarmLevel.INFO,
    "crowd": AlarmLevel.WARNING,
    "vehicle_intrusion": AlarmLevel.WARNING,
    "suspicious": AlarmLevel.WARNING,
}

EVENT_LEVEL_LABELS: dict[int, str] = {
    AlarmLevel.CRITICAL: "严重",
    AlarmLevel.WARNING: "警告",
    AlarmLevel.INFO: "提示",
}


@dataclass
class AlarmConfig:
    enabled: bool = True
    suppress_window_s: float = 30.0
    aggregate_window_s: float = 60.0
    escalate_after_s: float = 300.0
    max_alarms_per_minute: int = 20
    level_overrides: dict[str, int] = field(default_factory=dict)
    notifiers: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Alarm:
    id: int | None = None
    alarm_key: str = ""
    event_type: str = ""
    level: AlarmLevel = AlarmLevel.INFO
    status: str = AlarmStatus.ACTIVE
    count: int = 1
    first_event_time: float = 0.0
    last_event_time: float = 0.0
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    acknowledged_at: float | None = None
    resolved_at: float | None = None
    escalated_at: float | None = None
    notified_channels: list[str] = field(default_factory=list)


class AlarmStore(SQLiteBase):
    """SQLite-backed persistent storage for alarms."""

    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._create_schema()

    def _create_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alarms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alarm_key TEXT NOT NULL,
                event_type TEXT NOT NULL,
                level INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                count INTEGER NOT NULL DEFAULT 1,
                first_event_time REAL NOT NULL,
                last_event_time REAL NOT NULL,
                description TEXT,
                extra_json TEXT,
                acknowledged_at REAL,
                resolved_at REAL,
                escalated_at REAL,
                notified_channels_json TEXT
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_alarms_status ON alarms(status);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_alarms_level ON alarms(level);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_alarms_key_time ON alarms(alarm_key, last_event_time);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_alarms_event_type ON alarms(event_type);"
        )
        self.conn.commit()

    def insert(self, alarm: Alarm) -> int:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """INSERT INTO alarms
                   (alarm_key, event_type, level, status, count,
                    first_event_time, last_event_time, description, extra_json,
                    acknowledged_at, resolved_at, escalated_at, notified_channels_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    alarm.alarm_key,
                    alarm.event_type,
                    int(alarm.level),
                    alarm.status,
                    alarm.count,
                    alarm.first_event_time,
                    alarm.last_event_time,
                    alarm.description,
                    json.dumps(alarm.extra),
                    alarm.acknowledged_at,
                    alarm.resolved_at,
                    alarm.escalated_at,
                    json.dumps(alarm.notified_channels),
                ),
            )
            self.conn.commit()
            return cur.lastrowid

    def update(self, alarm: Alarm) -> None:
        if alarm.id is None:
            return
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """UPDATE alarms SET
                   alarm_key=?, event_type=?, level=?, status=?, count=?,
                    first_event_time=?, last_event_time=?, description=?, extra_json=?,
                    acknowledged_at=?, resolved_at=?, escalated_at=?, notified_channels_json=?
                   WHERE id=?""",
                (
                    alarm.alarm_key,
                    alarm.event_type,
                    int(alarm.level),
                    alarm.status,
                    alarm.count,
                    alarm.first_event_time,
                    alarm.last_event_time,
                    alarm.description,
                    json.dumps(alarm.extra),
                    alarm.acknowledged_at,
                    alarm.resolved_at,
                    alarm.escalated_at,
                    json.dumps(alarm.notified_channels),
                    alarm.id,
                ),
            )
            self.conn.commit()

    def get_active_by_key(self, alarm_key: str) -> Alarm | None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                f"SELECT * FROM alarms WHERE alarm_key = ? AND status IN {_LIVE_STATUSES_SQL} ORDER BY last_event_time DESC LIMIT 1",
                (alarm_key,),
            )
            row = cur.fetchone()
            if row:
                return self._row_to_alarm(row)
        return None

    def get_by_id(self, alarm_id: int) -> Alarm | None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM alarms WHERE id = ?", (alarm_id,))
            row = cur.fetchone()
            if row:
                return self._row_to_alarm(row)
        return None

    def query(
        self,
        status: str | None = None,
        level: int | None = None,
        event_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        conditions = []
        params: list[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if level is not None:
            conditions.append("level = ?")
            params.append(level)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._lock:
            cur = self.conn.cursor()
            sql = "SELECT *, COUNT(*) OVER() as _total FROM alarms{w} ORDER BY last_event_time DESC LIMIT ? OFFSET ?"
            cur.execute(sql.format(w=where), params + [limit, offset])
            rows = cur.fetchall()
            total = rows[0]["_total"] if rows else 0
            results = []
            for row in rows:
                d = dict(row)
                d.pop("_total", None)
                d["level_label"] = EVENT_LEVEL_LABELS.get(d["level"], "未知")
                if d.get("extra_json"):
                    d["extra"] = json.loads(d["extra_json"])
                else:
                    d["extra"] = {}
                d.pop("extra_json", None)
                if d.get("notified_channels_json"):
                    d["notified_channels"] = json.loads(d["notified_channels_json"])
                else:
                    d["notified_channels"] = []
                d.pop("notified_channels_json", None)
                results.append(d)
            return results, total

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT status, COUNT(*) as cnt FROM alarms GROUP BY status")
            by_status = {row["status"]: row["cnt"] for row in cur.fetchall()}
            cur.execute(
                f"SELECT level, COUNT(*) as cnt FROM alarms WHERE status IN {_LIVE_STATUSES_SQL} GROUP BY level"
            )
            active_by_level = {row["level"]: row["cnt"] for row in cur.fetchall()}
            cur.execute(
                f"SELECT COUNT(*) as cnt FROM alarms WHERE status IN {_LIVE_STATUSES_SQL}"
            )
            active_count = cur.fetchone()["cnt"]
            critical_count = active_by_level.get(3, 0)
            cur.execute(
                f"SELECT * FROM alarms WHERE status IN {_LIVE_STATUSES_SQL} ORDER BY last_event_time DESC LIMIT 5"
            )
            recent_rows = cur.fetchall()
            recent = []
            for row in recent_rows:
                d = dict(row)
                d.pop("extra_json", None)
                d.pop("notified_channels_json", None)
                d["level_label"] = EVENT_LEVEL_LABELS.get(d.get("level", 1), "未知")
                recent.append(d)
            return {
                "total_alarms": sum(by_status.values()),
                "by_status": by_status,
                "active_by_level": active_by_level,
                "active_count": active_count,
                "critical_count": critical_count,
                "recent": recent,
            }

    def acknowledge(self, alarm_id: int) -> bool:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE alarms SET status = ?, acknowledged_at = ? WHERE id = ? AND status IN ('active', 'escalated')",
                (AlarmStatus.ACKNOWLEDGED, time.time(), alarm_id),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def resolve(self, alarm_id: int) -> bool:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE alarms SET status = ?, resolved_at = ? WHERE id = ? AND status != 'resolved'",
                (AlarmStatus.RESOLVED, time.time(), alarm_id),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def resolve_by_type(self, event_type: str) -> int:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE alarms SET status = ?, resolved_at = ? WHERE event_type = ? AND status IN ('active', 'escalated', 'acknowledged')",
                (AlarmStatus.RESOLVED, time.time(), event_type),
            )
            self.conn.commit()
            return cur.rowcount

    def delete_before(self, before_timestamp: float) -> int:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "DELETE FROM alarms WHERE last_event_time < ?", (before_timestamp,)
            )
            deleted = cur.rowcount
            self.conn.commit()
            return deleted

    def clear_all(self) -> int:
        """Delete ALL alarms. Returns count of deleted rows."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM alarms")
            deleted = cur.rowcount
            self.conn.commit()
            return deleted

    def _row_to_alarm(self, row) -> Alarm:
        d = dict(row)
        return Alarm(
            id=d["id"],
            alarm_key=d["alarm_key"],
            event_type=d["event_type"],
            level=AlarmLevel(d["level"]),
            status=d["status"],
            count=d["count"],
            first_event_time=d["first_event_time"],
            last_event_time=d["last_event_time"],
            description=d.get("description", ""),
            extra=json.loads(d["extra_json"]) if d.get("extra_json") else {},
            acknowledged_at=d.get("acknowledged_at"),
            resolved_at=d.get("resolved_at"),
            escalated_at=d.get("escalated_at"),
            notified_channels=json.loads(d["notified_channels_json"])
            if d.get("notified_channels_json")
            else [],
        )

    def close(self) -> None:
        super().close()


class AlarmEngine:
    """Processes events into alarms with suppression, aggregation, and escalation."""

    def __init__(self, store: AlarmStore, config: AlarmConfig | None = None):
        self.store = store
        self.config = config or AlarmConfig()
        self._suppress_cache: dict[str, float] = {}
        self._rate_counter: list[float] = []
        self._cache_lock = threading.Lock()
        self._escalation_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._running = False
        # Two-stage cascade wiring. When an MLLM reviewer is attached,
        # new alarms start as SUSPECTED and wait for a verdict; otherwise
        # they go straight to ACTIVE (single-stage fallback).
        self._review_enabled = False
        self._shadow_mode = False

    def enable_review(self, shadow_mode: bool = False) -> None:
        """Turn on second-stage review. New alarms become SUSPECTED until the
        MLLM applies a verdict. In shadow_mode, a `dismiss` verdict is recorded
        but does NOT actually suppress the alarm (observe-only gray rollout)."""
        self._review_enabled = True
        self._shadow_mode = shadow_mode
        logger.info("Alarm second-stage review enabled (shadow_mode=%s)", shadow_mode)

    def start(self) -> None:
        self._running = True
        self._stop_event = threading.Event()
        self._escalation_thread = threading.Thread(
            target=self._escalation_loop, daemon=True
        )
        self._escalation_thread.start()
        logger.info("Alarm engine started")

    def stop(self) -> None:
        self._running = False
        # Wake the escalation loop immediately and wait for it to exit so it
        # can't query the store after the connection is closed.
        if self._stop_event is not None:
            self._stop_event.set()
        if self._escalation_thread is not None and self._escalation_thread.is_alive():
            self._escalation_thread.join(timeout=5.0)
        logger.info("Alarm engine stopped")

    def process_event(self, event: Event) -> Alarm | None:
        if not self.config.enabled:
            return None

        now = time.time()
        alarm_key = self._make_alarm_key(event)
        level = self._classify_level(event)

        if self._is_suppressed(alarm_key, now):
            return None

        if not self._check_rate_limit(now):
            logger.warning(
                "Alarm rate limit reached, dropping event: %s", event.event_type
            )
            return None

        existing = self.store.get_active_by_key(alarm_key)
        if (
            existing
            and (now - existing.last_event_time) < self.config.aggregate_window_s
        ):
            return self._aggregate(existing, event, now)

        alarm = Alarm(
            alarm_key=alarm_key,
            event_type=event.event_type,
            level=level,
            status=AlarmStatus.SUSPECTED
            if self._review_enabled
            else AlarmStatus.ACTIVE,
            count=1,
            first_event_time=event.timestamp_s,
            last_event_time=event.timestamp_s,
            description=event.description,
            extra=event.extra or {},
        )
        alarm.id = self.store.insert(alarm)
        with self._cache_lock:
            self._suppress_cache[alarm_key] = now
            self._rate_counter.append(now)
        logger.info(
            "New alarm: [%s] %s (level=%s)", alarm_key, event.event_type, level.name
        )
        return alarm

    def process_events(self, events: list[Event]) -> list[Alarm]:
        alarms = []
        for event in events:
            alarm = self.process_event(event)
            if alarm:
                alarms.append(alarm)
        return alarms

    def apply_verdict(
        self,
        alarm_id: int,
        verdict: str,
        confidence: float = 0.0,
        reasoning: str = "",
    ) -> Alarm | None:
        """Apply the MLLM second-stage verdict to a suspected alarm.

        This is the join point of the two-stage cascade: the first stage
        (rule engine) created a SUSPECTED alarm; here the second stage's
        decision actually changes its fate.

        - validate -> CONFIRMED
        - escalate -> CONFIRMED, level bumped up one tier
        - dismiss  -> DISMISSED (or, in shadow_mode, kept but flagged)

        The verdict + reasoning are persisted in extra_json so the link
        between the two stages is auditable in the UI.
        """
        alarm = self.store.get_by_id(alarm_id)
        if alarm is None:
            return None
        # Only review alarms still awaiting a verdict; never override a human
        # action (acknowledged/resolved) or an already-applied verdict.
        if alarm.status != AlarmStatus.SUSPECTED:
            return alarm

        alarm.extra["mllm_verdict"] = verdict
        alarm.extra["mllm_confidence"] = round(float(confidence), 3)
        alarm.extra["mllm_reasoning"] = reasoning
        alarm.extra["mllm_reviewed_at"] = time.time()

        if verdict == "dismiss":
            if self._shadow_mode:
                # Observe-only: record what the AI *would* have done, but keep
                # the alarm live so we can measure its accuracy first.
                alarm.extra["mllm_shadow_dismissed"] = True
                alarm.status = AlarmStatus.CONFIRMED
            else:
                alarm.status = AlarmStatus.DISMISSED
        elif verdict == "escalate":
            new_level = min(int(alarm.level) + 1, int(AlarmLevel.CRITICAL))
            alarm.level = AlarmLevel(new_level)
            alarm.status = AlarmStatus.CONFIRMED
        else:  # validate or anything unrecognized -> treat as confirmed
            alarm.status = AlarmStatus.CONFIRMED

        self.store.update(alarm)
        logger.info(
            "MLLM verdict '%s' applied to alarm #%d (%s) -> %s",
            verdict,
            alarm_id,
            alarm.event_type,
            alarm.status,
        )
        return alarm

    def _make_alarm_key(self, event: Event) -> str:
        parts = [event.event_type]
        if event.zone_name:
            parts.append(event.zone_name)
        if event.track_id is not None:
            parts.append(f"tid{event.track_id}")
        return ":".join(parts)

    def _classify_level(self, event: Event) -> AlarmLevel:
        if event.event_type in self.config.level_overrides:
            return AlarmLevel(self.config.level_overrides[event.event_type])
        return EVENT_LEVEL_MAP.get(event.event_type, AlarmLevel.INFO)

    def _is_suppressed(self, alarm_key: str, now: float) -> bool:
        with self._cache_lock:
            last_time = self._suppress_cache.get(alarm_key)
            suppressed = bool(
                last_time and (now - last_time) < self.config.suppress_window_s
            )
            # Periodic cleanup of expired suppression entries to prevent memory leak
            if len(self._suppress_cache) > 5000:
                cutoff = (
                    now
                    - max(self.config.suppress_window_s, self.config.aggregate_window_s)
                    * 2
                )
                expired = [k for k, t in self._suppress_cache.items() if t < cutoff]
                for k in expired:
                    del self._suppress_cache[k]
                if expired:
                    logger.debug(
                        "Cleaned %d expired suppression cache entries", len(expired)
                    )
        return suppressed

    def _check_rate_limit(self, now: float) -> bool:
        with self._cache_lock:
            cutoff = now - 60.0
            self._rate_counter = [t for t in self._rate_counter if t > cutoff]
            return len(self._rate_counter) < self.config.max_alarms_per_minute

    def _aggregate(self, existing: Alarm, event: Event, now: float) -> Alarm:
        existing.count += 1
        existing.last_event_time = event.timestamp_s
        if event.extra:
            existing.extra.update(event.extra)
        self.store.update(existing)
        with self._cache_lock:
            self._suppress_cache[existing.alarm_key] = now
            self._rate_counter.append(now)
        logger.debug(
            "Aggregated alarm: %s (count=%d)", existing.alarm_key, existing.count
        )
        return existing

    def _escalation_loop(self) -> None:
        while self._running:
            try:
                self._check_escalations()
            except Exception:
                logger.exception("Error in escalation check")
            # Interruptible wait: stop() sets the event to wake us immediately.
            if self._stop_event is not None:
                if self._stop_event.wait(timeout=30):
                    break
            else:
                time.sleep(30)

    def _check_escalations(self) -> None:
        now = time.time()
        # Escalate unresolved critical alarms whether single-stage (active) or
        # two-stage (confirmed by MLLM). Suspected alarms are still under review
        # and are not escalated yet.
        for live_status in ("active", "confirmed"):
            offset = 0
            page_size = 200
            while True:
                alarms, total = self.store.query(
                    status=live_status, limit=page_size, offset=offset
                )
                for alarm_data in alarms:
                    if alarm_data["level"] >= AlarmLevel.CRITICAL:
                        elapsed = now - alarm_data["first_event_time"]
                        if elapsed > self.config.escalate_after_s:
                            alarm = Alarm(
                                id=alarm_data["id"],
                                alarm_key=alarm_data["alarm_key"],
                                event_type=alarm_data["event_type"],
                                level=AlarmLevel(alarm_data["level"]),
                                status=AlarmStatus.ESCALATED,
                                count=alarm_data["count"],
                                first_event_time=alarm_data["first_event_time"],
                                last_event_time=alarm_data["last_event_time"],
                                description=alarm_data.get("description", ""),
                                extra=alarm_data.get("extra", {}),
                                escalated_at=now,
                            )
                            self.store.update(alarm)
                            logger.warning(
                                "Escalated alarm: %s after %.0fs",
                                alarm.alarm_key,
                                elapsed,
                            )
                offset += page_size
                if offset >= total:
                    break

    def get_alarms(self, **kwargs) -> tuple[list[dict], int]:
        return self.store.query(**kwargs)

    def get_stats(self) -> dict[str, Any]:
        return self.store.get_stats()

    def acknowledge(self, alarm_id: int) -> bool:
        return self.store.acknowledge(alarm_id)

    def resolve(self, alarm_id: int) -> bool:
        return self.store.resolve(alarm_id)
