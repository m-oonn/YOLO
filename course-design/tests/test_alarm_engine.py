# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the alarm engine: classification, suppression, aggregation, escalation."""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from core.alarm_engine import (
    AlarmConfig,
    AlarmEngine,
    AlarmLevel,
    AlarmStatus,
    AlarmStore,
    EVENT_LEVEL_MAP,
    EVENT_LEVEL_LABELS,
)
from core.rules import Event


@pytest.fixture
def alarm_store(tmp_path):
    db_path = str(tmp_path / "test_alarms.db")
    store = AlarmStore(db_path)
    yield store
    store.close()


@pytest.fixture
def alarm_engine(alarm_store):
    config = AlarmConfig(
        suppress_window_s=0.5,
        aggregate_window_s=1.0,
        escalate_after_s=2.0,
        max_alarms_per_minute=100,
    )
    engine = AlarmEngine(alarm_store, config)
    engine.start()
    yield engine
    engine.stop()


def _make_event(event_type="fight", timestamp_s=None, track_id=None, zone_name=None):
    return Event(
        event_type=event_type,
        timestamp_s=timestamp_s or time.time(),
        frame_index=1,
        track_id=track_id,
        zone_name=zone_name,
        description=f"Test {event_type}",
    )


class TestAlarmLevelClassification:
    def test_fight_is_critical(self):
        assert EVENT_LEVEL_MAP["fight"] == AlarmLevel.CRITICAL

    def test_fall_is_critical(self):
        assert EVENT_LEVEL_MAP["fall"] == AlarmLevel.CRITICAL

    def test_intrusion_is_warning(self):
        assert EVENT_LEVEL_MAP["intrusion"] == AlarmLevel.WARNING

    def test_crowd_is_warning(self):
        assert EVENT_LEVEL_MAP["crowd"] == AlarmLevel.WARNING

    def test_running_is_info(self):
        assert EVENT_LEVEL_MAP["running"] == AlarmLevel.INFO

    def test_unknown_is_info(self):
        assert EVENT_LEVEL_MAP.get("unknown", AlarmLevel.INFO) == AlarmLevel.INFO

    def test_level_labels(self):
        assert EVENT_LEVEL_LABELS[AlarmLevel.CRITICAL] == "严重"
        assert EVENT_LEVEL_LABELS[AlarmLevel.WARNING] == "警告"
        assert EVENT_LEVEL_LABELS[AlarmLevel.INFO] == "提示"

    def test_level_override(self, alarm_engine):
        alarm_engine.config.level_overrides = {"running": 3}
        event = _make_event("running")
        level = alarm_engine._classify_level(event)
        assert level == AlarmLevel.CRITICAL


class TestAlarmSuppression:
    def test_suppress_duplicate_within_window(self, alarm_engine):
        event = _make_event("fight")
        alarm1 = alarm_engine.process_event(event)
        assert alarm1 is not None
        alarm2 = alarm_engine.process_event(event)
        assert alarm2 is None

    def test_allow_after_suppress_window(self, alarm_engine):
        event = _make_event("fight")
        alarm1 = alarm_engine.process_event(event)
        assert alarm1 is not None
        time.sleep(0.6)
        alarm2 = alarm_engine.process_event(event)
        assert alarm2 is not None

    def test_different_keys_not_suppressed(self, alarm_engine):
        event1 = _make_event("fight", track_id=1)
        event2 = _make_event("fight", track_id=2)
        alarm1 = alarm_engine.process_event(event1)
        alarm2 = alarm_engine.process_event(event2)
        assert alarm1 is not None
        assert alarm2 is not None

    def test_disabled_engine_returns_none(self, alarm_engine):
        alarm_engine.config.enabled = False
        event = _make_event("fight")
        result = alarm_engine.process_event(event)
        assert result is None
        alarm_engine.config.enabled = True


class TestAlarmAggregation:
    def test_aggregate_same_key_within_window(self, alarm_engine):
        event1 = _make_event("fight", track_id=1)
        alarm1 = alarm_engine.process_event(event1)
        assert alarm1 is not None
        assert alarm1.count == 1
        time.sleep(0.6)
        event2 = _make_event("fight", track_id=1)
        alarm2 = alarm_engine.process_event(event2)
        assert alarm2 is not None
        assert alarm2.count == 2

    def test_new_alarm_after_aggregate_window(self, alarm_engine):
        event1 = _make_event("fight", track_id=1)
        alarm1 = alarm_engine.process_event(event1)
        assert alarm1 is not None
        time.sleep(1.2)
        event2 = _make_event("fight", track_id=1)
        alarm2 = alarm_engine.process_event(event2)
        assert alarm2 is not None
        assert alarm2.count == 1


class TestAlarmRateLimit:
    def test_rate_limit_blocks_excess(self, alarm_engine):
        alarm_engine.config.max_alarms_per_minute = 3
        for i in range(5):
            event = _make_event("fight", track_id=i)
            alarm_engine.process_event(event)
        alarms, total = alarm_engine.get_alarms()
        assert total <= 3


class TestAlarmStore:
    def test_insert_and_query(self, alarm_store):
        from core.alarm_engine import Alarm

        alarm = Alarm(
            alarm_key="fight:tid1",
            event_type="fight",
            level=AlarmLevel.CRITICAL,
            status=AlarmStatus.ACTIVE,
            count=1,
            first_event_time=time.time(),
            last_event_time=time.time(),
            description="Test fight",
        )
        alarm_id = alarm_store.insert(alarm)
        assert alarm_id > 0
        results, total = alarm_store.query()
        assert total == 1
        assert results[0]["event_type"] == "fight"

    def test_query_with_status_filter(self, alarm_store):
        from core.alarm_engine import Alarm

        for status in ["active", "resolved"]:
            alarm = Alarm(
                alarm_key=f"fight:{status}",
                event_type="fight",
                level=AlarmLevel.CRITICAL,
                status=status,
                count=1,
                first_event_time=time.time(),
                last_event_time=time.time(),
            )
            alarm_store.insert(alarm)
        results, total = alarm_store.query(status="active")
        assert total == 1
        assert results[0]["status"] == "active"

    def test_acknowledge_alarm(self, alarm_store):
        from core.alarm_engine import Alarm

        alarm = Alarm(
            alarm_key="fight:tid1",
            event_type="fight",
            level=AlarmLevel.CRITICAL,
            status=AlarmStatus.ACTIVE,
            count=1,
            first_event_time=time.time(),
            last_event_time=time.time(),
        )
        alarm_id = alarm_store.insert(alarm)
        success = alarm_store.acknowledge(alarm_id)
        assert success
        results, _ = alarm_store.query(status="acknowledged")
        assert len(results) == 1

    def test_resolve_alarm(self, alarm_store):
        from core.alarm_engine import Alarm

        alarm = Alarm(
            alarm_key="fight:tid1",
            event_type="fight",
            level=AlarmLevel.CRITICAL,
            status=AlarmStatus.ACTIVE,
            count=1,
            first_event_time=time.time(),
            last_event_time=time.time(),
        )
        alarm_id = alarm_store.insert(alarm)
        success = alarm_store.resolve(alarm_id)
        assert success
        results, _ = alarm_store.query(status="resolved")
        assert len(results) == 1

    def test_get_stats(self, alarm_store):
        from core.alarm_engine import Alarm

        for et in ["fight", "fall"]:
            alarm = Alarm(
                alarm_key=f"{et}:tid1",
                event_type=et,
                level=AlarmLevel.CRITICAL,
                status=AlarmStatus.ACTIVE,
                count=1,
                first_event_time=time.time(),
                last_event_time=time.time(),
            )
            alarm_store.insert(alarm)
        stats = alarm_store.get_stats()
        assert stats["active_count"] == 2
        assert stats["total_alarms"] == 2

    def test_get_active_by_key(self, alarm_store):
        from core.alarm_engine import Alarm

        alarm = Alarm(
            alarm_key="fight:tid1",
            event_type="fight",
            level=AlarmLevel.CRITICAL,
            status=AlarmStatus.ACTIVE,
            count=1,
            first_event_time=time.time(),
            last_event_time=time.time(),
        )
        alarm_store.insert(alarm)
        found = alarm_store.get_active_by_key("fight:tid1")
        assert found is not None
        assert found.event_type == "fight"

    def test_resolve_by_type(self, alarm_store):
        from core.alarm_engine import Alarm

        for i in range(3):
            alarm = Alarm(
                alarm_key=f"fight:tid{i}",
                event_type="fight",
                level=AlarmLevel.CRITICAL,
                status=AlarmStatus.ACTIVE,
                count=1,
                first_event_time=time.time(),
                last_event_time=time.time(),
            )
            alarm_store.insert(alarm)
        count = alarm_store.resolve_by_type("fight")
        assert count == 3


class TestAlarmEngineIntegration:
    def test_process_events_batch(self, alarm_engine):
        events = [_make_event("fight", track_id=i) for i in range(3)]
        alarms = alarm_engine.process_events(events)
        assert len(alarms) == 3

    def test_alarm_key_format(self, alarm_engine):
        event = _make_event("fight", track_id=5, zone_name="zone1")
        key = alarm_engine._make_alarm_key(event)
        assert key == "fight:zone1:tid5"

    def test_alarm_key_no_zone(self, alarm_engine):
        event = _make_event("fight", track_id=3)
        key = alarm_engine._make_alarm_key(event)
        assert key == "fight:tid3"

    def test_acknowledge_via_engine(self, alarm_engine):
        event = _make_event("fight")
        alarm = alarm_engine.process_event(event)
        assert alarm is not None
        success = alarm_engine.acknowledge(alarm.id)
        assert success

    def test_resolve_via_engine(self, alarm_engine):
        event = _make_event("fight")
        alarm = alarm_engine.process_event(event)
        assert alarm is not None
        success = alarm_engine.resolve(alarm.id)
        assert success
