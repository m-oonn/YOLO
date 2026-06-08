"""Tests for EventsStore."""

import os
import sys

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

import pytest

from core.events_store import EventsStore
from core.rules import Event


@pytest.fixture
def store(tmp_path):
    """Create a temporary EventsStore for each test."""
    db_path = str(tmp_path / "test_events.db")
    s = EventsStore(db_path)
    yield s
    s.close()


def _make_event(
    event_type="running",
    timestamp_s=100.0,
    frame_index=1,
    track_id=1,
    zone_name=None,
    confidence=0.9,
    bbox=None,
    description=None,
    extra=None,
):
    return Event(
        event_type=event_type,
        timestamp_s=timestamp_s,
        frame_index=frame_index,
        track_id=track_id,
        zone_name=zone_name,
        confidence=confidence,
        bbox=bbox or [0, 0, 10, 20],
        description=description or f"{event_type} (track {track_id})",
        extra=extra or {},
    )


def test_record_single_event(store):
    event = _make_event()
    assert store.record(event) is True
    assert store.count() == 1


def test_record_batch(store):
    events = [(_make_event(event_type="running", track_id=i), None) for i in range(5)]
    count = store.record_batch(events)
    assert count == 5
    assert store.count() == 5


def test_record_batch_empty(store):
    assert store.record_batch([]) == 0


def test_record_with_snapshot_path(store):
    event = _make_event()
    assert store.record(event, snapshot_path="/tmp/snap.jpg") is True
    results = store.query()
    assert len(results) == 1
    assert results[0]["snapshot_path"] == "/tmp/snap.jpg"


def test_query_all(store):
    for i in range(10):
        store.record(_make_event(timestamp_s=100.0 + i, frame_index=i))
    results = store.query()
    assert len(results) == 10
    # Default sort is DESC by timestamp
    assert results[0]["frame_index"] == 9
    assert results[-1]["frame_index"] == 0


def test_query_with_limit_offset(store):
    for i in range(10):
        store.record(_make_event(timestamp_s=100.0 + i, frame_index=i))
    results = store.query(limit=3, offset=2)
    assert len(results) == 3
    # DESC order, offset 2 means frame_index 7, 6, 5
    assert results[0]["frame_index"] == 7


def test_query_by_event_type(store):
    store.record(_make_event(event_type="running"))
    store.record(_make_event(event_type="fall"))
    store.record(_make_event(event_type="running"))

    results = store.query(event_type="running")
    assert len(results) == 2
    all_running = all(r["event_type"] == "running" for r in results)
    assert all_running is True


def test_query_by_time_range(store):
    store.record(_make_event(timestamp_s=100.0))
    store.record(_make_event(timestamp_s=200.0))
    store.record(_make_event(timestamp_s=300.0))

    results = store.query(start_time=150.0, end_time=250.0)
    assert len(results) == 1
    assert results[0]["timestamp_s"] == 200.0


def test_query_with_total(store):
    for i in range(5):
        store.record(_make_event(timestamp_s=100.0 + i))
    results, total = store.query_with_total(limit=2, offset=0)
    assert len(results) == 2
    assert total == 5


def test_query_with_total_empty(store):
    results, total = store.query_with_total()
    assert results == []
    assert total == 0


def test_count(store):
    assert store.count() == 0
    store.record(_make_event())
    assert store.count() == 1
    store.record(_make_event())
    assert store.count() == 2


def test_count_by_type(store):
    store.record(_make_event(event_type="running"))
    store.record(_make_event(event_type="running"))
    store.record(_make_event(event_type="fall"))
    assert store.count(event_type="running") == 2
    assert store.count(event_type="fall") == 1
    assert store.count(event_type="intrusion") == 0


def test_get_stats(store):
    store.record(_make_event(event_type="running"))
    store.record(_make_event(event_type="fall"))
    store.record(_make_event(event_type="running"))

    stats = store.get_stats()
    assert stats["total_events"] == 3
    assert stats["by_type"]["running"] == 2
    assert stats["by_type"]["fall"] == 1
    assert stats["first_event"] is not None
    assert stats["last_event"] is not None


def test_get_stats_empty(store):
    stats = store.get_stats()
    assert stats["total_events"] == 0
    assert stats["by_type"] == {}
    assert stats["first_event"] is None
    assert stats["last_event"] is None


def test_delete_events_by_type(store):
    store.record(_make_event(event_type="running"))
    store.record(_make_event(event_type="fall"))
    store.record(_make_event(event_type="running"))

    deleted = store.delete_events(event_type="running")
    assert deleted == 2
    assert store.count() == 1
    assert store.count(event_type="fall") == 1


def test_delete_events_before_timestamp(store):
    store.record(_make_event(timestamp_s=100.0))
    store.record(_make_event(timestamp_s=200.0))
    store.record(_make_event(timestamp_s=300.0))

    deleted = store.delete_events(before_timestamp=250.0)
    assert deleted == 2
    assert store.count() == 1


def test_delete_events_combined_filters(store):
    store.record(_make_event(event_type="running", timestamp_s=100.0))
    store.record(_make_event(event_type="running", timestamp_s=300.0))
    store.record(_make_event(event_type="fall", timestamp_s=100.0))

    deleted = store.delete_events(event_type="running", before_timestamp=200.0)
    assert deleted == 1
    assert store.count() == 2


def test_clear_all(store):
    for _ in range(5):
        store.record(_make_event())
    assert store.count() == 5
    deleted = store.clear_all()
    assert deleted == 5
    assert store.count() == 0


def test_query_returns_parsed_json(store):
    bbox = [10, 20, 30, 40]
    extra = {"detail": "test"}
    store.record(_make_event(bbox=bbox, extra=extra))
    results = store.query()
    assert results[0]["bbox"] == bbox
    assert results[0]["extra"] == extra


def test_query_returns_bbox_none_when_empty(store):
    event = _make_event()
    # Override bbox to None
    event = Event(
        event_type="running",
        timestamp_s=100.0,
        frame_index=1,
        track_id=1,
        bbox=None,
        description="no bbox",
    )
    store.record(event)
    results = store.query()
    assert results[0]["bbox"] is None


def test_schema_is_created_automatically(tmp_path):
    """EventsStore should create the database and schema on init."""
    db_path = str(tmp_path / "new_events.db")
    assert not os.path.exists(db_path)
    s = EventsStore(db_path)
    assert os.path.exists(db_path)
    s.close()


def test_empty_query_filters(store):
    """Query with no filters should return all events sorted by timestamp DESC."""
    for i in range(3):
        store.record(_make_event(timestamp_s=float(300 - i * 100), frame_index=i))
    results = store.query()
    assert len(results) == 3
    assert results[0]["frame_index"] == 0  # timestamp 300
    assert results[2]["frame_index"] == 2  # timestamp 100
