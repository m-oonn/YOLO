"""Tests for backend store singleton."""

import os
import sys

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

import pytest

from backend.store import close_store, get_store
from core.events_store import EventsStore


@pytest.fixture(autouse=True)
def cleanup_store():
    """Reset store singleton after each test."""
    import backend.store as store_module

    store_module._store = None
    yield
    if store_module._store is not None:
        store_module._store.close()
        store_module._store = None


class TestGetStore:
    """Test get_store singleton behavior."""

    def test_get_store_creates_instance(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = get_store(db_path)
        assert isinstance(store, EventsStore)

    def test_get_store_returns_same_instance(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s1 = get_store(db_path)
        s2 = get_store()
        assert s1 is s2

    def test_get_store_default_path(self):
        store = get_store()
        assert store.conn is not None
        close_store()

    def test_get_store_explicit_path(self, tmp_path):
        db_path = str(tmp_path / "custom.db")
        store = get_store(db_path)
        assert isinstance(store, EventsStore)


class TestCloseStore:
    """Test close_store behavior."""

    def test_close_store_when_open(self, tmp_path):
        import backend.store as store_module

        db_path = str(tmp_path / "test.db")
        store = get_store(db_path)
        assert store.conn is not None
        close_store()
        assert store_module._store is None

    def test_close_store_when_closed(self):
        import backend.store as store_module

        # Ensure no store exists
        store_module._store = None
        close_store()  # Should not raise

    def test_close_then_get_new(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s1 = get_store(db_path)
        close_store()
        s2 = get_store(db_path)
        assert s1 is not s2
