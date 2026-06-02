# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for mode switching between camera and video file detection.

Verifies that the system correctly handles transitions between
real-time monitoring (camera) and video detection modes.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from backend.detection_manager import DetectionManager
from core.pipeline import DetectionPipeline


@pytest.fixture
def manager():
    """Fresh DetectionManager for each test."""
    m = DetectionManager()
    # Reset internal state
    m._pipeline = None
    m._pipeline_thread = None
    m._detection_active = False
    m._current_source = None
    return m


class TestWaitForPreviousDetection:
    """Tests for DetectionManager._wait_for_previous — the cleanup coordinator."""

    def test_no_thread_returns_none(self, manager):
        """Should return None when there is no previous thread."""
        manager._pipeline_thread = None
        assert manager._wait_for_previous() is None

    def test_finished_thread_returns_none(self, manager):
        """Should return None when previous thread has already finished."""
        done = threading.Thread(target=lambda: None)
        done.start()
        done.join()
        manager._pipeline_thread = done
        assert manager._wait_for_previous() is None

    def test_does_not_hold_lock_during_join(self, manager):
        """Critical: _wait_for_previous must NOT hold _lock while calling join().
        The old thread's finally block needs that lock to complete;
        holding it during join results in a deadlock."""
        saved_thread = manager._pipeline_thread

        entered_join = threading.Event()
        lock_acquired = threading.Event()

        class SlowThread(threading.Thread):
            def run(self):
                entered_join.wait(timeout=5)
                ok = manager._lock.acquire(timeout=3.0)
                if ok:
                    manager._lock.release()
                lock_acquired.set() if ok else lock_acquired.clear()

        thread = SlowThread()
        thread.start()
        manager._pipeline_thread = thread

        try:
            waiter_result = []

            def waiter():
                waiter_result.append(manager._wait_for_previous())

            w = threading.Thread(target=waiter, daemon=True)
            w.start()

            time.sleep(0.5)
            entered_join.set()

            w.join(timeout=5.0)
            thread.join(timeout=1.0)

            assert waiter_result == [None], (
                f"_wait_for_previous returned {waiter_result}, expected [None]"
            )
            assert lock_acquired.is_set(), (
                "SlowThread could not acquire _lock during join — "
                "_wait_for_previous held the lock while joining, causing a deadlock"
            )
        finally:
            manager._pipeline_thread = saved_thread


class TestStopDetection:
    """Tests for DetectionManager.stop — ensures immediate flag cleanup."""

    def test_sets_detection_active_false_immediately(self, manager):
        """stop() must set _detection_active = False *before* returning,
        so a subsequent start() does not race with the old thread's
        finally-block cleanup."""
        manager._pipeline = MagicMock()
        manager._detection_active = True
        manager._pipeline_thread = MagicMock()
        manager._pipeline_thread.is_alive.return_value = False

        result = manager.stop()

        assert result["status"] == "stopped"
        assert manager._detection_active is False, (
            "_detection_active should be False immediately after stop()"
        )

    def test_idempotent_when_not_active(self, manager):
        """Calling stop() when no detection is active should return stopped."""
        manager._detection_active = False
        manager._pipeline = None

        result = manager.stop()
        assert result["status"] == "stopped"
        assert manager._detection_active is False

    def test_stop_delegates_to_detection_api(self):
        """The stop_detection route handler delegates to the manager."""
        from backend.api.detection import stop_detection

        result = stop_detection()
        assert "status" in result


class TestPipelineCloseDoesNotCloseStore:
    """DetectionPipeline.close() must NOT close the shared store singleton."""

    def test_close_does_not_call_store_close(self):
        """Pipeline uses a shared EventsStore singleton. Closing the pipeline
        must not close that store, otherwise subsequent detections will get
        a closed SQLite connection."""
        mock_store = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.model_path = "yolo11n.pt"
        mock_cfg.pose.enabled = False
        mock_cfg.pose.process_interval = 2
        mock_cfg.output_dir = "/tmp"
        mock_cfg.snapshots_dir = "/tmp/snapshots"
        mock_cfg.inference_scale = 1.0
        mock_cfg.jpeg_quality = 80

        with patch("core.pipeline.YOLO") as mock_yolo:
            pipeline = DetectionPipeline(mock_cfg, store=mock_store)
            pipeline.close()

        mock_store.close.assert_not_called()

    def test_no_store_provided_close_does_not_crash(self):
        """When no store is provided (pipeline creates its own), close() should
        still work without error."""
        mock_cfg = MagicMock()
        mock_cfg.model_path = "yolo11n.pt"
        mock_cfg.pose.enabled = False
        mock_cfg.pose.process_interval = 2
        mock_cfg.output_dir = "/tmp"
        mock_cfg.snapshots_dir = "/tmp/snapshots"
        mock_cfg.inference_scale = 1.0
        mock_cfg.jpeg_quality = 80

        with (
            patch("core.pipeline.YOLO"),
            patch("core.pipeline.EventsStore") as mock_events_store_cls,
        ):
            mock_events_store = MagicMock()
            mock_events_store_cls.return_value = mock_events_store
            pipeline = DetectionPipeline(mock_cfg, store=None)
            pipeline.close()

        assert pipeline.store is not None
        mock_events_store.close.assert_not_called()
