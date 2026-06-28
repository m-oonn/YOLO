# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for VideoClipRecorder."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.video_archiver import ArchivedClip, VideoClipRecorder


@pytest.fixture
def tmp_output_dir():
    # On Windows the SQLite/VideoWriter handles may still be held when the
    # test ends, causing PermissionError during temp-dir cleanup.  Swallow
    # that race instead of failing the teardown.
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        with contextlib.suppress(PermissionError):
            shutil.rmtree(d)


def _make_frame(w=160, h=120):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestInit:
    def test_defaults(self, tmp_output_dir):
        r = VideoClipRecorder(output_dir=tmp_output_dir)
        assert r.enabled
        assert r._pre_seconds == 8
        assert r._clip_fps == 15
        r.shutdown()

    def test_custom(self, tmp_output_dir):
        r = VideoClipRecorder(
            pre_seconds=5,
            post_seconds=2,
            clip_fps=10,
            max_clips=10,
            output_dir=tmp_output_dir,
        )
        assert r._pre_seconds == 5
        assert r._max_clips == 10
        r.shutdown()


class TestRingBuffer:
    def test_feed_appends(self, tmp_output_dir):
        r = VideoClipRecorder(pre_seconds=2, clip_fps=5, output_dir=tmp_output_dir)
        for _ in range(15):
            r.feed_frame(_make_frame())
        assert len(r._buffer) == 10  # 2*5 cap
        r.shutdown()

    def test_timestamp(self, tmp_output_dir):
        r = VideoClipRecorder(pre_seconds=2, clip_fps=5, output_dir=tmp_output_dir)
        r.feed_frame(_make_frame(), timestamp=100.5)
        _, ts = r._buffer[0]
        assert ts == 100.5
        r.shutdown()


class TestTrigger:
    def test_starts_recording(self, tmp_output_dir):
        r = VideoClipRecorder(
            pre_seconds=2, post_seconds=1, clip_fps=5, output_dir=tmp_output_dir
        )
        for _ in range(15):
            r.feed_frame(_make_frame())
        cid = r.trigger_clip("fall")
        assert cid and cid.startswith("fall_")
        assert r._recording
        r.shutdown()

    def test_merges_during_recording(self, tmp_output_dir):
        r = VideoClipRecorder(
            pre_seconds=2, post_seconds=3, clip_fps=5, output_dir=tmp_output_dir
        )
        for _ in range(15):
            r.feed_frame(_make_frame())
        assert r.trigger_clip("fall") is not None
        assert r.trigger_clip("fight") is None  # merged
        assert "fight" in r._current_tags
        r.shutdown()


class TestClipWrite:
    def test_writes_mp4(self, tmp_output_dir):
        r = VideoClipRecorder(
            pre_seconds=1,
            post_seconds=1,
            clip_fps=5,
            jpeg_quality=30,
            output_dir=tmp_output_dir,
        )
        for _ in range(10):
            r.feed_frame(_make_frame())
        r.trigger_clip("run")
        for _ in range(10):
            r.feed_frame(_make_frame())
        r.shutdown()
        mp4s = list(Path(tmp_output_dir).glob("*.mp4"))
        assert len(mp4s) >= 1


class TestCRUD:
    def test_get_clips(self, tmp_output_dir):
        r = VideoClipRecorder(
            pre_seconds=1,
            post_seconds=1,
            clip_fps=5,
            jpeg_quality=30,
            output_dir=tmp_output_dir,
        )
        for _ in range(10):
            r.feed_frame(_make_frame())
        r.trigger_clip("fight")
        for _ in range(10):
            r.feed_frame(_make_frame())
        r.shutdown()
        clips = r.get_clips()
        assert len(clips) > 0
        assert isinstance(clips[0], ArchivedClip)
        assert os.path.exists(clips[0].file_path)

    def test_filter_by_type(self, tmp_output_dir):
        r = VideoClipRecorder(
            pre_seconds=1,
            post_seconds=1,
            clip_fps=5,
            jpeg_quality=30,
            output_dir=tmp_output_dir,
        )
        for _ in range(10):
            r.feed_frame(_make_frame())
        r.trigger_clip("fall")
        for _ in range(10):
            r.feed_frame(_make_frame())
        r.shutdown()
        assert len(r.get_clips(event_type="fall")) > 0
        assert len(r.get_clips(event_type="run")) == 0

    def test_delete(self, tmp_output_dir):
        r = VideoClipRecorder(
            pre_seconds=1,
            post_seconds=1,
            clip_fps=5,
            jpeg_quality=30,
            output_dir=tmp_output_dir,
        )
        for _ in range(10):
            r.feed_frame(_make_frame())
        r.trigger_clip("intrusion")
        for _ in range(10):
            r.feed_frame(_make_frame())
        r.shutdown()
        clips = r.get_clips()
        assert clips
        assert r.delete_clip(clips[0].clip_id)
        assert not os.path.exists(clips[0].file_path)

    def test_delete_nonexistent(self, tmp_output_dir):
        r = VideoClipRecorder(output_dir=tmp_output_dir)
        assert not r.delete_clip("nonexistent")
        r.shutdown()

    def test_count(self, tmp_output_dir):
        r = VideoClipRecorder(
            pre_seconds=1,
            post_seconds=1,
            clip_fps=5,
            jpeg_quality=30,
            output_dir=tmp_output_dir,
        )
        assert r.get_clip_count() == 0
        for _ in range(10):
            r.feed_frame(_make_frame())
        r.trigger_clip("crowd")
        for _ in range(10):
            r.feed_frame(_make_frame())
        r.shutdown()
        assert r.get_clip_count() >= 1


class TestPruning:
    def test_caps_clips(self, tmp_output_dir):
        r = VideoClipRecorder(
            pre_seconds=1,
            post_seconds=1,
            clip_fps=5,
            jpeg_quality=30,
            max_clips=3,
            output_dir=tmp_output_dir,
        )
        for i in range(5):
            for _ in range(10):
                r.feed_frame(_make_frame())
            r.trigger_clip(f"e{i}")
            for _ in range(10):
                r.feed_frame(_make_frame())
        r.shutdown()
        assert r.get_clip_count() <= 3


class TestEdgeCases:
    def test_empty_trigger_no_write(self, tmp_output_dir):
        r = VideoClipRecorder(output_dir=tmp_output_dir)
        r.trigger_clip("fall")
        r.shutdown()
        assert len(list(Path(tmp_output_dir).glob("*.mp4"))) == 0

    def test_shutdown_during_recording(self, tmp_output_dir):
        r = VideoClipRecorder(
            pre_seconds=1, post_seconds=5, clip_fps=5, output_dir=tmp_output_dir
        )
        for _ in range(10):
            r.feed_frame(_make_frame())
        r.trigger_clip("fall")
        r.shutdown()  # no raise

    def test_idempotent_shutdown(self, tmp_output_dir):
        r = VideoClipRecorder(output_dir=tmp_output_dir)
        r.shutdown()
        r.shutdown()
