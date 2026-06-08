# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【核心引擎】video_archiver.py — 报警视频片段录制
# 依赖：OpenCV（cv2）
# 被调用：pipeline.py（检测到事件时触发录制）
# 核心职责：
#   ① 环形缓冲区（JPEG帧内存缓存，保留最近N秒）
#   ② 事件触发时，将缓冲区+后续帧写入MP4文件
#   ③ 报警视频可供前端下载回放
# 数据流：VideoCapture帧 → 环形缓冲 → 事件触发 → MP4文件
# ──────────────────────────────────────────────────────────

"""Event-triggered video clip recording with ring buffer.

Uses a compressed ring buffer (JPEG frames in memory) to keep the last N
seconds of video.  When an event fires the buffer is flushed to disk as an
MP4 clip together with post-event frames.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PRE_SECONDS = 8
DEFAULT_POST_SECONDS = 4
DEFAULT_CLIP_FPS = 15
DEFAULT_JPEG_QUALITY = 70
DEFAULT_OUTPUT_DIR = "outputs/clips"
MAX_CLIPS_DEFAULT = 200


@dataclass
class ArchivedClip:
    """Metadata for an archived video clip."""

    clip_id: str
    event_type: str
    timestamp: float
    duration_s: float
    file_path: str
    file_size_bytes: int = 0
    event_count: int = 1
    tags: list[str] = field(default_factory=list)


class VideoClipRecorder:
    """Lightweight video clip recorder using a JPEG ring buffer.

    Stores the last *pre_seconds* of frames as compressed JPEG bytes so
    memory stays bounded (~5-15 MB for typical settings).  When an event
    is signalled the pre-event buffer is decoded, combined with post-event
    frames, and written as an MP4 file in a background thread.
    """

    def __init__(
        self,
        pre_seconds: float = DEFAULT_PRE_SECONDS,
        post_seconds: float = DEFAULT_POST_SECONDS,
        clip_fps: int = DEFAULT_CLIP_FPS,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        max_clips: int = MAX_CLIPS_DEFAULT,
    ):
        self._pre_seconds = pre_seconds
        self._post_seconds = post_seconds
        self._clip_fps = clip_fps
        self._jpeg_quality = jpeg_quality
        self._output_dir = Path(output_dir)
        self._max_clips = max_clips

        self._max_buf = int(pre_seconds * clip_fps)
        self._buffer: deque[tuple[bytes, float]] = deque(maxlen=self._max_buf)
        self._buffer_lock = threading.Lock()

        self._recording = False
        self._post_frames: list[tuple[bytes, float]] = []
        self._post_remaining = 0
        self._current_tags: list[str] = []
        self._current_event_type = ""
        self._rec_lock = threading.Lock()

        self._db_path = self._output_dir / "clips.db"
        self._db_conn: sqlite3.Connection | None = None

        self._write_executor = None
        self._clip_count = 0
        self._total_bytes = 0

    # ── public API ───────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return True

    def feed_frame(self, frame: np.ndarray, timestamp: float | None = None) -> None:
        """Push a frame into the ring buffer (call from detection thread)."""
        ts = timestamp if timestamp is not None else time.time()
        _, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        with self._buffer_lock:
            self._buffer.append((jpeg.tobytes(), ts))

        with self._rec_lock:
            if self._recording and self._post_remaining > 0:
                self._post_frames.append((jpeg.tobytes(), ts))
                self._post_remaining -= 1
                if self._post_remaining <= 0:
                    self._finalize_clip()

    def trigger_clip(
        self, event_type: str, tags: list[str] | None = None
    ) -> str | None:
        """Signal an event and start collecting post-event frames."""
        with self._rec_lock:
            if self._recording:
                if event_type not in self._current_tags:
                    self._current_tags.append(event_type)
                return None
            self._recording = True
            self._current_event_type = event_type
            self._current_tags = [event_type] + (tags or [])
            self._post_remaining = int(self._post_seconds * self._clip_fps)
            self._post_frames = []

        clip_id = (
            f"{event_type}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        )
        self._pending_clip_id = clip_id
        return clip_id

    def get_clips(
        self,
        event_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ArchivedClip]:
        """Query archived clips with optional filters."""
        self._ensure_db()
        conn = self._db_conn
        if conn is None:
            return []
        try:
            if event_type:
                rows = conn.execute(
                    "SELECT clip_id, event_type, timestamp, duration_s, file_path, "
                    "file_size_bytes, event_count, tags FROM clips "
                    "WHERE event_type = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (event_type, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT clip_id, event_type, timestamp, duration_s, file_path, "
                    "file_size_bytes, event_count, tags FROM clips "
                    "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            return [_row_to_clip(r) for r in rows]
        except Exception as e:
            logger.error("Failed to query clips: %s", e)
            return []

    def get_clip_count(self) -> int:
        self._ensure_db()
        conn = self._db_conn
        if conn is None:
            return 0
        try:
            return conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
        except Exception:
            return 0

    def delete_clip(self, clip_id: str) -> bool:
        self._ensure_db()
        conn = self._db_conn
        if conn is None:
            return False
        try:
            row = conn.execute(
                "SELECT file_path FROM clips WHERE clip_id = ?", (clip_id,)
            ).fetchone()
            if row is None:
                return False
            file_path = row[0]
            if os.path.exists(file_path):
                os.remove(file_path)
            conn.execute("DELETE FROM clips WHERE clip_id = ?", (clip_id,))
            conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to delete clip %s: %s", clip_id, e)
            return False

    def shutdown(self) -> None:
        with self._rec_lock:
            if self._recording:
                self._finalize_clip()
        if self._write_executor is not None:
            self._write_executor.shutdown(wait=True)
            self._write_executor = None
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None

    # ── internal ─────────────────────────────────────────────────

    def _ensure_db(self) -> None:
        if self._db_conn is not None:
            return
        with self._rec_lock:
            if self._db_conn is not None:
                return
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._db_conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._db_conn.execute("PRAGMA journal_mode=WAL")
            self._db_conn.execute("PRAGMA synchronous=NORMAL")
            self._db_conn.execute(
                """CREATE TABLE IF NOT EXISTS clips (
                    clip_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    duration_s REAL NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size_bytes INTEGER DEFAULT 0,
                    event_count INTEGER DEFAULT 1,
                    tags TEXT DEFAULT '[]'
                )"""
            )
            self._db_conn.commit()

    def _finalize_clip(self) -> None:
        clip_id = getattr(self, '_pending_clip_id', None)
        if not clip_id:
            clip_id = (
                f"{self._current_event_type}_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
            )
        event_type = self._current_event_type
        tags = list(self._current_tags)

        with self._buffer_lock:
            pre_frames = list(self._buffer)
        post_frames = list(self._post_frames)
        all_frames = pre_frames + post_frames

        self._recording = False
        self._post_frames = []

        if not all_frames:
            return

        if self._write_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._write_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="clip-writer"
            )

        self._write_executor.submit(
            self._write_clip, clip_id, event_type, tags, all_frames,
        )

    def _write_clip(
        self,
        clip_id: str,
        event_type: str,
        tags: list[str],
        frames: list[tuple[bytes, float]],
    ) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        file_path = str(self._output_dir / f"{clip_id}.mp4")

        try:
            if not frames:
                return
            first = cv2.imdecode(
                np.frombuffer(frames[0][0], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if first is None:
                return
            h, w = first.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(file_path, fourcc, self._clip_fps, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(file_path, fourcc, self._clip_fps, (w, h))

            for jpeg_bytes, _ts in frames:
                img = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if img is not None:
                    writer.write(img)

            writer.release()
            file_size = os.path.getsize(file_path)
            duration_s = len(frames) / self._clip_fps

            self._ensure_db()
            conn = self._db_conn
            if conn is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO clips "
                    "(clip_id, event_type, timestamp, duration_s, file_path, "
                    "file_size_bytes, tags) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        clip_id, event_type, time.time(), duration_s, file_path,
                        file_size, str(tags),
                    ),
                )
                conn.commit()

            self._clip_count += 1
            self._total_bytes += file_size
            logger.info(
                "Clip saved: %s (%.1fs, %dKB)",
                clip_id, duration_s, file_size // 1024,
            )
            self._prune_old_clips()

        except Exception as e:
            logger.error("Failed to write clip %s: %s", clip_id, e)

    def _prune_old_clips(self) -> None:
        conn = self._db_conn
        if conn is None:
            return
        try:
            count = conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
            if count <= self._max_clips:
                return
            excess = count - self._max_clips
            rows = conn.execute(
                "SELECT clip_id, file_path FROM clips "
                "ORDER BY timestamp ASC LIMIT ?",
                (excess,),
            ).fetchall()
            for clip_id, file_path in rows:
                if os.path.exists(file_path):
                    os.remove(file_path)
            conn.execute(
                "DELETE FROM clips WHERE clip_id IN "
                "(SELECT clip_id FROM clips ORDER BY timestamp ASC LIMIT ?)",
                (excess,),
            )
            conn.commit()
            logger.info("Pruned %d old clips", excess)
        except Exception as e:
            logger.warning("Clip pruning failed: %s", e)


def _row_to_clip(row: tuple) -> ArchivedClip:
    import json

    (
        clip_id, event_type, timestamp, duration_s,
        file_path, file_size_bytes, event_count, tags_raw,
    ) = row
    try:
        tags = (
            json.loads(tags_raw) if isinstance(tags_raw, str) else (tags_raw or [])
        )
    except (json.JSONDecodeError, TypeError):
        tags = []
    return ArchivedClip(
        clip_id=clip_id,
        event_type=event_type,
        timestamp=timestamp,
        duration_s=duration_s,
        file_path=file_path,
        file_size_bytes=file_size_bytes,
        event_count=event_count,
        tags=tags,
    )
