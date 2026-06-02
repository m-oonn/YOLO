# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Scene context buffer for MLLM sidecar — sliding window of frames + detections."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SceneFrame:
    timestamp: float
    frame_data: Any = None
    detections: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    person_count: int = 0
    active_rule_types: list[str] = field(default_factory=list)


@dataclass
class SceneContext:
    frames: list[SceneFrame] = field(default_factory=list)
    total_person_count: int = 0
    max_person_count: int = 0
    event_counts: dict[str, int] = field(default_factory=dict)
    active_rule_types: set[str] = field(default_factory=set)
    duration_s: float = 0.0

    def to_prompt_context(self) -> str:
        parts = []
        if self.duration_s > 0:
            parts.append(f"时间窗口: 最近{self.duration_s:.1f}秒")
        if self.max_person_count > 0:
            parts.append(f"人数: 最多{self.max_person_count}人, 当前{self.total_person_count}人")
        if self.event_counts:
            event_strs = [f"{k}: {v}次" for k, v in self.event_counts.items() if v > 0]
            if event_strs:
                parts.append(f"事件: {', '.join(event_strs)}")
        if self.active_rule_types:
            parts.append(f"活跃规则: {', '.join(sorted(self.active_rule_types))}")
        return "; ".join(parts) if parts else "无显著活动"


class SceneContextBuffer:
    def __init__(self, max_frames: int = 5):
        self._buffer: deque[SceneFrame] = deque(maxlen=max_frames)
        self._max_frames = max_frames

    def add_frame(
        self,
        detections: list[dict] | None = None,
        events: list[dict] | None = None,
        frame_data: Any = None,
    ) -> None:
        person_count = 0
        active_rules: list[str] = []
        if detections:
            person_count = sum(
                1 for d in detections if d.get("class_name") == "person"
            )
        if events:
            active_rules = list({e.get("type", "unknown") for e in events})

        sf = SceneFrame(
            timestamp=time.time(),
            frame_data=frame_data,
            detections=detections or [],
            events=events or [],
            person_count=person_count,
            active_rule_types=active_rules,
        )
        self._buffer.append(sf)

    def get_context(self) -> SceneContext:
        if not self._buffer:
            return SceneContext()

        frames = list(self._buffer)
        total_persons = frames[-1].person_count if frames else 0
        max_persons = max((f.person_count for f in frames), default=0)
        event_counts: dict[str, int] = {}
        active_rules: set[str] = set()
        for f in frames:
            for e in f.events:
                etype = e.get("type", "unknown")
                event_counts[etype] = event_counts.get(etype, 0) + 1
            active_rules.update(f.active_rule_types)

        duration = frames[-1].timestamp - frames[0].timestamp if len(frames) > 1 else 0.0

        return SceneContext(
            frames=frames,
            total_person_count=total_persons,
            max_person_count=max_persons,
            event_counts=event_counts,
            active_rule_types=active_rules,
            duration_s=duration,
        )

    def has_alarm_events(self) -> bool:
        for f in self._buffer:
            if f.events:
                return True
        return False

    def get_latest_frame(self) -> SceneFrame | None:
        return self._buffer[-1] if self._buffer else None

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def size(self) -> int:
        return len(self._buffer)
