# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Alarm enhancer — VLM-based alarm verification with cooldown."""

from __future__ import annotations

import logging
import time
from typing import Any

from core.mllm.mllm_config import MLLMConfig
from core.mllm.scene_context_buffer import SceneContext
from core.mllm.scene_describer import (
    build_alarm_prompt,
    normalize_alarm_result,
    parse_json_response,
)

logger = logging.getLogger(__name__)


class AlarmEnhancer:
    def __init__(self, config: MLLMConfig):
        self._config = config
        self._last_enhance_time: dict[str, float] = {}
        self._stats = {
            "total_verifications": 0,
            "validated": 0,
            "dismissed": 0,
            "escalated": 0,
            "errors": 0,
        }

    def should_enhance(self, alarm_type: str) -> bool:
        if not self._config.alarm_enhance_enabled:
            return False
        last_time = self._last_enhance_time.get(alarm_type, 0.0)
        if time.time() - last_time < self._config.enhancement_cooldown_s:
            return False
        return True

    def enhance_alarm(
        self,
        alarm_type: str,
        alarm_details: str,
        context: SceneContext,
        vlm_generate_fn,
    ) -> dict[str, Any] | None:
        if not self.should_enhance(alarm_type):
            return None

        self._last_enhance_time[alarm_type] = time.time()
        self._stats["total_verifications"] += 1

        try:
            prompt = build_alarm_prompt(alarm_type, alarm_details, context)
            raw_text = vlm_generate_fn(prompt)
            if not raw_text:
                self._stats["errors"] += 1
                return None

            parsed = parse_json_response(raw_text)
            if not parsed:
                self._stats["errors"] += 1
                return None

            result = normalize_alarm_result(parsed)
            verdict = result.get("verdict", "validate")
            if verdict == "dismiss":
                self._stats["dismissed"] += 1
            elif verdict == "escalate":
                self._stats["escalated"] += 1
            else:
                self._stats["validated"] += 1

            result["shadow_mode"] = self._config.shadow_mode
            result["alarm_type"] = alarm_type
            return result

        except Exception as e:
            logger.error("Alarm enhancement failed: %s", e)
            self._stats["errors"] += 1
            return None

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def reset_cooldown(self, alarm_type: str | None = None) -> None:
        if alarm_type:
            self._last_enhance_time.pop(alarm_type, None)
        else:
            self._last_enhance_time.clear()
