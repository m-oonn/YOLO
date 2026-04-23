# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: MIT

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

from core.config import (
    AppConfig,
    CrowdRule,
    FallRule,
    FightRule,
    IntrusionRule,
    RulesConfig,
    RunningRule,
)


@pytest.fixture
def sample_config():
    return AppConfig(
        model_path="dummy.pt",
        rules=RulesConfig(
            running=RunningRule(enabled=True, speed_px_s=50, min_duration_s=0.0),
            fall=FallRule(enabled=True, upright_aspect_min=1.2, fallen_aspect_max=1.0),
            crowd=CrowdRule(enabled=True, min_people=3),
            intrusion=IntrusionRule(enabled=False),
            fight=FightRule(
                enabled=True, distance_threshold=150, movement_threshold=30
            ),
        ),
    )


@pytest.fixture
def mock_yolo():
    with patch("core.pipeline.YOLO") as mock:
        mock_instance = MagicMock()
        mock_instance.track.return_value = []
        mock.return_value = mock_instance
        yield mock_instance
