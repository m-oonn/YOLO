"""Tests for the rules engine behavior detection logic."""

import os
import sys

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
    Zone,
)
from core.rules import Detection, RulesEngine

PERSON = 0


def _make_cfg(
    running=True, speed_px_s=50, min_duration_s=0.0,
    fall=True, upright_aspect_min=1.2, fallen_aspect_max=1.0, transition_window_s=1.0,
    crowd=True, min_people=3,
    intrusion=False,
    fight=True, distance_threshold=150, movement_threshold=30, fight_min_duration_s=0.0,
) -> AppConfig:
    return AppConfig(
        model_path="dummy.pt",
        rules=RulesConfig(
            running=RunningRule(enabled=running, speed_px_s=speed_px_s, min_duration_s=min_duration_s),
            fall=FallRule(enabled=fall, upright_aspect_min=upright_aspect_min, fallen_aspect_max=fallen_aspect_max, transition_window_s=transition_window_s),
            crowd=CrowdRule(enabled=crowd, min_people=min_people),
            intrusion=IntrusionRule(enabled=intrusion),
            fight=FightRule(enabled=fight, distance_threshold=distance_threshold, movement_threshold=movement_threshold, min_duration_s=fight_min_duration_s),
        ),
    )


def _person(track_id, x1, y1, x2, y2, conf=0.9):
    return Detection(track_id=track_id, class_id=PERSON, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2)


# ---- Running Detection ----
# State machine: frame1 (build history) -> frame2 (start timer, continue) -> frame3+ (emit)

def test_running_detection_triggers():
    cfg = _make_cfg(speed_px_s=50, min_duration_s=0.0)
    engine = RulesEngine(cfg, person_class_id=PERSON)
    t = 100.0

    # Frame 1: establish position
    dets1 = [_person(1, 0, 0, 10, 20)]
    events = engine.update(dets1, 1, t)
    assert len(events) == 0

    # Frame 2: start running timer
    dets2 = [_person(1, 100, 100, 110, 120)]
    events = engine.update(dets2, 2, t + 0.01)
    assert len(events) == 0  # Timer started, continues

    # Frame 3: emit event (timer elapsed + min_duration_s met)
    dets3 = [_person(1, 200, 200, 210, 220)]
    events = engine.update(dets3, 3, t + 0.02)
    assert len(events) == 1
    assert events[0].event_type == "running"
    assert events[0].track_id == 1


def test_running_detection_below_threshold():
    cfg = _make_cfg(speed_px_s=500, min_duration_s=0.0)
    engine = RulesEngine(cfg, person_class_id=PERSON)
    t = 100.0

    dets1 = [_person(1, 0, 0, 10, 20)]
    dets2 = [_person(1, 5, 5, 15, 25)]

    engine.update(dets1, 1, t)
    engine.update(dets2, 2, t + 0.1)
    events = engine.update(dets2, 3, t + 0.2)  # same position = no speed
    assert len(events) == 0


# ---- Fall Detection ----

def test_fall_detection_triggers():
    cfg = _make_cfg(upright_aspect_min=1.2, fallen_aspect_max=1.0)
    engine = RulesEngine(cfg, person_class_id=PERSON)
    t = 100.0

    # Upright person (tall box)
    upright = _person(1, 0, 0, 30, 100)
    events = engine.update([upright], 1, t)
    assert len(events) == 0

    # Fallen person (wide box)
    fallen = _person(1, 0, 0, 100, 30)
    events = engine.update([fallen], 2, t + 0.5)
    assert len(events) == 1
    assert events[0].event_type == "fall"


def test_fall_no_false_positive():
    cfg = _make_cfg(upright_aspect_min=1.2, fallen_aspect_max=1.0)
    engine = RulesEngine(cfg, person_class_id=PERSON)

    dets = [_person(1, 0, 0, 30, 100)]  # aspect = 100/30 ~ 3.33
    events = engine.update(dets, 1, 100.0)
    events = engine.update(dets, 2, 100.5)
    assert len(events) == 0


# ---- Crowd Detection ----

def test_crowd_detection_triggers():
    cfg = _make_cfg(min_people=3)
    engine = RulesEngine(cfg, person_class_id=PERSON)

    dets = [
        _person(1, 0, 0, 10, 20),
        _person(2, 50, 50, 60, 70),
        _person(3, 100, 100, 110, 120),
    ]
    events = engine.update(dets, 1, 100.0)
    assert len(events) == 1
    assert events[0].event_type == "crowd"


def test_crowd_below_threshold():
    cfg = _make_cfg(min_people=5)
    engine = RulesEngine(cfg, person_class_id=PERSON)

    dets = [
        _person(1, 0, 0, 10, 20),
        _person(2, 50, 50, 60, 70),
        _person(3, 100, 100, 110, 120),
    ]
    events = engine.update(dets, 1, 100.0)
    assert len(events) == 0


# ---- Intrusion Detection ----

def test_intrusion_detection_triggers():
    cfg = AppConfig(
        model_path="dummy.pt",
        rules=RulesConfig(
            running=RunningRule(enabled=False),
            fall=FallRule(enabled=False),
            crowd=CrowdRule(enabled=False),
            intrusion=IntrusionRule(
                enabled=True,
                zones=[Zone(name="restricted", polygon=[[0,0],[100,0],[100,100],[0,100]])],
            ),
            fight=FightRule(enabled=False),
        ),
    )
    engine = RulesEngine(cfg, person_class_id=PERSON)

    dets = [_person(1, 45, 45, 55, 55)]  # Center of zone
    events = engine.update(dets, 1, 100.0)
    assert len(events) == 1
    assert events[0].event_type == "intrusion"
    assert events[0].zone_name == "restricted"


def test_intrusion_outside_zone():
    cfg = AppConfig(
        model_path="dummy.pt",
        rules=RulesConfig(
            running=RunningRule(enabled=False),
            fall=FallRule(enabled=False),
            crowd=CrowdRule(enabled=False),
            intrusion=IntrusionRule(
                enabled=True,
                zones=[Zone(name="restricted", polygon=[[0,0],[100,0],[100,100],[0,100]])],
            ),
            fight=FightRule(enabled=False),
        ),
    )
    engine = RulesEngine(cfg, person_class_id=PERSON)

    dets = [_person(1, 200, 200, 210, 220)]  # Outside zone
    events = engine.update(dets, 1, 100.0)
    assert len(events) == 0


# ---- Fight Detection ----
# State machine: frame1 (build history) -> frame2 (start timer, continue) -> frame3+ (emit)

def test_fight_detection_triggers():
    cfg = _make_cfg(distance_threshold=150, movement_threshold=30, fight_min_duration_s=0.0)
    engine = RulesEngine(cfg, person_class_id=PERSON)
    t = 100.0

    p1 = _person(1, 0, 0, 10, 20)
    p2 = _person(2, 30, 0, 40, 20)

    # Frame 1: establish history
    engine.update([p1, p2], 1, t)

    # Frame 2: close together, one moves fast -> start fight timer
    p1_moved = _person(1, 50, 50, 60, 70)
    p2_still = _person(2, 30, 0, 40, 20)
    events = engine.update([p1_moved, p2_still], 2, t + 0.01)
    assert len(events) == 0  # Timer started

    # Frame 3: both still close, one still moving -> emit (may also trigger running)
    p1_moved_more = _person(1, 100, 100, 110, 120)
    events = engine.update([p1_moved_more, p2_still], 3, t + 0.02)
    fight_events = [e for e in events if e.event_type == "fight"]
    assert len(fight_events) == 1
    assert fight_events[0].event_type == "fight"


# ---- Debounce ----

def test_debounce_prevents_duplicate_events():
    cfg = _make_cfg(speed_px_s=50, min_duration_s=0.0)
    engine = RulesEngine(cfg, person_class_id=PERSON)
    t = 100.0

    dets1 = [_person(1, 0, 0, 10, 20)]
    dets2 = [_person(1, 100, 100, 110, 120)]

    # Build history + start timer
    engine.update(dets1, 1, t)
    engine.update(dets2, 2, t + 0.01)

    # Emit first event
    dets3 = [_person(1, 200, 200, 210, 220)]
    events = engine.update(dets3, 3, t + 0.02)
    assert len(events) == 1

    # Same condition immediately after should be debounced (5s cooldown)
    dets4 = [_person(1, 300, 300, 310, 320)]
    events = engine.update(dets4, 4, t + 0.03)
    assert len(events) == 0


# ---- Rules can be disabled ----

def test_disabled_rule_does_not_fire():
    cfg = _make_cfg(running=False, speed_px_s=50, min_duration_s=0.0)
    engine = RulesEngine(cfg, person_class_id=PERSON)

    dets1 = [_person(1, 0, 0, 10, 20)]
    dets2 = [_person(1, 200, 200, 210, 220)]

    engine.update(dets1, 1, 100.0)
    engine.update(dets2, 2, 100.01)
    events = engine.update(dets2, 3, 100.02)
    assert len(events) == 0


def test_all_rules_disabled():
    cfg = _make_cfg(running=False, fall=False, crowd=False, intrusion=False, fight=False)
    engine = RulesEngine(cfg, person_class_id=PERSON)

    dets = [_person(1, 0, 0, 10, 20), _person(2, 50, 50, 60, 70)]
    events = engine.update(dets, 1, 100.0)
    assert len(events) == 0


def test_empty_detections_list():
    """Edge case: empty detections should not cause errors."""
    cfg = _make_cfg()
    engine = RulesEngine(cfg, person_class_id=PERSON)
    events = engine.update([], 1, 100.0)
    assert len(events) == 0


def test_non_person_detections():
    """Edge case: non-person detections should not trigger behavior rules."""
    cfg = _make_cfg()
    engine = RulesEngine(cfg, person_class_id=PERSON)
    # class_id=2 is 'car', not a person
    dets = [Detection(track_id=1, class_id=2, conf=0.9, x1=0, y1=0, x2=10, y2=20)]
    events = engine.update(dets, 1, 100.0)
    assert len(events) == 0


def test_crowd_with_sparse_people():
    """Crowd detection should not trigger when people are far apart."""
    cfg = _make_cfg(crowd=True, min_people=2, intrusion=False, fight=False, running=False, fall=False)
    engine = RulesEngine(cfg, person_class_id=PERSON)
    # Two people far apart (distance >> proximity_px of 200)
    dets = [
        _person(1, 0, 0, 10, 20),
        _person(2, 1000, 1000, 1010, 1020),
    ]
    events = engine.update(dets, 1, 100.0)
    assert len(events) == 0


def test_multiple_events_same_frame():
    """Multiple events of different types can fire in the same frame."""
    cfg = _make_cfg(speed_px_s=50, min_duration_s=0.0)
    engine = RulesEngine(cfg, person_class_id=PERSON)
    t = 100.0

    # Two people, one running and one stationary - frame 1
    engine.update([_person(1, 0, 0, 10, 20), _person(2, 100, 100, 110, 120)], 1, t)
    # Frame 2: person 1 moves fast
    engine.update([_person(1, 200, 200, 210, 220), _person(2, 100, 100, 110, 120)], 2, t + 0.01)
    # Frame 3: person 1 keeps moving fast
    dets = [_person(1, 400, 400, 410, 420), _person(2, 100, 100, 110, 120)]
    events = engine.update(dets, 3, t + 0.02)
    running_events = [e for e in events if e.event_type == "running"]
    assert len(running_events) >= 1
