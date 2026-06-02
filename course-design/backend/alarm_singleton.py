# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Singleton alarm engine instance shared across the application."""

from __future__ import annotations

import os
import threading

from core.alarm_engine import AlarmConfig, AlarmEngine, AlarmStore

_engine: AlarmEngine | None = None
_lock: threading.Lock = threading.Lock()


def get_alarm_engine() -> AlarmEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                db_path = os.path.join("outputs", "alarms.db")
                store = AlarmStore(db_path)
                config = _load_config_from_env()
                _engine = AlarmEngine(store, config)
                _engine.start()
    return _engine


def _load_config_from_env() -> AlarmConfig:
    config = AlarmConfig()
    config.enabled = os.environ.get("ALARM_ENABLED", "true").lower() == "true"
    if os.environ.get("ALARM_SUPPRESS_WINDOW"):
        config.suppress_window_s = float(os.environ["ALARM_SUPPRESS_WINDOW"])
    if os.environ.get("ALARM_AGGREGATE_WINDOW"):
        config.aggregate_window_s = float(os.environ["ALARM_AGGREGATE_WINDOW"])
    if os.environ.get("ALARM_ESCALATE_AFTER"):
        config.escalate_after_s = float(os.environ["ALARM_ESCALATE_AFTER"])
    if os.environ.get("ALARM_MAX_PER_MINUTE"):
        config.max_alarms_per_minute = int(os.environ["ALARM_MAX_PER_MINUTE"])

    webhook_url = os.environ.get("ALARM_WEBHOOK_URL")
    if webhook_url:
        config.notifiers["webhook"] = {"url": webhook_url}

    smtp_host = os.environ.get("ALARM_SMTP_HOST")
    if smtp_host:
        config.notifiers["email"] = {
            "smtp_host": smtp_host,
            "smtp_port": int(os.environ.get("ALARM_SMTP_PORT", "587")),
            "smtp_user": os.environ.get("ALARM_SMTP_USER", ""),
            "smtp_pass": os.environ.get("ALARM_SMTP_PASS", ""),
            "from_addr": os.environ.get("ALARM_FROM_ADDR", ""),
            "to_addrs": os.environ.get("ALARM_TO_ADDRS", "").split(",") if os.environ.get("ALARM_TO_ADDRS") else [],
            "use_tls": os.environ.get("ALARM_SMTP_TLS", "true").lower() == "true",
        }

    return config


def close_alarm_engine() -> None:
    global _engine
    with _lock:
        if _engine is not None:
            _engine.stop()
            _engine.store.close()
            _engine = None
