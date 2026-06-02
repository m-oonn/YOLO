# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for multi-channel alarm notification dispatchers."""

from __future__ import annotations

from unittest.mock import patch

from core.alarm_engine import Alarm, AlarmLevel
from core.notifiers import (
    ConsoleNotifier,
    LogNotifier,
    NotificationDispatcher,
    NotificationResult,
    WebhookNotifier,
)


def _make_alarm(level=AlarmLevel.WARNING, event_type="fight", **kwargs):
    return Alarm(
        alarm_key="test:key",
        event_type=event_type,
        level=level,
        count=1,
        description="test alarm",
        **kwargs,
    )


class TestNotificationResult:
    def test_to_dict(self):
        r = NotificationResult("test", True, "ok")
        d = r.to_dict()
        assert d == {"channel": "test", "success": True, "message": "ok"}

    def test_to_dict_failure(self):
        r = NotificationResult("test", False, "error msg")
        d = r.to_dict()
        assert d == {"channel": "test", "success": False, "message": "error msg"}


class TestLogNotifier:
    def test_send_returns_success(self):
        notifier = LogNotifier()
        result = notifier.send(_make_alarm())
        assert result.success is True
        assert result.channel == "log"

    def test_send_includes_channel_name(self):
        notifier = LogNotifier()
        result = notifier.send(_make_alarm())
        assert result.channel == "log"

    def test_send_critical_level(self):
        notifier = LogNotifier()
        result = notifier.send(_make_alarm(level=AlarmLevel.CRITICAL))
        assert result.success is True


class TestWebhookNotifier:
    def test_send_no_url_returns_failure(self):
        notifier = WebhookNotifier()
        result = notifier.send(_make_alarm())
        assert result.success is False
        assert "No URL configured" in result.message

    def test_send_success(self):
        notifier = WebhookNotifier({"url": "http://example.com/hook"})
        with patch("urllib.request.urlopen") as mock:
            mock.return_value.__enter__.return_value.status = 200
            result = notifier.send(_make_alarm())
        assert result.success is True

    def test_send_http_error(self):
        notifier = WebhookNotifier({"url": "http://example.com/hook"})
        with patch("urllib.request.urlopen") as mock:
            mock.return_value.__enter__.return_value.status = 500
            result = notifier.send(_make_alarm())
        assert result.success is False

    def test_send_timeout_returns_failure(self):
        notifier = WebhookNotifier({"url": "http://example.com/hook", "timeout": 1})
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timeout")):
            result = notifier.send(_make_alarm())
        assert result.success is False


class TestConsoleNotifier:
    def test_send_prints_to_stdout(self, capsys):
        notifier = ConsoleNotifier()
        result = notifier.send(_make_alarm())
        captured = capsys.readouterr()
        assert "ALARM" in captured.out
        assert result.success is True

    def test_send_returns_success(self):
        notifier = ConsoleNotifier()
        result = notifier.send(_make_alarm())
        assert result.success is True
        assert result.channel == "console"


class TestNotificationDispatcher:
    def test_dispatch_log_notifier(self):
        dispatcher = NotificationDispatcher()
        results = dispatcher.dispatch(_make_alarm(), channels=["log"])
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].channel == "log"

    def test_dispatch_all_configured_channels(self):
        dispatcher = NotificationDispatcher()
        results = dispatcher.dispatch(_make_alarm())
        assert len(results) >= 1
        assert all(r.success is True for r in results)

    def test_dispatch_unknown_channel_returns_failure(self):
        dispatcher = NotificationDispatcher()
        results = dispatcher.dispatch(_make_alarm(), channels=["nonexistent"])
        assert len(results) == 1
        assert results[0].success is False
        assert "not configured" in results[0].message

    def test_configure_adds_notifier(self):
        dispatcher = NotificationDispatcher()
        assert "webhook" not in dispatcher.get_configured_channels()
        dispatcher.configure({"webhook": {"url": "http://example.com/hook"}})
        assert "webhook" in dispatcher.get_configured_channels()

    def test_configure_console_when_disabled(self):
        dispatcher = NotificationDispatcher()
        dispatcher.configure({"console": {"enabled": False}})
        assert "console" not in dispatcher.get_configured_channels()

    def test_configure_console_when_enabled(self):
        dispatcher = NotificationDispatcher()
        dispatcher.configure({"console": {"enabled": True}})
        assert "console" in dispatcher.get_configured_channels()

    def test_get_configured_channels(self):
        dispatcher = NotificationDispatcher()
        channels = dispatcher.get_configured_channels()
        assert "log" in channels
