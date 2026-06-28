# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Multi-channel alarm notification dispatchers.

Supported channels:
- Log: Write alarm to application log (always available)
- Webhook: HTTP POST to configured URL
- Email: SMTP email (requires configuration)
- Console: Print to stdout for development
"""

from __future__ import annotations

import json
import logging
import smtplib
from concurrent.futures import ThreadPoolExecutor
from email.mime.text import MIMEText
from typing import Any

from .alarm_engine import EVENT_LEVEL_LABELS, Alarm, AlarmLevel

logger = logging.getLogger(__name__)


class NotificationResult:
    def __init__(self, channel: str, success: bool, message: str = ""):
        self.channel = channel
        self.success = success
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "success": self.success,
            "message": self.message,
        }


class LogNotifier:
    """Always-available notifier that writes alarms to the application log."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def send(self, alarm: Alarm) -> NotificationResult:
        level_name = EVENT_LEVEL_LABELS.get(alarm.level, "未知")
        log_level = (
            logging.CRITICAL if alarm.level >= AlarmLevel.CRITICAL else logging.WARNING
        )
        logger.log(
            log_level,
            "[ALARM][%s] %s | key=%s count=%d desc=%s",
            level_name,
            alarm.event_type,
            alarm.alarm_key,
            alarm.count,
            alarm.description,
        )
        return NotificationResult("log", True, "Logged successfully")


class WebhookNotifier:
    """Send alarm notifications via HTTP POST webhook."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.url = self.config.get("url", "")
        self.headers = self.config.get("headers", {"Content-Type": "application/json"})
        self.timeout = self.config.get("timeout", 10)

    def send(self, alarm: Alarm) -> NotificationResult:
        if not self.url:
            return NotificationResult("webhook", False, "No URL configured")

        try:
            import urllib.request

            payload = json.dumps(
                {
                    "alarm_key": alarm.alarm_key,
                    "event_type": alarm.event_type,
                    "level": int(alarm.level),
                    "level_label": EVENT_LEVEL_LABELS.get(alarm.level, "未知"),
                    "status": alarm.status,
                    "count": alarm.count,
                    "description": alarm.description,
                    "first_event_time": alarm.first_event_time,
                    "last_event_time": alarm.last_event_time,
                    "extra": alarm.extra,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                self.url,
                data=payload,
                headers=self.headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return NotificationResult(
                    "webhook", resp.status < 400, f"HTTP {resp.status}"
                )
        except Exception as e:
            return NotificationResult("webhook", False, str(e))


class EmailNotifier:
    """Send alarm notifications via SMTP email."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.smtp_host = self.config.get("smtp_host", "")
        self.smtp_port = int(self.config.get("smtp_port", 587))
        self.smtp_user = self.config.get("smtp_user", "")
        self.smtp_pass = self.config.get("smtp_pass", "")
        self.from_addr = self.config.get("from_addr", "")
        self.to_addrs = self.config.get("to_addrs", [])
        self.use_tls = self.config.get("use_tls", True)

    def send(self, alarm: Alarm) -> NotificationResult:
        if not self.smtp_host or not self.to_addrs:
            return NotificationResult("email", False, "SMTP not configured")

        try:
            level_label = EVENT_LEVEL_LABELS.get(alarm.level, "未知")
            subject = (
                f"[YOLO报警][{level_label}] {alarm.event_type} - {alarm.alarm_key}"
            )
            body = (
                f"报警类型: {alarm.event_type}\n"
                f"报警级别: {level_label}\n"
                f"报警状态: {alarm.status}\n"
                f"发生次数: {alarm.count}\n"
                f"首次时间: {alarm.first_event_time}\n"
                f"最近时间: {alarm.last_event_time}\n"
                f"描述: {alarm.description}\n"
            )
            if alarm.extra:
                body += f"附加信息: {json.dumps(alarm.extra, ensure_ascii=False)}\n"

            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_user and self.smtp_pass:
                    server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            return NotificationResult("email", True, "Email sent")
        except Exception as e:
            return NotificationResult("email", False, str(e))


class ConsoleNotifier:
    """Print alarm to console for development/testing."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def send(self, alarm: Alarm) -> NotificationResult:
        level_label = EVENT_LEVEL_LABELS.get(alarm.level, "未知")
        print(
            f"\033[91m[ALARM][{level_label}]\033[0m {alarm.event_type} "
            f"| key={alarm.alarm_key} count={alarm.count} desc={alarm.description}"
        )
        return NotificationResult("console", True, "Printed to console")


class NotificationDispatcher:
    """Dispatches alarm notifications to multiple channels in parallel."""

    def __init__(self):
        self._notifiers: dict[str, Any] = {
            "log": LogNotifier(),
        }
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="notify")

    def configure(self, config: dict[str, dict[str, Any]]) -> None:
        if "webhook" in config:
            self._notifiers["webhook"] = WebhookNotifier(config["webhook"])
        if "email" in config:
            self._notifiers["email"] = EmailNotifier(config["email"])
        if "console" in config and config["console"].get("enabled", False):
            self._notifiers["console"] = ConsoleNotifier(config["console"])

    def dispatch(
        self, alarm: Alarm, channels: list[str] | None = None
    ) -> list[NotificationResult]:
        targets = channels or list(self._notifiers.keys())
        results = []
        for channel in targets:
            notifier = self._notifiers.get(channel)
            if notifier is None:
                results.append(
                    NotificationResult(channel, False, "Channel not configured")
                )
                continue
            try:
                result = notifier.send(alarm)
                results.append(result)
            except Exception as e:
                results.append(NotificationResult(channel, False, str(e)))
        return results

    def dispatch_async(self, alarm: Alarm, channels: list[str] | None = None) -> None:
        self._executor.submit(self.dispatch, alarm, channels)

    def get_configured_channels(self) -> list[str]:
        return list(self._notifiers.keys())
