# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Structured JSON logging and request ID context management."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON with consistent field ordering."""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self._include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
        ts_ms = int(record.msec * 1000)
        record_dict: dict[str, Any] = {
            "timestamp": f"{ts}.{ts_ms:03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        request_id = _request_id_ctx.get()
        if request_id:
            record_dict["request_id"] = request_id
        if record.exc_info:
            record_dict["exception"] = self.formatException(record.exc_info)
        if self._include_extra:
            extra = {
                k: v
                for k, v in record.__dict__.items()
                if k not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__
                and not k.startswith("_")
            }
            if extra:
                record_dict["extra"] = extra
        return json.dumps(record_dict, ensure_ascii=False, separators=(",", ":"))


class PlainFormatter(logging.Formatter):
    """Human-readable formatter with request_id when available."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        request_id = _request_id_ctx.get()
        if request_id:
            record.msg = f"[req={request_id}] {record.msg}"
        return super().format(record)


def setup_logging(
    level: int = logging.INFO,
    log_file: str = "outputs/app.log",
    json_format: bool = False,
) -> None:
    """Configure root logger with structured JSON or plain text formatting.

    Parameters
    ----------
    level : int
        Logging level (e.g. logging.INFO).
    log_file : str
        Path to the log file.
    json_format : bool
        If True, use JSONFormatter for structured logs; otherwise PlainFormatter.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)

    formatter = JSONFormatter() if json_format else PlainFormatter()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)


def get_request_id() -> str | None:
    """Return the current request ID from context."""
    return _request_id_ctx.get()


def generate_request_id() -> str:
    """Generate a new request ID and set it in context."""
    rid = uuid.uuid4().hex[:16]
    _request_id_ctx.set(rid)
    return rid


def set_request_id(rid: str) -> None:
    """Set a known request ID in context."""
    _request_id_ctx.set(rid)


def clear_request_id() -> None:
    """Clear the request ID from context."""
    _request_id_ctx.set(None)
