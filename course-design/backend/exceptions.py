# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Custom exception classes for structured error handling."""

from typing import Any


class YOLOException(Exception):
    """Base exception for all YOLO application errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or type(self).__name__
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class PipelineNotRunningError(YOLOException):
    """Raised when a pipeline operation requires a running pipeline but none is active."""

    def __init__(
        self,
        message: str = "No active detection pipeline",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, code="PIPELINE_NOT_RUNNING", details=details)


class PipelineAlreadyRunningError(YOLOException):
    """Raised when attempting to start a pipeline that is already running."""

    def __init__(
        self,
        message: str = "Detection pipeline is already running",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, code="PIPELINE_ALREADY_RUNNING", details=details)


class ModelLoadError(YOLOException):
    """Raised when a model fails to load."""

    def __init__(self, model_path: str, cause: str | None = None):
        details = {"model_path": model_path}
        if cause:
            details["cause"] = cause
        super().__init__(
            f"Failed to load model: {model_path}",
            code="MODEL_LOAD_ERROR",
            details=details,
        )


class ModelNotFoundError(YOLOException):
    """Raised when a model file does not exist."""

    def __init__(self, model_path: str):
        super().__init__(
            f"Model file not found: {model_path}",
            code="MODEL_NOT_FOUND",
            details={"model_path": model_path},
        )


class InvalidSourceError(YOLOException):
    """Raised when a video source (camera/file/URL) is invalid or inaccessible."""

    def __init__(self, source: str, reason: str | None = None):
        details = {"source": source}
        if reason:
            details["reason"] = reason
        super().__init__(
            f"Invalid source: {source}",
            code="INVALID_SOURCE",
            details=details,
        )


class ConfigError(YOLOException):
    """Raised when configuration is invalid or cannot be loaded."""

    def __init__(self, message: str, config_path: str | None = None):
        details = {}
        if config_path:
            details["config_path"] = config_path
        super().__init__(message, code="CONFIG_ERROR", details=details)


class StorageError(YOLOException):
    """Raised when event storage operations fail."""

    def __init__(self, message: str, operation: str | None = None):
        details = {}
        if operation:
            details["operation"] = operation
        super().__init__(message, code="STORAGE_ERROR", details=details)


class AlarmEngineError(YOLOException):
    """Raised when alarm engine operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="ALARM_ENGINE_ERROR", details=details)


class MLLMError(YOLOException):
    """Raised when MLLM sidecar operations fail."""

    def __init__(self, message: str, model_type: str | None = None):
        details = {}
        if model_type:
            details["model_type"] = model_type
        super().__init__(message, code="MLLM_ERROR", details=details)
