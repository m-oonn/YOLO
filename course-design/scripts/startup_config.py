# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Startup configuration for one-click launch system.

This module provides configurable startup options for the YOLO detection system,
supporting different deployment scenarios and platform-specific settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendConfig:
    """Backend service configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    model_path: str = "models/yolov8n.pt"  # Match configs/default.yaml
    device: str = "auto"  # auto, cpu, cuda, mps
    camera_index: int = 0


@dataclass
class FrontendConfig:
    """Frontend service configuration."""

    enabled: bool = True
    port: int = 8080
    host: str = "127.0.0.1"
    dev_mode: bool = True  # True = npm run dev, False = npm run preview
    open_browser: bool = True
    browser_delay: float = 2.0  # seconds to wait before opening browser


@dataclass
class DatabaseConfig:
    """Database configuration."""

    path: str = "outputs/events.db"
    auto_migrate: bool = True  # Auto-add missing columns


@dataclass
class StartupOptions:
    """Main startup configuration container."""

    backend: BackendConfig = field(default_factory=BackendConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Startup behavior
    skip_health_checks: bool = False
    startup_timeout: int = 60  # seconds
    retry_attempts: int = 3
    retry_delay: float = 2.0  # seconds between retries

    # Resource checks
    check_gpu: bool = True
    check_ports: bool = True
    check_dependencies: bool = True
    check_camera: bool = True  # Check camera availability (Windows)

    # Output
    verbose: bool = True
    color_output: bool = True
    log_file: str = "outputs/startup.log"


def get_default_options() -> StartupOptions:
    """Get default startup options."""
    return StartupOptions()


def load_from_env() -> StartupOptions:
    """Load configuration from environment variables."""
    options = StartupOptions()

    # Backend settings
    if os.getenv("BACKEND_HOST"):
        options.backend.host = os.getenv("BACKEND_HOST")
    if os.getenv("BACKEND_PORT"):
        options.backend.port = int(os.getenv("BACKEND_PORT"))
    if os.getenv("MODEL_PATH"):
        options.backend.model_path = os.getenv("MODEL_PATH")
    if os.getenv("DEVICE"):
        options.backend.device = os.getenv("DEVICE")
    if os.getenv("CAMERA_INDEX"):
        options.backend.camera_index = int(os.getenv("CAMERA_INDEX"))

    # Frontend settings
    if os.getenv("FRONTEND_ENABLED"):
        options.frontend.enabled = os.getenv("FRONTEND_ENABLED").lower() in (
            "true",
            "1",
            "yes",
        )
    if os.getenv("FRONTEND_PORT"):
        options.frontend.port = int(os.getenv("FRONTEND_PORT"))

    # Behavior settings
    if os.getenv("SKIP_HEALTH_CHECKS"):
        options.skip_health_checks = True
    if os.getenv("VERBOSE"):
        options.verbose = os.getenv("VERBOSE").lower() in ("true", "1", "yes")

    return options


def detect_platform() -> str:
    """Detect current platform."""
    import platform

    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    else:
        return "linux"


def get_platform_config() -> dict[str, Any]:
    """Get platform-specific defaults."""
    platform_name = detect_platform()

    config = {
        "shell_extension": ".sh" if platform_name != "windows" else ".bat",
        "python_cmd": "python" if platform_name == "windows" else "python3",
        "node_cmd": "npm",
        "clipboard_cmd": "clip"
        if platform_name == "windows"
        else "pbcopy"
        if platform_name == "macos"
        else "xclip",
        "browser_cmd": "start"
        if platform_name == "windows"
        else "open"
        if platform_name == "macos"
        else "xdg-open",
    }

    return config
