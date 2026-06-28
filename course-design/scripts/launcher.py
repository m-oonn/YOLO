#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unified launcher for YOLO detection system with health checks and progress feedback.

This module provides a one-click startup experience with:
- Pre-flight system checks (Python version, dependencies, ports, GPU)
- Parallel service startup (backend + frontend)
- Health check polling with timeout
- Graceful shutdown on errors
- Cross-platform support (Windows, macOS, Linux)
"""

from __future__ import annotations

import argparse
import http.client
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent.parent
if str(COURSE_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_DIR))

import contextlib

from scripts.startup_config import (
    StartupOptions,
    detect_platform,
    get_default_options,
    get_platform_config,
    load_from_env,
)

logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def color_print(text: str, color: str = "", bold: bool = False, file=None):
    """Print colored text to terminal."""
    prefix = f"{Colors.BOLD}" if bold else ""
    suffix = Colors.ENDC
    end = "\n" if file is None else ""
    print(f"{prefix}{color}{text}{suffix}", end=end, flush=True, file=file)


def print_header(text: str):
    """Print section header."""
    print()
    color_print(f"{'=' * 60}", Colors.CYAN)
    color_print(f"  {text}", Colors.CYAN, bold=True)
    color_print(f"{'=' * 60}", Colors.CYAN)


def print_step(step: str, status: str = "..."):
    """Print a startup step."""
    print(
        f"  [{Colors.BLUE}STEP{Colors.ENDC}] {step:<40} {status}", end="\r", flush=True
    )


def print_status(text: str, success: bool = True):
    """Print completion status."""
    color = Colors.GREEN if success else Colors.RED
    symbol = "✓" if success else "✗"
    print(f"  [{color}{symbol}{Colors.ENDC}] {text}")


class SystemChecker:
    """Perform pre-flight system checks."""

    _torch_cache: bool | None = None  # class-level GPU availability cache

    def __init__(self, options: StartupOptions):
        self.options = options
        self.platform = detect_platform()
        self.config = get_platform_config()
        self.errors: list[str] = []

    def check_python_version(self) -> bool:
        """Verify Python version is 3.10 or higher."""
        print_step("Checking Python version")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            self.errors.append(
                f"Python 3.10+ required, found {version.major}.{version.minor}"
            )
            print_status(
                f"Python {version.major}.{version.minor}.{version.micro}", False
            )
            return False
        print_status(f"Python {version.major}.{version.minor}.{version.micro}")
        return True

    def check_dependencies_fast(self) -> bool:
        """Verify critical packages are importable (blocking, fast path)."""
        print_step("Checking critical dependencies")
        # Only check packages needed for API startup — fast builtins/lightweight libs
        lightweight = {"fastapi", "uvicorn", "yaml", "sqlite3", "requests", "filetype"}
        missing = []
        for pkg in lightweight:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            self.errors.append(f"Missing critical packages: {', '.join(missing)}")
            print_status(f"Missing: {', '.join(missing)}", False)
            return False
        print_status("Critical packages OK")
        return True

    def check_dependencies_slow(self) -> bool:
        """Verify heavy ML packages (torch, cv2, ultralytics, numpy). Non-blocking phase."""
        print_step("Checking ML dependencies")
        heavy = {
            "opencv-python": "cv2",
            "numpy": "numpy",
            "torch": "torch",
            "ultralytics": "ultralytics",
            "pillow": "PIL",
        }
        missing = []
        for name, import_name in heavy.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(name)

        if missing:
            print_status(f"Missing ML packages: {', '.join(missing)}", False)
            color_print(
                "  Detection and GPU features may not work. Install with: pip install -r requirements.txt",
                Colors.YELLOW,
            )
            return False
        print_status("ML packages OK")
        return True

    def check_model_file(self, model_path: str) -> bool:
        """Verify YOLO model file exists."""
        print_step("Checking model file")
        if not model_path:
            self.errors.append("Model path not configured")
            print_status("Model path not set", False)
            return False

        # Support both absolute and relative paths
        if not os.path.isabs(model_path):
            model_path = COURSE_DIR / model_path

        if not os.path.exists(model_path):
            self.errors.append(f"Model file not found: {model_path}")
            print_status(f"Model not found: {model_path}", False)
            return False

        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print_status(f"Model found ({size_mb:.1f} MB)")
        return True

    def check_port_available(self, port: int, service: str) -> bool:
        """Check if a port is available for binding."""
        print_step(f"Checking port {port} availability")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("", port))
            print_status(f"Port {port} available for {service}")
            return True
        except OSError as e:
            self.errors.append(f"Port {port} in use by {service}: {e}")
            print_status(f"Port {port} in use", False)
            return False
        finally:
            sock.close()

    def check_gpu(self) -> bool:
        """Check GPU availability using cached torch import."""
        if SystemChecker._torch_cache is not None:
            return SystemChecker._torch_cache

        print_step("Checking GPU availability")
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print_status(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print_status("Using Apple MPS GPU")
            else:
                print_status("No GPU found, using CPU")
            SystemChecker._torch_cache = True
            return True
        except ImportError:
            print_status("PyTorch not installed, assuming CPU")
            SystemChecker._torch_cache = True
            return True

    def check_camera(self) -> bool:
        """Check if default camera (index 0) is accessible.

        Uses a lightweight timeout-based check that won't hang.
        Only active on Windows; skipped on other platforms.
        """
        print_step("Checking camera availability")
        if self.platform != "windows":
            print_status("Skipped (non-Windows)")
            return True

        try:
            from backend.camera_utils import quick_camera_check

            available, message = quick_camera_check(0, timeout_sec=5.0)
            if available:
                print_status("Camera accessible")
                return True
            else:
                print_status("Camera not available", False)
                color_print("  Camera check results:", Colors.YELLOW)
                color_print(f"    {message}", Colors.YELLOW)
                color_print("  Windows camera troubleshooting:", Colors.YELLOW)
                color_print("    1. 检查摄像头是否已连接并安装驱动", Colors.YELLOW)
                color_print(
                    "    2. Windows 设置 > 隐私和安全性 > 摄像头 -- 确保已开启",
                    Colors.YELLOW,
                )
                color_print(
                    "    3. 关闭其他占用摄像头的程序 (Teams, Zoom, 浏览器等)",
                    Colors.YELLOW,
                )
                color_print(
                    "    4. 设备管理器 (devmgmt.msc) 检查摄像头设备状态", Colors.YELLOW
                )
                return True  # Don't block startup, just warn
        except ImportError:
            print_status("Skipped (camera_utils not yet loaded)", False)
            return True
        except Exception as e:
            logger.warning("Camera check failed: %s", e)
            print_status(f"Camera check error: {e}", False)
            return True  # Don't block startup

    def print_gpu_info(self) -> None:
        """Print GPU info using cached check if available."""
        if SystemChecker._torch_cache is not None:
            return  # Already printed during check_gpu
        if self.options.check_gpu:
            return  # Already checked
        # Lightweight informational only
        print_status("GPU: Skipped (use --check-gpu to verify)")

    def run_all_checks(self) -> bool:
        """Run all system checks (fast phase only — ML deps deferred)."""
        print_header("System Pre-Flight Checks")

        checks = [
            self.check_python_version,
            self.check_dependencies_fast,
        ]

        if self.options.check_gpu:
            checks.append(self.check_gpu)

        if self.options.check_camera:
            checks.append(self.check_camera)

        if self.options.check_ports:
            checks.append(
                lambda: self.check_port_available(self.options.backend.port, "Backend")
            )
            if self.options.frontend.enabled:
                checks.append(
                    lambda: self.check_port_available(
                        self.options.frontend.port, "Frontend"
                    )
                )

        checks.append(lambda: self.check_model_file(self.options.backend.model_path))

        all_passed = all(check() for check in checks)

        if not all_passed:
            print_header("Pre-Flight Checks Failed")
            for error in self.errors:
                print_status(error, False)
            print()
            color_print("Please fix the issues above before starting.", Colors.YELLOW)
            return False

        print_status("All pre-flight checks passed", True)
        return True

    def run_slow_checks(self) -> None:
        """Run heavy dependency checks in background (does not block startup)."""
        self.check_dependencies_slow()


class ServiceManager:
    """Manages service processes."""

    def __init__(self, options: StartupOptions):
        self.options = options
        self.platform = detect_platform()
        self.config = get_platform_config()
        self.processes: dict[str, subprocess.Popen] = {}
        self.startup_time: dict[str, float] = {}

    def _is_port_free(self, port: int) -> bool:
        """Check if a TCP port is available for binding."""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("", port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _free_port(self, port: int) -> None:
        """Kill any process listening on the given port to free it."""
        if self._is_port_free(port):
            return
        print_step(f"Port {port} is occupied, attempting to free it")
        try:
            if self.platform == "windows":
                killed = False
                # Try PowerShell Get-NetTCPConnection first (modern Windows)
                try:
                    ps_cmd = [
                        "powershell.exe",
                        "-Command",
                        f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | "
                        f"Select-Object -ExpandProperty OwningProcess",
                    ]
                    result = subprocess.run(
                        ps_cmd,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.stdout.strip():
                        for pid in result.stdout.strip().splitlines():
                            pid = pid.strip()
                            if pid:
                                logger.info(
                                    "Killing process PID %s holding port %s", pid, port
                                )
                                subprocess.run(
                                    ["taskkill", "/F", "/PID", pid],
                                    capture_output=True,
                                    timeout=5,
                                )
                                killed = True
                except Exception:
                    pass
                # Fallback to netstat if PowerShell method didn't work
                if not killed:
                    try:
                        result = subprocess.run(
                            ["netstat", "-ano"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                            encoding="gbk",
                            errors="ignore",
                        )
                        for line in result.stdout.splitlines():
                            parts = line.strip().split()
                            if (
                                len(parts) >= 5
                                and f":{port}" in parts[1]
                                and "LISTENING" in parts
                            ):
                                pid = parts[4]
                                logger.info(
                                    "Killing process PID %s holding port %s", pid, port
                                )
                                subprocess.run(
                                    ["taskkill", "/F", "/PID", pid],
                                    capture_output=True,
                                    timeout=5,
                                )
                                killed = True
                    except FileNotFoundError:
                        pass
                time.sleep(1)
                if self._is_port_free(port):
                    print_status(f"Freed port {port}")
                    return
                print_status(
                    f"Could not free port {port} (no process identified)", False
                )
            else:
                # Unix-like: try lsof, then fuser
                for cmd_template in [["lsof", "-ti", ":{port}"]]:
                    try:
                        result = subprocess.run(
                            cmd_template, capture_output=True, text=True, timeout=5
                        )
                        if result.stdout.strip():
                            for pid in result.stdout.strip().splitlines():
                                subprocess.run(
                                    ["kill", "-9", pid], capture_output=True, timeout=5
                                )
                            time.sleep(1)
                            if self._is_port_free(port):
                                print_status(f"Freed port {port}")
                                return
                    except FileNotFoundError:
                        continue
                print_status(f"Could not free port {port}", False)
        except Exception as e:
            logger.warning("Failed to free port %d: %s", port, e)

    def start_backend(self) -> bool:
        """Start the FastAPI backend service."""
        self._free_port(self.options.backend.port)
        print_step("Starting backend service")

        backend_dir = COURSE_DIR / "backend"
        log_file = COURSE_DIR / "outputs" / "backend.log"
        os.makedirs(backend_dir.parent / "outputs", exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            self.options.backend.host,
            "--port",
            str(self.options.backend.port),
            "--workers",
            str(self.options.backend.workers),
            "--log-level",
            self.options.backend.log_level,
        ]

        try:
            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(COURSE_DIR),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(COURSE_DIR),
                        "YOLO_DEVICE": self.options.backend.device,
                    },
                )

            self.processes["backend"] = process
            self.startup_time["backend"] = time.time()
            print_status(f"Backend starting (PID: {process.pid})")
            return True

        except Exception as e:
            print_status(f"Failed to start backend: {e}", False)
            return False

    def wait_for_backend_healthy(self, timeout: int = 30) -> bool:
        """Wait for backend to become healthy."""
        print_step("Waiting for backend to be ready")

        start = time.time()
        url = f"http://127.0.0.1:{self.options.backend.port}/health"
        proc = self.processes.get("backend")
        spinner = ["|", "/", "-", "\\"]
        idx = 0
        interval = 0.1

        while time.time() - start < timeout:
            # Check our process is still alive first
            if proc is not None and proc.poll() is not None:
                print_status("Backend process died", False)
                return False

            try:
                import urllib.request

                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        # Verify OUR process is healthy, not a stale one
                        if proc is not None and proc.poll() is not None:
                            print_status("Backend process died", False)
                            return False
                        elapsed = time.time() - self.startup_time["backend"]
                        print_status(f"Backend ready ({elapsed:.1f}s)")
                        return True
            except (
                TimeoutError,
                urllib.error.URLError,
                ConnectionRefusedError,
                ConnectionResetError,
                http.client.RemoteDisconnected,
            ):
                pass

            print(
                f"\r  [WAIT] Backend starting {spinner[idx]} ({time.time() - start:.0f}s)",
                end="",
                flush=True,
            )
            idx = (idx + 1) % 4
            time.sleep(interval)
            interval = min(interval * 1.5, 1.0)

        print()
        print_status("Backend health check timeout", False)
        return False

    def start_frontend(self) -> bool:
        """Start the Vue.js frontend development server."""
        if not self.options.frontend.enabled:
            return True

        print_step("Starting frontend service")

        frontend_dir = COURSE_DIR / "frontend"

        # Find npm executable
        import shutil

        npm_cmd = shutil.which("npm")
        # Fallback: search in common node installation paths and frontend node_modules
        if not npm_cmd:
            search_paths = [
                Path("C:/Program Files/nodejs/npm.cmd"),
                Path("C:/Program Files (x86)/nodejs/npm.cmd"),
                Path(os.environ.get("LOCALAPPDATA", "")) / "nodejs/npm.cmd",
                COURSE_DIR / "frontend/node_modules/.bin/npm.cmd",
            ]
            for p in search_paths:
                if p.exists():
                    npm_cmd = str(p)
                    break
        # Final fallback: use npx directly from node_modules if npm is missing but npx exists
        if not npm_cmd:
            npx_cmd = shutil.which("npx")
            if not npx_cmd:
                npx_paths = [
                    Path("C:/Program Files/nodejs/npx.cmd"),
                    Path("C:/Program Files (x86)/nodejs/npx.cmd"),
                    Path(os.environ.get("LOCALAPPDATA", "")) / "nodejs/npx.cmd",
                    COURSE_DIR / "frontend/node_modules/.bin/npx.cmd",
                ]
                for p in npx_paths:
                    if p.exists():
                        npx_cmd = str(p)
                        break
            if npx_cmd:
                npm_cmd = npx_cmd  # use npx as npm substitute for running scripts
        # Ultra fallback: directly use vite from node_modules/.bin if available
        if not npm_cmd:
            vite_cmd = shutil.which("vite")
            if not vite_cmd:
                vite_paths = [
                    COURSE_DIR / "frontend/node_modules/.bin/vite.cmd",
                    COURSE_DIR / "frontend/node_modules/.bin/vite",
                ]
                for p in vite_paths:
                    if p.exists():
                        vite_cmd = str(p)
                        break
            if vite_cmd:
                # Use vite directly, bypassing npm entirely
                print_status(f"Using direct vite: {vite_cmd}")
                try:
                    log_file = COURSE_DIR / "outputs" / "frontend.log"
                    with open(log_file, "w") as log:
                        process = subprocess.Popen(
                            [
                                vite_cmd,
                                "--host",
                                self.options.frontend.host,
                                "--port",
                                str(self.options.frontend.port),
                            ],
                            cwd=str(frontend_dir),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                    self.processes["frontend"] = process
                    self.startup_time["frontend"] = time.time()
                    print_status(f"Frontend starting (PID: {process.pid})")
                    return True
                except Exception as e:
                    print_status(f"Failed to start frontend: {e}", False)
                    return False
        if not npm_cmd:
            print_status("npm not found in PATH", False)
            return False

        # Check if node_modules exists
        if not os.path.exists(frontend_dir / "node_modules"):
            print_status("Installing frontend dependencies...", False)
            install_process = subprocess.run(
                [npm_cmd, "install"],
                cwd=str(frontend_dir),
                capture_output=True,
            )
            if install_process.returncode != 0:
                print_status("Failed to install npm packages", False)
                return False
            print_status("Dependencies installed")

        cmd = [
            npm_cmd,
            "run",
            "dev" if self.options.frontend.dev_mode else "preview",
            "--",
            "--host",
            self.options.frontend.host,
            "--port",
            str(self.options.frontend.port),
        ]

        try:
            log_file = COURSE_DIR / "outputs" / "frontend.log"
            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(frontend_dir),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

            self.processes["frontend"] = process
            self.startup_time["frontend"] = time.time()
            print_status(f"Frontend starting (PID: {process.pid})")
            return True

        except Exception as e:
            print_status(f"Failed to start frontend: {e}", False)
            return False

    def wait_for_frontend_healthy(self, timeout: int = 60) -> bool:
        """Wait for frontend dev server to be ready."""
        if not self.options.frontend.enabled:
            return True

        print_step("Waiting for frontend to be ready")

        start = time.time()
        spinner = ["|", "/", "-", "\\"]
        idx = 0
        interval = 0.5

        while time.time() - start < timeout:
            # Check if process is still running
            if self.processes["frontend"].poll() is not None:
                print_status("Frontend process died", False)
                return False

            # Primary check: a raw TCP connect proves the dev server is
            # listening. This is more reliable than an HTTP request because
            # Vite may return non-200 or stall during initial warmup compile.
            try:
                with socket.create_connection(
                    ("127.0.0.1", self.options.frontend.port), timeout=2
                ):
                    elapsed = time.time() - self.startup_time["frontend"]
                    print_status(f"Frontend ready ({elapsed:.1f}s)")
                    return True
            except (TimeoutError, ConnectionRefusedError, OSError):
                pass

            print(
                f"\r  [WAIT] Frontend starting {spinner[idx]} ({time.time() - start:.0f}s)",
                end="",
                flush=True,
            )
            idx = (idx + 1) % 4
            time.sleep(interval)
            interval = min(interval * 1.5, 2.0)

        print()
        print_status("Frontend health check timeout", False)
        return False

    def open_browser(self):
        """Open default browser to frontend URL."""
        if not self.options.frontend.enabled or not self.options.frontend.open_browser:
            return

        time.sleep(self.options.frontend.browser_delay)

        url = f"http://127.0.0.1:{self.options.frontend.port}"
        try:
            if self.platform == "windows":
                os.system(f"start {url}")
            elif self.platform == "macos":
                subprocess.run(["open", url])
            else:
                subprocess.run(["xdg-open", url])
        except Exception:
            pass  # Silently fail if browser can't be opened

    def shutdown(self):
        """Gracefully shutdown all services."""
        print_header("Shutting Down Services")

        for name, process in self.processes.items():
            if process.poll() is None:  # Still running
                print_step(f"Stopping {name}")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    print_status(f"{name} stopped")
                except subprocess.TimeoutExpired:
                    process.kill()
                    print_status(f"{name} killed")
                except Exception as e:
                    print_status(f"Error stopping {name}: {e}", False)


def run_startup(options: StartupOptions) -> int:
    """Main startup routine with parallel service launch and timing."""
    overall_start = time.time()
    phases: dict[str, float] = {}

    print_header("YOLO Detection System Launcher")
    color_print("Starting one-click launch...", Colors.GREEN, bold=True)
    print(f"  Backend:  http://{options.backend.host}:{options.backend.port}")
    if options.frontend.enabled:
        print(f"  Frontend: http://{options.frontend.host}:{options.frontend.port}")
    print(f"  Platform: {detect_platform()}")

    # Run pre-flight checks (fast phase only)
    checker = SystemChecker(options)
    if options.check_dependencies and not checker.run_all_checks():
        return 1

    phases["preflight"] = time.time() - overall_start

    # Always show GPU info even when checks are skipped
    checker.print_gpu_info()

    # Start services in parallel
    print_header("Starting Services")
    manager = ServiceManager(options)

    # Shared results for parallel launch
    backend_ok = False
    frontend_ok = not options.frontend.enabled  # True if not starting frontend

    def _start_backend():
        nonlocal backend_ok
        if manager.start_backend():
            backend_ok = manager.wait_for_backend_healthy(
                timeout=options.startup_timeout
            )
        if not backend_ok:
            color_print("\nBackend failed to start. Check logs:", Colors.RED)
            print(f"  {COURSE_DIR / 'outputs' / 'backend.log'}")

    def _start_frontend():
        nonlocal frontend_ok
        if options.frontend.enabled:
            if manager.start_frontend():
                frontend_ok = manager.wait_for_frontend_healthy(
                    timeout=options.startup_timeout * 2
                )
            if not frontend_ok:
                color_print("\nFrontend failed to start. Check logs:", Colors.RED)
                print(f"  {COURSE_DIR / 'outputs' / 'frontend.log'}")

    # Run slow dependency checks in background thread (non-blocking)
    def _slow_checks():
        with contextlib.suppress(Exception):
            checker.run_slow_checks()

    slow_check_thread = threading.Thread(target=_slow_checks, daemon=True)
    slow_check_thread.start()

    # Launch backend and frontend in parallel
    backend_thread = threading.Thread(target=_start_backend, daemon=True)
    frontend_thread = threading.Thread(target=_start_frontend, daemon=True)
    backend_thread.start()
    frontend_thread.start()

    backend_thread.join(timeout=options.startup_timeout + 5)
    frontend_thread.join(timeout=options.startup_timeout * 2 + 5)

    phases["services"] = time.time() - overall_start - phases["preflight"]

    if not backend_ok or not frontend_ok:
        manager.shutdown()
        return 1

    # Open browser
    manager.open_browser()

    # Success!
    total_elapsed = time.time() - overall_start
    phases["total"] = total_elapsed

    print_header("System Ready!")
    print()
    color_print("YOLO Detection System is now running!", Colors.GREEN, bold=True)
    print()
    print(f"  Backend API:  http://localhost:{options.backend.port}")
    if options.frontend.enabled:
        print(f"  Frontend UI:  http://localhost:{options.frontend.port}")
    print(f"  API Docs:     http://localhost:{options.backend.port}/docs")
    print()
    print(
        f"  Startup phases: preflight={phases['preflight']:.1f}s, services={phases['services']:.1f}s, total={phases['total']:.1f}s"
    )
    print()
    color_print("Press Ctrl+C to stop the system.", Colors.YELLOW)
    print()

    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
            # Check if any process died
            for name, process in manager.processes.items():
                if process.poll() is not None:
                    color_print(f"\n{name} process has stopped!", Colors.RED)
                    manager.shutdown()
                    return 1
    except KeyboardInterrupt:
        manager.shutdown()
        print("\nShutdown complete.")
        return 0

    return 0


def main():
    """Entry point for launcher."""
    parser = argparse.ArgumentParser(
        description="YOLO Detection System - One-Click Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Start with default settings
  %(prog)s --backend-only     # Start only backend (no frontend)
  %(prog)s --port 9000        # Use custom backend port
  %(prog)s --no-browser      # Don't open browser automatically
  %(prog)s --skip-checks     # Skip pre-flight system checks
        """,
    )

    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Start only backend service (no frontend)",
    )
    parser.add_argument("--port", type=int, help="Backend port (default: 8000)")
    parser.add_argument(
        "--frontend-port", type=int, help="Frontend port (default: 8080)"
    )
    parser.add_argument(
        "--model", type=str, help="Model path (default: models/yolov11n.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for inference (default: auto)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )
    parser.add_argument(
        "--skip-checks", action="store_true", help="Skip pre-flight system checks"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Build options
    options = get_default_options()
    options = load_from_env()  # Override with environment variables

    # Apply command-line overrides
    if args.backend_only:
        options.frontend.enabled = False
    if args.port:
        options.backend.port = args.port
    if args.frontend_port:
        options.frontend.port = args.frontend_port
    if args.model:
        options.backend.model_path = args.model
    if args.device:
        options.backend.device = args.device
    if args.no_browser:
        options.frontend.open_browser = False
    if args.skip_checks:
        options.check_dependencies = False
        options.check_gpu = False
        options.check_ports = False
    if args.verbose:
        options.verbose = True

    # Run startup
    exit_code = run_startup(options)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
