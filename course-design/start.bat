@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   YOLO Detection System - Windows Launcher
echo ============================================================
echo.

REM --- Locate Python interpreter (prefer the CUDA-enabled yolovll conda env) ---
set "PYTHON_EXE=E:\Miniconda3\envs\yolovll\python.exe"
if not exist "%PYTHON_EXE%" (
    REM Fall back to whatever python is on PATH
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found. Please install Python 3.10 or higher.
        echo         Download from: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

for /f "tokens=2" %%v in ('"%PYTHON_EXE%" --version 2^>^&1') do set PY_VER=%%v
echo [INFO] Using Python %PY_VER% (%PYTHON_EXE%)

REM --- Quick camera diagnostic (non-blocking) ---
echo [INFO] Checking camera availability...
powershell -NoProfile -Command "try { $cam = Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' -or $_.Name -match 'camera|webcam' }; if ($cam) { Write-Host '  Camera detected: ' -NoNewline; Write-Host $cam.Name } else { Write-Host '  [WARN] No camera device found in Device Manager' } } catch { Write-Host '  [WARN] Could not query camera devices' }" 2>nul
echo.

REM --- Run the Python launcher with safe defaults ---
"%PYTHON_EXE%" "%~dp0scripts\launcher.py" --skip-checks %*

if errorlevel 1 (
    echo.
    echo [ERROR] Startup failed. Check:
    echo         1. Log files: outputs\backend.log
    echo         2. Camera: Windows Settings ^> Privacy ^> Camera must be ON
    echo         3. Camera: Device Manager for driver issues
    pause
    exit /b 1
)

exit /b 0
