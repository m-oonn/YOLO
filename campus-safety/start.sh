#!/bin/bash
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0
#
# YOLO Detection System - One-Click Startup Script for Unix/macOS
# This script provides a simple way to start the YOLO detection system
# with automatic health checks and progress feedback.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo "============================================================"
echo "  YOLO Detection System - Unix/macOS Launcher"
echo "============================================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python 3 not found. Please install Python 3.10 or higher.${NC}"
    echo "         Download from: https://www.python.org/downloads/"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python launcher
echo -e "${BLUE}[INFO]${NC} Starting YOLO Detection System..."
echo ""

python3 "$SCRIPT_DIR/scripts/launcher.py" "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR]${NC} Startup failed. Check the error messages above."
    exit $EXIT_CODE
fi

exit 0
