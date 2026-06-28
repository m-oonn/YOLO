#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Quick API health check script.

Iterates over a list of core API endpoints and reports their HTTP status
codes and response times. Useful for smoke-testing a running backend.

Usage:
    python scripts/check_api.py
"""

import requests

BASE_URL = "http://127.0.0.1:8000"
endpoints = [
    "/health",
    "/api/events",
    "/api/events/stats",
    "/api/events/types",
    "/api/detection/status",
    "/api/cameras",
]

print("API Endpoint Health Check\n")
print("=" * 60)

for endpoint in endpoints:
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", timeout=3)
        status = "✓" if r.status_code == 200 else "✗"
        print(f"{status} {endpoint:30s} -> {r.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ {endpoint:30s} -> ERROR: {e}")

print("=" * 60)
