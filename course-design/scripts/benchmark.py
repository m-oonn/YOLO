#!/usr/bin/env python3
# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark script for YOLO Detection System.

This script measures performance metrics including:
- Detection FPS
- MJPEG stream latency
- API response times
- Concurrent client handling

Usage:
    python scripts/benchmark.py                    # Run all benchmarks
    python scripts/benchmark.py --quick          # Quick benchmark
    python scripts/benchmark.py --fps            # FPS benchmark only
    python scripts/benchmark.py --api             # API benchmark only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import requests

BASE_URL = os.environ.get("API_URL", "http://localhost:8000")
API_PREFIX = f"{BASE_URL}/api/detection"


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""

    name: str
    iterations: int
    times: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.times) * 1000 if self.times else 0

    @property
    def min_ms(self) -> float:
        return min(self.times) * 1000 if self.times else 0

    @property
    def max_ms(self) -> float:
        return max(self.times) * 1000 if self.times else 0

    @property
    def p95_ms(self) -> float:
        if not self.times:
            return 0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx] * 1000

    @property
    def success_rate(self) -> float:
        total = len(self.times) + len(self.errors)
        return len(self.times) / total if total > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "success_rate": round(self.success_rate * 100, 1),
            "errors": self.errors[:5],
        }


def benchmark_api_latency(endpoint: str, iterations: int = 100) -> BenchmarkResults:
    """Measure API endpoint latency.

    Args:
        endpoint: API endpoint path (relative to API_PREFIX)
        iterations: Number of requests to make

    Returns:
        BenchmarkResults with timing statistics
    """
    result = BenchmarkResults(name=f"API: {endpoint}", iterations=iterations)
    url = f"{API_PREFIX}{endpoint}"

    for _i in range(iterations):
        try:
            start = time.perf_counter()
            response = requests.get(url, timeout=5)
            elapsed = time.perf_counter() - start

            if response.status_code == 200:
                result.times.append(elapsed)
            else:
                result.errors.append(f"HTTP {response.status_code}")
        except Exception as e:
            result.errors.append(str(e)[:100])

    return result


def benchmark_status_polling(iterations: int = 50) -> BenchmarkResults:
    """Measure status polling performance.

    Args:
        iterations: Number of polls to perform

    Returns:
        BenchmarkResults with timing statistics
    """
    result = BenchmarkResults(name="Status Polling", iterations=iterations)

    for _i in range(iterations):
        try:
            start = time.perf_counter()
            response = requests.get(f"{API_PREFIX}/status", timeout=5)
            elapsed = time.perf_counter() - start

            if response.status_code == 200:
                result.times.append(elapsed)
            else:
                result.errors.append(f"HTTP {response.status_code}")
        except Exception as e:
            result.errors.append(str(e)[:100])

        time.sleep(0.1)

    return result


def benchmark_monitoring_endpoint(iterations: int = 30) -> BenchmarkResults:
    """Measure monitoring endpoint performance.

    Args:
        iterations: Number of requests to make

    Returns:
        BenchmarkResults with timing statistics
    """
    result = BenchmarkResults(name="Monitoring Endpoint", iterations=iterations)
    url = f"{API_PREFIX}/monitoring"

    for _i in range(iterations):
        try:
            start = time.perf_counter()
            response = requests.get(url, timeout=5)
            elapsed = time.perf_counter() - start

            if response.status_code == 200:
                result.times.append(elapsed)
            else:
                result.errors.append(f"HTTP {response.status_code}")
        except Exception as e:
            result.errors.append(str(e)[:100])

        time.sleep(0.2)

    return result


def benchmark_concurrent_api(
    iterations: int = 100, concurrent: int = 10
) -> dict[str, Any]:
    """Measure concurrent API request handling.

    Args:
        iterations: Total number of requests
        concurrent: Number of concurrent requests

    Returns:
        Dictionary with concurrency test results
    """
    import concurrent.futures

    result = {
        "name": f"Concurrent API ({concurrent} parallel)",
        "total_requests": iterations,
        "successful": 0,
        "failed": 0,
        "total_time_s": 0,
        "requests_per_sec": 0,
    }

    start_time = time.perf_counter()

    def make_request():
        try:
            response = requests.get(f"{API_PREFIX}/status", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(make_request) for _ in range(iterations)]
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                result["successful"] += 1
            else:
                result["failed"] += 1

    result["total_time_s"] = time.perf_counter() - start_time
    result["requests_per_sec"] = iterations / result["total_time_s"]

    return result


async def benchmark_websocket_throughput(duration_s: int = 10) -> dict[str, Any]:
    """Measure WebSocket message throughput.

    Args:
        duration_s: Test duration in seconds

    Returns:
        Dictionary with WebSocket test results
    """
    import websocket

    result = {
        "name": "WebSocket Throughput",
        "duration_s": duration_s,
        "messages_received": 0,
        "messages_per_sec": 0,
        "errors": 0,
    }

    ws_url = "ws://localhost:8000/api/detection/stream"
    ws = None

    try:
        ws = websocket.create_connection(ws_url, timeout=5)
        start_time = time.perf_counter()
        end_time = start_time + duration_s

        while time.time() < end_time:
            try:
                msg = ws.recv()
                if msg:
                    result["messages_received"] += 1
            except Exception:
                result["errors"] += 1
                break

        result["messages_per_sec"] = result["messages_received"] / duration_s

    except Exception as e:
        result["errors"] += 1
        result["error_message"] = str(e)[:200]
    finally:
        if ws:
            ws.close()

    return result


def run_all_benchmarks(quick: bool = False) -> list[dict[str, Any]]:
    """Run all benchmarks.

    Args:
        quick: If True, reduce iterations for faster testing

    Returns:
        List of benchmark results dictionaries
    """
    iterations = 20 if quick else 100
    concurrent = 5 if quick else 10
    duration = 5 if quick else 10

    print("\n" + "=" * 60)
    print("YOLO Detection System - Performance Benchmark")
    print("=" * 60)

    results = []

    print("\n[1/5] Testing API latency (status endpoint)...")
    r = benchmark_status_polling(iterations=iterations)
    results.append(r.to_dict())
    print(
        f"  Avg: {r.avg_ms:.2f}ms | P95: {r.p95_ms:.2f}ms | Success: {r.success_rate * 100:.1f}%"
    )

    print("\n[2/5] Testing API latency (models endpoint)...")
    r = benchmark_api_latency("/models", iterations=iterations)
    results.append(r.to_dict())
    print(
        f"  Avg: {r.avg_ms:.2f}ms | P95: {r.p95_ms:.2f}ms | Success: {r.success_rate * 100:.1f}%"
    )

    print("\n[3/5] Testing monitoring endpoint...")
    r = benchmark_monitoring_endpoint(iterations=iterations // 2)
    results.append(r.to_dict())
    print(
        f"  Avg: {r.avg_ms:.2f}ms | P95: {r.p95_ms:.2f}ms | Success: {r.success_rate * 100:.1f}%"
    )

    print("\n[4/5] Testing concurrent requests...")
    r = benchmark_concurrent_api(iterations=iterations, concurrent=concurrent)
    results.append(r)
    print(
        f"  Total: {r['total_requests']} | Success: {r['successful']} | {r['requests_per_sec']:.1f} req/s"
    )

    print("\n[5/5] Testing WebSocket throughput...")
    try:
        ws_result = asyncio.run(benchmark_websocket_throughput(duration_s=duration))
        results.append(ws_result)
        print(
            f"  Messages: {ws_result['messages_received']} | {ws_result['messages_per_sec']:.1f} msg/s"
        )
    except Exception as e:
        results.append({"name": "WebSocket Throughput", "error": str(e)})
        print(f"  WebSocket test failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="YOLO Detection System Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    args = parser.parse_args()

    results = run_all_benchmarks(quick=args.quick)

    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "-" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
