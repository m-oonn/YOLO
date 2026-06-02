"""Performance benchmark script - measures key metrics before/after optimization."""
import time
import requests
import json
import sys

BASE = "http://127.0.0.1:8000/api"
RESULTS = {}

def bench(label, fn, iterations=5):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    RESULTS[label] = {"avg_ms": round(avg, 1), "min_ms": round(mn, 1), "max_ms": round(mx, 1)}
    print(f"  {label}: avg={avg:.1f}ms, min={mn:.1f}ms, max={mx:.1f}ms")
    return result

print("=" * 60)
print("Performance Benchmark")
print("=" * 60)

# 1. Health check
print("\n1. Health Check")
bench("GET /health", lambda: requests.get(f"{BASE.replace('/api','')}/health", timeout=5))

# 2. Detection status
print("\n2. Detection Status")
bench("GET /detection/status", lambda: requests.get(f"{BASE}/detection/status", timeout=5))

# 3. Events API
print("\n3. Events API")
bench("GET /events/", lambda: requests.get(f"{BASE}/events/", timeout=5))
bench("GET /events/stats", lambda: requests.get(f"{BASE}/events/stats", timeout=5))
bench("GET /events/types", lambda: requests.get(f"{BASE}/events/types", timeout=5))

# 4. Alarms API
print("\n4. Alarms API")
bench("GET /alarms/", lambda: requests.get(f"{BASE}/alarms/", timeout=5))
bench("GET /alarms/stats", lambda: requests.get(f"{BASE}/alarms/stats", timeout=5))

# 5. Cameras API
print("\n5. Cameras API")
bench("GET /cameras/", lambda: requests.get(f"{BASE}/cameras/", timeout=5))

# 6. MLLM Status
print("\n6. MLLM Status")
bench("GET /mllm/status", lambda: requests.get(f"{BASE}/mllm/status", timeout=5))

# 7. Detection start/stop cycle
print("\n7. Detection Start/Stop Cycle")
bench("POST /detection/start", lambda: requests.post(f"{BASE}/detection/start", json={"source": "0", "config": "configs/default.yaml"}, timeout=30))
time.sleep(5)
bench("GET /detection/status (running)", lambda: requests.get(f"{BASE}/detection/status", timeout=5))
bench("POST /detection/stop", lambda: requests.post(f"{BASE}/detection/stop", timeout=10))

# Summary
print("\n" + "=" * 60)
print("Benchmark Summary")
print("=" * 60)
for k, v in RESULTS.items():
    print(f"  {k}: {v['avg_ms']}ms (min={v['min_ms']}ms, max={v['max_ms']}ms)")

# Save results
with open("outputs/benchmark_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nResults saved to outputs/benchmark_results.json")
