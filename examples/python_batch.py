#!/usr/bin/env python3
"""
ALICE-Edge Python batch processing demo.

Demonstrates calling ALICE-Edge from Python via PyO3 bindings.
Processes NumPy arrays with zero-copy for maximum throughput.

Prerequisites:
    cd ALICE-Edge
    pip install maturin numpy
    maturin develop --features pyo3

Usage:
    python examples/python_batch.py

Author: Moroya Sakamoto
"""

import time
import sys

try:
    import numpy as np
except ImportError:
    print("numpy required: pip install numpy")
    sys.exit(1)

try:
    import alice_edge
except ImportError:
    print("alice_edge not installed. Run:")
    print("  cd ALICE-Edge && maturin develop --features pyo3")
    print("\nRunning with pure Python fallback...\n")
    alice_edge = None


def fit_linear_python(data):
    """Pure Python reference implementation."""
    n = len(data)
    if n < 2:
        return (0, data[0] if n == 1 else 0)

    sum_x = n * (n - 1) // 2
    sum_xx = n * (n - 1) * (2 * n - 1) // 6
    sum_y = sum(data)
    sum_xy = sum(i * y for i, y in enumerate(data))

    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return (0, sum_y // n)

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return (slope, intercept)


def main():
    print("=== ALICE-Edge: Python Batch Processing ===\n")

    # Generate simulated sensor data
    np.random.seed(42)
    sample_counts = [100, 1000, 10000, 100000]

    for n in sample_counts:
        # Simulate temperature: 25.00°C with slight upward trend + noise
        trend = np.linspace(2500, 2510, n)
        noise = np.random.randint(-5, 6, n)
        data = (trend + noise).astype(np.int32)

        print(f"--- {n:,} samples ---")

        # Python reference
        start = time.perf_counter()
        py_slope, py_intercept = fit_linear_python(data.tolist())
        py_time = (time.perf_counter() - start) * 1e6

        print(f"  Python:     slope={py_slope:.4f} intercept={py_intercept:.2f} ({py_time:.0f} µs)")

        # Rust via PyO3
        if alice_edge:
            start = time.perf_counter()
            rust_slope, rust_intercept = alice_edge.fit_linear_fixed(data)
            rust_time = (time.perf_counter() - start) * 1e6
            speedup = py_time / rust_time if rust_time > 0 else float("inf")
            print(f"  Rust+SIMD:  slope={rust_slope} intercept={rust_intercept} ({rust_time:.0f} µs) [{speedup:.0f}x faster]")

        # Compression stats
        raw_bytes = n * 4
        compressed_bytes = 8
        print(f"  Compression: {raw_bytes:,} → {compressed_bytes} bytes ({raw_bytes // compressed_bytes}x)\n")

    # Throughput benchmark
    print("--- Throughput Benchmark ---")
    n = 1_000_000
    data = np.random.randint(0, 10000, n, dtype=np.int32)

    if alice_edge:
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            alice_edge.fit_linear_fixed(data)
        elapsed = time.perf_counter() - start

        samples_per_sec = (n * iterations) / elapsed
        print(f"  {n:,} samples × {iterations} iterations = {elapsed:.2f}s")
        print(f"  Throughput: {samples_per_sec:,.0f} samples/sec")
        print(f"  Latency: {elapsed / iterations * 1e6:.0f} µs per fit")
    else:
        print("  (Skipped — alice_edge module not installed)")


if __name__ == "__main__":
    main()
