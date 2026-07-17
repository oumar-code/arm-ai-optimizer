"""
latency_benchmark.py
Measures end-to-end inference latency for the PdM models on the
current hardware and prints a summary table.

Targets:
  Jetson AGX Orin    ~25 ms total
  Graviton3          ~43 ms total
  Raspberry Pi 5     ~135 ms total (warning: above 50 ms target)

Usage:
    python benchmarks/latency_benchmark.py [--model-dir edge/model] [--runs 100]
"""

import argparse
import logging
import os
import pickle
import platform
import time
from pathlib import Path
from typing import List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("benchmark")

N_FEATURES = 21
WINDOW_LEN = 60


def detect_hardware() -> str:
    machine = platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            info = f.read()
        if "Graviton" in info or "aarch64" in machine.lower():
            if "Graviton3" in info:
                return "Graviton3"
            if "Orin" in info or "Xavier" in info:
                return "Jetson AGX Orin"
            if "Raspberry Pi 5" in info:
                return "Raspberry Pi 5"
            return f"Arm64 ({machine})"
    except FileNotFoundError:
        pass
    return f"{platform.processor() or platform.machine()}"


def benchmark_isolation_forest(model_dir: Path, n_runs: int) -> List[float]:
    if_path = model_dir / "isolation_forest.pkl"
    scaler_path = model_dir / "scaler.pkl"
    if not if_path.exists():
        log.warning("isolation_forest.pkl not found — skipping IF benchmark")
        return []

    with open(if_path, "rb") as fh:
        clf = pickle.load(fh)
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)

    dummy = np.random.randn(1, N_FEATURES).astype(np.float32)
    if scaler:
        dummy = scaler.transform(dummy)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        clf.predict(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    warmup = min(10, len(latencies) // 2)
    return latencies[warmup:]   # drop warm-up runs


def benchmark_onnx(model_dir: Path, n_runs: int, model_name: str) -> List[float]:
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping ONNX benchmark")
        return []

    onnx_path = model_dir / model_name
    if not onnx_path.exists():
        log.warning("%s not found — skipping", model_name)
        return []

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy_seq = np.random.randn(1, WINDOW_LEN, N_FEATURES).astype(np.float32)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"features": dummy_seq})
        latencies.append((time.perf_counter() - t0) * 1000)

    warmup = min(10, len(latencies) // 2)
    return latencies[warmup:]


def summarise(name: str, latencies: List[float]) -> None:
    if not latencies:
        return
    arr = np.array(latencies)
    log.info(
        "%-35s  p50=%6.1f ms  p95=%6.1f ms  p99=%6.1f ms  mean=%6.1f ms",
        name,
        np.percentile(arr, 50),
        np.percentile(arr, 95),
        np.percentile(arr, 99),
        np.mean(arr),
    )


def print_table(results: dict, target_ms: float = 50.0) -> None:
    print("\n" + "=" * 65)
    print(f"  Arm64 Inference Latency Benchmark — {detect_hardware()}")
    print("=" * 65)
    total = 0.0
    for name, latencies in results.items():
        if not latencies:
            continue
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        total += p50
        status = "✅" if p50 < target_ms else "⚠️ "
        print(f"  {name:<35} p50={p50:6.1f} ms  p95={p95:6.1f} ms  {status}")
    print("-" * 65)
    status = "✅" if total < target_ms else "⚠️ "
    print(f"  {'TOTAL (p50 sum)':<35} {total:6.1f} ms  {status} (target < {target_ms:.0f} ms)")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="edge/model", help="Model directory")
    parser.add_argument("--runs", type=int, default=100, help="Number of inference runs")
    parser.add_argument("--target-ms", type=float, default=50.0, help="Latency target in ms")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    log.info("Hardware: %s", detect_hardware())
    log.info("Running %d inference iterations per model …", args.runs)

    results = {
        "Isolation Forest (sklearn)": benchmark_isolation_forest(model_dir, args.runs),
        "LSTM FP32 (ONNX)": benchmark_onnx(model_dir, args.runs, "lstm_rul_fp32.onnx"),
        "LSTM INT8 (ONNX quantised)": benchmark_onnx(model_dir, args.runs, "lstm_rul_int8.onnx"),
    }

    for name, latencies in results.items():
        summarise(name, latencies)

    print_table(results, args.target_ms)


if __name__ == "__main__":
    main()
