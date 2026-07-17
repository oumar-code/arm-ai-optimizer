# Arm64 Optimisation — Coo-Cah Edge Twin

## Overview

The Coo-Cah Edge Twin is purpose-built to run on Arm64 hardware. Every layer of the inference pipeline is optimised to hit the **< 50 ms per 60-second window** KPI target on production Arm hardware.

---

## Quantisation: INT8 Post-Training Quantisation

The LSTM RUL predictor is exported from PyTorch to ONNX FP32 and then quantised to INT8 using ONNX Runtime's `quantize_dynamic` API.

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "lstm_rul_fp32.onnx",
    "lstm_rul_int8.onnx",
    weight_type=QuantType.QInt8,
)
```

**Measured latency reduction:**

| Model | FP32 | INT8 | Reduction |
|-------|------|------|-----------|
| LSTM (Graviton3) | ~58 ms | ~35 ms | ~40% |
| LSTM (Jetson AGX Orin) | ~32 ms | ~20 ms | ~38% |

Dynamic quantisation does not require a calibration dataset and works immediately — ideal for hackathon demo day.

---

## ONNX Runtime Arm64 Execution Provider

ONNX Runtime ships native Arm64 kernels tuned for Cortex-A and Neoverse microarchitectures. The key difference is:

- **No x86 emulation** — instructions execute natively on the Arm pipeline.
- **NEON SIMD** — matrix multiplications use `vfmaq_f32` / `vmlaq_s8` instructions.
- **KleidiAI micro-kernels** — from ORT 1.18+, Arm's KleidiAI library provides further optimised INT8 GeMM routines on Cortex-A520/A720 and Neoverse V2.

```python
import onnxruntime as ort

sess = ort.InferenceSession(
    "pdm_model.onnx",
    providers=["CPUExecutionProvider"],   # native on Arm64 — no ArmNN needed
)
```

---

## Memory-Mapped Feature Store

The ingestor writes features into InfluxDB, but the inference service queries only the most recent window row — a single lightweight Flux query. No large in-memory buffers are held; `deque(maxlen=70)` rolling windows per sensor keep RAM usage bounded at ~2 MB per machine.

---

## Lightweight MQTT Pipeline

Mosquitto 2.0 runs at **< 1% CPU** on Cortex-A55 cores, publishing 1-second telemetry frames at ~500 bytes/message. QoS 1 ensures exactly-once delivery without TCP overhead of QoS 2.

---

## Latency Benchmarks

Collected via `benchmarks/latency_benchmark.py` (p50 over 100 runs):

| Hardware | Isolation Forest | LSTM INT8 | **Total** | Target |
|----------|-----------------|-----------|-----------|--------|
| Jetson AGX Orin (Cortex-A78AE) | ~5 ms | ~20 ms | **~25 ms** | ✅ < 50 ms |
| Graviton3 (c7g.xlarge, Neoverse V1) | ~8 ms | ~35 ms | **~43 ms** | ✅ < 50 ms |
| Raspberry Pi 5 (Cortex-A76) | ~15 ms | ~120 ms | **~135 ms** | ⚠️ > 50 ms |

> **Note:** Raspberry Pi 5 misses the 50 ms target due to limited L2 cache (512 KB per core). On Pi 5 the Isolation Forest is fast but the LSTM exceeds budget. Recommended minimum hardware: Jetson Orin Nano or equivalent Cortex-A78 device.

---

## Run the Benchmark

```bash
# Inside the model container (after training):
python benchmarks/latency_benchmark.py --model-dir edge/model --runs 200

# Or directly on Arm hardware after pip install:
pip install onnxruntime scikit-learn numpy
python benchmarks/latency_benchmark.py
```

---

## Further Optimisation Path (Post-Hackathon)

| Technique | Expected Gain |
|-----------|---------------|
| Static INT8 quantisation with calibration data | Additional ~10% over dynamic INT8 |
| KleidiAI explicit provider (`ArmNNExecutionProvider`) | ~15% on Cortex-A520+ |
| TensorRT on Jetson | ~50% LSTM reduction |
| Feature pruning (reduce from 21 → 14) | Proportional linear reduction |

---

*Coo-Cah Technologies Holdings — Arm AI Developer Challenge 2026*
