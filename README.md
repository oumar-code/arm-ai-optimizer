## What This Is

This repository contains the **runnable implementation** of the Coo-Cah Edge Twin for Arm — an edge-first AI system for factory predictive maintenance and energy-aware scheduling, purpose-built for Arm64 hardware.

For the full architecture documentation, hackathon strategy, and factory context see the master docs repo:

- **Strategy & concept:** [`docs/hackathon/index.md`](https://github.com/oumar-code/Coo-Kah-Doks/blob/main/docs/hackathon/index.md)
- **Demo architecture:** [`docs/hackathon/demo-architecture.md`](https://github.com/oumar-code/Coo-Kah-Doks/blob/main/docs/hackathon/demo-architecture.md)
- **Submission concept:** [`docs/hackathon/submission-concept.md`](https://github.com/oumar-code/Coo-Kah-Doks/blob/main/docs/hackathon/submission-concept.md)
- **AI platform architecture:** [`docs/08-ai-platform/index.md`](https://github.com/oumar-code/Coo-Kah-Doks/blob/main/docs/08-ai-platform/index.md)

---

## Quick Start

```bash
git clone https://github.com/oumar-code/arm-ai-optimizer.git
cd arm-ai-optimizer
docker compose up
```

Then open Grafana at `http://localhost:3000` — default credentials: `admin / admin`.

---

## What It Does

| Component | Description |
|-----------|-------------|
| **Telemetry Simulator** | Generates realistic sensor streams (vibration, temperature, current) for 2 SMT line machines with configurable fault injection |
| **Energy Simulator** | Simulates solar generation and BESS state-of-charge (SoC) curves |
| **Ingestor Service** | MQTT → InfluxDB with 60-second windowed feature extraction |
| **PdM Model** | Isolation Forest (anomaly detection) + LSTM (RUL estimation) — exported to ONNX INT8, running via ONNX Runtime Arm64 |
| **Energy Advisor** | Rule engine + Prophet forecast recommending batch-start decisions based on BESS/solar state |
| **Cloud Sync Agent** | Offline-resilient delta sync to Rwanda cloud hub (≥ 80% bandwidth reduction vs. raw stream) |
| **Grafana Dashboard** | Machine health, sensor telemetry, and energy status — fully local, no cloud dependency |

---

## KPI Targets

| KPI | Target |
|-----|--------|
| Edge inference latency | < 50 ms per 60-second sensor window on Arm64 |
| Cloud bandwidth reduction | ≥ 80% vs. raw stream upload |
| Fault prediction lead time | ≥ 48 hours before failure |
| Energy recommendation accuracy | ≥ 85% correct decisions vs. manual baseline |

---

## Arm64 Optimisation

See [`docs/arm-optimisation.md`](docs/arm-optimisation.md) for full details.

Key optimisations:
- **INT8 post-training quantisation** via ONNX Runtime quantisation tools (~40% latency reduction vs. FP32)
- **ONNX Runtime ARM64 execution provider** — native Arm64 path, no x86 emulation
- **Memory-mapped feature store** — efficient feature access on edge nodes with limited RAM
- **Lightweight MQTT pipeline** — Mosquitto at < 1% CPU on Cortex-A

### Latency Benchmarks

| Hardware | Isolation Forest (INT8) | LSTM (FP16) | Total per window |
|----------|------------------------|-------------|-----------------|
| Jetson AGX Orin | ~5 ms | ~20 ms | ~25 ms ✅ |
| Graviton3 (c7g.xlarge) | ~8 ms | ~35 ms | ~43 ms ✅ |
| Raspberry Pi 5 | ~15 ms | ~120 ms | ~135 ms ⚠️ |

---

## Repository Structure

```
arm-ai-optimizer/
├── README.md
├── docker-compose.yml               ← One-command demo stack
├── simulator/
│   ├── telemetry_simulator.py       ← Sensor stream generator
│   ├── energy_simulator.py          ← BESS + solar state generator
│   └── config.yaml                  ← Machine profiles + failure injection
├── edge/
│   ├── ingestor/                    ← MQTT → InfluxDB + feature extraction
│   ├── model/
│   │   ├── train.py                 ← Model training
│   │   ├── export_onnx.py           ← ONNX export + INT8 quantisation
│   │   └── pdm_model.onnx           ← Pre-trained model
│   ├── energy_advisor/              ← Energy recommendation service
│   └── sync_agent/                  ← Cloud sync + model pull
├── dashboards/
│   └── grafana/
│       └── factory-ops.json         ← Grafana dashboard JSON
├── cloud/
│   └── mlflow_setup/                ← MLflow server config for Rwanda hub
├── docs/
│   └── arm-optimisation.md          ← Arm64 optimisation details
└── benchmarks/
    └── latency_benchmark.py         ← Hardware comparison script
```

---

## Factory Context

This demo targets the **Personal Electronics pilot factory — SMT line, Sagamu, Nigeria**.  
Full factory specification: [`docs/factories/electronics/personal-electronics/`](https://github.com/oumar-code/Coo-Kah-Doks/tree/main/docs/factories/electronics/personal-electronics)

MQTT topics follow the Coo-Cah group schema: `coocah/pe-sagamu/{asset_id}/telemetry`  
Topic schema reference: [`platform/mqtt-topic-schema.md`](https://github.com/oumar-code/Coo-Kah-Doks/blob/main/platform/mqtt-topic-schema.md)

---

*Coo-Cah Technologies Holdings — Arm AI Developer Challenge 2026*
<shellId: 1 completed with exit code 0>

