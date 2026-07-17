# Simulator Dockerfile — multi-target build
# Target 'telemetry': runs telemetry_simulator.py
# Target 'energy':    runs energy_simulator.py

FROM python:3.11-slim AS base

WORKDIR /app

COPY simulator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simulator/ .

# ── Telemetry target ──────────────────────────────────────────────────────────
FROM base AS telemetry
CMD ["python", "telemetry_simulator.py"]

# ── Energy target ─────────────────────────────────────────────────────────────
FROM base AS energy
CMD ["python", "energy_simulator.py"]
