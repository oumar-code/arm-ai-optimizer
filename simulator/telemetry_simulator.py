"""
telemetry_simulator.py
Generates realistic sensor streams for SMT line machines and publishes
them to MQTT. Supports configurable fault injection to simulate bearing
degradation / thermal runaway.

Topic schema: coocah/pe-sagamu/{machine_id}/telemetry
"""

import json
import math
import random
import time
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any

import paho.mqtt.client as mqtt
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("telemetry-sim")

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config.yaml")


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def fault_progress(elapsed_s: float, start_offset_s: float, ramp_duration_s: float) -> float:
    """Return a fault severity in [0, 1]. 0 = healthy, 1 = fully degraded."""
    if elapsed_s < start_offset_s:
        return 0.0
    ramp = (elapsed_s - start_offset_s) / ramp_duration_s
    return min(ramp, 1.0)


def generate_reading(sensor_cfg: Dict, severity: float) -> float:
    mean = sensor_cfg["normal_mean"]
    std = sensor_cfg["normal_std"]
    multiplier = sensor_cfg.get("fault_multiplier", 1.0)
    # Linearly interpolate between normal and faulted mean
    effective_mean = mean * (1 + (multiplier - 1) * severity)
    # Add Gaussian noise
    return random.gauss(effective_mean, std * (1 + severity * 0.5))


def make_payload(machine_id: str, readings: Dict[str, float], severity: float) -> str:
    rul_hours = max(0.0, 48.0 * (1.0 - severity))  # 48 h at full health
    payload = {
        "machine_id": machine_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vibration_g": round(readings["vibration"], 4),
        "temperature_c": round(readings["temperature"], 2),
        "current_a": round(readings["current"], 3),
        "fault_severity": round(severity, 4),
        "rul_hours_true": round(rul_hours, 2),   # ground truth for evaluation
    }
    return json.dumps(payload)


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    mqtt_cfg = cfg["mqtt"]

    client = mqtt.Client(client_id="telemetry-sim")
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    broker = mqtt_cfg["broker"]
    port = mqtt_cfg["port"]
    log.info("Connecting to MQTT broker %s:%d …", broker, port)

    # Retry connection on startup
    for attempt in range(30):
        try:
            client.connect(broker, port, keepalive=60)
            break
        except OSError as exc:
            log.warning("Broker not ready (attempt %d): %s", attempt + 1, exc)
            time.sleep(2)
    else:
        raise RuntimeError("Could not connect to MQTT broker after 30 attempts")

    client.loop_start()
    log.info("Connected. Starting telemetry stream …")

    start_time = time.monotonic()
    interval = mqtt_cfg.get("publish_interval_s", 1)

    while True:
        elapsed = time.monotonic() - start_time

        for machine in cfg["machines"]:
            mid = machine["id"]
            fi = machine["fault_injection"]
            severity = 0.0
            if fi.get("enabled", False):
                severity = fault_progress(
                    elapsed,
                    fi["start_offset_s"],
                    fi["ramp_duration_s"],
                )

            readings = {
                sensor: generate_reading(sensor_cfg, severity)
                for sensor, sensor_cfg in machine["sensors"].items()
            }

            payload = make_payload(mid, readings, severity)
            topic = f"{mqtt_cfg['topic_prefix']}/{mid}/telemetry"
            client.publish(topic, payload, qos=1)

            if int(elapsed) % 30 == 0:
                log.info(
                    "[%s] t=%.0fs severity=%.3f vib=%.3fg temp=%.1f°C I=%.2fA",
                    mid,
                    elapsed,
                    severity,
                    readings["vibration"],
                    readings["temperature"],
                    readings["current"],
                )

        time.sleep(interval)


if __name__ == "__main__":
    main()
