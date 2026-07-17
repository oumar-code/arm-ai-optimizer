"""
ingestor.py
MQTT → InfluxDB bridge with 60-second windowed feature extraction.

Subscribes to:
  coocah/pe-sagamu/+/telemetry
  coocah/pe-sagamu/energy/state

Writes raw measurements and extracted features to InfluxDB every 60 s.
"""

import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List

import numpy as np
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ingestor")

# ── Environment ──────────────────────────────────────────────────────────────
MQTT_BROKER = os.environ.get("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
TOPIC_PREFIX = os.environ.get("TOPIC_PREFIX", "coocah/pe-sagamu")

INFLUX_URL = os.environ.get("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "edge-demo-token")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "coocah")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "factory")

WINDOW_SECONDS = int(os.environ.get("WINDOW_SECONDS", "60"))
# ─────────────────────────────────────────────────────────────────────────────

SENSOR_FIELDS = ["vibration_g", "temperature_c", "current_a"]
MAX_WINDOW = WINDOW_SECONDS + 10   # keep a little extra headroom

# Rolling buffers: machine_id → {field: deque of (timestamp, value)}
buffers: Dict[str, Dict[str, Deque]] = defaultdict(
    lambda: {f: deque(maxlen=MAX_WINDOW) for f in SENSOR_FIELDS}
)

# Latest energy state
energy_state: Dict[str, Any] = {}


def extract_features(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "peak_to_peak": float(np.ptp(arr)),
        "kurtosis": float(
            np.mean(((arr - np.mean(arr)) / (np.std(arr) + 1e-9)) ** 4)
        ),
    }


def on_message(client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
    try:
        payload = json.loads(msg.payload.decode())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning("Bad payload on %s: %s", msg.topic, exc)
        return

    now = time.monotonic()

    if msg.topic.endswith("/telemetry"):
        mid = payload.get("machine_id", "unknown")
        for field in SENSOR_FIELDS:
            if field in payload:
                buffers[mid][field].append((now, payload[field]))

    elif msg.topic.endswith("/energy/state"):
        energy_state.update(payload)


def flush_features(write_api: Any) -> None:
    """Extract features from the last WINDOW_SECONDS of data and write to InfluxDB."""
    cutoff = time.monotonic() - WINDOW_SECONDS
    now_utc = datetime.now(timezone.utc)

    for mid, fields in buffers.items():
        point = (
            Point("machine_features")
            .tag("machine_id", mid)
            .time(now_utc, WritePrecision.NANOSECONDS)
        )
        feature_count = 0
        for field, buf in fields.items():
            window_vals = [v for ts, v in buf if ts >= cutoff]
            if not window_vals:
                continue
            feats = extract_features(window_vals)
            for feat_name, feat_val in feats.items():
                point = point.field(f"{field}__{feat_name}", feat_val)
                feature_count += 1
            # Also write the raw mean as a convenience
            point = point.field(f"{field}_mean", feats.get("mean", 0.0))

        if feature_count > 0:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
            log.debug("Wrote %d features for %s", feature_count, mid)

    # Write energy snapshot
    if energy_state:
        ep = Point("energy_state").time(now_utc, WritePrecision.NANOSECONDS)
        for k, v in energy_state.items():
            if isinstance(v, (int, float)):
                ep = ep.field(k, float(v))
            elif isinstance(v, bool):
                ep = ep.field(k, int(v))
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=ep)

    log.info("Feature flush complete for %d machine(s)", len(buffers))


def setup_influx() -> Any:
    log.info("Connecting to InfluxDB at %s …", INFLUX_URL)
    for attempt in range(30):
        try:
            influx = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            # Probe health
            influx.ping()
            log.info("InfluxDB ready")
            return influx.write_api(write_options=SYNCHRONOUS)
        except Exception as exc:
            log.warning("InfluxDB not ready (attempt %d): %s", attempt + 1, exc)
            time.sleep(3)
    raise RuntimeError("Could not connect to InfluxDB after 30 attempts")


def main() -> None:
    write_api = setup_influx()

    mqttc = mqtt.Client(client_id="ingestor")
    mqttc.on_message = on_message

    log.info("Connecting to MQTT broker %s:%d …", MQTT_BROKER, MQTT_PORT)
    for attempt in range(30):
        try:
            mqttc.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except OSError as exc:
            log.warning("Broker not ready (attempt %d): %s", attempt + 1, exc)
            time.sleep(2)
    else:
        raise RuntimeError("Could not connect to MQTT broker after 30 attempts")

    mqttc.subscribe(f"{TOPIC_PREFIX}/+/telemetry", qos=1)
    mqttc.subscribe(f"{TOPIC_PREFIX}/energy/state", qos=1)
    mqttc.loop_start()
    log.info("Subscribed. Flushing features every %d s …", WINDOW_SECONDS)

    while True:
        time.sleep(WINDOW_SECONDS)
        try:
            flush_features(write_api)
        except Exception as exc:
            log.error("Flush error: %s", exc)


if __name__ == "__main__":
    main()
