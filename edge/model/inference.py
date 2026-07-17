"""
inference.py
PdM inference service.

Every INFER_INTERVAL_S seconds:
  1. Reads the latest machine_features from InfluxDB.
  2. Runs Isolation Forest anomaly score.
  3. Runs LSTM INT8 RUL prediction via ONNX Runtime.
  4. Writes results back to InfluxDB (machine_health measurement).
  5. Publishes alerts to MQTT if anomaly detected.
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("pdm-inference")

# ── Environment ───────────────────────────────────────────────────────────────
INFLUX_URL = os.environ.get("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "edge-demo-token")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "coocah")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "factory")

MQTT_BROKER = os.environ.get("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
TOPIC_PREFIX = os.environ.get("TOPIC_PREFIX", "coocah/pe-sagamu")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
INFER_INTERVAL_S = int(os.environ.get("INFER_INTERVAL_S", "60"))
WINDOW_LEN = int(os.environ.get("WINDOW_LEN", "60"))
# ─────────────────────────────────────────────────────────────────────────────

MACHINE_IDS = ["smt-pick-place-01", "smt-reflow-oven-02"]
SENSOR_FIELDS = ["vibration_g", "temperature_c", "current_a"]
FEATURE_SUFFIXES = ["mean", "std", "min", "max", "rms", "peak_to_peak", "kurtosis"]
N_FEATURES = len(SENSOR_FIELDS) * len(FEATURE_SUFFIXES)


def load_models():
    scaler_path = MODEL_DIR / "scaler.pkl"
    if_path = MODEL_DIR / "isolation_forest.pkl"
    onnx_path = MODEL_DIR / "pdm_model.onnx"

    scaler = None
    iforest = None
    ort_session = None

    if scaler_path.exists():
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
        log.info("Scaler loaded from %s", scaler_path)

    if if_path.exists():
        with open(if_path, "rb") as fh:
            iforest = pickle.load(fh)
        log.info("Isolation Forest loaded from %s", if_path)

    if onnx_path.exists():
        providers = ["CPUExecutionProvider"]
        ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
        log.info("ONNX session loaded from %s (providers=%s)", onnx_path, providers)
    else:
        log.warning("pdm_model.onnx not found — RUL prediction will be skipped")

    return scaler, iforest, ort_session


def query_latest_features(influx: InfluxDBClient, machine_id: str) -> Optional[np.ndarray]:
    """Query last WINDOW_LEN feature rows from InfluxDB and return as a numpy array."""
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{WINDOW_LEN * 2}s)
      |> filter(fn: (r) => r._measurement == "machine_features")
      |> filter(fn: (r) => r.machine_id == "{machine_id}")
      |> last()
    """
    tables = influx.query_api().query(query, org=INFLUX_ORG)
    if not tables:
        return None

    row: Dict[str, float] = {}
    for table in tables:
        for record in table.records:
            row[record.get_field()] = record.get_value()

    if not row:
        return None

    # Build feature vector in a consistent order
    vec = []
    for sensor in SENSOR_FIELDS:
        for suffix in FEATURE_SUFFIXES:
            key = f"{sensor}__{suffix}"
            vec.append(float(row.get(key, 0.0)))

    return np.array(vec, dtype=np.float32).reshape(1, -1)


def run_inference(
    feat_vec: np.ndarray,
    scaler: Any,
    iforest: Any,
    ort_session: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"anomaly_score": 0.0, "is_anomaly": False, "rul_hours": 48.0}

    x = scaler.transform(feat_vec) if scaler else feat_vec

    if iforest is not None:
        score = iforest.decision_function(x)[0]      # higher = more normal
        is_anomaly = iforest.predict(x)[0] == -1
        result["anomaly_score"] = float(score)
        result["is_anomaly"] = bool(is_anomaly)

    if ort_session is not None:
        # LSTM expects (batch, window, features) — replicate feature window
        seq = np.tile(x, (WINDOW_LEN, 1))[np.newaxis].astype(np.float32)  # (1, 60, 21)
        t0 = time.perf_counter()
        rul_out = ort_session.run(None, {"features": seq})[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        result["rul_hours"] = float(max(0.0, rul_out[0]))
        result["latency_ms"] = round(latency_ms, 2)
        log.debug("Inference latency: %.2f ms", latency_ms)

    return result


def write_health(write_api: Any, machine_id: str, result: Dict[str, Any]) -> None:
    point = (
        Point("machine_health")
        .tag("machine_id", machine_id)
        .time(datetime.now(timezone.utc), WritePrecision.NANOSECONDS)
        .field("anomaly_score", result["anomaly_score"])
        .field("is_anomaly", int(result["is_anomaly"]))
        .field("rul_hours", result["rul_hours"])
    )
    if "latency_ms" in result:
        point = point.field("inference_latency_ms", result["latency_ms"])
    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


def publish_alert(mqttc: mqtt.Client, machine_id: str, result: Dict[str, Any]) -> None:
    if result["is_anomaly"] or result["rul_hours"] < 48.0:
        alert = {
            "machine_id": machine_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "anomaly_score": result["anomaly_score"],
            "is_anomaly": result["is_anomaly"],
            "rul_hours": result["rul_hours"],
            "severity": "warning" if result["rul_hours"] > 12 else "critical",
        }
        topic = f"{TOPIC_PREFIX}/{machine_id}/alert"
        mqttc.publish(topic, json.dumps(alert), qos=1)
        log.warning(
            "ALERT [%s] anomaly=%s RUL=%.1f h",
            machine_id,
            result["is_anomaly"],
            result["rul_hours"],
        )


def main() -> None:
    scaler, iforest, ort_session = load_models()

    influx = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    for attempt in range(30):
        try:
            influx.ping()
            break
        except Exception as exc:
            log.warning("InfluxDB not ready (attempt %d): %s", attempt + 1, exc)
            time.sleep(3)
    write_api = influx.write_api(write_options=SYNCHRONOUS)

    mqttc = mqtt.Client(client_id="pdm-inference")
    for attempt in range(30):
        try:
            mqttc.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except OSError as exc:
            log.warning("MQTT not ready (attempt %d): %s", attempt + 1, exc)
            time.sleep(2)
    mqttc.loop_start()

    log.info("Inference service ready — running every %d s", INFER_INTERVAL_S)
    while True:
        for mid in MACHINE_IDS:
            try:
                feat_vec = query_latest_features(influx, mid)
                if feat_vec is None:
                    log.debug("No features yet for %s", mid)
                    continue
                result = run_inference(feat_vec, scaler, iforest, ort_session)
                write_health(write_api, mid, result)
                publish_alert(mqttc, mid, result)
                log.info(
                    "[%s] anomaly=%-5s RUL=%.1f h latency=%s ms",
                    mid,
                    result["is_anomaly"],
                    result["rul_hours"],
                    result.get("latency_ms", "n/a"),
                )
            except Exception as exc:
                log.error("Error processing %s: %s", mid, exc)

        time.sleep(INFER_INTERVAL_S)


if __name__ == "__main__":
    main()
