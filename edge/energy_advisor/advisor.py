"""
advisor.py
Energy Advisor service.

Reads the latest energy state from InfluxDB (written by the ingestor),
applies a rule engine and a Prophet forecast, then publishes a
"safe_to_run_batch" recommendation to MQTT and writes it to InfluxDB.

Decision logic:
  1. Rule engine: immediate veto/go conditions based on current SoC and solar.
  2. Prophet: short-range solar forecast to detect a coming good window.
  3. Combined: recommend starting batch only when conditions look stable.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from prophet import Prophet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("energy-advisor")

# ── Environment ───────────────────────────────────────────────────────────────
INFLUX_URL = os.environ.get("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "edge-demo-token")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "coocah")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "factory")

MQTT_BROKER = os.environ.get("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
TOPIC_PREFIX = os.environ.get("TOPIC_PREFIX", "coocah/pe-sagamu")

ADVISE_INTERVAL_S = int(os.environ.get("ADVISE_INTERVAL_S", "60"))

# Rule engine thresholds
SOC_MIN_GO = 0.50          # minimum SoC to start a batch
SOLAR_MIN_GO_KW = 5.0      # minimum solar output to start a batch
SOC_VETO = 0.20            # always veto if SoC below this
FORECAST_HORIZON_MIN = 30  # how many minutes ahead to forecast
# ─────────────────────────────────────────────────────────────────────────────


def query_energy_history(influx: InfluxDBClient, minutes: int = 120) -> pd.DataFrame:
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{minutes}m)
      |> filter(fn: (r) => r._measurement == "energy_state")
      |> filter(fn: (r) => r._field == "solar_kw" or r._field == "bess_soc")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    tables = influx.query_api().query(query, org=INFLUX_ORG)
    rows = []
    for table in tables:
        for record in table.records:
            rows.append({
                "ds": record.get_time(),
                "solar_kw": record.values.get("solar_kw", 0.0),
                "bess_soc": record.values.get("bess_soc", 0.5),
            })
    if not rows:
        return pd.DataFrame(columns=["ds", "solar_kw", "bess_soc"])
    df = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    return df


def query_latest_energy(influx: InfluxDBClient) -> Optional[Dict[str, Any]]:
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -5m)
      |> filter(fn: (r) => r._measurement == "energy_state")
      |> last()
    """
    tables = influx.query_api().query(query, org=INFLUX_ORG)
    row: Dict[str, Any] = {}
    for table in tables:
        for record in table.records:
            row[record.get_field()] = record.get_value()
    return row if row else None


def prophet_forecast_solar(history_df: pd.DataFrame, horizon_min: int) -> float:
    """Return the mean forecast solar output over the next horizon_min minutes."""
    if len(history_df) < 10:
        log.debug("Not enough history for Prophet — returning last known solar")
        if len(history_df) > 0:
            return float(history_df["solar_kw"].iloc[-1])
        return 0.0

    df_fit = history_df[["ds", "solar_kw"]].rename(columns={"solar_kw": "y"})
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=True,
        interval_width=0.8,
        changepoint_prior_scale=0.1,
    )
    m.fit(df_fit)

    future = m.make_future_dataframe(periods=horizon_min, freq="min")
    forecast = m.predict(future)
    # Take the average forecast over the horizon window
    tail = forecast.tail(horizon_min)
    mean_forecast = float(tail["yhat"].clip(lower=0).mean())
    return mean_forecast


def make_decision(
    current: Dict[str, Any],
    forecast_solar_kw: float,
) -> Tuple[bool, str]:
    """
    Returns (safe_to_run_batch, reason_string).
    """
    soc = float(current.get("bess_soc", 0.0))
    solar_kw = float(current.get("solar_kw", 0.0))

    # Hard veto conditions
    if soc < SOC_VETO:
        return False, f"VETO: SoC too low ({soc:.1%} < {SOC_VETO:.1%})"

    # Green: current state is good AND forecast is good
    current_ok = soc >= SOC_MIN_GO and solar_kw >= SOLAR_MIN_GO_KW
    forecast_ok = forecast_solar_kw >= SOLAR_MIN_GO_KW * 0.8   # allow 20% drop

    if current_ok and forecast_ok:
        return True, (
            f"GO: SoC={soc:.1%} solar={solar_kw:.1f}kW "
            f"forecast={forecast_solar_kw:.1f}kW"
        )

    if current_ok and not forecast_ok:
        return False, (
            f"WAIT: solar dropping soon — forecast={forecast_solar_kw:.1f}kW"
        )

    return False, (
        f"WAIT: SoC={soc:.1%} (need {SOC_MIN_GO:.0%}) "
        f"solar={solar_kw:.1f}kW (need {SOLAR_MIN_GO_KW:.0f}kW)"
    )


def main() -> None:
    influx = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    for attempt in range(30):
        try:
            influx.ping()
            break
        except Exception as exc:
            log.warning("InfluxDB not ready (attempt %d): %s", attempt + 1, exc)
            time.sleep(3)
    write_api = influx.write_api(write_options=SYNCHRONOUS)

    mqttc = mqtt.Client(client_id="energy-advisor")
    for attempt in range(30):
        try:
            mqttc.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except OSError as exc:
            log.warning("MQTT not ready (attempt %d): %s", attempt + 1, exc)
            time.sleep(2)
    mqttc.loop_start()

    log.info("Energy Advisor ready — advising every %d s", ADVISE_INTERVAL_S)

    while True:
        try:
            current = query_latest_energy(influx)
            if current is None:
                log.debug("No energy state yet — waiting …")
                time.sleep(ADVISE_INTERVAL_S)
                continue

            history = query_energy_history(influx, minutes=120)
            forecast_solar = prophet_forecast_solar(history, FORECAST_HORIZON_MIN)
            safe, reason = make_decision(current, forecast_solar)

            log.info("Decision: safe=%s | %s", safe, reason)

            # Write to InfluxDB
            point = (
                Point("energy_advice")
                .time(datetime.now(timezone.utc), WritePrecision.NANOSECONDS)
                .field("safe_to_run_batch", int(safe))
                .field("forecast_solar_kw", forecast_solar)
                .field("reason", reason)
            )
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

            # Publish to MQTT
            payload = json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "safe_to_run_batch": safe,
                "forecast_solar_kw": round(forecast_solar, 2),
                "reason": reason,
            })
            mqttc.publish(f"{TOPIC_PREFIX}/energy/advice", payload, qos=1)

        except Exception as exc:
            log.error("Advisor error: %s", exc)

        time.sleep(ADVISE_INTERVAL_S)


if __name__ == "__main__":
    main()
