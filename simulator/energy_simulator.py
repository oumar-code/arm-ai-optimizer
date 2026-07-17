"""
energy_simulator.py
Simulates solar panel output and Battery Energy Storage System (BESS)
state-of-charge (SoC) for the Sagamu factory site. Publishes energy
state to MQTT every second.

Topic: coocah/pe-sagamu/energy/state
"""

import json
import math
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
log = logging.getLogger("energy-sim")

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config.yaml")

# Compress a full day into SIMULATION_SPEEDUP × real-time
# (60 = 1 sim-minute per real-second → full day in 24 real-minutes)
SIMULATION_SPEEDUP = float(os.environ.get("SIM_SPEEDUP", "60"))


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def solar_output_kw(sim_hour: float, peak_kw: float, efficiency: float, derating: float) -> float:
    """Simplified clear-sky solar curve centred on solar noon (12:00)."""
    sunrise, sunset = 6.0, 18.5   # Sagamu approximate
    if sim_hour < sunrise or sim_hour > sunset:
        return 0.0
    # Bell curve approximation
    angle = math.pi * (sim_hour - sunrise) / (sunset - sunrise)
    raw = peak_kw * math.sin(angle) ** 1.5
    return raw * efficiency * (1.0 - derating)


def make_payload(
    sim_hour: float,
    solar_kw: float,
    bess_soc: float,
    bess_kw: float,
    baseline_kw: float,
    grid_kw: float,
    safe_to_batch: bool,
) -> str:
    return json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sim_hour": round(sim_hour % 24, 3),
        "solar_kw": round(solar_kw, 3),
        "bess_soc": round(bess_soc, 4),
        "bess_kw": round(bess_kw, 3),         # positive = charging, negative = discharging
        "baseline_load_kw": round(baseline_kw, 2),
        "grid_import_kw": round(grid_kw, 3),   # negative = export
        "safe_to_run_batch": safe_to_batch,
    })


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    mqtt_cfg = cfg["mqtt"]
    ecfg = cfg["energy"]
    scfg = ecfg["solar"]
    bcfg = ecfg["bess"]

    client = mqtt.Client(client_id="energy-sim")
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    broker = mqtt_cfg["broker"]
    port = mqtt_cfg["port"]
    log.info("Connecting to MQTT broker %s:%d …", broker, port)

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
    log.info("Connected. Starting energy simulation (speedup=%.0fx) …", SIMULATION_SPEEDUP)

    bess_soc = bcfg["initial_soc"]
    start_real = time.monotonic()
    # Start simulation at 06:00
    sim_start_hour = 6.0
    interval = mqtt_cfg.get("publish_interval_s", 1)
    topic = f"{mqtt_cfg['topic_prefix']}/energy/state"

    while True:
        real_elapsed = time.monotonic() - start_real
        sim_elapsed_hours = (real_elapsed * SIMULATION_SPEEDUP) / 3600.0
        sim_hour = sim_start_hour + sim_elapsed_hours

        # Solar output
        solar_kw = solar_output_kw(
            sim_hour,
            scfg["peak_kw"],
            scfg["panel_efficiency"],
            scfg["dust_derating"],
        )

        baseline_kw = ecfg["baseline_load_kw"]
        net_kw = solar_kw - baseline_kw   # positive → surplus for BESS

        # BESS charge/discharge decision
        dt_hours = (interval * SIMULATION_SPEEDUP) / 3600.0
        max_charge = bcfg["max_charge_rate_kw"]
        max_discharge = bcfg["max_discharge_rate_kw"]
        efficiency = bcfg["round_trip_efficiency"]
        min_soc = bcfg["min_soc"]
        max_soc = bcfg["max_soc"]
        capacity = bcfg["capacity_kwh"]

        if net_kw > 0:
            # Surplus — charge BESS
            charge_kw = min(net_kw, max_charge)
            charge_kw = min(charge_kw, (max_soc - bess_soc) * capacity / dt_hours) if dt_hours > 0 else charge_kw
            bess_kw = max(0.0, charge_kw)
            bess_soc = min(max_soc, bess_soc + bess_kw * efficiency * dt_hours / capacity)
            grid_kw = net_kw - bess_kw  # residual exported
        else:
            # Deficit — discharge BESS
            deficit = abs(net_kw)
            discharge_kw = min(deficit, max_discharge)
            discharge_kw = min(discharge_kw, (bess_soc - min_soc) * capacity / dt_hours) if dt_hours > 0 else discharge_kw
            discharge_kw = max(0.0, discharge_kw)
            bess_kw = -discharge_kw
            bess_soc = max(min_soc, bess_soc - discharge_kw * dt_hours / capacity)
            grid_kw = deficit - discharge_kw  # still needed from grid

        # "Safe to run batch" heuristic: SoC > 50% AND solar > 5 kW
        safe_to_batch = bess_soc >= 0.50 and solar_kw >= 5.0

        payload = make_payload(
            sim_hour, solar_kw, bess_soc, bess_kw, baseline_kw, grid_kw, safe_to_batch
        )
        client.publish(topic, payload, qos=1)

        if int(real_elapsed) % 30 == 0:
            log.info(
                "sim %05.2f h | solar %.1f kW | BESS SoC %.1f%% | safe=%s",
                sim_hour % 24,
                solar_kw,
                bess_soc * 100,
                safe_to_batch,
            )

        time.sleep(interval)


if __name__ == "__main__":
    main()
