#!/usr/bin/env bash
# InfluxDB setup script — run once to create org, bucket, and API token
# Usage: INFLUX_PASSWORD=<your-password> bash influx_setup.sh

set -euo pipefail

INFLUX_URL="${INFLUX_URL:-http://localhost:8086}"
ADMIN_TOKEN="${ADMIN_TOKEN:-edge-demo-token}"
ORG="${ORG:-coocah}"
BUCKET="${BUCKET:-factory}"
USERNAME="${USERNAME:-admin}"
# PASSWORD must be supplied via environment variable: export INFLUX_PASSWORD=...
: "${INFLUX_PASSWORD:?INFLUX_PASSWORD env var is required}"

influx setup \
  --host "${INFLUX_URL}" \
  --username "${USERNAME}" \
  --password "${INFLUX_PASSWORD}" \
  --org "${ORG}" \
  --bucket "${BUCKET}" \
  --token "${ADMIN_TOKEN}" \
  --force
