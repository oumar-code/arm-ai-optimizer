# InfluxDB setup script — run once to create org, bucket, and API token
INFLUX_URL="http://localhost:8086"
ADMIN_TOKEN="edge-demo-token"
ORG="coocah"
BUCKET="factory"
USERNAME="admin"
******

influx setup \
  --host "${INFLUX_URL}" \
  --username "${USERNAME}" \
  --password "${PASSWORD}" \
  --org "${ORG}" \
  --bucket "${BUCKET}" \
  --token "${ADMIN_TOKEN}" \
  --force
