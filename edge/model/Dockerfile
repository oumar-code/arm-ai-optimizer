FROM python:3.11-slim

WORKDIR /app

# Build dependencies for torch/onnx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY edge/model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY edge/model/ .

# Default command: run inference (override for training/export)
CMD ["python", "inference.py"]
