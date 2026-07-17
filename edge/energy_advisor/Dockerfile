FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY edge/energy_advisor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY edge/energy_advisor/ .

CMD ["python", "advisor.py"]
