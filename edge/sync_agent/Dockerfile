FROM python:3.11-slim

WORKDIR /app

COPY edge/sync_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY edge/sync_agent/ .

CMD ["python", "sync_agent.py"]
