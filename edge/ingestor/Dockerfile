FROM python:3.11-slim

WORKDIR /app

COPY edge/ingestor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY edge/ingestor/ .

CMD ["python", "ingestor.py"]
