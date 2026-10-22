FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY core ./core
COPY services ./services

RUN useradd --create-home voxa && chown -R voxa /app
USER voxa

# Cloud Run injects PORT; one worker keeps all call state in a single event loop.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --ws websockets --no-access-log"]
