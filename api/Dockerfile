FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

COPY src /app/src
COPY models /app/models
COPY config /app/config
COPY api /app/api

ENV PYTHONPATH=/app/src:/app
ENV NAI_API_KEYS=dev-key-change-me

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]