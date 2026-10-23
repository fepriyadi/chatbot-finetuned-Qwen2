FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/root/.cache/huggingface \
    DATA_DIR=/app/data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY legal_rag/ legal_rag/
COPY streamlit_app.py docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["./docker-entrypoint.sh"]
