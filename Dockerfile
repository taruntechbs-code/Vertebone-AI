FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY inference.py .
COPY openenv.yaml .
COPY Dataset/ Dataset/
COPY server/ server/

EXPOSE 7860

RUN test -d Dataset || (echo "Dataset missing" && exit 1)

ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
ENV HF_TOKEN=""
ENV DATASET_DIR="Dataset"
ENV WORKERS=2
ENV MAX_CONCURRENT_ENVS=100
ENV PORT=7860
ENV HOST=0.0.0.0

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
