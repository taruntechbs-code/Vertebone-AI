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
COPY server/ server/
COPY Dataset/ Dataset/

RUN test -d Dataset || (echo "Dataset missing" && exit 1)

ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
ENV HF_TOKEN=""
ENV DATASET_DIR="Dataset"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
