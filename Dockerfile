FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/yolo26_ppe.onnx \
    MODEL_METADATA_PATH=/app/models/model_metadata.json \
    INPUT_SIZE=640

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY inference_service ./inference_service
COPY stream_ingestion ./stream_ingestion
COPY protos ./protos
COPY models/model_metadata.json ./models/model_metadata.json
COPY models/yolo26_ppe.onnx ./models/yolo26_ppe.onnx
COPY models/yolo26_ppe_fp16.onnx ./models/yolo26_ppe_fp16.onnx

FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM base AS inference
EXPOSE 50051
CMD ["python", "-m", "inference_service.server", "--host", "0.0.0.0", "--port", "50051"]

FROM base AS ingestion
CMD ["python", "-m", "stream_ingestion.ingest", "--grpc-target", "inference-service:50051", "--source", "sample_videos/demo.mp4", "--reconnect"]
