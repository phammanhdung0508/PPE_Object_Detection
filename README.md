# PPE Object Detection System

Distributed construction-site PPE object detection system using a 5-class YOLO26 ONNX model.

Primary lab demo architecture:

```text
Video file -> StreamIngestion -> gRPC ObjectDetectionInference -> YOLO26 ONNX -> bounding boxes
```

The same ingestion service also supports webcam, RTSP, and smartphone IP camera URLs. The FastAPI app is kept as an optional browser/API demo.

The service performs the full request lifecycle in one deployable unit:

```text
Image upload -> preprocess -> ONNX Runtime inference -> NMS/postprocess -> log -> JSON response
```

## Project Structure

```text
protos/                      gRPC protobuf contract and generated modules
inference_service/           gRPC ObjectDetectionInference service
stream_ingestion/            IP camera/video ingestion gRPC client
app/                         Optional FastAPI API/demo service
models/                      Model metadata and exported ONNX model
models/kaggle/               Kaggle training kernel files
data/                        YOLO dataset config and Kaggle dataset metadata
tests/                       Unit and gRPC integration tests
Dockerfile                   Multi-target container image
docker-compose.yml           gRPC services + optional API/monitoring stack
requirements.txt             Runtime dependencies
requirements-dev.txt         Test and benchmark dependencies
```

## gRPC Contract

Contract file:

```text
protos/detector.proto
```

Compile generated Python modules after editing the contract:

```bash
.venv/bin/python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  --grpc_python_out=. \
  protos/detector.proto
```

## Local Environment

The local project environment is created at:

```text
.venv/
```

Activate it:

```bash
conda activate /home/sunf/FSB/DDM501/PPE_Object_Detection/.venv
```

Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-dev.txt
```

For native NVIDIA GPU inference, use the GPU requirements instead of the CPU runtime requirements:

```bash
.venv/bin/python -m pip install -r requirements-gpu.txt
```

The API automatically prefers `CUDAExecutionProvider` when `onnxruntime-gpu` and compatible NVIDIA drivers/CUDA libraries are available.

## Quick Start Video Demo

Place a demo video at:

```text
sample_videos/demo.mp4
```

Terminal 1: start the gRPC model-serving microservice:

```bash
.venv/bin/python -m inference_service.server --host 0.0.0.0 --port 50051
```

It loads:

```text
models/yolo26_ppe.onnx
```

Terminal 2: stream frames from the video to the inference service and log detections:

```bash
.venv/bin/python -m stream_ingestion.ingest \
  --grpc-target localhost:50051 \
  --source sample_videos/demo.mp4 \
  --frame-stride 10 --show
```

Expected log format:

```text
frame=10 detections=['person:0.95', 'helmet:0.91', 'vest:0.63'] inference_ms=663.17 total_ms=671.59
```

If no PPE is detected on a frame, `detections=[]` is still a valid response.

Display annotated video live:

```bash
.venv/bin/python -m stream_ingestion.ingest \
  --grpc-target localhost:50051 \
  --source sample_videos/demo.mp4 \
  --frame-stride 10 \
  --show
```

Press `q` or `Esc` to close the live display window.

Save annotated video:

```bash
.venv/bin/python -m stream_ingestion.ingest \
  --grpc-target localhost:50051 \
  --source sample_videos/demo.mp4 \
  --frame-stride 10 \
  --output outputs/demo_annotated.mp4
```

The video still displays/writes every frame. Inference runs every `--frame-stride` frames and the latest detections are reused on skipped frames.

Lower `--frame-stride` for more frequent detection updates:

```text
--frame-stride 5
```

Increase `--frame-stride` if CPU inference is too slow:

```text
--frame-stride 15
--frame-stride 30
```

Validate the demo video path if ingestion cannot open it:

```bash
.venv/bin/python - <<'PY'
import cv2
cap = cv2.VideoCapture('sample_videos/demo.mp4')
print('opened:', cap.isOpened())
print('frames:', cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('fps:', cap.get(cv2.CAP_PROP_FPS))
print('width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
PY
```

## Other Stream Sources

The same ingestion command also supports webcam, RTSP URL, or smartphone IP camera URL:

```bash
.venv/bin/python -m stream_ingestion.ingest \
  --grpc-target localhost:50051 \
  --source 0 \
  --frame-stride 15 \
  --reconnect
```

Examples:

```bash
.venv/bin/python -m stream_ingestion.ingest --grpc-target localhost:50051 --source sample_videos/demo.mp4
.venv/bin/python -m stream_ingestion.ingest --grpc-target localhost:50051 --source 0 --show
.venv/bin/python -m stream_ingestion.ingest --grpc-target localhost:50051 --source rtsp://user:pass@camera-ip/stream --reconnect
.venv/bin/python -m stream_ingestion.ingest --grpc-target localhost:50051 --source http://phone-ip:8080/video --reconnect
```

The ingestion service downsizes frames, JPEG-encodes them, sends binary payloads over gRPC, and logs returned detections.

## Lab 2 Accelerated Batch Inference

The accelerated gRPC server supports both single-frame `Detect` and batched `BatchDetect` requests. It automatically selects the best available ONNX artifact:

```text
CUDA available + models/yolo26_ppe_fp16.onnx exists -> FP16 ONNX
otherwise -> FP32 ONNX
```

Run accelerated batched inference service:

```bash
.venv/bin/python -m inference_service.server_accelerated \
  --host 0.0.0.0 \
  --port 50051 \
  --max-batch-size 8
```

Simulate multiple video streams with client-side batching:

```bash
.venv/bin/python -m stream_ingestion.ingest_batched \
  --grpc-target localhost:50051 \
  --source sample_videos/demo.mp4 \
  --camera-count 4 \
  --batch-size 4 \
  --frame-stride 10 \
  --max-batches 10
```

Use explicit sources instead of repeated simulation:

```bash
.venv/bin/python -m stream_ingestion.ingest_batched \
  --grpc-target localhost:50051 \
  --sources sample_videos/cam1.mp4 sample_videos/cam2.mp4 sample_videos/cam3.mp4 sample_videos/cam4.mp4 \
  --batch-size 4
```

Export accelerated model artifacts from PyTorch weights:

```bash
.venv/bin/python export_accelerated.py \
  --weights models/kaggle/construction_output/runs/construction_safety_yolo26_finetune/weights/best.pt \
  --formats onnx-fp32 onnx-fp16 \
  --output-dir models/accelerated
```

Optional formats, if hardware/runtime is installed:

```bash
.venv/bin/python export_accelerated.py --formats engine
.venv/bin/python export_accelerated.py --formats openvino
```

Docker CPU accelerated target:

```bash
docker build -f Dockerfile.cpu -t ppe-inference-cpu .
docker run --rm -p 50051:50051 ppe-inference-cpu
```

Docker batched pipeline:

```bash
docker compose --profile batched up --build inference-service stream-ingestion-batched
```

Current benchmark snapshot on local CPU using ONNX Runtime CPU provider:

```text
batch size | precision | provider             | observed behavior
1          | FP32      | CPUExecutionProvider | ~500-700 ms per frame on local CPU
4          | FP32      | CPUExecutionProvider | supported via BatchDetect; throughput depends on CPU
FP16       | FP16      | CUDAExecutionProvider | intended for GPU; slower on CPU
```

Run benchmark-related tests:

```bash
.venv/bin/pytest tests/test_benchmark.py
```

## Run API Locally

Place the trained model at:

```text
models/yolo26_ppe.onnx
```

Current classes:

```text
0: helmet
1: no_helmet
2: no_vest
3: person
4: vest
```

Run:

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Predict:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_images/test.jpg" \
  -F "confidence_threshold=0.25"
```

Predict and return an image with bounding boxes drawn:

```bash
curl -X POST "http://localhost:8000/predict-annotated" \
  -F "file=@sample_images/test.jpg" \
  -F "confidence_threshold=0.25" \
  --output annotated.jpg
```

Batch predict:

```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -F "files=@sample_images/cam1.jpg" \
  -F "files=@sample_images/cam2.jpg" \
  -F "confidence_threshold=0.25"
```

`/predict-batch` attempts true ONNX batch inference first. If the exported model only supports batch size 1, it falls back to per-image inference and returns `used_fallback: true`.

Startup warmup is enabled by default with:

```text
ENABLE_MODEL_WARMUP=true
```

## Colab Demo

The Colab demo setup is in:

```text
colab/
```

It runs the same FastAPI app on Google Colab and exposes `/docs`, `/health`, and `/predict` through a temporary Cloudflare tunnel.

See:

```text
colab/README.md
colab/ppe_fastapi_demo.ipynb
```

## Docker Compose

Create `.env` from the example:

```bash
cp .env.example .env
```

Make sure the demo video exists before starting the ingestion container:

```text
sample_videos/demo.mp4
```

Run only the gRPC video pipeline:

```bash
docker compose up --build inference-service stream-ingestion
```

Start the full local stack:

```bash
docker compose up --build
```

Services:

```text
gRPC inference:  localhost:50051
FastAPI demo:    http://localhost:8000/docs
Metrics:         http://localhost:8000/metrics
Prometheus:      http://localhost:9090
Grafana:         http://localhost:3000
```

Set video source for Docker ingestion in `.env`:

```text
STREAM_SOURCE=sample_videos/demo.mp4
FRAME_STRIDE=15
```

Grafana default login:

```text
admin / admin
```

## Monitoring

The API exposes Prometheus metrics at `/metrics` and writes structured JSON logs to:

```text
logs/predictions.log
```

Key metrics:

```text
ppe_request_latency_ms
ppe_inference_latency_ms
ppe_requests_total
ppe_errors_total
ppe_detections_total
ppe_image_brightness
ppe_model_loaded
```

## Validate ONNX Model

```bash
.venv/bin/python scripts/check_onnx.py --model models/yolo26_ppe.onnx
```

## Benchmark

Run after the API is available and `benchmarks/sample.jpg` exists:

```bash
.venv/bin/locust -f benchmarks/locustfile.py \
  --host http://localhost:8000 \
  --users 5 \
  --spawn-rate 1 \
  --run-time 1m \
  --headless
```

## Tests

```bash
.venv/bin/pytest tests
```
