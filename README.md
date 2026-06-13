# PPE Object Detection API

FastAPI model-as-a-service for construction-site PPE object detection using a 5-class YOLO26 ONNX model.

The service performs the full request lifecycle in one deployable unit:

```text
Image upload -> preprocess -> ONNX Runtime inference -> NMS/postprocess -> log -> JSON response
```

## Project Structure

```text
app/                         FastAPI API service
models/                      Model metadata and exported ONNX model
models/kaggle/               Kaggle training kernel files
scripts/                     Utility scripts
benchmarks/                  Locust benchmark files
deploy/                      Prometheus/Grafana config
tests/                       Unit tests
Dockerfile                   API container image
docker-compose.yml           API + monitoring stack
requirements.txt             Runtime dependencies
requirements-dev.txt         Test and benchmark dependencies
```

## Train On Kaggle

Training kernel:

```text
models/kaggle/train_yolo26_ppe.py
```

Push to Kaggle:

```bash
kaggle kernels push -p models/kaggle
```

Check status:

```bash
kaggle kernels status dungsunf/construction-safety-yolo26-fine-tune
```

Download output after finish:

```bash
python scripts/fetch_kaggle_model.py \
  --kernel dungsunf/construction-safety-yolo26-fine-tune \
  --release v2 \
  --include-fp16
```

Kaggle GPU note: the P100 runtime may fail with newer PyTorch builds because P100 uses CUDA architecture `sm_60`. If that happens, switch the Kaggle accelerator to T4 or another newer GPU.

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

Start API with monitoring:

```bash
docker compose up --build
```

Services:

```text
API:        http://localhost:8000
Metrics:    http://localhost:8000/metrics
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000
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
