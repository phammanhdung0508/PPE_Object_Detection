# API Benchmark

This benchmark uses Locust because `/predict` requires multipart image upload.

## Prepare

Add a representative test image:

```text
benchmarks/sample.jpg
```

Start the API first:

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run Headless Benchmark

```bash
.venv/bin/locust -f benchmarks/locustfile.py \
  --host http://localhost:8000 \
  --users 5 \
  --spawn-rate 1 \
  --run-time 1m \
  --headless
```

## Recommended Scenarios

```text
1 user: baseline latency
5 users: small site simulation
10 users: multiple cameras/users
20 users: stress test
```

Primary metrics to watch:

```text
p50 latency
p95 latency
p99 latency
requests/sec
failure rate
```

Target for MVP:

```text
p95 below 150ms if possible
failure rate 0% for valid images
```
