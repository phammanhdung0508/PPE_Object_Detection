# Deployment Checklist

## Model Artifact

- Confirm `models/yolo26_ppe.onnx` exists.
- Run `scripts/check_onnx.py` successfully.
- Confirm model metadata in `models/model_metadata.json` matches the ONNX model.
- Confirm class order is correct for the 5-class construction-safety model.

## API Validation

- Start API locally.
- Check `/health` returns `model_loaded: true`.
- Send at least one valid image to `/predict`.
- Send one corrupted image and confirm HTTP 400.
- Confirm predictions are logged to `logs/predictions.log`.
- Confirm `/metrics` returns Prometheus text metrics.

## Performance

- Run Locust baseline with 1 user.
- Run Locust with expected camera/user concurrency.
- Confirm p95 latency is acceptable.
- Confirm error rate is 0% under normal test load.

## Docker

- Build image successfully.
- Run with `docker compose up --build`.
- Confirm API, Prometheus, and Grafana start correctly.
- Confirm Prometheus target `ppe-api` is UP.
- Confirm Grafana dashboard receives data.

## Operations

- Decide model rollback path.
- Backup model artifact and metadata.
- Rotate or ship `logs/predictions.log`.
- Document server IP, ports, and restart procedure.
- Monitor disk usage for logs and Prometheus data.

## Security

- Restrict public access to `/predict` if deployed outside local network.
- Set a non-default Grafana admin password.
- Limit upload size using `MAX_UPLOAD_BYTES`.
- Do not commit Kaggle credentials or private model weights.
