from typing import Any

from prometheus_client import Counter, Gauge, Histogram, generate_latest


REQUESTS_TOTAL = Counter(
    "ppe_requests_total",
    "Total prediction requests.",
    ["status"],
)
ERRORS_TOTAL = Counter(
    "ppe_errors_total",
    "Total prediction errors.",
    ["error_type"],
)
INVALID_IMAGES_TOTAL = Counter(
    "ppe_invalid_images_total",
    "Total invalid image requests.",
    ["reason"],
)
DETECTIONS_TOTAL = Counter(
    "ppe_detections_total",
    "Total detected objects by class.",
    ["class_name"],
)
NO_DETECTIONS_TOTAL = Counter(
    "ppe_no_detections_total",
    "Total successful requests with no detections.",
)
BATCH_FALLBACK_TOTAL = Counter(
    "ppe_batch_fallback_total",
    "Total batch requests that fell back to per-image inference.",
)

REQUEST_LATENCY_MS = Histogram(
    "ppe_request_latency_ms",
    "Total prediction request latency in milliseconds.",
    buckets=(10, 25, 50, 75, 100, 150, 250, 500, 1000, 2500, 5000),
)
PREPROCESS_LATENCY_MS = Histogram(
    "ppe_preprocess_latency_ms",
    "Preprocessing latency in milliseconds.",
    buckets=(1, 2.5, 5, 10, 25, 50, 100, 250, 500),
)
INFERENCE_LATENCY_MS = Histogram(
    "ppe_inference_latency_ms",
    "ONNX Runtime inference latency in milliseconds.",
    buckets=(5, 10, 25, 50, 75, 100, 150, 250, 500, 1000),
)
POSTPROCESS_LATENCY_MS = Histogram(
    "ppe_postprocess_latency_ms",
    "Postprocessing latency in milliseconds.",
    buckets=(1, 2.5, 5, 10, 25, 50, 100, 250),
)

IMAGE_WIDTH = Histogram(
    "ppe_image_width_pixels",
    "Input image width in pixels.",
    buckets=(64, 128, 256, 512, 640, 1024, 1280, 1920, 2560, 3840),
)
IMAGE_HEIGHT = Histogram(
    "ppe_image_height_pixels",
    "Input image height in pixels.",
    buckets=(64, 128, 256, 512, 640, 720, 1080, 1440, 2160),
)
IMAGE_BRIGHTNESS = Histogram(
    "ppe_image_brightness",
    "Mean input image brightness.",
    buckets=(0, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255),
)
MODEL_LOADED = Gauge("ppe_model_loaded", "Whether the model is loaded into memory.")
BATCH_SIZE = Histogram(
    "ppe_batch_size",
    "Number of images in batch prediction requests.",
    buckets=(1, 2, 4, 8, 16, 32),
)


def record_success(
    detections: list[dict[str, Any]],
    original_size: tuple[int, int],
    brightness: float,
    timing: dict[str, float],
) -> None:
    REQUESTS_TOTAL.labels(status="success").inc()
    REQUEST_LATENCY_MS.observe(timing["total_ms"])
    PREPROCESS_LATENCY_MS.observe(timing["preprocess_ms"])
    INFERENCE_LATENCY_MS.observe(timing["inference_ms"])
    POSTPROCESS_LATENCY_MS.observe(timing["postprocess_ms"])

    width, height = original_size
    IMAGE_WIDTH.observe(width)
    IMAGE_HEIGHT.observe(height)
    IMAGE_BRIGHTNESS.observe(brightness)

    if not detections:
        NO_DETECTIONS_TOTAL.inc()
    for detection in detections:
        DETECTIONS_TOTAL.labels(class_name=detection["class"]).inc()


def record_batch_success(
    results: list[dict[str, Any]],
    image_stats: list[dict[str, Any]],
    timing: dict[str, float],
    used_fallback: bool,
) -> None:
    REQUESTS_TOTAL.labels(status="success").inc()
    REQUEST_LATENCY_MS.observe(timing["total_ms"])
    PREPROCESS_LATENCY_MS.observe(timing["preprocess_ms"])
    INFERENCE_LATENCY_MS.observe(timing["inference_ms"])
    POSTPROCESS_LATENCY_MS.observe(timing["postprocess_ms"])
    BATCH_SIZE.observe(len(results))

    if used_fallback:
        BATCH_FALLBACK_TOTAL.inc()

    for result, stats in zip(results, image_stats):
        width, height = stats["original_size"]
        IMAGE_WIDTH.observe(width)
        IMAGE_HEIGHT.observe(height)
        IMAGE_BRIGHTNESS.observe(stats["brightness"])

        detections = result["detections"]
        if not detections:
            NO_DETECTIONS_TOTAL.inc()
        for detection in detections:
            DETECTIONS_TOTAL.labels(class_name=detection["class"]).inc()


def record_invalid_image(reason: str) -> None:
    REQUESTS_TOTAL.labels(status="error").inc()
    ERRORS_TOTAL.labels(error_type="invalid_image").inc()
    INVALID_IMAGES_TOTAL.labels(reason=reason).inc()


def record_prediction_error() -> None:
    REQUESTS_TOTAL.labels(status="error").inc()
    ERRORS_TOTAL.labels(error_type="prediction_error").inc()


def set_model_loaded(is_loaded: bool) -> None:
    MODEL_LOADED.set(1 if is_loaded else 0)


def metrics_payload() -> bytes:
    return generate_latest()
