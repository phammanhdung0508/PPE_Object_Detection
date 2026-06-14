import time
from typing import Any

from app.model import YoloOnnxModel
from app.postprocessing import postprocess
from app.preprocessing import preprocess_image


class ObjectDetector:
    def __init__(self) -> None:
        self.model = YoloOnnxModel()

    def load(self) -> None:
        self.model.load()
        self.model.warmup()

    def detect(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.25,
    ) -> tuple[list[dict[str, Any]], float]:
        started_at = time.perf_counter()
        input_tensor, original_size, scale, pad, _ = preprocess_image(image_bytes)
        predictions = self.model.predict(input_tensor)
        detections = postprocess(
            predictions,
            original_size,
            scale,
            pad,
            confidence_threshold,
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return detections, latency_ms
