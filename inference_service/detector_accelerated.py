import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from app.model import YoloOnnxModel
from app.postprocessing import postprocess
from app.preprocessing import preprocess_image


class AcceleratedObjectDetector:
    def __init__(self) -> None:
        self.model = YoloOnnxModel()
        self.model_path = self._select_model_path()
        self.model_precision = "FP16" if self.model_path.name.endswith("fp16.onnx") else "FP32"

    def _select_model_path(self) -> Path:
        explicit_path = os.getenv("ACCELERATED_MODEL_PATH")
        if explicit_path:
            return Path(explicit_path)

        fp16_path = Path(os.getenv("FP16_MODEL_PATH", "models/yolo26_ppe_fp16.onnx"))
        fp32_path = Path(os.getenv("MODEL_PATH", "models/yolo26_ppe.onnx"))
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers and fp16_path.exists():
            return fp16_path
        return fp32_path

    def load(self) -> None:
        os.environ["MODEL_PATH"] = str(self.model_path)
        self.model.load()
        self.model.warmup()

    @property
    def execution_provider(self) -> str:
        if self.model.session is None:
            return "not_loaded"
        providers = self.model.session.get_providers()
        return providers[0] if providers else "unknown"

    def detect_batch(
        self,
        image_payloads: list[bytes],
        confidence_threshold: float = 0.25,
    ) -> tuple[list[list[dict[str, Any]]], float]:
        if not image_payloads:
            raise ValueError("At least one frame is required")

        started_at = time.perf_counter()
        tensors: list[np.ndarray] = []
        image_stats: list[tuple[tuple[int, int], float, tuple[int, int]]] = []

        for image_bytes in image_payloads:
            input_tensor, original_size, scale, pad, _ = preprocess_image(image_bytes)
            tensors.append(input_tensor)
            image_stats.append((original_size, scale, pad))

        batch_tensor = np.concatenate(tensors, axis=0)
        try:
            batch_predictions = self.model.predict_batch(batch_tensor)
            if len(batch_predictions) != len(image_payloads):
                raise RuntimeError("Batch prediction count mismatch")
        except Exception:
            batch_predictions = [self.model.predict(tensor) for tensor in tensors]

        results: list[list[dict[str, Any]]] = []
        for predictions, (original_size, scale, pad) in zip(batch_predictions, image_stats):
            results.append(postprocess(predictions, original_size, scale, pad, confidence_threshold))

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return results, latency_ms
