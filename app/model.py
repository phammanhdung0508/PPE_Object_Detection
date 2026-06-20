from pathlib import Path
import numpy as np

from app.onnxruntime_dlls import add_nvidia_dll_directories

add_nvidia_dll_directories()
import onnxruntime as ort

from app.config import (
    INPUT_SIZE,
    MODEL_PATH,
    ORT_INTER_OP_THREADS,
    ORT_INTRA_OP_THREADS,
)
from app.logging_config import get_logger

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(directory="")


logger = get_logger()


class YoloOnnxModel:
    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.input_dtype: np.dtype = np.float32
        self.output_names: list[str] | None = None

    def load(self) -> None:
        model_file = Path(MODEL_PATH)
        if not model_file.exists():
            raise RuntimeError(f"ONNX model file not found: {model_file}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available_providers = ort.get_available_providers()
        selected_providers = [p for p in providers if p in available_providers]

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.intra_op_num_threads = ORT_INTRA_OP_THREADS
        session_options.inter_op_num_threads = ORT_INTER_OP_THREADS

        self.session = ort.InferenceSession(
            str(model_file), sess_options=session_options, providers=selected_providers
        )
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_dtype = (
            np.float16 if input_info.type == "tensor(float16)" else np.float32
        )
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(
            "Model loaded path=%s providers=%s input=%s input_dtype=%s outputs=%s",
            model_file,
            self.session.get_providers(),
            self.input_name,
            self.input_dtype,
            self.output_names,
        )

    def is_loaded(self) -> bool:
        return self.session is not None

    def warmup(self) -> None:
        dummy_tensor = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=self.input_dtype)
        self.predict(dummy_tensor)
        logger.info("Model warmup completed input_shape=%s", dummy_tensor.shape)

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        return self.predict_batch(input_tensor)[0]

    def predict_batch(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        if self.session is None or self.input_name is None or self.output_names is None:
            raise RuntimeError("Model is not loaded")

        if input_tensor.dtype != self.input_dtype:
            input_tensor = input_tensor.astype(self.input_dtype, copy=False)

        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        predictions = outputs[0]

        if predictions.ndim == 2:
            return [self._normalize_prediction(predictions)]

        if predictions.ndim == 3:
            batch_size = input_tensor.shape[0]
            if predictions.shape[0] == batch_size:
                return [
                    self._normalize_prediction(predictions[index])
                    for index in range(batch_size)
                ]
            if batch_size == 1 and predictions.shape[0] == 1:
                return [self._normalize_prediction(predictions[0])]
            if predictions.shape[-1] == batch_size:
                return [
                    self._normalize_prediction(predictions[..., index])
                    for index in range(batch_size)
                ]

        raise RuntimeError(f"Unsupported model output shape: {predictions.shape}")

    @staticmethod
    def _normalize_prediction(predictions: np.ndarray) -> np.ndarray:
        if predictions.ndim != 2:
            raise RuntimeError(f"Unsupported prediction shape: {predictions.shape}")
        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] in {
            6,
            7,
            84,
            85,
        }:
            predictions = predictions.T
        return predictions


model = YoloOnnxModel()
