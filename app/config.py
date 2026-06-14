import os
import json
from pathlib import Path


MODEL_PATH = os.getenv("MODEL_PATH", "models/yolo26_ppe.onnx")
MODEL_METADATA_PATH = os.getenv("MODEL_METADATA_PATH", "models/model_metadata.json")
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "640"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MIN_IMAGE_SIZE = int(os.getenv("MIN_IMAGE_SIZE", "32"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
ORT_INTRA_OP_THREADS = int(os.getenv("ORT_INTRA_OP_THREADS", "0"))
ORT_INTER_OP_THREADS = int(os.getenv("ORT_INTER_OP_THREADS", "0"))
ENABLE_MODEL_WARMUP = os.getenv("ENABLE_MODEL_WARMUP", "true").lower() == "true"
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))

DEFAULT_CLASS_NAMES = [
    "helmet",
    "no_helmet",
    "no_vest",
    "person",
    "vest",
]
CLASS_NAMES = [
    name.strip()
    for name in os.getenv("CLASS_NAMES", ",".join(DEFAULT_CLASS_NAMES)).split(",")
    if name.strip()
]


def load_model_metadata() -> dict:
    metadata_path = Path(MODEL_METADATA_PATH)
    if not metadata_path.exists():
        return {
            "model_name": "unknown",
            "model_version": "unknown",
            "model_path": MODEL_PATH,
            "input_size": INPUT_SIZE,
            "class_names": CLASS_NAMES,
        }

    metadata = json.loads(metadata_path.read_text())
    metadata.setdefault("model_path", MODEL_PATH)
    metadata.setdefault("input_size", INPUT_SIZE)
    metadata.setdefault("class_names", CLASS_NAMES)
    return metadata
