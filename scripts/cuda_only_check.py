import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import INPUT_SIZE, MODEL_PATH
from app.onnxruntime_dlls import add_nvidia_dll_directories

add_nvidia_dll_directories()
import onnxruntime as ort

ort.preload_dlls(directory="")


def smi(label: str) -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,pstate,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"{label}: {result.stdout.strip()}")


def main() -> None:
    model_path = Path(MODEL_PATH)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    smi("before")
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CUDAExecutionProvider"],
    )
    session.disable_fallback()

    input_info = session.get_inputs()[0]
    input_name = input_info.name
    output_names = [output.name for output in session.get_outputs()]
    input_dtype = np.float16 if input_info.type == "tensor(float16)" else np.float32
    tensor = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=input_dtype)

    print(f"model={model_path}")
    print(f"providers={session.get_providers()}")
    print(f"input={input_name} dtype={input_dtype}")

    session.run(output_names, {input_name: tensor})
    smi("after_warmup")

    started_at = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started_at < 10:
        session.run(output_names, {input_name: tensor})
        iterations += 1
        if iterations % 20 == 0:
            smi(f"iter_{iterations}")

    elapsed = time.perf_counter() - started_at
    print(f"iterations={iterations} seconds={elapsed:.2f} ips={iterations / elapsed:.2f}")
    smi("after")


if __name__ == "__main__":
    main()
