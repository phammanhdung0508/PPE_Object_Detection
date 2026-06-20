import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.model import YoloOnnxModel


def print_nvidia_smi(label: str) -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    print(f"{label}_nvidia_smi={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated ONNX inference for GPU checks.")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--sample-smi", action="store_true")
    args = parser.parse_args()

    if args.sample_smi:
        print_nvidia_smi("before_load")

    model = YoloOnnxModel()
    model.load()
    if model.session is None:
        raise RuntimeError("Model session was not created")

    print(f"providers={model.session.get_providers()}")
    tensor = np.zeros((1, 3, 640, 640), dtype=model.input_dtype)
    model.predict(tensor)
    if args.sample_smi:
        print_nvidia_smi("after_warmup")

    started_at = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started_at < args.seconds:
        model.predict(tensor)
        iterations += 1
        if args.sample_smi and iterations % 20 == 0:
            print_nvidia_smi(f"iter_{iterations}")

    elapsed = time.perf_counter() - started_at
    print(f"iterations={iterations} seconds={elapsed:.2f} ips={iterations / elapsed:.2f}")
    if args.sample_smi:
        print_nvidia_smi("after_loop")


if __name__ == "__main__":
    main()
