import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.onnxruntime_dlls import add_nvidia_dll_directories

add_nvidia_dll_directories()
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(directory="")

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an ONNX model for ONNX Runtime.")
    parser.add_argument("--model", default="models/yolo26_ppe.onnx", help="Path to ONNX model")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected_providers = [p for p in providers if p in ort.get_available_providers()]

    session = ort.InferenceSession(
        str(model_path), sess_options=session_options, providers=selected_providers
    )

    print(f"model: {model_path}")
    print(f"available_providers: {ort.get_available_providers()}")
    print(f"session_providers: {session.get_providers()}")
    print("inputs:")
    for item in session.get_inputs():
        print(f"- name={item.name} shape={item.shape} type={item.type}")
    print("outputs:")
    for item in session.get_outputs():
        print(f"- name={item.name} shape={item.shape} type={item.type}")


if __name__ == "__main__":
    main()
