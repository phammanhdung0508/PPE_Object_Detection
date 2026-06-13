import argparse
from pathlib import Path

import onnxruntime as ort


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
