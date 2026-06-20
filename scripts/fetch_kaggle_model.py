import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_KERNEL = "dungsunf/ppe-yolo26-training"
DEFAULT_OUTPUT_DIR = Path("models/kaggle/output")
DEFAULT_MODEL_NAME = "yolo26_ppe.onnx"
DEFAULT_FP16_MODEL_NAME = "yolo26_ppe_fp16.onnx"


def kaggle_executable() -> str:
    candidate = Path(sys.executable).with_name("kaggle")
    if candidate.exists():
        return str(candidate)
    return "kaggle"


def run_kaggle_output(kernel: str, output_dir: Path, model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            kaggle_executable(),
            "kernels",
            "output",
            kernel,
            "-p",
            str(output_dir),
            "--file-pattern",
            f".*{model_name}$",
        ]
    )


def find_exported_model(output_dir: Path, model_name: str) -> Path:
    matches = sorted(output_dir.rglob(model_name))
    if not matches:
        raise FileNotFoundError(f"Could not find {model_name} under {output_dir}")
    return matches[0]


def publish_model(source: Path, release: str, model_name: str) -> None:
    serving_path = Path("models") / DEFAULT_MODEL_NAME
    release_dir = Path("models/releases") / release
    release_path = release_dir / model_name

    release_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, release_path)
    if model_name == DEFAULT_MODEL_NAME:
        shutil.copy2(source, serving_path)

    print(f"Downloaded model: {source}")
    if model_name == DEFAULT_MODEL_NAME:
        print(f"Published serving model: {serving_path}")
    print(f"Published release model: {release_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle kernel output and publish ONNX model.")
    parser.add_argument("--kernel", default=DEFAULT_KERNEL, help="Kaggle kernel slug")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Local output directory")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Expected ONNX model file name")
    parser.add_argument("--release", default="v1", help="Model release directory name")
    parser.add_argument(
        "--include-fp16",
        action="store_true",
        help=f"Also download and publish {DEFAULT_FP16_MODEL_NAME} to the release directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_kaggle_output(args.kernel, output_dir, args.model_name)
    model_path = find_exported_model(output_dir, args.model_name)
    publish_model(model_path, args.release, args.model_name)

    if args.include_fp16:
        run_kaggle_output(args.kernel, output_dir, DEFAULT_FP16_MODEL_NAME)
        fp16_model_path = find_exported_model(output_dir, DEFAULT_FP16_MODEL_NAME)
        publish_model(fp16_model_path, args.release, DEFAULT_FP16_MODEL_NAME)


if __name__ == "__main__":
    main()
