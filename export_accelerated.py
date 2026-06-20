import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO26 weights to accelerated inference formats."
    )
    parser.add_argument(
        "--weights",
        default="models/kaggle/construction_output/runs/construction_safety_yolo26_finetune/weights/best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--output-dir", default="models/accelerated")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx-fp32", "onnx-fp16"],
        choices=["onnx-fp32", "onnx-fp16", "engine", "openvino"],
    )
    return parser.parse_args()


def export_onnx(model, args: argparse.Namespace, half: bool, output_name: str) -> Path:
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=True,
            dynamic=True,
            half=half,
            device=0 if half else None,
        )
    )
    output_path = Path(args.output_dir) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output_path)
    return output_path


def export_other(model, args: argparse.Namespace, export_format: str) -> Path:
    exported = Path(model.export(format=export_format, imgsz=args.imgsz))
    output_path = Path(args.output_dir) / exported.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported.is_dir():
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(exported, output_path)
    else:
        shutil.copy2(exported, output_path)
    return output_path


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    model = YOLO(args.weights)
    outputs: list[Path] = []

    if "onnx-fp32" in args.formats:
        outputs.append(
            export_onnx(model, args, half=False, output_name="yolo26_ppe.onnx")
        )
    if "onnx-fp16" in args.formats:
        outputs.append(
            export_onnx(model, args, half=True, output_name="yolo26_ppe_fp16.onnx")
        )
    if "engine" in args.formats:
        outputs.append(export_other(model, args, "engine"))
    if "openvino" in args.formats:
        outputs.append(export_other(model, args, "openvino"))

    for output in outputs:
        print(f"exported: {output}")


if __name__ == "__main__":
    main()
