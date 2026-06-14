import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Adjust imports to local code
sys.path.append(str(Path(__file__).parent.parent))
from app.model import YoloOnnxModel
from app.preprocessing import preprocess_image
from app.postprocessing import postprocess

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install ultralytics to run parity test")
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--pt-model", type=str, required=True, help="Path to PyTorch model (.pt)")
    parser.add_argument("--onnx-model", type=str, required=True, help="Path to ONNX model")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image {args.image} not found")
        sys.exit(1)

    pt_model = YOLO(args.pt_model)
    print("Running PyTorch model...")
    pt_results = pt_model(args.image, conf=0.25, iou=0.45, verbose=False)
    pt_boxes = []
    if pt_results and len(pt_results) > 0:
        boxes = pt_results[0].boxes
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                pt_boxes.append(xyxy)

    print("Running ONNX model...")
    with open(args.image, "rb") as f:
        image_bytes = f.read()

    input_tensor, original_size, scale, pad, _ = preprocess_image(image_bytes)

    import os
    os.environ["MODEL_PATH"] = args.onnx_model
    onnx_model = YoloOnnxModel()
    onnx_model.load()

    predictions = onnx_model.predict(input_tensor)
    onnx_detections = postprocess(predictions, original_size, scale, pad, 0.25)

    onnx_boxes = []
    for d in onnx_detections:
        x, y, w, h = d["coordinates"]
        onnx_boxes.append([x, y, x + w, y + h])

    print("\n--- Results ---")
    print(f"PyTorch detections: {len(pt_boxes)}")
    print(f"ONNX detections: {len(onnx_boxes)}")

    for i, pt_box in enumerate(pt_boxes):
        print(f"PT Box {i}: {pt_box}")
    for i, onnx_box in enumerate(onnx_boxes):
        print(f"ONNX Box {i}: {onnx_box}")

if __name__ == "__main__":
    main()
