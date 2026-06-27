import time
import numpy as np
import cv2
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.postprocessing import postprocess
from app.config import CLASS_NAMES

def benchmark_postprocess(n_boxes=8400, iterations=100):
    # Simulate YOLO output: [n_boxes, 5 + len(CLASS_NAMES)]
    # Or [n_boxes, 6]

    # Test case 1: CXCYWH format (e.g. YOLOv5/v8 without NMS exported)
    n_classes = len(CLASS_NAMES)
    predictions_cxcywh = np.random.rand(n_boxes, 5 + n_classes).astype(np.float32) * 0.1
    # Set objectness and class scores for some boxes to ensure they pass the threshold
    predictions_cxcywh[:10, 4] = 0.9 # objectness
    predictions_cxcywh[:10, 5:] = 0.9 # class scores

    original_size = (1920, 1080)
    scale = 0.5
    pad = (0, 0)
    confidence_threshold = 0.25

    print(f"Benchmarking with {n_boxes} boxes, {iterations} iterations...")

    # Warmup
    for _ in range(10):
        postprocess(predictions_cxcywh, original_size, scale, pad, confidence_threshold)

    start = time.perf_counter()
    for _ in range(iterations):
        postprocess(predictions_cxcywh, original_size, scale, pad, confidence_threshold)
    end = time.perf_counter()

    avg_time = (end - start) * 1000 / iterations
    print(f"Average Postprocess time (CXCYWH): {avg_time:.2f} ms")

    # Test case 2: XYXY format (e.g. YOLOv8 with NMS exported)
    predictions_xyxy = np.random.rand(n_boxes, 6).astype(np.float32) * 0.1
    predictions_xyxy[:10, 4] = 0.9 # confidence
    predictions_xyxy[:10, 5] = 0 # class_id

    # Warmup
    for _ in range(10):
        postprocess(predictions_xyxy, original_size, scale, pad, confidence_threshold)

    start = time.perf_counter()
    for _ in range(iterations):
        postprocess(predictions_xyxy, original_size, scale, pad, confidence_threshold)
    end = time.perf_counter()

    avg_time_xyxy = (end - start) * 1000 / iterations
    print(f"Average Postprocess time (XYXY): {avg_time_xyxy:.2f} ms")

if __name__ == "__main__":
    benchmark_postprocess()
