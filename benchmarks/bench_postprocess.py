import time
import numpy as np
import cv2
from app.postprocessing import postprocess
from app.config import CLASS_NAMES

def benchmark_postprocess():
    num_classes = len(CLASS_NAMES)
    # Simulate YOLOv8 output: 8400 predictions, each with 4 box coords + num_classes scores
    num_predictions = 8400
    predictions = np.random.rand(num_predictions, 4 + num_classes).astype(np.float32)
    # Make some predictions have high confidence to avoid empty results
    predictions[:10, 4:] = 0.9

    original_size = (1920, 1080)
    scale = 0.33
    pad = (10, 10)
    confidence_threshold = 0.25

    # Warmup
    for _ in range(10):
        postprocess(predictions, original_size, scale, pad, confidence_threshold)

    start_time = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        postprocess(predictions, original_size, scale, pad, confidence_threshold)
    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) / iterations * 1000
    print(f"Average postprocess time: {avg_time_ms:.2f} ms")

if __name__ == "__main__":
    benchmark_postprocess()
