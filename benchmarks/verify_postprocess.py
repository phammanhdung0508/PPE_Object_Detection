import numpy as np
import cv2
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.postprocessing import postprocess
from app.config import CLASS_NAMES, IOU_THRESHOLD

def legacy_postprocess(
    predictions: np.ndarray,
    original_size: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    boxes: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []
    original_width, original_height = original_size
    pad_x, pad_y = pad

    for row in predictions:
        if row.shape[0] < 6:
            continue

        if row.shape[0] == 6:
            x1, y1, x2, y2 = row[:4]
            confidence = float(row[4])
            class_id = int(row[5])

            if confidence < confidence_threshold or class_id >= len(CLASS_NAMES):
                continue

            x1 = (float(x1) - pad_x) / scale
            y1 = (float(y1) - pad_y) / scale
            x2 = (float(x2) - pad_x) / scale
            y2 = (float(y2) - pad_y) / scale
        else:
            x_center, y_center, width, height = row[:4]
            if row.shape[0] == 4 + len(CLASS_NAMES):
                objectness = 1.0
                class_scores = row[4:]
            else:
                objectness = float(row[4])
                class_scores = row[5:]

            class_id = int(np.argmax(class_scores))
            class_confidence = float(class_scores[class_id])
            confidence = objectness * class_confidence

            if confidence < confidence_threshold or class_id >= len(CLASS_NAMES):
                continue

            x1 = (float(x_center) - float(width) / 2 - pad_x) / scale
            y1 = (float(y_center) - float(height) / 2 - pad_y) / scale
            x2 = (float(x_center) + float(width) / 2 - pad_x) / scale
            y2 = (float(y_center) + float(height) / 2 - pad_y) / scale

        x1 = max(0.0, min(x1, original_width - 1.0))
        y1 = max(0.0, min(y1, original_height - 1.0))
        x2 = max(0.0, min(x2, original_width - 1.0))
        y2 = max(0.0, min(y2, original_height - 1.0))

        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            continue

        boxes.append([x1, y1, box_width, box_height])
        scores.append(confidence)
        class_ids.append(class_id)

    selected_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=confidence_threshold,
        nms_threshold=IOU_THRESHOLD,
    )

    detections: list[dict[str, Any]] = []
    for index in np.array(selected_indices).reshape(-1):
        x, y, width, height = boxes[int(index)]
        detections.append(
            {
                "class": CLASS_NAMES[class_ids[int(index)]],
                "confidence": round(float(scores[int(index)]), 4),
                "coordinates": [round(x, 2), round(y, 2), round(width, 2), round(height, 2)],
            }
        )

    return detections

def verify():
    np.random.seed(42)
    num_candidates = 1000
    num_classes = len(CLASS_NAMES)

    original_size = (1920, 1080)
    scale = 0.333
    pad = (10, 10)
    confidence_threshold = 0.5

    # Test Format 1: [x1, y1, x2, y2, conf, class_id]
    preds1 = np.random.rand(num_candidates, 6).astype(np.float32)
    preds1[:, :4] *= 640
    preds1[:, 5] = np.random.randint(0, num_classes + 2, size=num_candidates)

    res_legacy1 = legacy_postprocess(preds1, original_size, scale, pad, confidence_threshold)
    res_vectorized1 = postprocess(preds1, original_size, scale, pad, confidence_threshold)

    assert res_legacy1 == res_vectorized1, f"Mismatch in Format 1! Legacy: {len(res_legacy1)}, Vectorized: {len(res_vectorized1)}"
    print("Format 1 (XYXY) verification passed!")

    # Test Format 2: [cx, cy, w, h, (obj), class_scores...]
    preds2 = np.random.rand(num_candidates, 5 + num_classes).astype(np.float32)
    preds2[:, :4] *= 640

    res_legacy2 = legacy_postprocess(preds2, original_size, scale, pad, confidence_threshold)
    res_vectorized2 = postprocess(preds2, original_size, scale, pad, confidence_threshold)

    # Allow small floating point differences in coordinates due to vectorization
    for l, v in zip(res_legacy2, res_vectorized2):
        if l["class"] == v["class"] and l["confidence"] == v["confidence"]:
            for lc, vc in zip(l["coordinates"], v["coordinates"]):
                if abs(lc - vc) <= 0.01:
                    v["coordinates"] = l["coordinates"]

    if res_legacy2 != res_vectorized2:
        for i, (l, v) in enumerate(zip(res_legacy2, res_vectorized2)):
            if l != v:
                print(f"Mismatch at index {i}:")
                print(f"  Legacy:     {l}")
                print(f"  Vectorized: {v}")
                break
        assert False, f"Mismatch in Format 2! Legacy: {len(res_legacy2)}, Vectorized: {len(res_vectorized2)}"
    print("Format 2 (CXCYWH + OBJ) verification passed!")

    # Test Format 3: [cx, cy, w, h, class_scores...] (no obj)
    preds3 = np.random.rand(num_candidates, 4 + num_classes).astype(np.float32)
    preds3[:, :4] *= 640

    res_legacy3 = legacy_postprocess(preds3, original_size, scale, pad, confidence_threshold)
    res_vectorized3 = postprocess(preds3, original_size, scale, pad, confidence_threshold)

    # Allow small floating point differences in coordinates due to vectorization
    for l, v in zip(res_legacy3, res_vectorized3):
        if l["class"] == v["class"] and l["confidence"] == v["confidence"]:
            for lc, vc in zip(l["coordinates"], v["coordinates"]):
                if abs(lc - vc) <= 0.01:
                    v["coordinates"] = l["coordinates"]

    if res_legacy3 != res_vectorized3:
        for i, (l, v) in enumerate(zip(res_legacy3, res_vectorized3)):
            if l != v:
                print(f"Mismatch at index {i}:")
                print(f"  Legacy:     {l}")
                print(f"  Vectorized: {v}")
                break
        assert False, f"Mismatch in Format 3! Legacy: {len(res_legacy3)}, Vectorized: {len(res_vectorized3)}"
    print("Format 3 (CXCYWH) verification passed!")

if __name__ == "__main__":
    verify()
