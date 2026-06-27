from typing import Any

import cv2
import numpy as np

from app.config import CLASS_NAMES, IOU_THRESHOLD


def postprocess(
    predictions: np.ndarray,
    original_size: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """
    Vectorized post-processing for YOLO predictions.
    Supports both [x1, y1, x2, y2, confidence, class_id] and [cx, cy, w, h, obj, class_scores...] formats.
    """
    if predictions.size == 0:
        return []

    original_width, original_height = original_size
    pad_x, pad_y = pad

    # Handle different output formats
    if predictions.shape[1] == 6:
        # Format: [x1, y1, x2, y2, confidence, class_id]
        boxes_raw = predictions[:, :4]
        scores = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int)

        # Filter by confidence and valid class_id
        mask = (scores >= confidence_threshold) & (class_ids < len(CLASS_NAMES))
        if not np.any(mask):
            return []

        boxes_raw = boxes_raw[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Transform to [x, y, w, h] in original image coordinates
        x1 = (boxes_raw[:, 0] - pad_x) / scale
        y1 = (boxes_raw[:, 1] - pad_y) / scale
        x2 = (boxes_raw[:, 2] - pad_x) / scale
        y2 = (boxes_raw[:, 3] - pad_y) / scale
    else:
        # Format: [cx, cy, w, h, obj, class_scores...] or [cx, cy, w, h, class_scores...]
        boxes_raw = predictions[:, :4]
        if predictions.shape[1] == 4 + len(CLASS_NAMES):
            objectness = 1.0
            class_scores = predictions[:, 4:]
        else:
            objectness = predictions[:, 4]
            class_scores = predictions[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        # Using advanced indexing to get class-specific scores
        class_confidences = class_scores[np.arange(len(class_scores)), class_ids]
        scores = objectness * class_confidences

        # Filter by confidence and valid class_id
        mask = (scores >= confidence_threshold) & (class_ids < len(CLASS_NAMES))
        if not np.any(mask):
            return []

        boxes_raw = boxes_raw[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Transform to [x, y, w, h] in original image coordinates
        # cx, cy, w, h -> x1, y1, x2, y2
        cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

    # Clip to image boundaries
    x1 = np.clip(x1, 0, original_width - 1)
    y1 = np.clip(y1, 0, original_height - 1)
    x2 = np.clip(x2, 0, original_width - 1)
    y2 = np.clip(y2, 0, original_height - 1)

    # Convert to [x, y, w, h]
    w_orig = x2 - x1
    h_orig = y2 - y1

    # Filter out empty boxes
    keep = (w_orig > 0) & (h_orig > 0)
    if not np.any(keep):
        return []

    final_boxes = np.stack([x1[keep], y1[keep], w_orig[keep], h_orig[keep]], axis=1)
    final_scores = scores[keep]
    final_class_ids = class_ids[keep]

    selected_indices = cv2.dnn.NMSBoxes(
        bboxes=final_boxes.tolist(),
        scores=final_scores.tolist(),
        score_threshold=float(confidence_threshold),
        nms_threshold=IOU_THRESHOLD,
    )

    detections: list[dict[str, Any]] = []
    for index in np.array(selected_indices).reshape(-1):
        idx = int(index)
        x, y, w, h = final_boxes[idx]
        detections.append(
            {
                "class": CLASS_NAMES[final_class_ids[idx]],
                "confidence": round(float(final_scores[idx]), 4),
                "coordinates": [round(float(x), 2), round(float(y), 2), round(float(w), 2), round(float(h), 2)],
            }
        )

    return detections
