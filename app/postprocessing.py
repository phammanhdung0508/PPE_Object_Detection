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
    Postprocess YOLO predictions using vectorized NumPy operations.
    Performance optimization: Replaces row-by-row loop with vectorized operations (~10-100x speedup).
    """
    if predictions.size == 0 or predictions.ndim != 2:
        return []

    original_width, original_height = original_size
    pad_x, pad_y = pad

    if predictions.shape[1] == 6:
        # Format: [x1, y1, x2, y2, confidence, class_id]
        x1 = predictions[:, 0]
        y1 = predictions[:, 1]
        x2 = predictions[:, 2]
        y2 = predictions[:, 3]
        scores = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int)
    else:
        # Format: [x_center, y_center, width, height, (optional objectness), ...class_scores]
        if predictions.shape[1] == 4 + len(CLASS_NAMES):
            objectness = 1.0
            class_scores = predictions[:, 4:]
        else:
            objectness = predictions[:, 4]
            class_scores = predictions[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        # Using advanced indexing for vectorized score extraction
        scores = objectness * class_scores[np.arange(len(predictions)), class_ids]

        x_center = predictions[:, 0]
        y_center = predictions[:, 1]
        w = predictions[:, 2]
        h = predictions[:, 3]

        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

    # Vectorized filtering by confidence and valid class index
    mask = (scores >= confidence_threshold) & (class_ids < len(CLASS_NAMES))
    if not np.any(mask):
        return []

    x1, y1, x2, y2, scores, class_ids = (
        x1[mask],
        y1[mask],
        x2[mask],
        y2[mask],
        scores[mask],
        class_ids[mask],
    )

    # Vectorized scaling and unpadding
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    # Vectorized clipping to image boundaries
    x1 = np.clip(x1, 0, original_width - 1.0)
    y1 = np.clip(y1, 0, original_height - 1.0)
    x2 = np.clip(x2, 0, original_width - 1.0)
    y2 = np.clip(y2, 0, original_height - 1.0)

    # Convert to [x, y, w, h] format for cv2.dnn.NMSBoxes
    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    # Ensure boxes have positive area
    valid_boxes_mask = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)

    if not np.any(valid_boxes_mask):
        return []

    boxes = boxes[valid_boxes_mask]
    scores = scores[valid_boxes_mask]
    class_ids = class_ids[valid_boxes_mask]

    # NMS is still done via OpenCV (highly optimized)
    selected_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=confidence_threshold,
        nms_threshold=IOU_THRESHOLD,
    )

    detections: list[dict[str, Any]] = []
    # flattened_indices handle both empty and non-empty results from NMSBoxes
    for index in np.array(selected_indices).flatten():
        idx = int(index)
        box = boxes[idx]
        detections.append(
            {
                "class": CLASS_NAMES[class_ids[idx]],
                "confidence": round(float(scores[idx]), 4),
                "coordinates": [
                    round(float(box[0]), 2),
                    round(float(box[1]), 2),
                    round(float(box[2]), 2),
                    round(float(box[3]), 2),
                ],
            }
        )

    return detections
