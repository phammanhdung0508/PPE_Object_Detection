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
    Vectorized post-processing for YOLO model outputs.
    Eliminates the row-by-row Python loop for significant performance speedup.
    """
    if predictions.size == 0 or predictions.ndim != 2:
        return []

    original_width, original_height = original_size
    pad_x, pad_y = pad
    num_cols = predictions.shape[1]

    if num_cols < 6:
        return []

    if num_cols == 6:
        # Format: [x1, y1, x2, y2, confidence, class_id] (e.g. YOLOv8/v10)
        x1 = (predictions[:, 0] - pad_x) / scale
        y1 = (predictions[:, 1] - pad_y) / scale
        x2 = (predictions[:, 2] - pad_x) / scale
        y2 = (predictions[:, 3] - pad_y) / scale
        scores = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int)
    else:
        # Format: [x_center, y_center, width, height, (objectness), class_scores...]
        xc, yc, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]

        if num_cols == 4 + len(CLASS_NAMES):
            # No objectness (e.g. some YOLOv8 variants)
            objectness = 1.0
            class_scores = predictions[:, 4:]
        else:
            # With objectness (e.g. YOLOv5)
            objectness = predictions[:, 4]
            class_scores = predictions[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        # Use advanced indexing to get scores for selected class_ids
        class_confidences = class_scores[np.arange(len(predictions)), class_ids]
        scores = objectness * class_confidences

        x1 = (xc - w / 2 - pad_x) / scale
        y1 = (yc - h / 2 - pad_y) / scale
        x2 = (xc + w / 2 - pad_x) / scale
        y2 = (yc + h / 2 - pad_y) / scale

    # Filter by confidence and class_id
    mask = (scores >= confidence_threshold) & (class_ids < len(CLASS_NAMES))
    if not np.any(mask):
        return []

    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    scores, class_ids = scores[mask], class_ids[mask]

    # Clip to image boundaries
    x1 = np.clip(x1, 0, original_width - 1.0)
    y1 = np.clip(y1, 0, original_height - 1.0)
    x2 = np.clip(x2, 0, original_width - 1.0)
    y2 = np.clip(y2, 0, original_height - 1.0)

    widths = x2 - x1
    heights = y2 - y1

    # Final filter for valid boxes
    keep = (widths > 0) & (heights > 0)
    if not np.any(keep):
        return []

    x1, y1, widths, heights = x1[keep], y1[keep], widths[keep], heights[keep]
    scores, class_ids = scores[keep], class_ids[keep]

    # Convert to list for cv2.dnn.NMSBoxes
    boxes = np.stack([x1, y1, widths, heights], axis=1).tolist()
    scores_list = scores.astype(float).tolist()

    selected_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores_list,
        score_threshold=confidence_threshold,
        nms_threshold=IOU_THRESHOLD,
    )

    detections: list[dict[str, Any]] = []
    if len(selected_indices) > 0:
        for index in np.array(selected_indices).reshape(-1):
            idx = int(index)
            detections.append(
                {
                    "class": CLASS_NAMES[class_ids[idx]],
                    "confidence": round(float(scores[idx]), 4),
                    "coordinates": [
                        round(float(boxes[idx][0]), 2),
                        round(float(boxes[idx][1]), 2),
                        round(float(boxes[idx][2]), 2),
                        round(float(boxes[idx][3]), 2),
                    ],
                }
            )

    return detections
