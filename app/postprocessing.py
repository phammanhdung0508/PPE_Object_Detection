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
    Postprocess YOLO predictions using NumPy vectorization for performance.
    """
    if predictions.shape[0] == 0:
        return []

    original_width, original_height = original_size
    pad_x, pad_y = pad

    # Type 1: [x1, y1, x2, y2, confidence, class_id]
    if predictions.shape[1] == 6:
        scores = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int)
        mask = (scores >= confidence_threshold) & (class_ids < len(CLASS_NAMES))
        predictions = predictions[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if predictions.shape[0] == 0:
            return []

        x1 = (predictions[:, 0] - pad_x) / scale
        y1 = (predictions[:, 1] - pad_y) / scale
        x2 = (predictions[:, 2] - pad_x) / scale
        y2 = (predictions[:, 3] - pad_y) / scale
    # Type 2: [x_center, y_center, width, height, (optional objectness), class_scores...]
    else:
        if predictions.shape[1] == 4 + len(CLASS_NAMES):
            objectness = 1.0
            class_scores = predictions[:, 4:]
        else:
            objectness = predictions[:, 4]
            class_scores = predictions[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        # Use advanced indexing to get the confidence for the predicted class
        class_confidences = class_scores[np.arange(len(class_scores)), class_ids]
        scores = objectness * class_confidences

        mask = (scores >= confidence_threshold) & (class_ids < len(CLASS_NAMES))
        predictions = predictions[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if predictions.shape[0] == 0:
            return []

        x_center, y_center, width, height = (
            predictions[:, 0],
            predictions[:, 1],
            predictions[:, 2],
            predictions[:, 3],
        )
        x1 = (x_center - width / 2 - pad_x) / scale
        y1 = (y_center - height / 2 - pad_y) / scale
        x2 = (x_center + width / 2 - pad_x) / scale
        y2 = (y_center + height / 2 - pad_y) / scale

    # Common clipping and validation
    x1 = np.clip(x1, 0, original_width - 1.0)
    y1 = np.clip(y1, 0, original_height - 1.0)
    x2 = np.clip(x2, 0, original_width - 1.0)
    y2 = np.clip(y2, 0, original_height - 1.0)

    box_w = x2 - x1
    box_h = y2 - y1

    valid_mask = (box_w > 0) & (box_h > 0)
    if not np.any(valid_mask):
        return []

    x1, y1, box_w, box_h = x1[valid_mask], y1[valid_mask], box_w[valid_mask], box_h[valid_mask]
    scores, class_ids = scores[valid_mask], class_ids[valid_mask]

    # cv2.dnn.NMSBoxes expects list of boxes in [x, y, w, h] format
    boxes = np.stack([x1, y1, box_w, box_h], axis=1).tolist()
    scores_list = scores.tolist()

    selected_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores_list,
        score_threshold=confidence_threshold,
        nms_threshold=IOU_THRESHOLD,
    )

    detections: list[dict[str, Any]] = []
    # Handle both single and multi-dimensional output from NMSBoxes
    for index in np.array(selected_indices).reshape(-1):
        idx = int(index)
        detections.append(
            {
                "class": CLASS_NAMES[class_ids[idx]],
                "confidence": round(float(scores_list[idx]), 4),
                "coordinates": [
                    round(boxes[idx][0], 2),
                    round(boxes[idx][1], 2),
                    round(boxes[idx][2], 2),
                    round(boxes[idx][3], 2),
                ],
            }
        )

    return detections
