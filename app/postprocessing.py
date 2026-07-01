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
    Supports both XYXY (6-column) and CXCYWH (variable-column) formats.
    """
    if predictions.size == 0:
        return []

    original_width, original_height = original_size
    pad_x, pad_y = pad

    # 1. Calculate scores and class IDs
    if predictions.shape[1] == 6:
        # Format: [x1, y1, x2, y2, confidence, class_id]
        scores = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int)
        boxes_raw = predictions[:, :4]
    else:
        # Format: [x_center, y_center, width, height, (optional objectness), ...class_scores]
        if predictions.shape[1] == 4 + len(CLASS_NAMES):
            objectness = 1.0
            class_scores = predictions[:, 4:]
        else:
            objectness = predictions[:, 4]
            class_scores = predictions[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        # Use vectorized indexing to get scores for the predicted classes
        class_confidences = class_scores[np.arange(len(predictions)), class_ids]
        scores = objectness * class_confidences
        boxes_raw = predictions[:, :4]

    # 2. Early filter by confidence and class validity
    mask = (scores >= confidence_threshold) & (class_ids < len(CLASS_NAMES))
    if not np.any(mask):
        return []

    scores = scores[mask]
    class_ids = class_ids[mask]
    boxes_raw = boxes_raw[mask]

    # 3. Transform coordinates to original image space
    if predictions.shape[1] == 6:
        x1 = (boxes_raw[:, 0] - pad_x) / scale
        y1 = (boxes_raw[:, 1] - pad_y) / scale
        x2 = (boxes_raw[:, 2] - pad_x) / scale
        y2 = (boxes_raw[:, 3] - pad_y) / scale
    else:
        x_center, y_center, w, h = boxes_raw.T
        x1 = (x_center - w / 2 - pad_x) / scale
        y1 = (y_center - h / 2 - pad_y) / scale
        x2 = (x_center + w / 2 - pad_x) / scale
        y2 = (y_center + h / 2 - pad_y) / scale

    # 4. Clip to image boundaries
    x1 = np.clip(x1, 0, original_width - 1)
    y1 = np.clip(y1, 0, original_height - 1)
    x2 = np.clip(x2, 0, original_width - 1)
    y2 = np.clip(y2, 0, original_height - 1)

    # 5. Calculate width and height for NMS
    w = x2 - x1
    h = y2 - y1

    # Filter out invalid boxes (zero or negative area)
    valid_mask = (w > 0) & (h > 0)
    if not np.any(valid_mask):
        return []

    x1 = x1[valid_mask]
    y1 = y1[valid_mask]
    w = w[valid_mask]
    h = h[valid_mask]
    scores = scores[valid_mask]
    class_ids = class_ids[valid_mask]

    # 6. Apply Non-Maximum Suppression (NMS)
    # cv2.dnn.NMSBoxes expects boxes as [x, y, w, h]
    nms_boxes = np.stack([x1, y1, w, h], axis=1).tolist()
    nms_scores = scores.tolist()

    selected_indices = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes,
        scores=nms_scores,
        score_threshold=confidence_threshold,
        nms_threshold=IOU_THRESHOLD,
    )

    # 7. Format output detections
    detections: list[dict[str, Any]] = []
    if len(selected_indices) > 0:
        # Handle different return types from NMSBoxes depending on OpenCV version
        selected_indices = np.array(selected_indices).flatten()
        for idx in selected_indices:
            detections.append(
                {
                    "class": CLASS_NAMES[class_ids[idx]],
                    "confidence": round(float(scores[idx]), 4),
                    "coordinates": [
                        round(float(x1[idx]), 2),
                        round(float(y1[idx]), 2),
                        round(float(w[idx]), 2),
                        round(float(h[idx]), 2),
                    ],
                }
            )

    return detections
