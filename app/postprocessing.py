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
    if predictions.size == 0:
        return []

    original_width, original_height = original_size
    pad_x, pad_y = pad

    if predictions.shape[1] == 6:
        # Vectorized Case 1: [x1, y1, x2, y2, conf, class_id]
        scores = predictions[:, 4]
        mask = scores >= confidence_threshold
        if not np.any(mask):
            return []

        predictions = predictions[mask]
        scores = scores[mask]
        class_ids = predictions[:, 5].astype(int)

        valid_classes = class_ids < len(CLASS_NAMES)
        if not np.any(valid_classes):
            return []

        predictions = predictions[valid_classes]
        scores = scores[valid_classes]
        class_ids = class_ids[valid_classes]

        x1 = (predictions[:, 0] - pad_x) / scale
        y1 = (predictions[:, 1] - pad_y) / scale
        x2 = (predictions[:, 2] - pad_x) / scale
        y2 = (predictions[:, 3] - pad_y) / scale
    else:
        # Vectorized Case 2: [cx, cy, w, h, (obj), scores...]
        if predictions.shape[1] == 4 + len(CLASS_NAMES):
            objectness = 1.0
            class_scores = predictions[:, 4:]
        elif predictions.shape[1] > 5:
            objectness = predictions[:, 4]
            class_scores = predictions[:, 5:]
        else:
            return [] # Should not happen based on existing logic

        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = class_scores[np.arange(len(class_scores)), class_ids]
        scores = objectness * class_confidences

        mask = (scores >= confidence_threshold) & (class_ids < len(CLASS_NAMES))
        if not np.any(mask):
            return []

        predictions = predictions[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        x1 = (cx - (w / 2) - pad_x) / scale
        y1 = (cy - (h / 2) - pad_y) / scale
        x2 = (cx + (w / 2) - pad_x) / scale
        y2 = (cy + (h / 2) - pad_y) / scale

    # Clip coordinates
    x1 = np.clip(x1, 0.0, original_width - 1.0)
    y1 = np.clip(y1, 0.0, original_height - 1.0)
    x2 = np.clip(x2, 0.0, original_width - 1.0)
    y2 = np.clip(y2, 0.0, original_height - 1.0)

    bw = x2 - x1
    bh = y2 - y1

    valid_boxes_mask = (bw > 0) & (bh > 0)
    if not np.any(valid_boxes_mask):
        return []

    final_boxes = np.stack([x1, y1, bw, bh], axis=1)[valid_boxes_mask]
    final_scores = scores[valid_boxes_mask]
    final_class_ids = class_ids[valid_boxes_mask]

    # NMS
    selected_indices = cv2.dnn.NMSBoxes(
        bboxes=final_boxes,
        scores=final_scores,
        score_threshold=confidence_threshold,
        nms_threshold=IOU_THRESHOLD,
    )

    detections: list[dict[str, Any]] = []
    if len(selected_indices) > 0:
        for index in np.array(selected_indices).reshape(-1):
            x, y, width, height = final_boxes[index]
            detections.append(
                {
                    "class": CLASS_NAMES[final_class_ids[index]],
                    "confidence": round(float(final_scores[index]), 4),
                    "coordinates": [round(float(x), 2), round(float(y), 2), round(float(width), 2), round(float(height), 2)],
                }
            )

    return detections
