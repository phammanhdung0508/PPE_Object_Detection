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
        # Format 1: [x1, y1, x2, y2, confidence, class_id]
        x1_raw, y1_raw, x2_raw, y2_raw = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        confidences = predictions[:, 4].astype(float)
        class_ids = predictions[:, 5].astype(int)

        x1 = (x1_raw - pad_x) / scale
        y1 = (y1_raw - pad_y) / scale
        x2 = (x2_raw - pad_x) / scale
        y2 = (y2_raw - pad_y) / scale
    else:
        # Format 2 & 3: [x_center, y_center, width, height, (optional objectness), class_scores...]
        x_center, y_center, width, height = (
            predictions[:, 0],
            predictions[:, 1],
            predictions[:, 2],
            predictions[:, 3],
        )

        if predictions.shape[1] == 4 + len(CLASS_NAMES):
            objectness = 1.0
            class_scores = predictions[:, 4:]
        else:
            objectness = predictions[:, 4]
            class_scores = predictions[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = class_scores[np.arange(len(predictions)), class_ids]
        confidences = (objectness * class_confidences).astype(float)

        x1 = (x_center - width / 2 - pad_x) / scale
        y1 = (y_center - height / 2 - pad_y) / scale
        x2 = (x_center + width / 2 - pad_x) / scale
        y2 = (y_center + height / 2 - pad_y) / scale

    # Vectorized filtering and clipping
    mask = (confidences >= confidence_threshold) & (class_ids < len(CLASS_NAMES))
    if not np.any(mask):
        return []

    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    scores = confidences[mask]
    class_ids = class_ids[mask]

    x1 = np.clip(x1, 0, original_width - 1.0)
    y1 = np.clip(y1, 0, original_height - 1.0)
    x2 = np.clip(x2, 0, original_width - 1.0)
    y2 = np.clip(y2, 0, original_height - 1.0)

    box_widths = x2 - x1
    box_heights = y2 - y1

    valid_mask = (box_widths > 0) & (box_heights > 0)
    if not np.any(valid_mask):
        return []

    final_boxes = np.stack([x1, y1, box_widths, box_heights], axis=1)[valid_mask]
    final_scores = scores[valid_mask]
    final_class_ids = class_ids[valid_mask]

    selected_indices = cv2.dnn.NMSBoxes(
        bboxes=final_boxes.tolist(),
        scores=final_scores.tolist(),
        score_threshold=float(confidence_threshold),
        nms_threshold=float(IOU_THRESHOLD),
    )

    detections: list[dict[str, Any]] = []
    if len(selected_indices) > 0:
        for index in np.array(selected_indices).flatten():
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
