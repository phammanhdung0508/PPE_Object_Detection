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
    Vectorized post-processing of YOLO predictions.
    Supports both [x1, y1, x2, y2, conf, class_id] and
    [x_center, y_center, w, h, (objectness), class_scores...] formats.
    """
    if predictions.size == 0:
        return []

    original_width, original_height = original_size
    pad_x, pad_y = pad

    if predictions.shape[1] == 6:
        # Format: [x1, y1, x2, y2, confidence, class_id]
        confs = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int)

        mask = (confs >= confidence_threshold) & (class_ids < len(CLASS_NAMES))
        if not np.any(mask):
            return []

        filtered_preds = predictions[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        x1 = (filtered_preds[:, 0] - pad_x) / scale
        y1 = (filtered_preds[:, 1] - pad_y) / scale
        x2 = (filtered_preds[:, 2] - pad_x) / scale
        y2 = (filtered_preds[:, 3] - pad_y) / scale

    else:
        # Format: [x_center, y_center, width, height, (optional objectness), class_scores...]
        if predictions.shape[1] == 4 + len(CLASS_NAMES):
            objectness = 1.0
            class_scores = predictions[:, 4:]
        else:
            objectness = predictions[:, 4]
            class_scores = predictions[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        class_confs = class_scores[np.arange(len(class_scores)), class_ids]
        confs = objectness * class_confs

        mask = (confs >= confidence_threshold) & (class_ids < len(CLASS_NAMES))
        if not np.any(mask):
            return []

        filtered_preds = predictions[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        x_center = filtered_preds[:, 0]
        y_center = filtered_preds[:, 1]
        w = filtered_preds[:, 2]
        h = filtered_preds[:, 3]

        x1 = (x_center - w / 2 - pad_x) / scale
        y1 = (y_center - h / 2 - pad_y) / scale
        x2 = (x_center + w / 2 - pad_x) / scale
        y2 = (y_center + h / 2 - pad_y) / scale

    # Clip coordinates to image boundaries
    x1 = np.clip(x1, 0, original_width - 1.0)
    y1 = np.clip(y1, 0, original_height - 1.0)
    x2 = np.clip(x2, 0, original_width - 1.0)
    y2 = np.clip(y2, 0, original_height - 1.0)

    widths = x2 - x1
    heights = y2 - y1

    # Filter out invalid boxes with non-positive dimensions
    valid_mask = (widths > 0) & (heights > 0)
    if not np.any(valid_mask):
        return []

    final_boxes = np.stack([x1[valid_mask], y1[valid_mask], widths[valid_mask], heights[valid_mask]], axis=1)
    final_confs = confs[valid_mask]
    final_class_ids = class_ids[valid_mask]

    # Apply Non-Maximum Suppression (NMS)
    selected_indices = cv2.dnn.NMSBoxes(
        bboxes=final_boxes.tolist(),
        scores=final_confs.tolist(),
        score_threshold=confidence_threshold,
        nms_threshold=IOU_THRESHOLD,
    )

    detections: list[dict[str, Any]] = []
    # Handle different return types of NMSBoxes across OpenCV versions
    for index in np.array(selected_indices).flatten():
        idx = int(index)
        detections.append(
            {
                "class": CLASS_NAMES[final_class_ids[idx]],
                "confidence": round(float(final_confs[idx]), 4),
                "coordinates": [
                    round(float(final_boxes[idx, 0]), 2),
                    round(float(final_boxes[idx, 1]), 2),
                    round(float(final_boxes[idx, 2]), 2),
                    round(float(final_boxes[idx, 3]), 2),
                ],
            }
        )

    return detections
