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
                "coordinates": [
                    round(x, 2),
                    round(y, 2),
                    round(width, 2),
                    round(height, 2),
                ],
            }
        )

    return detections
