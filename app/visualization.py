from typing import Any

import cv2
import numpy as np


def draw_detections(image_bytes: bytes, detections: list[dict[str, Any]]) -> bytes:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image, please check the camera")

    for detection in detections:
        x, y, width, height = detection["coordinates"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + width))
        y2 = int(round(y + height))
        label = f"{detection['class']} {detection['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_width, text_height = text_size
        label_y1 = max(0, y1 - text_height - baseline - 6)
        label_y2 = y1 if y1 > text_height + baseline + 6 else text_height + baseline + 6
        cv2.rectangle(
            image,
            (x1, label_y1),
            (x1 + text_width + 6, label_y2),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 3, label_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Could not encode annotated image")
    return encoded.tobytes()
