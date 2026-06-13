import cv2
import numpy as np
import pytest

from app.visualization import draw_detections


def encode_image(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_draw_detections_returns_jpeg_bytes() -> None:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    detections = [
        {
            "class": "helmet",
            "confidence": 0.9,
            "coordinates": [20, 30, 50, 40],
        }
    ]

    annotated = draw_detections(encode_image(image), detections)

    decoded = cv2.imdecode(np.frombuffer(annotated, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == image.shape


def test_draw_detections_rejects_invalid_image() -> None:
    with pytest.raises(ValueError, match="Invalid image"):
        draw_detections(b"not an image", [])
