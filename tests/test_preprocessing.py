import cv2
import numpy as np
import pytest

from app.config import INPUT_SIZE
from app.preprocessing import preprocess_image


def encode_image(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_preprocess_valid_image_returns_yolo_tensor() -> None:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    tensor, original_size, scale, pad, brightness = preprocess_image(
        encode_image(image)
    )

    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == np.float32
    assert original_size == (160, 120)
    assert scale > 0
    assert len(pad) == 2
    assert brightness == pytest.approx(128.0, abs=1.0)
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_preprocess_rejects_corrupted_image() -> None:
    with pytest.raises(ValueError, match="Invalid image"):
        preprocess_image(b"not an image")


def test_preprocess_rejects_dark_image() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Invalid image"):
        preprocess_image(encode_image(image))


def test_preprocess_rejects_small_image() -> None:
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        preprocess_image(encode_image(image))
