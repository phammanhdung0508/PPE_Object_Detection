import cv2
import numpy as np

from app.config import INPUT_SIZE, MAX_UPLOAD_BYTES, MIN_IMAGE_SIZE


def preprocess_image(
    image_bytes: bytes,
) -> tuple[np.ndarray, tuple[int, int], float, tuple[int, int], float]:
    if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Invalid image size")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image, please check the camera")

    original_height, original_width = image.shape[:2]
    if original_height < MIN_IMAGE_SIZE or original_width < MIN_IMAGE_SIZE:
        raise ValueError("Invalid image, image is too small")

    brightness = float(np.mean(image))
    if brightness < 3.0:
        raise ValueError("Invalid image, please check the camera")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = min(INPUT_SIZE / original_width, INPUT_SIZE / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    resized = cv2.resize(
        rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )

    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (INPUT_SIZE - resized_width) // 2
    pad_y = (INPUT_SIZE - resized_height) // 2
    padded[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

    normalized = padded.astype(np.float32) / 255.0
    tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
    return (
        np.ascontiguousarray(tensor),
        (original_width, original_height),
        scale,
        (pad_x, pad_y),
        brightness,
    )
