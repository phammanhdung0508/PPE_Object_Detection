import numpy as np

from app.config import CLASS_NAMES
from app.postprocessing import postprocess


def test_postprocess_returns_detection_for_high_confidence_prediction() -> None:
    class_scores = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    class_scores[0] = 0.9
    prediction = np.array([[320, 320, 100, 80, 0.95, *class_scores]], dtype=np.float32)

    detections = postprocess(
        prediction,
        original_size=(640, 640),
        scale=1.0,
        pad=(0, 0),
        confidence_threshold=0.25,
    )

    assert len(detections) == 1
    assert detections[0]["class"] == CLASS_NAMES[0]
    assert detections[0]["confidence"] > 0.8
    assert detections[0]["coordinates"] == [270.0, 280.0, 100.0, 80.0]


def test_postprocess_filters_low_confidence_prediction() -> None:
    class_scores = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    class_scores[0] = 0.1
    prediction = np.array([[320, 320, 100, 80, 0.5, *class_scores]], dtype=np.float32)

    detections = postprocess(
        prediction,
        original_size=(640, 640),
        scale=1.0,
        pad=(0, 0),
        confidence_threshold=0.25,
    )

    assert detections == []


def test_postprocess_applies_nms() -> None:
    class_scores = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    class_scores[0] = 0.9
    predictions = np.array(
        [
            [320, 320, 100, 80, 0.95, *class_scores],
            [322, 322, 100, 80, 0.90, *class_scores],
        ],
        dtype=np.float32,
    )

    detections = postprocess(
        predictions,
        original_size=(640, 640),
        scale=1.0,
        pad=(0, 0),
        confidence_threshold=0.25,
    )

    assert len(detections) == 1


def test_postprocess_supports_exported_nms_xyxy_format() -> None:
    prediction = np.array([[270, 280, 370, 360, 0.9, 0]], dtype=np.float32)

    detections = postprocess(
        prediction,
        original_size=(640, 640),
        scale=1.0,
        pad=(0, 0),
        confidence_threshold=0.25,
    )

    assert len(detections) == 1
    assert detections[0]["class"] == "helmet"
    assert detections[0]["confidence"] == 0.9
    assert detections[0]["coordinates"] == [270.0, 280.0, 100.0, 80.0]
