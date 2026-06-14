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


def test_postprocess_class_aware_nms_retains_overlapping_different_classes() -> None:
    # Create two predictions with high overlap but different classes
    # e.g., 'person' and 'vest'
    person_class_id = CLASS_NAMES.index("person") if "person" in CLASS_NAMES else 3
    vest_class_id = CLASS_NAMES.index("vest") if "vest" in CLASS_NAMES else 4

    class_scores_person = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    class_scores_person[person_class_id] = 0.9

    class_scores_vest = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    class_scores_vest[vest_class_id] = 0.85

    predictions = np.array(
        [
            [320, 320, 100, 80, 0.95, *class_scores_person],
            [320, 320, 100, 80, 0.90, *class_scores_vest],  # Identical box, different class
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

    # Class-aware NMS should keep both boxes because their classes are different
    assert len(detections) == 2
    detected_classes = {d["class"] for d in detections}
    assert CLASS_NAMES[person_class_id] in detected_classes
    assert CLASS_NAMES[vest_class_id] in detected_classes
