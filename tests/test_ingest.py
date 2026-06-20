import numpy as np

from protos import detector_pb2
from stream_ingestion.ingest import draw_detections, resize_for_inference


def test_resize_for_inference_preserves_aspect_ratio() -> None:
    frame = np.zeros((432, 768, 3), dtype=np.uint8)

    resized = resize_for_inference(frame, max_width=640)

    assert resized.shape[:2] == (360, 640)


def test_draw_detections_scales_from_inference_size() -> None:
    frame = np.zeros((432, 768, 3), dtype=np.uint8)
    detection = detector_pb2.Detection(
        class_name="helmet",
        confidence=0.9,
        x=320,
        y=180,
        width=100,
        height=60,
    )

    draw_detections(frame, [detection], source_size=(640, 360))

    # Top-left corner should scale from (320, 180) to (384, 216).
    assert np.any(frame[216, 384] != 0)


def test_draw_detections_without_scaling() -> None:
    frame = np.zeros((432, 768, 3), dtype=np.uint8)
    detection = detector_pb2.Detection(
        class_name="helmet",
        confidence=0.9,
        x=20,
        y=30,
        width=100,
        height=60,
    )

    draw_detections(frame, [detection])

    assert np.any(frame[30, 20] != 0)
