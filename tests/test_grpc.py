from concurrent import futures

import cv2
import grpc
import numpy as np
import pytest

from inference_service.server import create_server
from protos import detector_pb2, detector_pb2_grpc


class FakeDetector:
    def detect(self, image_bytes: bytes, confidence_threshold: float = 0.25):
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image, please check the camera")
        return [
            {
                "class": "helmet",
                "confidence": 0.9,
                "coordinates": [10.0, 20.0, 30.0, 40.0],
            }
        ], 12.5


@pytest.fixture()
def grpc_stub():
    server = create_server(FakeDetector())
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield detector_pb2_grpc.DetectorServiceStub(channel)
    finally:
        channel.close()
        server.stop(grace=0)


def encode_image() -> bytes:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_grpc_detect_valid_jpeg_returns_detections(grpc_stub) -> None:
    response = grpc_stub.Detect(
        detector_pb2.DetectRequest(
            image_bytes=encode_image(),
            confidence_threshold=0.25,
        )
    )

    assert response.latency_ms == 12.5
    assert len(response.detections) == 1
    assert response.detections[0].class_name == "helmet"
    assert response.detections[0].confidence == pytest.approx(0.9)


def test_grpc_detect_rejects_empty_payload(grpc_stub) -> None:
    with pytest.raises(grpc.RpcError) as exc_info:
        grpc_stub.Detect(detector_pb2.DetectRequest(image_bytes=b""))

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_grpc_detect_rejects_corrupted_image(grpc_stub) -> None:
    with pytest.raises(grpc.RpcError) as exc_info:
        grpc_stub.Detect(detector_pb2.DetectRequest(image_bytes=b"not an image"))

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_grpc_detect_rejects_invalid_threshold(grpc_stub) -> None:
    with pytest.raises(grpc.RpcError) as exc_info:
        grpc_stub.Detect(
            detector_pb2.DetectRequest(
                image_bytes=encode_image(),
                confidence_threshold=1.5,
            )
        )

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_grpc_batch_detect_valid_jpegs_returns_results(grpc_stub) -> None:
    response = grpc_stub.BatchDetect(
        detector_pb2.BatchDetectRequest(
            frames=[
                detector_pb2.FrameRequest(
                    image_bytes=encode_image(),
                    frame_id="frame-1",
                    camera_id="camera-1",
                ),
                detector_pb2.FrameRequest(
                    image_bytes=encode_image(),
                    frame_id="frame-2",
                    camera_id="camera-2",
                ),
            ],
            confidence_threshold=0.25,
        )
    )

    assert response.batch_size == 2
    assert len(response.results) == 2
    assert response.results[0].frame_id == "frame-1"
    assert response.results[0].camera_id == "camera-1"
