import time

import cv2
import grpc
import numpy as np

from inference_service.server_accelerated import create_server
from protos import detector_pb2, detector_pb2_grpc


class FastBatchDetector:
    model_precision = "FP32"
    execution_provider = "CPUExecutionProvider"

    def detect_batch(self, image_payloads: list[bytes], confidence_threshold: float = 0.25):
        return [[] for _ in image_payloads], 5.0


def encode_image() -> bytes:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_batch_detect_latency_boundary() -> None:
    server = create_server(FastBatchDetector(), max_batch_size=8)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = detector_pb2_grpc.DetectorServiceStub(channel)
    image_bytes = encode_image()

    started_at = time.perf_counter()
    response = stub.BatchDetect(
        detector_pb2.BatchDetectRequest(
            frames=[
                detector_pb2.FrameRequest(image_bytes=image_bytes, frame_id=str(index), camera_id=f"cam_{index}")
                for index in range(4)
            ],
            confidence_threshold=0.25,
        ),
        timeout=5,
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    channel.close()
    server.stop(grace=0)

    assert response.batch_size == 4
    assert response.model_precision == "FP32"
    assert response.execution_provider == "CPUExecutionProvider"
    assert elapsed_ms < 1000


def test_batch_detect_rejects_oversized_batch() -> None:
    server = create_server(FastBatchDetector(), max_batch_size=2)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = detector_pb2_grpc.DetectorServiceStub(channel)
    image_bytes = encode_image()

    try:
        try:
            stub.BatchDetect(
                detector_pb2.BatchDetectRequest(
                    frames=[detector_pb2.FrameRequest(image_bytes=image_bytes) for _ in range(3)]
                ),
                timeout=5,
            )
            raise AssertionError("BatchDetect should reject oversized batches")
        except grpc.RpcError as exc:
            assert exc.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        channel.close()
        server.stop(grace=0)
