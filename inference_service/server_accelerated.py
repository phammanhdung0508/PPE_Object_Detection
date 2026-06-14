from concurrent import futures
import argparse
import logging
from typing import Protocol

import grpc

from inference_service.detector_accelerated import AcceleratedObjectDetector
from protos import detector_pb2, detector_pb2_grpc


class BatchDetectorProtocol(Protocol):
    model_precision: str

    @property
    def execution_provider(self) -> str: ...

    def detect_batch(
        self,
        image_payloads: list[bytes],
        confidence_threshold: float = 0.25,
    ) -> tuple[list[list[dict]], float]: ...


def to_detection_message(detection: dict) -> detector_pb2.Detection:
    return detector_pb2.Detection(
        class_name=detection["class"],
        confidence=float(detection["confidence"]),
        x=float(detection["coordinates"][0]),
        y=float(detection["coordinates"][1]),
        width=float(detection["coordinates"][2]),
        height=float(detection["coordinates"][3]),
    )


class AcceleratedDetectorServicer(detector_pb2_grpc.DetectorServiceServicer):
    def __init__(self, detector: BatchDetectorProtocol, max_batch_size: int) -> None:
        self.detector = detector
        self.max_batch_size = max_batch_size

    def Detect(self, request, context):
        batch_response = self.BatchDetect(
            detector_pb2.BatchDetectRequest(
                frames=[
                    detector_pb2.FrameRequest(
                        image_bytes=request.image_bytes,
                        frame_id=request.frame_id,
                        camera_id=request.camera_id,
                    )
                ],
                confidence_threshold=request.confidence_threshold,
            ),
            context,
        )
        result = batch_response.results[0] if batch_response.results else detector_pb2.FrameDetections()
        return detector_pb2.DetectResponse(
            detections=result.detections,
            latency_ms=batch_response.latency_ms,
        )

    def BatchDetect(self, request, context):
        if not request.frames:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "frames are required")
        if len(request.frames) > self.max_batch_size:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"batch size exceeds max_batch_size={self.max_batch_size}",
            )

        confidence_threshold = request.confidence_threshold or 0.25
        if not 0.0 <= confidence_threshold <= 1.0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "confidence_threshold must be between 0 and 1")
        if any(not frame.image_bytes for frame in request.frames):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "all frames require image_bytes")

        try:
            detections_by_frame, latency_ms = self.detector.detect_batch(
                [frame.image_bytes for frame in request.frames],
                confidence_threshold,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            logging.exception("Batched inference failed")
            context.abort(grpc.StatusCode.INTERNAL, f"batched inference failed: {exc}")

        return detector_pb2.BatchDetectResponse(
            results=[
                detector_pb2.FrameDetections(
                    frame_id=frame.frame_id,
                    camera_id=frame.camera_id,
                    detections=[to_detection_message(detection) for detection in detections],
                )
                for frame, detections in zip(request.frames, detections_by_frame)
            ],
            latency_ms=latency_ms,
            batch_size=len(request.frames),
            model_precision=self.detector.model_precision,
            execution_provider=self.detector.execution_provider,
        )


def create_server(detector: BatchDetectorProtocol, max_workers: int = 4, max_batch_size: int = 8) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    detector_pb2_grpc.add_DetectorServiceServicer_to_server(
        AcceleratedDetectorServicer(detector, max_batch_size),
        server,
    )
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run accelerated batched YOLO26 gRPC inference service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    detector = AcceleratedObjectDetector()
    detector.load()
    server = create_server(detector, args.workers, args.max_batch_size)
    address = f"{args.host}:{args.port}"
    server.add_insecure_port(address)
    server.start()
    logging.info(
        "accelerated gRPC inference listening on %s precision=%s provider=%s",
        address,
        detector.model_precision,
        detector.execution_provider,
    )
    server.wait_for_termination()


if __name__ == "__main__":
    main()
