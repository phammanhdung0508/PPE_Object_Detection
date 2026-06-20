from concurrent import futures
import argparse
import logging
from typing import Protocol

import grpc

from protos import detector_pb2, detector_pb2_grpc
from inference_service.detector import ObjectDetector


class DetectorProtocol(Protocol):
    def detect(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.25,
    ) -> tuple[list[dict], float]: ...


class DetectorServicer(detector_pb2_grpc.DetectorServiceServicer):
    def __init__(self, detector: DetectorProtocol) -> None:
        self.detector = detector

    def Detect(self, request, context):
        if not request.image_bytes:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "image_bytes is required")

        confidence_threshold = request.confidence_threshold or 0.25
        if not 0.0 <= confidence_threshold <= 1.0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "confidence_threshold must be between 0 and 1",
            )

        try:
            detections, latency_ms = self.detector.detect(
                request.image_bytes,
                confidence_threshold,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception:
            logging.exception("Inference failed")
            context.abort(
                grpc.StatusCode.INTERNAL, "An internal error occurred during inference"
            )

        return detector_pb2.DetectResponse(
            detections=[
                detector_pb2.Detection(
                    class_name=detection["class"],
                    confidence=float(detection["confidence"]),
                    x=float(detection["coordinates"][0]),
                    y=float(detection["coordinates"][1]),
                    width=float(detection["coordinates"][2]),
                    height=float(detection["coordinates"][3]),
                )
                for detection in detections
            ],
            latency_ms=latency_ms,
        )

    def BatchDetect(self, request, context):
        if not request.frames:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "frames are required")

        confidence_threshold = request.confidence_threshold or 0.25
        if not 0.0 <= confidence_threshold <= 1.0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "confidence_threshold must be between 0 and 1",
            )

        results = []
        total_latency_ms = 0.0
        for frame in request.frames:
            if not frame.image_bytes:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "all frames require image_bytes"
                )
            try:
                detections, latency_ms = self.detector.detect(
                    frame.image_bytes,
                    confidence_threshold,
                )
            except ValueError as exc:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except Exception:
                logging.exception("Inference failed")
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    "An internal error occurred during inference",
                )

            total_latency_ms += latency_ms
            results.append(
                detector_pb2.FrameDetections(
                    frame_id=frame.frame_id,
                    camera_id=frame.camera_id,
                    detections=[
                        detector_pb2.Detection(
                            class_name=detection["class"],
                            confidence=float(detection["confidence"]),
                            x=float(detection["coordinates"][0]),
                            y=float(detection["coordinates"][1]),
                            width=float(detection["coordinates"][2]),
                            height=float(detection["coordinates"][3]),
                        )
                        for detection in detections
                    ],
                )
            )

        return detector_pb2.BatchDetectResponse(
            results=results,
            latency_ms=round(total_latency_ms, 2),
            batch_size=len(request.frames),
            model_precision="FP32",
            execution_provider="sequential",
        )


def create_server(detector: DetectorProtocol, max_workers: int = 4) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    detector_pb2_grpc.add_DetectorServiceServicer_to_server(
        DetectorServicer(detector),
        server,
    )
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO26 gRPC inference service.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=50051, help="Bind port")
    parser.add_argument("--workers", type=int, default=4, help="gRPC worker threads")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()

    detector = ObjectDetector()
    detector.load()

    server = create_server(detector, max_workers=args.workers)
    address = f"{args.host}:{args.port}"
    server.add_insecure_port(address)
    server.start()
    logging.info("gRPC inference service listening on %s", address)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
