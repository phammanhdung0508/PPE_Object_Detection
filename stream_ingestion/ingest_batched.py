import argparse
import logging
import time

import cv2
import grpc

from protos import detector_pb2, detector_pb2_grpc
from stream_ingestion.ingest import encode_frame, resize_for_inference, resolve_source


def open_captures(sources: list[str]) -> list[cv2.VideoCapture]:
    captures: list[cv2.VideoCapture] = []
    for source in sources:
        capture = cv2.VideoCapture(resolve_source(source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source: {source}")
        captures.append(capture)
    return captures


def read_batch_frames(
    captures: list[cv2.VideoCapture],
    sources: list[str],
    frame_index: int,
    max_width: int,
    jpeg_quality: int,
) -> list[detector_pb2.FrameRequest]:
    frames: list[detector_pb2.FrameRequest] = []
    for camera_index, capture in enumerate(captures):
        ok, frame = capture.read()
        if not ok:
            logging.warning(
                "source ended camera_id=%s source=%s",
                camera_index,
                sources[camera_index],
            )
            continue
        inference_frame = resize_for_inference(frame, max_width)
        frames.append(
            detector_pb2.FrameRequest(
                image_bytes=encode_frame(inference_frame, jpeg_quality),
                frame_id=str(frame_index),
                camera_id=f"camera_{camera_index}",
            )
        )
    return frames


def run_batched_ingestion(args: argparse.Namespace) -> None:
    sources = args.sources or [args.source] * args.camera_count
    channel = grpc.insecure_channel(args.grpc_target)
    stub = detector_pb2_grpc.DetectorServiceStub(channel)
    captures = open_captures(sources)
    logging.info("Opened %s stream sources", len(captures))

    try:
        frame_index = 0
        while True:
            frame_index += 1
            frames = read_batch_frames(
                captures,
                sources,
                frame_index,
                args.max_width,
                args.jpeg_quality,
            )
            if not frames:
                break
            if frame_index % args.frame_stride != 0:
                continue

            started_at = time.perf_counter()
            response = stub.BatchDetect(
                detector_pb2.BatchDetectRequest(
                    frames=frames[: args.batch_size],
                    confidence_threshold=args.confidence,
                ),
                timeout=args.timeout,
            )
            total_ms = round((time.perf_counter() - started_at) * 1000, 2)
            detection_count = sum(len(result.detections) for result in response.results)
            logging.info(
                "batch frame=%s batch_size=%s detections=%s precision=%s provider=%s inference_ms=%.2f total_ms=%.2f",
                frame_index,
                response.batch_size,
                detection_count,
                response.model_precision,
                response.execution_provider,
                response.latency_ms,
                total_ms,
            )

            if (
                args.max_batches
                and frame_index // args.frame_stride >= args.max_batches
            ):
                break
    finally:
        for capture in captures:
            capture.release()
        channel.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate multi-camera ingestion with client-side gRPC batching."
    )
    parser.add_argument(
        "--source",
        default="sample_videos/demo.mp4",
        help="Source repeated for simulated cameras",
    )
    parser.add_argument(
        "--sources", nargs="*", help="Explicit list of video/camera sources"
    )
    parser.add_argument(
        "--camera-count",
        type=int,
        default=4,
        help="Number of simulated cameras when --sources is omitted",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Frames per BatchDetect request"
    )
    parser.add_argument("--grpc-target", default="localhost:50051")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop after N batch requests; 0 runs until streams end",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_batched_ingestion(parse_args())


if __name__ == "__main__":
    main()
