import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import grpc
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protos import detector_pb2, detector_pb2_grpc


def encode_image(image_path: Path) -> bytes:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError(f"Could not encode image: {image_path}")
    return encoded.tobytes()


def synthetic_image_bytes() -> bytes:
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not encode synthetic image")
    return encoded.tobytes()


def benchmark_batch(
    stub: detector_pb2_grpc.DetectorServiceStub,
    image_bytes: bytes,
    batch_size: int,
    iterations: int,
    warmup: int,
    confidence: float,
    timeout: float,
) -> dict[str, float | int | str]:
    frames = [
        detector_pb2.FrameRequest(
            image_bytes=image_bytes,
            frame_id=str(index),
            camera_id=f"camera_{index}",
        )
        for index in range(batch_size)
    ]

    for _ in range(warmup):
        stub.BatchDetect(
            detector_pb2.BatchDetectRequest(
                frames=frames, confidence_threshold=confidence
            ),
            timeout=timeout,
        )

    latencies: list[float] = []
    provider = "unknown"
    precision = "unknown"
    for _ in range(iterations):
        started_at = time.perf_counter()
        response = stub.BatchDetect(
            detector_pb2.BatchDetectRequest(
                frames=frames, confidence_threshold=confidence
            ),
            timeout=timeout,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        latencies.append(elapsed_ms)
        provider = response.execution_provider
        precision = response.model_precision

    avg_latency = statistics.mean(latencies)
    p95_latency = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
    throughput_fps = (batch_size * 1000.0) / avg_latency
    return {
        "batch_size": batch_size,
        "precision": precision,
        "provider": provider,
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "throughput_fps": round(throughput_fps, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark real batched gRPC inference latency."
    )
    parser.add_argument("--grpc-target", default="localhost:50051")
    parser.add_argument("--image", type=Path, help="Optional benchmark image path")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_bytes = encode_image(args.image) if args.image else synthetic_image_bytes()
    channel = grpc.insecure_channel(args.grpc_target)
    stub = detector_pb2_grpc.DetectorServiceStub(channel)

    rows = [
        benchmark_batch(
            stub,
            image_bytes,
            batch_size,
            args.iterations,
            args.warmup,
            args.confidence,
            args.timeout,
        )
        for batch_size in args.batch_sizes
    ]
    channel.close()

    print(
        "| batch_size | precision | provider | avg_latency_ms | p95_latency_ms | throughput_fps |"
    )
    print("|---:|---|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['batch_size']} | {row['precision']} | {row['provider']} | "
            f"{row['avg_latency_ms']} | {row['p95_latency_ms']} | {row['throughput_fps']} |"
        )


if __name__ == "__main__":
    main()
