import argparse
import logging
import time
from pathlib import Path

import cv2
import grpc

from protos import detector_pb2, detector_pb2_grpc


def resolve_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def resize_for_inference(frame, max_width: int):
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        return cv2.resize(
            frame,
            (max_width, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def encode_frame(frame, jpeg_quality: int) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        raise ValueError("Could not encode frame as JPEG")
    return encoded.tobytes()


def color_for(label: str) -> tuple[int, int, int]:
    palette = {
        "helmet": (0, 220, 0),
        "no_helmet": (0, 0, 255),
        "no_vest": (0, 0, 255),
        "person": (255, 200, 0),
        "vest": (0, 160, 255),
    }
    return palette.get(label.lower(), (220, 220, 220))


def draw_detections(
    frame,
    detections: list[detector_pb2.Detection],
    source_size: tuple[int, int] | None = None,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    if source_size is None:
        scale_x = 1.0
        scale_y = 1.0
    else:
        source_width, source_height = source_size
        scale_x = frame_width / max(source_width, 1)
        scale_y = frame_height / max(source_height, 1)

    for detection in detections:
        x1 = int(round(detection.x * scale_x))
        y1 = int(round(detection.y * scale_y))
        x2 = int(round((detection.x + detection.width) * scale_x))
        y2 = int(round((detection.y + detection.height) * scale_y))
        color = color_for(detection.class_name)
        label = f"{detection.class_name} {detection.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        top = max(0, y1 - text_height - baseline - 6)
        bottom = top + text_height + baseline + 6
        cv2.rectangle(frame, (x1, top), (x1 + text_width + 8, bottom), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )


def open_video_writer(output_path: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, max(fps, 1.0), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output_path}")
    return writer


def send_frame(
    stub: detector_pb2_grpc.DetectorServiceStub,
    image_bytes: bytes,
    confidence_threshold: float,
    timeout: float,
) -> detector_pb2.DetectResponse:
    return stub.Detect(
        detector_pb2.DetectRequest(
            image_bytes=image_bytes,
            confidence_threshold=confidence_threshold,
        ),
        timeout=timeout,
    )


def open_capture(source: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(resolve_source(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return capture


def run_ingestion(args: argparse.Namespace) -> None:
    channel = grpc.insecure_channel(args.grpc_target)
    stub = detector_pb2_grpc.DetectorServiceStub(channel)
    frame_index = 0
    latest_detections: list[detector_pb2.Detection] = []
    latest_detection_size: tuple[int, int] | None = None

    while True:
        writer = None
        try:
            capture = open_capture(args.source)
            logging.info("Opened stream source=%s", args.source)
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Stream ended or frame read failed")

                frame_index += 1

                if frame_index % args.frame_stride == 0:
                    inference_frame = resize_for_inference(frame, args.max_width)
                    inference_height, inference_width = inference_frame.shape[:2]
                    image_bytes = encode_frame(inference_frame, args.jpeg_quality)
                    started_at = time.perf_counter()
                    response = send_frame(
                        stub,
                        image_bytes,
                        args.confidence,
                        args.timeout,
                    )
                    total_ms = round((time.perf_counter() - started_at) * 1000, 2)
                    latest_detections = list(response.detections)
                    latest_detection_size = (inference_width, inference_height)
                    detections = [
                        f"{det.class_name}:{det.confidence:.2f}"
                        for det in latest_detections
                    ]
                    logging.info(
                        "frame=%s detections=%s inference_ms=%.2f total_ms=%.2f",
                        frame_index,
                        detections,
                        response.latency_ms,
                        total_ms,
                    )

                display_frame = frame.copy()
                draw_detections(display_frame, latest_detections, latest_detection_size)

                if args.output:
                    if writer is None:
                        height, width = display_frame.shape[:2]
                        writer = open_video_writer(args.output, fps, (width, height))
                    writer.write(display_frame)

                if args.show:
                    cv2.imshow("PPE gRPC Video Demo", display_frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        return

                if args.max_frames and frame_index >= args.max_frames:
                    return
        except grpc.RpcError as exc:
            logging.warning("gRPC request failed: %s", exc)
        except Exception as exc:
            logging.warning("Stream ingestion error: %s", exc)
        finally:
            if "capture" in locals():
                capture.release()
            if writer is not None:
                writer.release()
            if args.show:
                cv2.destroyAllWindows()

        if not args.reconnect:
            break
        logging.info("Reconnecting in %.1f seconds", args.reconnect_delay)
        time.sleep(args.reconnect_delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read camera/video frames and send them to gRPC inference.")
    parser.add_argument("--source", default="0", help="Camera index, video path, RTSP URL, or HTTP stream URL")
    parser.add_argument("--grpc-target", default="localhost:50051", help="Detector gRPC target")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--frame-stride", type=int, default=15, help="Send every Nth frame")
    parser.add_argument("--max-width", type=int, default=640, help="Downscale frames above this width")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for frame payloads")
    parser.add_argument("--timeout", type=float, default=10.0, help="gRPC request timeout seconds")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N source frames; 0 runs forever")
    parser.add_argument("--show", action="store_true", help="Display annotated video frames in a window")
    parser.add_argument("--output", help="Write annotated video to this MP4 path")
    parser.add_argument("--reconnect", action="store_true", help="Reconnect when the stream drops")
    parser.add_argument("--reconnect-delay", type=float, default=3.0, help="Seconds between reconnect attempts")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_ingestion(parse_args())


if __name__ == "__main__":
    main()
