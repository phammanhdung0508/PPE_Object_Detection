from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run realtime PPE detection with a fine-tuned YOLO26 model."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=project_dir / "weights" / "best.pt",
        help="Path to the fine-tuned YOLO26 weights.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, or RTSP/HTTP stream URL. Default: 0",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Inference device, e.g. "cpu", "cuda:0". Default lets Ultralytics choose.',
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror (flip horizontally) the video stream.",
    )
    return parser.parse_args()


def resolve_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def color_for(label: str) -> tuple[int, int, int]:
    palette = {
        "human": (255, 200, 0),
        "person": (255, 200, 0),
        "helmet": (0, 220, 0),
        "hardhat": (0, 220, 0),
        "vest": (0, 160, 255),
    }
    return palette.get(label.lower(), (220, 220, 220))


def find_class_ids(names: dict[int, str]) -> dict[str, set[int]]:
    aliases = {
        "person": {"person", "human", "worker"},
        "helmet": {"helmet", "hardhat", "hat"},
        "vest": {"vest", "safety_vest", "jacket", "reflective_vest"},
    }
    found: dict[str, set[int]] = {key: set() for key in aliases}
    for class_id, class_name in names.items():
        normalized = class_name.lower().strip().replace(" ", "_")
        for target, options in aliases.items():
            if normalized in options:
                found[target].add(class_id)
    return found


def extract_detections(result) -> list[Detection]:
    detections: list[Detection] = []
    boxes = result.boxes
    if boxes is None or boxes.cls is None:
        return detections

    xyxy = boxes.xyxy.int().cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    class_ids = boxes.cls.int().cpu().tolist()

    for box, conf, class_id in zip(xyxy, confs, class_ids):
        detections.append(
            Detection(
                class_id=class_id,
                class_name=str(result.names[class_id]),
                confidence=float(conf),
                box=tuple(box),
            )
        )
    return detections


def point_in_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def belongs_to_region(
    det: Detection, person_box: tuple[int, int, int, int], target: str
) -> bool:
    center_x, center_y = det.center
    x1, y1, x2, y2 = person_box
    height = max(1, y2 - y1)
    relative_y = (center_y - y1) / height

    if not point_in_box((center_x, center_y), person_box):
        return False
    if target == "helmet":
        return relative_y <= 0.45
    if target == "vest":
        return 0.2 <= relative_y <= 0.9
    return True


def select_person_for_item(
    item: Detection, people: list[Detection], target: str
) -> Detection | None:
    matches: list[tuple[int, Detection]] = []
    for person in people:
        if belongs_to_region(item, person.box, target):
            x1, y1, x2, y2 = person.box
            area = max(1, (x2 - x1) * (y2 - y1))
            matches.append((area, person))
    if not matches:
        return None
    matches.sort(key=lambda pair: pair[0])
    return matches[0][1]


def draw_label(
    frame,
    text: str,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    top = max(0, y1 - text_height - 10)
    cv2.rectangle(
        frame,
        (x1, top),
        (x1 + text_width + 10, top + text_height + 8),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 5, top + text_height + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )


def annotate_frame(
    frame,
    detections: list[Detection],
    class_ids: dict[str, set[int]],
    line_width: int,
    fps: float,
) -> None:
    people = [d for d in detections if d.class_id in class_ids["person"]]
    helmets = [d for d in detections if d.class_id in class_ids["helmet"]]
    vests = [d for d in detections if d.class_id in class_ids["vest"]]

    matched_helmets: dict[int, list[Detection]] = {}
    matched_vests: dict[int, list[Detection]] = {}

    for helmet in helmets:
        person = select_person_for_item(helmet, people, "helmet")
        if person is not None:
            matched_helmets.setdefault(id(person), []).append(helmet)

    for vest in vests:
        person = select_person_for_item(vest, people, "vest")
        if person is not None:
            matched_vests.setdefault(id(person), []).append(vest)

    for detection in detections:
        label = f"{detection.class_name} {detection.confidence:.2f}"
        draw_label(
            frame,
            label,
            detection.box,
            color_for(detection.class_name),
            line_width,
        )

    compliant_count = 0
    for person in people:
        has_helmet = bool(matched_helmets.get(id(person)))
        has_vest = bool(matched_vests.get(id(person)))

        missing: list[str] = []
        if not has_helmet:
            missing.append("helmet")
        if not has_vest:
            missing.append("vest")

        if missing:
            status = "Missing " + ", ".join(missing)
            color = (0, 0, 255)
        else:
            status = "PPE OK"
            color = (0, 200, 0)
            compliant_count += 1

        x1, y1, x2, y2 = person.box
        cv2.putText(
            frame,
            status,
            (x1, min(frame.shape[0] - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    summary = (
        f"People: {len(people)} | Helmets: {len(helmets)} | "
        f"Vests: {len(vests)} | PPE OK: {compliant_count} | FPS: {fps:.1f}"
    )
    cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 700), 45), (30, 30, 30), -1)
    cv2.putText(
        frame,
        summary,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    weights_path = args.weights.resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    class_ids = find_class_ids(model.names)

    if not class_ids["person"]:
        raise ValueError(
            f'Could not find a "person"/"human" class in model names: {model.names}'
        )

    source = resolve_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            detections = extract_detections(results[0])

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            annotate_frame(frame, detections, class_ids, args.line_width, fps)
            cv2.imshow("YOLO26 PPE Realtime Inference", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
