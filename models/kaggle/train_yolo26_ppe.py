import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "/kaggle/input/construction-safety-ppe"))
KAGGLE_INPUT_ROOT = Path(os.getenv("KAGGLE_INPUT_ROOT", "/kaggle/input"))
WORK_DIR = Path(os.getenv("WORK_DIR", "/kaggle/working"))
SCRIPT_DIR = Path(__file__).resolve().parent
PREPARED_DIR = Path(os.getenv("PREPARED_DIR", "/kaggle/temp/ppe_yolo_dataset"))
RUNS_DIR = WORK_DIR / "runs"
OUTPUT_MODEL = WORK_DIR / "yolo26_ppe.onnx"
OUTPUT_FP16_MODEL = WORK_DIR / "yolo26_ppe_fp16.onnx"

MODEL_ARCH = os.getenv("MODEL_ARCH", "yolo26s.pt")
EPOCHS = int(os.getenv("EPOCHS", "30"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
SEED = int(os.getenv("SEED", "42"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "0"))
DEVICE = os.getenv("DEVICE")
EXPORT_DYNAMIC = os.getenv("EXPORT_DYNAMIC", "true").lower() == "true"
EXPORT_FP16 = os.getenv("EXPORT_FP16", "true").lower() == "true"
LR0 = float(os.getenv("LR0", "0.001"))
PATIENCE = int(os.getenv("PATIENCE", "15"))
RUN_NAME = os.getenv("RUN_NAME", "construction_safety_yolo26_finetune")
USE_EXISTING_DATA_YAML = os.getenv("USE_EXISTING_DATA_YAML", "true").lower() == "true"
FREEZE = int(os.getenv("FREEZE", "10"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_FILE_NAMES = {"classes.txt", "obj.names", "data.names", "labels.txt"}


def ensure_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])


def normalize_label(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def load_yaml(yaml_path: Path) -> dict[str, Any] | None:
    try:
        import yaml
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
        import yaml

    try:
        return yaml.safe_load(yaml_path.read_text()) or {}
    except Exception:
        return None


def load_names_from_yaml(yaml_path: Path) -> list[str] | None:
    payload = load_yaml(yaml_path)
    if not payload:
        return None

    names = payload.get("names")
    if isinstance(names, list):
        return [normalize_label(str(name)) for name in names]
    if isinstance(names, dict):
        return [normalize_label(str(names[index])) for index in sorted(names)]
    return None


def class_count_is_valid(names: list[str]) -> bool:
    return bool(names) and (NUM_CLASSES <= 0 or len(names) == NUM_CLASSES)


def discover_voc_class_names(root: Path) -> list[str]:
    names: set[str] = set()
    for xml_path in root.rglob("*.xml"):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        for label_node in tree.findall(".//object/name"):
            if label_node.text:
                names.add(normalize_label(label_node.text))
    return sorted(names)


def discover_class_names(root: Path) -> list[str]:
    env_names = os.getenv("PPE_CLASS_NAMES")
    if env_names:
        names = [normalize_label(name) for name in env_names.split(",") if name.strip()]
        if NUM_CLASSES > 0 and len(names) != NUM_CLASSES:
            raise ValueError(f"PPE_CLASS_NAMES must contain exactly {NUM_CLASSES} names")
        return names

    preferred_yaml = root / "data.yaml"
    if preferred_yaml.exists():
        names = load_names_from_yaml(preferred_yaml)
        if names and class_count_is_valid(names):
            return names

    for path in root.rglob("*"):
        if path.name in CLASS_FILE_NAMES:
            names = [normalize_label(line) for line in path.read_text().splitlines() if line.strip()]
            if class_count_is_valid(names):
                return names

    for path in root.rglob("*.yaml"):
        names = load_names_from_yaml(path)
        if names and class_count_is_valid(names):
            return names

    voc_names = discover_voc_class_names(root)
    if class_count_is_valid(voc_names):
        return voc_names

    if NUM_CLASSES <= 0:
        raise ValueError(
            "Could not discover class names. Add data.yaml with names or set PPE_CLASS_NAMES."
        )
    return [f"ppe_class_{index}" for index in range(NUM_CLASSES)]


def find_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_model_arch() -> str:
    model_path = Path(MODEL_ARCH)
    if model_path.exists():
        return str(model_path)

    source_relative_path = SCRIPT_DIR / MODEL_ARCH
    if source_relative_path.exists():
        return str(source_relative_path)

    if MODEL_ARCH.endswith(".pt") and MODEL_ARCH != "yolo26s.pt":
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_ARCH}")

    return MODEL_ARCH


def resolve_dataset_root() -> Path:
    if DATASET_ROOT.exists():
        return DATASET_ROOT

    if not KAGGLE_INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {DATASET_ROOT}. Kaggle input root also not found: {KAGGLE_INPUT_ROOT}"
        )

    print(f"Requested dataset root not found: {DATASET_ROOT}")
    print("Available Kaggle input folders:")
    for child in sorted(KAGGLE_INPUT_ROOT.iterdir()):
        if child.is_dir():
            image_count = len(find_images(child))
            print(f"- {child} images={image_count}")

    candidates = [child for child in KAGGLE_INPUT_ROOT.iterdir() if child.is_dir()]
    candidates = sorted(candidates, key=lambda path: len(find_images(path)), reverse=True)
    for candidate in candidates:
        if find_images(candidate):
            print(f"Auto-selected dataset root: {candidate}")
            return candidate

    raise FileNotFoundError(
        f"No image dataset found under {KAGGLE_INPUT_ROOT}. Check kernel dataset_sources."
    )


def parse_voc_xml(
    xml_path: Path,
    image_width: int,
    image_height: int,
    class_to_id: dict[str, int],
) -> list[str]:
    rows: list[str] = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        label_node = obj.find("name")
        box_node = obj.find("bndbox")
        if label_node is None or box_node is None:
            continue

        class_id = class_to_id.get(normalize_label(label_node.text or ""))
        if class_id is None:
            continue

        xmin = float(box_node.findtext("xmin", "0"))
        ymin = float(box_node.findtext("ymin", "0"))
        xmax = float(box_node.findtext("xmax", "0"))
        ymax = float(box_node.findtext("ymax", "0"))

        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        if width <= 0 or height <= 0:
            continue
        rows.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return rows


def parse_yolo_txt(label_path: Path, class_to_id: dict[str, int]) -> list[str]:
    rows: list[str] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        raw_class = parts[0]
        if raw_class.isdigit():
            class_id = int(raw_class)
            if not 0 <= class_id < len(class_to_id):
                continue
        else:
            mapped_class = class_to_id.get(normalize_label(raw_class))
            if mapped_class is None:
                continue
            class_id = mapped_class

        rows.append(" ".join([str(class_id), *parts[1:5]]))
    return rows


def matching_label_path(image_path: Path) -> Path | None:
    candidates = [
        image_path.with_suffix(".txt"),
        image_path.with_suffix(".xml"),
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        image_path.parent.parent / "annotations" / f"{image_path.stem}.xml",
        image_path.parent.parent / "Annotations" / f"{image_path.stem}.xml",
    ]
    return next((path for path in candidates if path.exists()), None)


def read_image_size(image_path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def convert_labels(image_path: Path, label_path: Path, class_to_id: dict[str, int]) -> list[str]:
    if label_path.suffix.lower() == ".txt":
        return parse_yolo_txt(label_path, class_to_id)
    if label_path.suffix.lower() == ".xml":
        width, height = read_image_size(image_path)
        return parse_voc_xml(label_path, width, height, class_to_id)
    return []


def write_data_yaml(path: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {PREPARED_DIR}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    path.write_text("\n".join(lines) + "\n")


def resolve_yaml_split_path(dataset_root: Path, yaml_path: Path, split: str) -> Path | None:
    payload = load_yaml(yaml_path)
    if not payload:
        return None

    value = payload.get(split)
    if not isinstance(value, str):
        return None

    base_path = Path(str(payload.get("path", yaml_path.parent)))
    if not base_path.is_absolute():
        base_path = (yaml_path.parent / base_path).resolve()

    split_path = Path(value)
    if split_path.is_absolute():
        return split_path

    candidates = [
        (base_path / split_path).resolve(),
        (yaml_path.parent / split_path).resolve(),
        (dataset_root / split_path).resolve(),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def existing_yolo_data_yaml(dataset_root: Path) -> Path | None:
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        return None

    names = load_names_from_yaml(yaml_path)
    train_path = resolve_yaml_split_path(dataset_root, yaml_path, "train")
    val_path = resolve_yaml_split_path(dataset_root, yaml_path, "val")
    if names and train_path and train_path.exists() and val_path and val_path.exists():
        print(f"Using existing YOLO data.yaml: {yaml_path}")
        print(f"Dataset classes ({len(names)}): {names}")
        return yaml_path
    return None


def prepare_dataset() -> Path:
    dataset_root = resolve_dataset_root()

    if USE_EXISTING_DATA_YAML:
        data_yaml = existing_yolo_data_yaml(dataset_root)
        if data_yaml is not None:
            return data_yaml

    if PREPARED_DIR.exists():
        shutil.rmtree(PREPARED_DIR)

    class_names = discover_class_names(dataset_root)
    class_to_id = {normalize_label(name): index for index, name in enumerate(class_names)}
    print(f"Using {len(class_names)} classes: {class_names}")

    for split in ["train", "val"]:
        (PREPARED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PREPARED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = find_images(dataset_root)
    random.Random(SEED).shuffle(images)
    train_cutoff = int(len(images) * 0.85)

    copied = 0
    skipped = 0
    for index, image_path in enumerate(images):
        label_path = matching_label_path(image_path)
        if label_path is None:
            skipped += 1
            continue

        rows = convert_labels(image_path, label_path, class_to_id)
        if not rows:
            skipped += 1
            continue

        split = "train" if index < train_cutoff else "val"
        target_image = PREPARED_DIR / "images" / split / image_path.name
        target_label = PREPARED_DIR / "labels" / split / f"{image_path.stem}.txt"

        shutil.copy2(image_path, target_image)
        target_label.write_text("\n".join(rows) + "\n")
        copied += 1

    if copied == 0:
        raise RuntimeError(
            "No usable labels were found. Inspect label format and set PPE_CLASS_NAMES if needed."
        )

    data_yaml = PREPARED_DIR / "data.yaml"
    write_data_yaml(data_yaml, class_names)
    print(f"Prepared dataset: copied={copied} skipped={skipped} path={PREPARED_DIR}")
    return data_yaml


def train_and_export(data_yaml: Path) -> None:
    from ultralytics import YOLO

    model_path = resolve_model_arch()
    print(f"Using model weights: {model_path}")
    model = YOLO(model_path)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": BATCH_SIZE,
        "project": str(RUNS_DIR),
        "name": RUN_NAME,
        "seed": SEED,
        "patience": PATIENCE,
        "cache": True,
        "lr0": LR0,
        "freeze": FREEZE,
    }
    if DEVICE:
        train_kwargs["device"] = DEVICE

    results = model.train(
        **train_kwargs,
    )

    best_model = Path(results.save_dir) / "weights" / "best.pt"
    trained_model = YOLO(str(best_model))
    trained_model.val(data=str(data_yaml), imgsz=IMG_SIZE, conf=CONF_THRESHOLD)
    exported_fp32 = trained_model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=12,
        simplify=True,
        dynamic=EXPORT_DYNAMIC,
        half=False,
    )
    shutil.copy2(exported_fp32, OUTPUT_MODEL)
    print(f"Exported ONNX model: {OUTPUT_MODEL}")

    if EXPORT_FP16:
        try:
            exported_fp16 = trained_model.export(
                format="onnx",
                imgsz=IMG_SIZE,
                opset=12,
                simplify=True,
                dynamic=EXPORT_DYNAMIC,
                half=True,
                device=DEVICE or 0,
            )
            shutil.copy2(exported_fp16, OUTPUT_FP16_MODEL)
            print(f"Exported FP16 ONNX model: {OUTPUT_FP16_MODEL}")
        except Exception as exc:
            print(f"FP16 ONNX export skipped: {exc}")


def main() -> None:
    ensure_ultralytics()
    data_yaml = prepare_dataset()
    train_and_export(data_yaml)


if __name__ == "__main__":
    main()
