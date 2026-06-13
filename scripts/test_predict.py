import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a test image to the PPE API.")
    parser.add_argument("image", help="Path to test image")
    parser.add_argument("--url", default="http://localhost:8000/predict", help="Prediction endpoint")
    parser.add_argument("--confidence", default="0.25", help="Confidence threshold")
    parser.add_argument("--field-name", default="file", help="Multipart upload field name")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with image_path.open("rb") as image_file:
        response = requests.post(
            args.url,
            files={args.field_name: (image_path.name, image_file, "application/octet-stream")},
            data={"confidence_threshold": args.confidence},
            timeout=30,
        )

    print(f"status_code: {response.status_code}")
    print(response.text)


if __name__ == "__main__":
    main()
