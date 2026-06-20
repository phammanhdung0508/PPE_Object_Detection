from pathlib import Path

from locust import HttpUser, between, task

SAMPLE_IMAGE = Path(__file__).with_name("sample.jpg")


class PPEApiUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="GET /health")

    @task(5)
    def predict(self) -> None:
        if not SAMPLE_IMAGE.exists():
            raise FileNotFoundError(
                f"Benchmark image not found: {SAMPLE_IMAGE}. Add a test image named sample.jpg."
            )

        with SAMPLE_IMAGE.open("rb") as image_file:
            self.client.post(
                "/predict",
                name="POST /predict",
                files={"file": (SAMPLE_IMAGE.name, image_file, "image/jpeg")},
                data={"confidence_threshold": "0.25"},
                timeout=30,
            )
