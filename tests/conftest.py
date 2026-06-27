import pytest
from unittest.mock import MagicMock

@pytest.fixture(autouse=True)
def mock_model_load_and_warmup(monkeypatch):
    monkeypatch.setattr("app.model.model.load", lambda: None)
    monkeypatch.setattr("app.model.model.warmup", lambda: None)
    monkeypatch.setattr("app.model.model.is_loaded", lambda: True)

    # Mock inference_service as well if needed
    monkeypatch.setattr("inference_service.detector.ObjectDetector.load", lambda self: None)
    monkeypatch.setattr("inference_service.detector_accelerated.AcceleratedObjectDetector.load", lambda self: None)
