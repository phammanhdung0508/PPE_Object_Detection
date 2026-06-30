import pytest
from unittest.mock import patch, MagicMock

# Mock the model BEFORE importing app
with patch("app.model.YoloOnnxModel.load"), \
     patch("app.model.YoloOnnxModel.warmup"), \
     patch("app.model.ort.InferenceSession"):
    from app.main import app

from fastapi.testclient import TestClient

def test_demo_ui_returns_html():
    # Patch the global model instance in app.main
    with patch("app.main.model") as mock_model:
        with TestClient(app) as client:
            response = client.get("/demo")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            assert "PPE SOC gRPC-Web Surveillance Dashboard" in response.text
            assert "http://localhost:8080/protos.DetectorService/BatchDetect" in response.text


def test_root_redirects_to_demo():
    with patch("app.main.model") as mock_model:
        with TestClient(app) as client:
            response = client.get("/", follow_redirects=False)
            assert response.status_code == 307
            assert response.headers["location"] == "/demo"
