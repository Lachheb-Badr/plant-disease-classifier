"""API tests with a stand-in predictor (no model or checkpoint needed)."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, UnidentifiedImageError

from src.api.main import app


class FakePredictor:
    name = "fake_model"
    classes = ["Healthy", "Powdery", "Rust"]

    def predict(self, image_bytes: bytes) -> dict:
        Image.open(io.BytesIO(image_bytes)).verify()  # raises on invalid images, like the real one
        return {
            "label": "Rust",
            "confidence": 0.9,
            "probabilities": {"Healthy": 0.05, "Powdery": 0.05, "Rust": 0.9},
            "model_name": self.name,
            "inference_time_ms": 1.0,
        }


def jpeg_bytes(size=(60, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 160, 60)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as c:
        c.app.state.predictor = FakePredictor()
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["classes"] == ["Healthy", "Powdery", "Rust"]


def test_predict_valid_image(client):
    r = client.post("/predict", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "Rust"
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-6


def test_predict_not_an_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_predict_empty_file(client):
    r = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_predict_missing_file(client):
    assert client.post("/predict").status_code == 422


def test_model_not_loaded():
    with TestClient(app) as c:
        c.app.state.predictor = None
        assert c.get("/health").json()["status"] == "degraded"
        r = c.post("/predict", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")})
        assert r.status_code == 503


def test_website_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "LeafScan" in r.text
    assert client.get("/health").status_code == 200  # API routes still win over the site


def test_website_assets_are_served(client):
    assert client.get("/style.css").status_code == 200
    assert client.get("/app.js").status_code == 200
