"""Web application: routing, upload validation, error handling."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from app.app import create_app
from src.model.build_model import build_model, compile_model
from src.utils.config import load_config


def _jpeg_bytes(size=(120, 120), colour=(80, 130, 70)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    """App configured to look for models in an empty directory."""
    import src.utils.config as config_module

    original = config_module.load_config

    def patched(path=None, overrides=None):
        merged = {"paths": {"models_dir": str(tmp_path / "empty_models")}}
        if overrides:
            merged.update(overrides)
        return original(path, overrides=merged)

    monkeypatch.setattr("app.app.load_config", patched)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(scope="module")
def stub_models_dir(tmp_path_factory):
    """A saved (untrained) model so prediction endpoints can be exercised."""
    root = tmp_path_factory.mktemp("app_models")
    version_dir = root / "v1"
    version_dir.mkdir()
    config = load_config(overrides={"data": {"image_size": 96}})
    compile_model(build_model(config), config).save(version_dir / "model.keras")
    (version_dir / "labels.json").write_text(json.dumps({
        "class_names": config.class_names,
        "image_size": 96,
        "confidence_threshold": 0.6,
        "model_version": "v1",
        "architecture": "MobileNetV2",
        "trained_utc": "2000-01-01T00:00:00Z",
    }), encoding="utf-8")
    return root


@pytest.fixture
def client_with_model(stub_models_dir, monkeypatch):
    import src.utils.config as config_module

    original = config_module.load_config

    def patched(path=None, overrides=None):
        merged = {"paths": {"models_dir": str(stub_models_dir)}}
        if overrides:
            merged.update(overrides)
        return original(path, overrides=merged)

    monkeypatch.setattr("app.app.load_config", patched)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# --------------------------------------------------------------------------
# Pages and status
# --------------------------------------------------------------------------

def test_index_renders_with_all_classes(client_without_model):
    response = client_without_model.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    for name in load_config().class_names:
        assert name in body
    assert "LEAF" in body and "PREDICT" in body
    assert "experimental machine-learning prototype" in body


def test_index_warns_when_no_model(client_without_model):
    body = client_without_model.get("/").get_data(as_text=True)
    assert "No trained model is available yet" in body


def test_status_reports_missing_model(client_without_model):
    payload = client_without_model.get("/api/status").get_json()
    assert payload["model_available"] is False
    assert payload["class_names"] == load_config().class_names
    assert "src.model.train" in payload["message"]


def test_status_reports_loaded_model(client_with_model):
    payload = client_with_model.get("/api/status").get_json()
    assert payload["model_available"] is True
    assert payload["image_size"] == 96
    assert payload["class_names"] == load_config().class_names


# --------------------------------------------------------------------------
# Prediction endpoint
# --------------------------------------------------------------------------

def test_predict_without_model_returns_503(client_without_model):
    response = client_without_model.post(
        "/api/predict",
        data={"image": (io.BytesIO(_jpeg_bytes()), "leaf.jpg")},
        content_type="multipart/form-data")
    assert response.status_code == 503
    assert response.get_json()["error"] is True


def test_predict_returns_full_distribution(client_with_model):
    response = client_with_model.post(
        "/api/predict",
        data={"image": (io.BytesIO(_jpeg_bytes()), "leaf.jpg")},
        content_type="multipart/form-data")
    assert response.status_code == 200
    payload = response.get_json()

    names = load_config().class_names
    assert payload["predicted_class"] in names
    assert 0.0 <= payload["confidence"] <= 1.0
    assert set(payload["probabilities"]) == set(names)
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-5
    assert [row["class"] for row in payload["probabilities_ordered"]] == names
    ranked = [row["probability"] for row in payload["probabilities_sorted"]]
    assert ranked == sorted(ranked, reverse=True)
    assert payload["filename"] == "leaf.jpg"
    assert "disclaimer" in payload


@pytest.mark.parametrize("extension", ["png", "webp", "bmp"])
def test_other_supported_formats_accepted(client_with_model, extension):
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), (90, 140, 70)).save(buffer, format=extension.upper())
    response = client_with_model.post(
        "/api/predict",
        data={"image": (io.BytesIO(buffer.getvalue()), f"leaf.{extension}")},
        content_type="multipart/form-data")
    assert response.status_code == 200, response.get_json()


def test_unsupported_extension_rejected(client_with_model):
    response = client_with_model.post(
        "/api/predict",
        data={"image": (io.BytesIO(b"GIF89a fake"), "animation.gif")},
        content_type="multipart/form-data")
    assert response.status_code == 415
    assert "not supported" in response.get_json()["message"]


def test_corrupt_image_rejected_gracefully(client_with_model):
    response = client_with_model.post(
        "/api/predict",
        data={"image": (io.BytesIO(b"definitely not an image"), "leaf.jpg")},
        content_type="multipart/form-data")
    assert response.status_code == 400
    message = response.get_json()["message"]
    assert "could not be read as an image" in message
    assert "Traceback" not in message


def test_empty_file_rejected(client_with_model):
    response = client_with_model.post(
        "/api/predict",
        data={"image": (io.BytesIO(b""), "leaf.jpg")},
        content_type="multipart/form-data")
    assert response.status_code == 400


def test_missing_field_rejected(client_with_model):
    response = client_with_model.post("/api/predict", data={},
                                      content_type="multipart/form-data")
    assert response.status_code == 400
    assert "No image" in response.get_json()["message"]


def test_tiny_image_rejected(client_with_model):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    response = client_with_model.post(
        "/api/predict",
        data={"image": (io.BytesIO(buffer.getvalue()), "tiny.png")},
        content_type="multipart/form-data")
    assert response.status_code == 400
    assert "too small" in response.get_json()["message"]


def test_oversized_upload_rejected(client_with_model):
    payload = b"\xff\xd8\xff" + b"0" * (11 * 1024 * 1024)
    response = client_with_model.post(
        "/api/predict",
        data={"image": (io.BytesIO(payload), "huge.jpg")},
        content_type="multipart/form-data")
    assert response.status_code == 413
    assert "MB limit" in response.get_json()["message"]


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------

def test_unknown_api_route_returns_json(client_with_model):
    response = client_with_model.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["error"] is True


def test_static_assets_are_served(client_without_model):
    for asset in ("style.css", "script.js"):
        response = client_without_model.get(f"/static/{asset}")
        assert response.status_code == 200
        assert len(response.data) > 100


def test_hidden_attribute_is_not_overridden_by_display_rules(client_without_model):
    """Regression: `.loading { display: flex }` used to beat `[hidden]`.

    The interface shows and hides every section with the `hidden` attribute.
    Author-level `display` rules outrank the user-agent `[hidden]` rule, so an
    explicit reset is required or the spinner stays on screen after a result.
    """
    css = client_without_model.get("/static/style.css").get_data(as_text=True)
    assert "[hidden]" in css and "display: none !important" in css


def test_every_toggled_element_exists_in_the_page(client_without_model):
    """The JS toggles these by id; a rename would silently break the flow."""
    body = client_without_model.get("/").get_data(as_text=True)
    for element_id in ("dropzone", "file-input", "preview-area", "preview-image",
                       "predict-button", "reset-button", "loading", "error-box",
                       "result-card", "result-class", "result-confidence",
                       "low-confidence", "uncertain-box", "probability-list"):
        assert f'id="{element_id}"' in body, f"missing element: {element_id}"
