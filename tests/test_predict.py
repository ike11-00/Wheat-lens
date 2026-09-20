"""Prediction engine behaviour, including the no-model case."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.model.build_model import build_model, compile_model
from src.model.predict import (ModelNotAvailableError, Predictor,
                               format_human_readable)
from src.utils.config import load_config
from src.utils.image_io import InvalidImageError


@pytest.fixture(scope="module")
def trained_stub(tmp_path_factory):
    """An untrained-but-saved model, enough to exercise the prediction path.

    The weights are random; only the plumbing is under test here. The
    end-to-end pipeline test covers a model that has actually learned.
    """
    directory = tmp_path_factory.mktemp("stub_model") / "v1"
    directory.mkdir(parents=True)
    config = load_config(overrides={"data": {"image_size": 96}})
    model = compile_model(build_model(config), config)
    model.save(directory / "model.keras")
    (directory / "labels.json").write_text(json.dumps({
        "class_names": config.class_names,
        "image_size": 96,
        "confidence_threshold": 0.6,
        "model_version": "stub",
        "architecture": "MobileNetV2",
        "trained_utc": "2000-01-01T00:00:00Z",
    }), encoding="utf-8")
    return directory / "model.keras"


def test_reports_unavailable_without_a_model(tmp_path):
    config = load_config(overrides={"paths": {"models_dir": str(tmp_path)}})
    predictor = Predictor(config)
    assert not predictor.is_available()
    with pytest.raises(ModelNotAvailableError) as excinfo:
        predictor.load()
    assert "src.model.train" in str(excinfo.value)


def test_info_available_without_a_model(tmp_path):
    config = load_config(overrides={"paths": {"models_dir": str(tmp_path)}})
    info = Predictor(config).info().to_dict()
    assert info["class_names"] == load_config().class_names
    assert info["confidence_threshold"] > 0


def test_labels_json_drives_class_order(trained_stub):
    predictor = Predictor(load_config(), model_path=trained_stub)
    assert predictor.model_version == "stub"
    assert predictor.image_size == 96
    assert predictor.class_names == load_config().class_names


def test_prediction_result_shape(trained_stub, sample_image):
    predictor = Predictor(load_config(), model_path=trained_stub)
    result = predictor.predict(sample_image)

    assert set(result) >= {"predicted_class", "confidence", "probabilities",
                           "low_confidence", "uncertain", "uncertainty", "disclaimer"}
    assert result["predicted_class"] in predictor.class_names
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == set(predictor.class_names)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
    # The reported confidence must be the probability of the reported class.
    assert result["confidence"] == pytest.approx(
        result["probabilities"][result["predicted_class"]])
    assert result["confidence"] == pytest.approx(max(result["probabilities"].values()))


def test_predictions_are_deterministic(trained_stub, sample_image):
    predictor = Predictor(load_config(), model_path=trained_stub)
    first = predictor.predict(sample_image)
    second = predictor.predict(sample_image)
    assert first["probabilities"] == second["probabilities"]


def test_batch_matches_single(trained_stub, tmp_path):
    predictor = Predictor(load_config(), model_path=trained_stub)
    paths = []
    for index in range(3):
        path = tmp_path / f"img_{index}.png"
        Image.new("RGB", (100, 100), (index * 60, 120, 80)).save(path)
        paths.append(path)
    batched = predictor.predict_many(paths)
    for path, batch_result in zip(paths, batched):
        single = predictor.predict(path)
        assert single["predicted_class"] == batch_result["predicted_class"]
        assert single["confidence"] == pytest.approx(batch_result["confidence"], abs=1e-6)


def test_low_confidence_flag_follows_threshold(trained_stub, sample_image):
    predictor = Predictor(load_config(), model_path=trained_stub)
    predictor.confidence_threshold = 0.0
    assert predictor.predict(sample_image)["low_confidence"] is False
    predictor.confidence_threshold = 1.0
    assert predictor.predict(sample_image)["low_confidence"] is True


def test_uncertainty_signals_are_computed(trained_stub, sample_image):
    predictor = Predictor(load_config(), model_path=trained_stub)
    uncertainty = predictor.predict(sample_image)["uncertainty"]
    assert 0.0 <= uncertainty["normalised_entropy"] <= 1.0
    assert 0.0 <= uncertainty["margin"] <= 1.0
    assert "limitation" in uncertainty


def test_uniform_distribution_is_flagged_uncertain(trained_stub):
    predictor = Predictor(load_config(), model_path=trained_stub)
    uniform = np.full(len(predictor.class_names), 1.0 / len(predictor.class_names))
    assert predictor._uncertainty(uniform)["uncertain"] is True


def test_confident_distribution_is_not_flagged(trained_stub):
    predictor = Predictor(load_config(), model_path=trained_stub)
    confident = np.zeros(len(predictor.class_names))
    confident[0] = 0.97
    confident[1:] = 0.03 / max(1, len(confident) - 1)
    assert predictor._uncertainty(confident)["uncertain"] is False


def test_invalid_image_raises_user_safe_error(trained_stub, corrupt_image):
    predictor = Predictor(load_config(), model_path=trained_stub)
    with pytest.raises(InvalidImageError):
        predictor.predict(corrupt_image)


def test_class_count_mismatch_is_detected(trained_stub, tmp_path):
    """A model whose head does not match the label list must be refused."""
    config = load_config(overrides={
        "classes": [{"name": n} for n in ["A", "B", "C"]],
        "paths": {"models_dir": str(tmp_path)},
    })
    predictor = Predictor(config, model_path=trained_stub)
    predictor.class_names = ["A", "B", "C"]  # 3 names vs a 5-output model
    with pytest.raises(ModelNotAvailableError) as excinfo:
        predictor.load()
    assert "outputs" in str(excinfo.value)


def test_human_readable_output_contains_every_class(trained_stub, sample_image):
    predictor = Predictor(load_config(), model_path=trained_stub)
    text = format_human_readable(predictor.predict(sample_image))
    assert "Prediction:" in text and "Confidence:" in text
    for name in predictor.class_names:
        assert name in text
