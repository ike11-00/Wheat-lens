"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """A small valid JPEG."""
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (120, 90), (70, 120, 60)).save(path)
    return path


@pytest.fixture
def corrupt_image(tmp_path: Path) -> Path:
    """A file with an image extension that is not an image."""
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is definitely not a JPEG payload")
    return path


@pytest.fixture
def truncated_image(tmp_path: Path, sample_image: Path) -> Path:
    """A real JPEG cut in half."""
    data = sample_image.read_bytes()
    path = tmp_path / "truncated.jpg"
    path.write_bytes(data[: len(data) // 2])
    return path


# ---------------------------------------------------------------------------
# Pretrained weights and the test suite
# ---------------------------------------------------------------------------
# The production configuration builds MobileNetV2 with ImageNet weights, which
# Keras downloads on first use. The test suite must not inherit that
# dependency: it would make every model test require internet access, fail on
# an air-gapped machine or in CI without egress, and make results depend on a
# cache that may or may not be warm.
#
# The tests therefore ask for `weights=None` **explicitly**. This is not a
# silent substitution and not a weakening of the assertions:
#
#   * The tests that use it verify architectural properties - output shape,
#     probability normalisation, preprocessing inside the graph, augmentation
#     being inactive at inference, backbone freezing and unfreezing, the loss
#     and metrics, label ordering. Every one of these is determined by how the
#     graph is assembled, not by the numeric values in the backbone.
#   * The tests that check prediction plumbing, serialisation and the HTTP API
#     only need a model that loads and emits a valid distribution.
#
# The production path - MobileNetV2 *with* ImageNet weights - is covered
# separately by `test_imagenet_weights_path_builds` in tests/test_model.py,
# which runs whenever the weights are obtainable and skips when they are not.
#
# To make the full suite, including that test, run offline, warm the cache once
# on a machine with network access:
#
#     python -m src.model.build_model --prefetch
#
# and copy ~/.keras/models to the offline machine (about 9 MB).
# ---------------------------------------------------------------------------

#: Override that selects a randomly initialised backbone - no download needed.
NO_PRETRAINED_WEIGHTS = {"model": {"weights": None}}


def offline_safe_config(**overrides):
    """``load_config`` with pretrained weights explicitly switched off.

    Use for any test that constructs a model but does not specifically test
    pretrained-weight loading.
    """
    merged = {"model": {"weights": None}}
    for key, value in overrides.items():
        if key == "model" and isinstance(value, dict):
            merged["model"] = {**merged["model"], **value}
        else:
            merged[key] = value
    return load_config(overrides=merged)


@pytest.fixture(scope="session")
def imagenet_weights_available(config):
    """True when the backbone's ImageNet weights are cached or downloadable."""
    from src.model.build_model import imagenet_weights_available as probe

    return probe(config)
