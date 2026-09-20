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
