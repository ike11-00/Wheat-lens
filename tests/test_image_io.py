"""Image validation and preprocessing."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.utils.image_io import (
    InvalidImageError,
    file_sha256,
    hamming_distance,
    inspect_image,
    iter_image_files,
    load_batch,
    load_image_array,
    open_image,
    perceptual_hash,
)


def test_valid_image_accepted(sample_image):
    info = inspect_image(sample_image)
    assert info.ok
    assert (info.width, info.height) == (120, 90)
    assert info.image_format == "JPEG"


def test_corrupt_image_rejected_without_raising(corrupt_image):
    info = inspect_image(corrupt_image)
    assert not info.ok
    assert "corrupt" in info.reason or "truncated" in info.reason


def test_truncated_image_rejected(truncated_image):
    info = inspect_image(truncated_image)
    assert not info.ok


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.jpg"
    path.write_bytes(b"")
    info = inspect_image(path)
    assert not info.ok and "empty" in info.reason


def test_too_small_image_rejected(tmp_path):
    path = tmp_path / "tiny.png"
    Image.new("RGB", (10, 10)).save(path)
    info = inspect_image(path, min_pixels=32)
    assert not info.ok and "too small" in info.reason


def test_unsupported_extension_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    info = inspect_image(path, extensions=[".jpg", ".png"])
    assert not info.ok and "unsupported" in info.reason


def test_missing_file_reported(tmp_path):
    info = inspect_image(tmp_path / "nope.jpg")
    assert not info.ok


def test_open_image_from_bytes_and_path(sample_image):
    from_path = open_image(sample_image)
    from_bytes = open_image(sample_image.read_bytes())
    assert from_path.size == from_bytes.size == (120, 90)
    assert from_path.mode == "RGB"


def test_open_image_converts_to_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (80, 80), 128).save(path)
    assert open_image(path).mode == "RGB"


def test_open_image_raises_user_safe_error(corrupt_image):
    with pytest.raises(InvalidImageError) as excinfo:
        open_image(corrupt_image)
    assert "could not be read" in str(excinfo.value)


def test_open_image_missing_path_raises():
    with pytest.raises(InvalidImageError):
        open_image("/definitely/not/here.jpg")


def test_load_image_array_shape_and_range(sample_image):
    array = load_image_array(sample_image, 224)
    assert array.shape == (224, 224, 3)
    assert array.dtype == np.float32
    # Preprocessing happens inside the model, so values stay in 0-255 here.
    assert 0.0 <= array.min() and array.max() <= 255.0


def test_load_batch_stacks(sample_image):
    batch = load_batch([sample_image, sample_image], 64)
    assert batch.shape == (2, 64, 64, 3)
    assert load_batch([], 64).shape == (0, 64, 64, 3)


def test_load_image_from_bytesio(sample_image):
    buffer = io.BytesIO(sample_image.read_bytes())
    assert load_image_array(buffer, 32).shape == (32, 32, 3)


def test_sha256_detects_identical_files(tmp_path, sample_image):
    copy = tmp_path / "copy.jpg"
    copy.write_bytes(sample_image.read_bytes())
    assert file_sha256(copy) == file_sha256(sample_image)


def test_perceptual_hash_similar_images_are_close(tmp_path, sample_image):
    resaved = tmp_path / "resaved.jpg"
    Image.open(sample_image).resize((240, 180)).save(resaved, quality=70)
    a, b = perceptual_hash(sample_image), perceptual_hash(resaved)
    if a is None or b is None:
        pytest.skip("imagehash is not installed")
    assert hamming_distance(a, b) <= 5


def test_iter_image_files_filters_by_extension(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.PNG").write_bytes(b"x")
    found = {p.name for p in iter_image_files(tmp_path, [".jpg", ".png"])}
    assert found == {"a.jpg", "c.PNG"}
