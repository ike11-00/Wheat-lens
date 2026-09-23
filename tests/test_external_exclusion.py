"""The external test set must be invisible to everything except evaluation.

These images are the only measurement in this project that has tracked
real-world behaviour. If any part of the training pipeline could read them,
the number they produce would be worthless.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.data.augment_backgrounds import AugmentationError, augment_training_split
from src.data.prepare_dataset import discover_raw_items, prepare
from src.data.validate_dataset import validate_dataset
from src.model.datasets import list_split_files
from src.testing.external_test import external_dir, read_manifest, write_template
from src.utils.config import load_config
from src.utils.image_io import iter_image_files


@pytest.fixture
def isolated(tmp_path):
    """Config whose data directories are all inside tmp_path."""
    return load_config(overrides={"paths": {
        "raw_dir": str(tmp_path / "raw"),
        "train_dir": str(tmp_path / "train"),
        "validation_dir": str(tmp_path / "validation"),
        "test_dir": str(tmp_path / "test"),
        "results_dir": str(tmp_path / "results"),
        "models_dir": str(tmp_path / "models"),
    }, "data": {"min_images_per_class": 1}})


def test_external_dir_is_outside_every_configured_data_path(isolated):
    """It must not sit under raw, train, validation or test."""
    external = external_dir(load_config()).resolve()
    real = load_config()
    for key in ("raw_dir", "train_dir", "validation_dir", "test_dir"):
        configured = real.path(key).resolve()
        assert configured != external
        assert configured not in external.parents
        assert external not in configured.parents


def test_prepare_dataset_never_reads_the_external_directory(isolated, tmp_path, monkeypatch):
    """Spy on every image opened during preparation."""
    raw = tmp_path / "raw"
    for spec in isolated.classes:
        d = raw / spec.directory
        d.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (64, 64), (60, 120, 60)).save(d / f"{i}.jpg")

    opened = []
    import src.data.prepare_dataset as module
    original = module.inspect_image

    def spy(path, **kwargs):
        opened.append(Path(path).resolve())
        return original(path, **kwargs)

    monkeypatch.setattr(module, "inspect_image", spy)
    prepare(isolated, raw_root=raw)

    external = external_dir(load_config()).resolve()
    assert opened, "preparation opened no images at all"
    for path in opened:
        assert external not in path.parents, f"preparation read the external set: {path}"


def test_training_split_listing_excludes_external(isolated, tmp_path):
    """The training loader only ever sees data/train."""
    for split in ("train", "validation", "test"):
        for spec in isolated.classes:
            d = isolated.path(f"{split}_dir") / spec.directory
            d.mkdir(parents=True)
            Image.new("RGB", (64, 64)).save(d / "a.jpg")

    external = external_dir(load_config()).resolve()
    for split in ("train", "validation", "test"):
        paths, _ = list_split_files(isolated, isolated.path(f"{split}_dir"))
        for path in paths:
            assert external not in Path(path).resolve().parents


def test_augmentation_never_writes_into_external(isolated, tmp_path):
    """Augmentation targets data/train only; external must be untouched."""
    external = external_dir(load_config())
    before = sorted(p.name for p in external.iterdir()) if external.exists() else []

    config = load_config(overrides={
        "paths": {k: str(tmp_path / k.replace("_dir", "")) for k in
                  ("raw_dir", "train_dir", "validation_dir", "test_dir",
                   "results_dir", "models_dir")},
        "data": {"background_augmentation": {
            "enabled": True, "classes": ["Healthy"], "background_sources": ["procedural"]}},
    })
    # No split exists, so it must refuse rather than wander elsewhere.
    with pytest.raises(AugmentationError, match="prepare_dataset"):
        augment_training_split(config)

    after = sorted(p.name for p in external.iterdir()) if external.exists() else []
    assert before == after, "augmentation altered the external directory"


def test_manifest_template_round_trips(tmp_path, monkeypatch):
    """The template lists present images and preserves labels already entered."""
    import src.testing.external_test as module
    fake = tmp_path / "external_test"
    fake.mkdir()
    monkeypatch.setattr(module, "external_dir", lambda config: fake)

    config = load_config()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (80, 80), (90, 140, 70)).save(fake / name)

    write_template(config)
    text = (fake / "manifest.csv").read_text()
    assert "a.jpg" in text and "b.jpg" in text

    # Fill one label, regenerate, and confirm it survived.
    rows = text.replace("a.jpg,,", "a.jpg,Brown Rust,")
    (fake / "manifest.csv").write_text(rows)
    write_template(config)
    cases = read_manifest(config)
    labelled = {c["file"]: c["true_class"] for c in cases}
    assert labelled["a.jpg"] == "Brown Rust", "an entered label was lost on regeneration"
    assert labelled["b.jpg"] is None


def test_invalid_true_class_is_rejected_not_guessed(tmp_path, monkeypatch):
    import src.testing.external_test as module
    fake = tmp_path / "external_test"
    fake.mkdir()
    monkeypatch.setattr(module, "external_dir", lambda config: fake)
    Image.new("RGB", (80, 80)).save(fake / "x.jpg")
    (fake / "manifest.csv").write_text(
        "file,true_class,lighting,background,distance,angle,quality,notes\n"
        "x.jpg,Septoria,,,,,,\n")

    cases = read_manifest(load_config())
    assert cases[0]["true_class"] is None, "an unconfigured class was silently accepted"


def test_aliases_are_accepted_in_the_manifest(tmp_path, monkeypatch):
    """'stripe rust' should resolve to Yellow Rust, as everywhere else."""
    import src.testing.external_test as module
    fake = tmp_path / "external_test"
    fake.mkdir()
    monkeypatch.setattr(module, "external_dir", lambda config: fake)
    Image.new("RGB", (80, 80)).save(fake / "y.jpg")
    (fake / "manifest.csv").write_text(
        "file,true_class,lighting,background,distance,angle,quality,notes\n"
        "y.jpg,stripe rust,,,,,,\n")

    cases = read_manifest(load_config())
    assert cases[0]["true_class"] == "Yellow Rust"
