"""Background augmentation: configuration, segmentation and leakage safety.

The leakage tests are the important ones. Augmentation generates derivatives of
training photographs; if one reached validation or test, the model would be
scored on a transformation of something it trained on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.augment_backgrounds import (AUGMENTED_PREFIX, AugmentationError,
                                          augment_training_split, composite,
                                          count_augmented, harvest_backgrounds,
                                          leaf_mask, procedural_background,
                                          remove_augmented, resolve_plan,
                                          verify_no_augmented_outside_train)
from src.utils.config import load_config
from src.utils.image_io import iter_image_files

STUDIO_CLASSES = ["Healthy", "Brown Rust", "Powdery Mildew"]
FIELD_CLASS = "Yellow Rust"


def _studio_image(size=(200, 200)):
    """A dark leaf shape on a pale background, like the WPLDD images."""
    array = np.full((size[1], size[0], 3), 235, dtype="uint8")
    array[40:160, 85:115] = (45, 95, 40)
    return Image.fromarray(array)


def _field_image(size=(200, 200), seed=0):
    """A textured image with no pale background, like the field photographs."""
    rng = np.random.default_rng(seed)
    base = np.array([110, 75, 50], dtype="float32")
    noise = rng.normal(0, 22, (size[1], size[0], 3))
    return Image.fromarray(np.clip(base + noise, 0, 255).astype("uint8"))


@pytest.fixture
def v2_config(tmp_path):
    """A config with tmp_path data directories and augmentation enabled."""
    return load_config(overrides={
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "train_dir": str(tmp_path / "train"),
            "validation_dir": str(tmp_path / "validation"),
            "test_dir": str(tmp_path / "test"),
            "results_dir": str(tmp_path / "results"),
            "models_dir": str(tmp_path / "models"),
        },
        "data": {
            "min_images_per_class": 1,
            "background_augmentation": {
                "enabled": True,
                "classes": STUDIO_CLASSES,
                "harvest_from": FIELD_CLASS,
                "background_sources": ["harvested", "procedural"],
                "copies_per_image": 1,
                "seed": 42,
            },
        },
    })


def _populate(config, per_class=4):
    """Create a realistic split: studio classes and one field class."""
    for split in ("train", "validation", "test"):
        root = config.path(f"{split}_dir")
        for spec in config.classes:
            directory = root / spec.directory
            directory.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                if spec.name == FIELD_CLASS:
                    _field_image(seed=hash((split, i)) % 1000).save(directory / f"f_{i}.jpg")
                else:
                    _studio_image().save(directory / f"s_{i}.jpg")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_disabled_by_default_in_the_shipped_config():
    """v1 behaviour must be unaffected by the feature existing."""
    plan = resolve_plan(load_config())
    assert plan.enabled is False
    assert plan.classes == []


def test_v2_config_enables_it():
    plan = resolve_plan(load_config("config/model_v2.yaml"))
    assert plan.enabled is True
    assert set(plan.classes) == set(STUDIO_CLASSES)
    assert plan.harvest_from == FIELD_CLASS


def test_unknown_class_rejected():
    config = load_config(overrides={"data": {"background_augmentation": {
        "enabled": True, "classes": ["Not A Class"]}}})
    with pytest.raises(AugmentationError, match="not configured"):
        resolve_plan(config)


def test_unknown_background_source_rejected():
    config = load_config(overrides={"data": {"background_augmentation": {
        "enabled": True, "classes": ["Healthy"], "background_sources": ["magic"]}}})
    with pytest.raises(AugmentationError, match="background_sources"):
        resolve_plan(config)


def test_harvesting_from_an_augmented_class_rejected():
    """Pasting a class's own backgrounds behind itself would not de-confound."""
    config = load_config(overrides={"data": {"background_augmentation": {
        "enabled": True, "classes": ["Healthy"], "harvest_from": "Healthy",
        "background_sources": ["harvested"]}}})
    with pytest.raises(AugmentationError, match="not break the correlation"):
        resolve_plan(config)


def test_harvested_without_source_rejected():
    config = load_config(overrides={"data": {"background_augmentation": {
        "enabled": True, "classes": ["Healthy"], "background_sources": ["harvested"]}}})
    with pytest.raises(AugmentationError, match="harvest_from"):
        resolve_plan(config)


# ---------------------------------------------------------------------------
# Segmentation and compositing
# ---------------------------------------------------------------------------

def test_leaf_mask_separates_leaf_from_pale_background():
    mask = leaf_mask(_studio_image(), 0.70, 0.35)
    assert mask[100, 100]          # leaf pixel
    assert not mask[10, 10]        # pale background
    assert 0.02 < mask.mean() < 0.5


def test_procedural_backgrounds_are_varied_and_deterministic():
    import random
    a = procedural_background(random.Random(1), 96)
    b = procedural_background(random.Random(1), 96)
    c = procedural_background(random.Random(2), 96)
    assert np.array_equal(np.asarray(a), np.asarray(b))     # same seed
    assert not np.array_equal(np.asarray(a), np.asarray(c))  # different seed


def test_composite_keeps_the_leaf_and_replaces_the_background(v2_config):
    import random
    plan = resolve_plan(v2_config)
    result = composite(_studio_image(), _field_image(), random.Random(0), plan)
    assert result is not None
    array = np.asarray(result, dtype="float32")
    # The pale studio background must be gone.
    pale = ((array.mean(axis=2) > 220)).mean()
    assert pale < 0.05, "pale studio background survived compositing"


def test_composite_skips_frames_with_almost_no_leaf(v2_config):
    import random
    plan = resolve_plan(v2_config)
    blank = Image.fromarray(np.full((200, 200, 3), 240, dtype="uint8"))
    assert composite(blank, _field_image(), random.Random(0), plan) is None


# ---------------------------------------------------------------------------
# Leakage - the critical guarantees
# ---------------------------------------------------------------------------

def test_augmentation_writes_only_into_train(v2_config):
    _populate(v2_config)
    before = {s: sum(1 for _ in iter_image_files(v2_config.path(f"{s}_dir"),
                                                 v2_config.supported_extensions))
              for s in ("validation", "test")}

    result = augment_training_split(v2_config)
    assert result.total > 0

    after = {s: sum(1 for _ in iter_image_files(v2_config.path(f"{s}_dir"),
                                                v2_config.supported_extensions))
             for s in ("validation", "test")}
    assert before == after, "augmentation touched validation or test"
    assert verify_no_augmented_outside_train(v2_config) == {"validation": 0, "test": 0}


def test_no_augmented_file_appears_outside_train(v2_config):
    _populate(v2_config)
    augment_training_split(v2_config)
    for split in ("validation", "test"):
        for path in iter_image_files(v2_config.path(f"{split}_dir"),
                                     v2_config.supported_extensions):
            assert not path.name.startswith(f"{AUGMENTED_PREFIX}__"), path


def test_backgrounds_are_harvested_only_from_train(v2_config, monkeypatch):
    """Harvesting from validation or test would import their information."""
    _populate(v2_config)
    opened = []
    import src.data.augment_backgrounds as module
    original = module.open_image

    def spy(path):
        opened.append(Path(path))
        return original(path)

    monkeypatch.setattr(module, "open_image", spy)
    harvest_backgrounds(v2_config, resolve_plan(v2_config))

    assert opened, "no images were opened while harvesting"
    train_root = v2_config.path("train_dir").resolve()
    for path in opened:
        assert train_root in path.resolve().parents, f"harvested from outside train: {path}"


def test_augmentation_refuses_to_run_before_the_split(v2_config):
    """It is a post-split step; running it first is what would cause leakage."""
    with pytest.raises(AugmentationError, match="prepare_dataset"):
        augment_training_split(v2_config)


# ---------------------------------------------------------------------------
# Reproducibility and provenance
# ---------------------------------------------------------------------------

def test_augmentation_is_reproducible(v2_config, tmp_path):
    _populate(v2_config)
    first = augment_training_split(v2_config)
    digests_a = {
        p.name: p.read_bytes()
        for p in iter_image_files(v2_config.path("train_dir"), v2_config.supported_extensions)
        if p.name.startswith(f"{AUGMENTED_PREFIX}__")
    }
    remove_augmented(v2_config)
    second = augment_training_split(v2_config)
    digests_b = {
        p.name: p.read_bytes()
        for p in iter_image_files(v2_config.path("train_dir"), v2_config.supported_extensions)
        if p.name.startswith(f"{AUGMENTED_PREFIX}__")
    }
    assert first.total == second.total
    assert digests_a.keys() == digests_b.keys()
    assert digests_a == digests_b, "same seed produced different images"


def test_original_filenames_remain_traceable(v2_config):
    _populate(v2_config)
    result = augment_training_split(v2_config)
    for entry in result.manifest:
        augmented = Path(entry["augmented"]).name
        original = Path(entry["original"]).stem
        assert augmented.startswith(f"{AUGMENTED_PREFIX}__")
        assert original in augmented, "augmented filename does not identify its original"


def test_remove_augmented_deletes_only_generated_files(v2_config):
    _populate(v2_config)
    originals = {p.name for p in iter_image_files(v2_config.path("train_dir"),
                                                  v2_config.supported_extensions)}
    augment_training_split(v2_config)
    removed = remove_augmented(v2_config)
    assert removed > 0
    remaining = {p.name for p in iter_image_files(v2_config.path("train_dir"),
                                                  v2_config.supported_extensions)}
    assert remaining == originals, "an original photograph was deleted"


def test_disabled_augmentation_generates_nothing(v2_config):
    _populate(v2_config)
    plan = resolve_plan(v2_config)
    plan.enabled = False
    result = augment_training_split(v2_config, plan=plan)
    assert result.total == 0
    assert count_augmented(v2_config, "train") == {n: 0 for n in v2_config.class_names}


def test_yellow_rust_is_not_augmented(v2_config):
    """The field class already has varied backgrounds; augmenting it is pointless."""
    _populate(v2_config)
    augment_training_split(v2_config)
    counts = count_augmented(v2_config, "train")
    assert counts[FIELD_CLASS] == 0
    assert all(counts[c] > 0 for c in STUDIO_CLASSES)
