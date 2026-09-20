"""Configuration loading and class handling."""

from __future__ import annotations

import pytest

from src.utils.config import ConfigError, load_config

REQUIRED_CLASSES = ["Healthy", "Yellow Rust", "Karnal Bunt", "Brown Rust", "Powdery Mildew"]


def test_required_classes_present_and_ordered(config):
    assert config.class_names == REQUIRED_CLASSES
    assert config.num_classes == 5


def test_class_directories_are_filesystem_safe(config):
    for directory in config.class_dirs:
        assert " " not in directory
        assert directory


def test_aliases_resolve_to_display_names(config):
    assert config.display_name("Yellow_Rust") == "Yellow Rust"
    assert config.display_name("stripe_rust") == "Yellow Rust"
    assert config.display_name("leaf rust") == "Brown Rust"
    assert config.display_name("Unknown_Thing") == "Unknown_Thing"


def test_class_index_round_trip(config):
    for index, name in enumerate(config.class_names):
        assert config.class_index(name) == index
    with pytest.raises(ConfigError):
        config.class_index("Not A Class")


def test_split_fractions_sum_to_one(config):
    fractions = config.split_fractions()
    assert abs(sum(fractions.values()) - 1.0) < 1e-9
    assert all(value > 0 for value in fractions.values())


def test_paths_resolve_to_absolute(config):
    for key in ("raw_dir", "train_dir", "validation_dir", "test_dir", "models_dir", "results_dir"):
        assert config.path(key).is_absolute()


def test_scalars_have_expected_types(config):
    assert isinstance(config.image_size, int) and config.image_size >= 32
    assert isinstance(config.batch_size, int) and config.batch_size >= 1
    assert isinstance(config.seed, int)
    assert 0.0 < config.confidence_threshold <= 1.0


def test_variant_config_deep_merges_without_losing_classes():
    v2 = load_config("config/model_v2.yaml")
    assert v2.model_version == "v2"
    assert v2.class_names == REQUIRED_CLASSES          # inherited from the base file
    assert v2.get("model", "dropout") == 0.3           # overridden
    assert v2.path("raw_dir") == load_config().path("raw_dir")


def test_overrides_are_applied():
    config = load_config(overrides={"model": {"version": "test-version"}})
    assert config.model_version == "test-version"
    assert config.model_file.name == "model.keras"


def test_extensible_to_a_sixth_class():
    """Adding a class must not require code changes."""
    base = load_config()
    extended = load_config(overrides={
        "classes": [
            *[{"name": c.name, "directory": c.directory, "aliases": c.aliases}
              for c in base.classes],
            {"name": "Septoria", "directory": "Septoria", "aliases": ["septoria"]},
        ]
    })
    assert extended.num_classes == 6
    assert extended.class_names[-1] == "Septoria"
    assert extended.class_index("septoria") == 5


def test_duplicate_class_names_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides={"classes": [{"name": "A"}, {"name": "A"}]})


def test_invalid_image_size_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides={"data": {"image_size": 8}})
