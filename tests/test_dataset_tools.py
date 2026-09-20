"""Dataset validation and leakage-free preparation."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.data.prepare_dataset import prepare
from src.data.validate_dataset import _cross_split_leakage, validate_dataset
from src.utils.config import load_config


def _make_dataset(root: Path, class_dirs, per_class=8, size=(64, 64)):
    for index, directory in enumerate(class_dirs):
        folder = root / directory
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            colour = ((index * 40) % 255, (i * 25) % 255, 100)
            Image.new("RGB", size, colour).save(folder / f"{directory}_{i}.jpg")


@pytest.fixture
def scratch_config(tmp_path):
    """A config whose data paths point inside tmp_path."""
    return load_config(overrides={
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "train_dir": str(tmp_path / "train"),
            "validation_dir": str(tmp_path / "validation"),
            "test_dir": str(tmp_path / "test"),
            "results_dir": str(tmp_path / "results"),
            "models_dir": str(tmp_path / "models"),
        },
        "data": {"min_images_per_class": 1},
    })


def test_missing_root_reported(scratch_config):
    report = validate_dataset(scratch_config, Path(scratch_config.path("raw_dir")))
    assert not report.usable
    assert any("does not exist" in error for error in report.errors)


def test_missing_classes_are_named(scratch_config, tmp_path):
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs[:2])
    report = validate_dataset(scratch_config, raw)
    assert not report.usable
    assert set(report.classes_missing) == set(scratch_config.class_names[2:])
    assert set(report.classes_present) == set(scratch_config.class_names[:2])


def test_full_dataset_is_usable_and_counted(scratch_config, tmp_path):
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs, per_class=6)
    report = validate_dataset(scratch_config, raw)
    n_classes = scratch_config.num_classes
    assert report.usable, report.errors
    assert report.total_valid_images == 6 * n_classes
    assert all(entry.valid_images == 6 for entry in report.per_class)
    expected_share = 100.0 / n_classes
    assert all(abs(entry.percentage_of_valid - expected_share) < 0.01
               for entry in report.per_class)


def test_corrupt_and_unsupported_files_are_reported(scratch_config, tmp_path):
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs, per_class=4)
    (raw / scratch_config.class_dirs[0] / "broken.jpg").write_bytes(b"not an image")
    (raw / scratch_config.class_dirs[0] / "notes.txt").write_text("ignore me")
    report = validate_dataset(scratch_config, raw)
    reasons = " ".join(entry["reason"] for entry in report.invalid_files)
    assert "corrupt" in reasons or "truncated" in reasons
    assert "unsupported" in reasons


def test_exact_duplicates_detected(scratch_config, tmp_path):
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs, per_class=4)
    source = raw / scratch_config.class_dirs[0] / f"{scratch_config.class_dirs[0]}_0.jpg"
    (raw / scratch_config.class_dirs[0] / "copy.jpg").write_bytes(source.read_bytes())
    report = validate_dataset(scratch_config, raw)
    assert report.exact_duplicate_groups


def test_imbalance_is_flagged(scratch_config, tmp_path):
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs[:1], per_class=40)
    _make_dataset(raw, scratch_config.class_dirs[1:], per_class=4)
    report = validate_dataset(scratch_config, raw)
    assert report.imbalance_ratio == 10.0
    assert any("imbalance" in warning for warning in report.warnings)


def test_alias_directory_is_recognised(scratch_config, tmp_path):
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs, per_class=3)
    # Rename Brown_Rust -> leaf_rust (a configured alias)
    (raw / "Brown_Rust").rename(raw / "leaf_rust")
    report = validate_dataset(scratch_config, raw)
    assert any("alias" in warning for warning in report.warnings)
    # prepare_dataset resolves the alias, so preparation still works
    result = prepare(scratch_config, raw_root=raw)
    assert sum(result.counts["Brown Rust"].values()) == 3


def test_preparation_splits_and_leaves_no_leakage(scratch_config, tmp_path):
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs, per_class=12)
    result = prepare(scratch_config, raw_root=raw)

    assert sum(result.totals.values()) > 0
    for class_name in scratch_config.class_names:
        assert sum(result.counts[class_name].values()) > 0

    reports = {
        split: validate_dataset(scratch_config, scratch_config.path(f"{split}_dir"),
                                check_duplicates=False)
        for split in ("train", "validation", "test")
    }
    leak = _cross_split_leakage(scratch_config, reports)
    assert not leak["leakage_detected"], leak


def test_near_duplicates_never_straddle_splits(scratch_config, tmp_path):
    """The core leakage guarantee: copies of one photo land in one split."""
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs, per_class=10)
    # Add near-identical re-saves of one image (different file bytes).
    original = raw / scratch_config.class_dirs[0] / f"{scratch_config.class_dirs[0]}_0.jpg"
    for i in range(4):
        Image.open(original).resize((70, 70)).save(
            raw / scratch_config.class_dirs[0] / f"nearcopy_{i}.jpg", quality=80 - i * 5)

    result = prepare(scratch_config, raw_root=raw)
    group_splits = {}
    for entry in result.manifest:
        group_splits.setdefault(entry["group_id"], set()).add(entry["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert result.near_duplicate_groups >= 1


def test_exact_duplicates_dropped_during_preparation(scratch_config, tmp_path):
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs, per_class=6)
    source = raw / scratch_config.class_dirs[0] / f"{scratch_config.class_dirs[0]}_0.jpg"
    (raw / scratch_config.class_dirs[0] / "dup.jpg").write_bytes(source.read_bytes())
    result = prepare(scratch_config, raw_root=raw)
    assert len(result.dropped_exact_duplicates) == 1
    assert sum(result.counts[scratch_config.class_names[0]].values()) == 6


def test_preparation_is_deterministic(scratch_config, tmp_path):
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs, per_class=10)
    first = {e["source"]: e["split"] for e in prepare(scratch_config, raw_root=raw).manifest}
    second = {e["source"]: e["split"] for e in prepare(scratch_config, raw_root=raw).manifest}
    assert first == second


def test_dry_run_writes_nothing(scratch_config, tmp_path):
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs, per_class=6)
    result = prepare(scratch_config, raw_root=raw, dry_run=True)
    assert result.manifest
    assert not any(Path(scratch_config.path("train_dir")).glob("**/*.jpg"))


def test_empty_raw_directory_is_handled(scratch_config, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    result = prepare(scratch_config, raw_root=raw)
    assert not result.manifest
    assert any("no usable images" in warning for warning in result.warnings)


def test_macos_artefacts_are_not_reported_as_corrupt(scratch_config, tmp_path):
    """A dataset copied or unzipped on macOS must validate cleanly.

    Finder leaves .DS_Store in every folder and unzipping leaves an AppleDouble
    "._name" sidecar beside every file. Before these were filtered, a real
    dataset produced hundreds of spurious "corrupt image" entries that buried
    the genuine findings.
    """
    raw = tmp_path / "raw"
    _make_dataset(raw, scratch_config.class_dirs, per_class=4)

    (raw / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    (raw / "__MACOSX").mkdir()
    for directory in scratch_config.class_dirs:
        (raw / directory / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
        (raw / directory / "._placeholder.jpg").write_bytes(b"\x00\x05\x16\x07")

    report = validate_dataset(scratch_config, raw, check_duplicates=False)
    assert report.usable, report.errors
    assert report.total_valid_images == 4 * scratch_config.num_classes
    assert report.invalid_files == [], report.invalid_files
    assert "__MACOSX" not in report.unexpected_directories

    # Preparation must also ignore them rather than fail or copy them across.
    result = prepare(scratch_config, raw_root=raw)
    assert sum(result.totals.values()) == 4 * scratch_config.num_classes
    assert not any(Path(e["destination"]).name.startswith("._") for e in result.manifest)
