"""The V3 dataset builder: de-duplication, source families, splits, isolation.

Two layers of tests.

* **Synthetic** tests build a small multi-source corpus in a temporary
  directory with deliberately planted exact duplicates, near duplicates,
  cross-source copies and label conflicts, run the real builder on it, and
  check every guarantee the V3 report makes. They need no network and no real
  data, so they run everywhere.
* **Real-manifest** tests check the committed ``data/v3_manifest.csv`` - the
  record of the actual V3 dataset. Counts and leakage are provable from the
  manifest alone, so these also run in a fresh clone. The few checks that need
  the images on disk skip, with a reason, when the images are absent.
"""

from __future__ import annotations

import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import src.data.build_v3 as build_v3
from src.data.build_v3 import (
    MANIFEST_PATH, POOL_DIR, REMOVED_PATH, SPLITS, VARIANT_DIRS, build, check_external_isolation,
    class_weights, hash_matrix, near_pairs_between, read_manifest, staged_name, verify,
)
from src.data.sources_v3 import DatasetSource, SourceFolder, active_sources, source_families
from src.utils.config import load_config
from src.utils.image_io import perceptual_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 5


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------
def _smooth(seed: int, size: int = 96) -> Image.Image:
    """A smooth random image: strong low-frequency content, so its pHash is stable."""
    grid = np.random.RandomState(seed).randint(0, 256, (8, 8, 3)).astype(np.uint8)
    return Image.fromarray(grid).resize((size, size), Image.BICUBIC)


def _near(image: Image.Image) -> Image.Image:
    """Different bytes, same photograph as far as pHash is concerned.

    A one-pixel nudge: measured at 0 bits of pHash distance, against 112-164 bits
    between unrelated fixture images. (A global +3 brightness shift was tried
    first and moved the hash 4-8 bits - straddling the threshold of 5, so it
    could not serve as an unambiguous near duplicate.)
    """
    array = np.asarray(image).copy()
    array[0, 0] = (array[0, 0].astype(np.int16) + 1) % 256
    return Image.fromarray(array)


class Corpus:
    """Builds a clone directory and the registry entries that describe it."""

    def __init__(self, root: Path):
        self.root = root
        self.folders: dict = defaultdict(list)     # source -> [(folder, class, n)]
        self.next_seed = 1000

    def fresh(self) -> Image.Image:
        self.next_seed += 1
        return _smooth(self.next_seed)

    def put(self, source: str, folder: str, name: str, image: Image.Image) -> Path:
        path = self.root / source / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    def copy(self, source: str, folder: str, name: str, original: Path) -> Path:
        path = self.root / source / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, path)
        return path


def _source(key: str, folders, family: str = "") -> DatasetSource:
    return DatasetSource(
        key=key, name=key, url="", clone_url="", acquired="test", licence="test",
        imagery="test", family=family, commit=f"{key}-commit",
        folders=[SourceFolder(path, cls, 0, "test") for path, cls in folders])


@pytest.fixture
def corpus(tmp_path):
    """Three sources, one of them a family member of another, with planted problems.

    Planted, and what the builder must do with each:
      * alpha/Healthy has an exact duplicate pair          -> drop one copy
      * alpha/Healthy has a near-duplicate triple           -> keep all three, one group
      * beta (family alpha) copies 4 alpha images, slightly -> drop those 4 from beta
      * beta has one exact copy of an alpha image           -> dropped as exact dup
      * beta has 8 genuinely new images                     -> keep
      * one byte-identical file is labelled both Healthy and Brown Rust in gamma
                                                            -> drop from both classes
    """
    clones = tmp_path / "clones"
    c = Corpus(clones)
    alpha_healthy = [c.put("alpha", "H", f"h{i:03d}.png", c.fresh()) for i in range(40)]
    c.copy("alpha", "H", "h_dup.png", alpha_healthy[0])                   # exact dup
    triple = c.fresh()
    c.put("alpha", "H", "t1.png", triple)
    c.put("alpha", "H", "t2.png", _near(triple))
    c.put("alpha", "H", "t3.png", _near(_near(triple)))
    alpha_br = [c.put("alpha", "BR", f"b{i:03d}.png", c.fresh()) for i in range(40)]
    for i in range(30):
        c.put("alpha", "PM", f"p{i:03d}.png", c.fresh())

    for i in range(4):                                                    # beta near-copies
        c.put("beta", "H", f"copy{i}.png", _near(Image.open(alpha_healthy[i + 1])))
    c.copy("beta", "H", "exact.png", alpha_healthy[10])                  # cross-source exact
    for i in range(8):
        c.put("beta", "H", f"new{i}.png", c.fresh())
    for i in range(6):
        c.put("beta", "BR", f"new{i}.png", c.fresh())

    for i in range(12):
        c.put("gamma", "YR", f"y{i:03d}.png", c.fresh())
    for i in range(10):
        c.put("gamma", "H", f"g{i:03d}.png", c.fresh())
    for i in range(10):
        c.put("gamma", "BR", f"g{i:03d}.png", c.fresh())
    conflict = c.put("gamma", "H", "conflict.png", c.fresh())
    c.copy("gamma", "BR", "conflict.png", conflict)

    sources = [
        _source("alpha", [("H", "Healthy"), ("BR", "Brown Rust"), ("PM", "Powdery Mildew")]),
        _source("beta", [("H", "Healthy"), ("BR", "Brown Rust")], family="alpha"),
        _source("gamma", [("YR", "Yellow Rust"), ("H", "Healthy"), ("BR", "Brown Rust")]),
    ]
    return {"clones": clones, "sources": sources, "root": tmp_path / "repo",
            "alpha_healthy": alpha_healthy, "alpha_br": alpha_br}


def _config(seed: int = 42):
    return load_config(overrides={
        "random_seed": seed,
        "data": {"split": {"train": 0.75, "validation": 0.125, "test": 0.125},
                 "deduplication": {"enabled": True, "hash_size": 16,
                                   "hash_threshold": THRESHOLD,
                                   "drop_exact_duplicates": True},
                 "imbalance_warn_ratio": 3.0},
    })


def _build(corpus, seed: int = 42, dry_run: bool = False):
    return build(_config(seed), corpus["clones"], root=corpus["root"],
                 sources=corpus["sources"], workers=1, dry_run=dry_run)


# ---------------------------------------------------------------------------
# The fixture's own premise
# ---------------------------------------------------------------------------
def test_fixture_premise_distinct_images_far_near_copies_close(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _smooth(1).save(a)
    _smooth(2).save(b)
    near = tmp_path / "near.png"
    _near(_smooth(1)).save(near)
    far = near_pairs_between(hash_matrix([perceptual_hash(a, 16)]),
                             hash_matrix([perceptual_hash(b, 16)]), THRESHOLD)
    close = near_pairs_between(hash_matrix([perceptual_hash(a, 16)]),
                               hash_matrix([perceptual_hash(near, 16)]), THRESHOLD)
    assert far == [], "two unrelated images must not count as near duplicates"
    assert close == [(0, 0)], "a slightly altered copy must count as a near duplicate"
    assert a.read_bytes() != near.read_bytes()


# ---------------------------------------------------------------------------
# Removal rules
# ---------------------------------------------------------------------------
def test_exact_duplicates_are_removed_and_recorded(corpus):
    result = _build(corpus, dry_run=True)
    exact = [r for r in result.removed if r["reason"] == "exact_duplicate"]
    names = {Path(r["source_path"]).name for r in exact}
    assert names == {"h_dup.png", "exact.png"}
    # The first copy in registry order is the one kept.
    kept = {r["kept_instead"] for r in exact}
    assert all(k.startswith("alpha/") for k in kept)


def test_family_member_loses_only_its_near_duplicates_of_the_primary(corpus):
    result = _build(corpus, dry_run=True)
    family = [r for r in result.removed if r["reason"] == "near_duplicate_of_alpha"]
    assert sorted(Path(r["source_path"]).name for r in family) == [
        "copy0.png", "copy1.png", "copy2.png", "copy3.png"]
    beta = [im for im in result.images if im.source == "beta"]
    assert len(beta) == 8 + 6, "beta's genuinely new images must all survive"
    assert all(im.family == "alpha" for im in beta)


def test_primary_source_is_never_trimmed_by_the_family_rule(corpus):
    result = _build(corpus, dry_run=True)
    assert not [r for r in result.removed
                if r["source"] == "alpha" and r["reason"].startswith("near_duplicate")]
    alpha_healthy = [im for im in result.images
                     if im.source == "alpha" and im.class_name == "Healthy"]
    assert len(alpha_healthy) == 40 + 3        # 41 files minus one exact dup, plus the triple


def test_label_conflicts_are_removed_from_both_classes(corpus):
    result = _build(corpus, dry_run=True)
    conflicts = [r for r in result.removed if r["reason"] == "label_conflict_exact"]
    assert sorted(r["class"] for r in conflicts) == ["Brown Rust", "Healthy"]
    assert not [im for im in result.images if Path(im.source_path).name == "conflict.png"]


def test_near_duplicates_that_remain_are_grouped_not_removed(corpus):
    result = _build(corpus, dry_run=True)
    triple = [im for im in result.images if Path(im.source_path).name in {"t1.png", "t2.png", "t3.png"}]
    assert len(triple) == 3
    assert len({im.group_id for im in triple}) == 1


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def test_near_duplicate_groups_never_cross_splits(corpus):
    result = _build(corpus, dry_run=True)
    splits_by_group = defaultdict(set)
    for image in result.images:
        splits_by_group[image.group_id].add(image.split)
    assert all(len(s) == 1 for s in splits_by_group.values())


def test_no_near_duplicate_pair_straddles_two_splits(corpus):
    result = _build(corpus, dry_run=True)
    by_split = {s: [im.phash for im in result.images if im.split == s] for s in SPLITS}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert near_pairs_between(hash_matrix(by_split[a]), hash_matrix(by_split[b]),
                                  THRESHOLD) == []


def test_every_split_is_populated_for_every_class(corpus):
    result = _build(corpus, dry_run=True)
    present = Counter((im.class_name, im.split) for im in result.images)
    for class_name in {im.class_name for im in result.images}:
        for split in SPLITS:
            assert present[(class_name, split)] > 0, (class_name, split)


def test_split_is_reproducible_with_the_same_seed(corpus):
    first = {im.uid: (im.split, im.in_balanced) for im in _build(corpus, seed=42, dry_run=True).images}
    second = {im.uid: (im.split, im.in_balanced) for im in _build(corpus, seed=42, dry_run=True).images}
    assert first == second


def test_a_different_seed_gives_a_different_split(corpus):
    first = {im.uid: im.split for im in _build(corpus, seed=42, dry_run=True).images}
    other = {im.uid: im.split for im in _build(corpus, seed=7, dry_run=True).images}
    assert first != other


# ---------------------------------------------------------------------------
# The balanced variant
# ---------------------------------------------------------------------------
def test_balanced_cap_is_ratio_times_the_scarcest_class(corpus):
    result = _build(corpus, dry_run=True)
    totals = Counter(im.class_name for im in result.images)
    assert result.balanced_cap == int(min(totals.values()) * 3.0)
    chosen = Counter(im.class_name for im in result.images if im.in_balanced)
    for class_name, total in totals.items():
        assert chosen[class_name] == min(total, result.balanced_cap)


def test_balanced_is_nested_in_full_split_for_split(corpus):
    """Every balanced image keeps its full-variant split - checked on disk."""
    _build(corpus)
    root = corpus["root"]
    for row in read_manifest(root / MANIFEST_PATH):
        if row["in_balanced"] != "1":
            continue
        rel = Path(row["split"]) / row["class_dir"] / row["staged_name"]
        assert (root / VARIANT_DIRS["balanced"] / rel).exists()
        assert (root / VARIANT_DIRS["full"] / rel).exists()
        for other in SPLITS:
            if other != row["split"]:
                assert not (root / VARIANT_DIRS["full"] / other / row["class_dir"]
                            / row["staged_name"]).exists()


def test_nothing_is_duplicated_to_fill_a_quota(corpus):
    result = _build(corpus, dry_run=True)
    shas = [im.sha256 for im in result.images if im.in_balanced]
    assert len(shas) == len(set(shas))


# ---------------------------------------------------------------------------
# Staging, manifest and verification
# ---------------------------------------------------------------------------
def test_verify_passes_on_a_clean_build(corpus):
    _build(corpus)
    checks = verify(corpus["root"], THRESHOLD)
    assert checks["passed"], checks


def test_staged_files_are_byte_identical_to_the_originals(corpus):
    result = _build(corpus)
    for image in result.images:
        staged = corpus["root"] / POOL_DIR / image.class_dir / image.staged_name
        assert staged.read_bytes() == image.origin.read_bytes()


def test_verify_detects_a_file_missing_from_disk(corpus):
    _build(corpus)
    row = read_manifest(corpus["root"] / MANIFEST_PATH)[0]
    (corpus["root"] / VARIANT_DIRS["full"] / row["split"] / row["class_dir"]
     / row["staged_name"]).unlink()
    checks = verify(corpus["root"], THRESHOLD)
    assert not checks["passed"]
    assert checks["disk_mismatches"]["full"]["missing"] == 1


def test_verify_detects_a_group_split_across_two_splits(corpus):
    """Move one member of the near-duplicate triple into another split."""
    _build(corpus)
    root = corpus["root"]
    manifest = root / MANIFEST_PATH
    rows = read_manifest(manifest)
    member = next(r for r in rows if r["staged_name"].endswith("__t1.png"))
    old_split = member["split"]
    new_split = next(s for s in SPLITS if s != old_split)
    old = root / VARIANT_DIRS["full"] / old_split / member["class_dir"] / member["staged_name"]
    new = root / VARIANT_DIRS["full"] / new_split / member["class_dir"] / member["staged_name"]
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)
    member["split"] = new_split
    if member["in_balanced"] == "1":
        (root / VARIANT_DIRS["balanced"] / old_split / member["class_dir"]
         / member["staged_name"]).unlink()
        member["in_balanced"] = "0"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=build_v3.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    checks = verify(root, THRESHOLD)
    assert not checks["passed"]
    assert checks["groups_spanning_splits"] >= 1
    assert checks["near_duplicates_across_splits"] >= 1


def test_verify_tolerates_background_composites_only_in_train(corpus):
    _build(corpus)
    root = corpus["root"]
    row = read_manifest(root / MANIFEST_PATH)[0]
    train = root / VARIANT_DIRS["full"] / "train" / row["class_dir"]
    train.mkdir(parents=True, exist_ok=True)
    _smooth(9).save(train / f"{build_v3.AUGMENTED_PREFIX}_0001.png")
    assert verify(root, THRESHOLD)["passed"]
    test_dir = root / VARIANT_DIRS["full"] / "test" / row["class_dir"]
    test_dir.mkdir(parents=True, exist_ok=True)
    _smooth(9).save(test_dir / f"{build_v3.AUGMENTED_PREFIX}_0002.png")
    checks = verify(root, THRESHOLD)
    assert not checks["passed"]
    assert checks["augmented_outside_train"]


def test_rebuild_is_identical(corpus):
    _build(corpus)
    first = (corpus["root"] / MANIFEST_PATH).read_bytes()
    _build(corpus)
    assert (corpus["root"] / MANIFEST_PATH).read_bytes() == first


def test_staging_refuses_to_write_outside_the_v3_directories(tmp_path):
    with pytest.raises(ValueError):
        build_v3._guarded(tmp_path, "data/train")
    with pytest.raises(ValueError):
        build_v3._guarded(tmp_path, "data/external_test")
    assert build_v3._guarded(tmp_path, POOL_DIR) == (tmp_path / POOL_DIR).resolve()


def test_staged_names_are_filesystem_safe():
    image = build_v3.PoolImage("Brown Rust", "Brown_Rust", "suhas", "suhas", "",
                               "Images/Leaf Rust/lolr (17).JPG", Path("/x"))
    name = staged_name(image)
    assert name == "suhas__Images_Leaf-Rust__lolr-17.jpg"
    assert not name.startswith(build_v3.AUGMENTED_PREFIX)


def test_class_weights_match_the_training_formula(config):
    from src.model.datasets import compute_class_weights
    counts = {"Healthy": 1790, "Yellow Rust": 335, "Brown Rust": 1866, "Powdery Mildew": 1265}
    labels = [i for i, name in enumerate(config.class_names) for _ in range(counts[name])]
    expected = compute_class_weights(config, labels)
    ours = class_weights(counts, config.class_names)
    for index, name in enumerate(config.class_names):
        assert ours[name] == pytest.approx(expected[index])


# ---------------------------------------------------------------------------
# External test set isolation
# ---------------------------------------------------------------------------
def test_build_never_reads_the_external_test_directory(corpus, monkeypatch):
    root = corpus["root"]
    external = root / "data" / "external_test"
    external.mkdir(parents=True)
    _smooth(424242).save(external / "ext_01.png")

    touched = []
    for name in ("inspect_image", "file_sha256", "perceptual_hash", "iter_image_files"):
        original = getattr(build_v3, name)

        def spy(path, *args, _original=original, **kwargs):
            touched.append(Path(path).resolve())
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(build_v3, name, spy)

    _build(corpus)
    assert touched, "spy saw no file access - the test would be vacuous"
    ext = external.resolve()
    assert not [p for p in touched if p == ext or ext in p.parents]


def test_external_isolation_check_finds_a_planted_contaminant(corpus):
    """Prove the content check is not vacuous: plant a pool image as an external one."""
    result = _build(corpus)
    root = corpus["root"]
    external = root / "data" / "external_test"
    external.mkdir(parents=True)
    _smooth(424242).save(external / "clean.png")
    assert check_external_isolation(root, _config(), THRESHOLD)["isolated"]

    shutil.copyfile(result.images[0].origin, external / "leaked.png")
    report = check_external_isolation(root, _config(), THRESHOLD)
    assert not report["isolated"]
    assert report["exact_matches_in_pool"] == ["leaked.png"]


def test_external_directory_is_outside_every_v3_path():
    for variant in ("balanced", "full"):
        cfg = load_config(PROJECT_ROOT / "config" / f"model_v3_{variant}.yaml")
        external = (PROJECT_ROOT / "data" / "external_test").resolve()
        for key in ("raw_dir", "train_dir", "validation_dir", "test_dir"):
            path = cfg.path(key).resolve()
            assert external != path and external not in path.parents and path not in external.parents


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_folds_mubashar_into_the_suhas_family():
    families = source_families()
    assert families["suhas"] == ["suhas", "mubashar"]
    assert set(families) == {"wpldd", "suhas", "hg3817", "kiran"}
    assert all(s.commit for s in active_sources())


# ---------------------------------------------------------------------------
# The real V3 dataset, from the committed manifest
# ---------------------------------------------------------------------------
REAL_MANIFEST = PROJECT_ROOT / MANIFEST_PATH
REAL_REMOVED = PROJECT_ROOT / REMOVED_PATH
needs_manifest = pytest.mark.skipif(not REAL_MANIFEST.exists(),
                                    reason="data/v3_manifest.csv not built yet")
needs_images = pytest.mark.skipif(not (PROJECT_ROOT / POOL_DIR).exists(),
                                  reason="V3 images not staged in this checkout "
                                         "(rebuild with python -m src.data.build_v3)")


@pytest.fixture(scope="module")
def real_rows():
    return read_manifest(REAL_MANIFEST)


@needs_manifest
def test_real_full_variant_counts(real_rows):
    totals = Counter(r["class"] for r in real_rows)
    assert totals == {"Healthy": 2388, "Yellow Rust": 447,
                      "Brown Rust": 2488, "Powdery Mildew": 1687}
    assert len(real_rows) == 7010


@needs_manifest
def test_real_balanced_variant_counts(real_rows):
    totals = Counter(r["class"] for r in real_rows if r["in_balanced"] == "1")
    assert totals == {"Healthy": 1341, "Yellow Rust": 447,
                      "Brown Rust": 1341, "Powdery Mildew": 1341}


@needs_manifest
def test_real_removals_match_the_approved_decision():
    removed = read_manifest(REAL_REMOVED)
    reasons = Counter(r["reason"] for r in removed)
    assert reasons["exact_duplicate"] == 1328
    assert reasons["near_duplicate_of_suhas"] == 267
    family = Counter(r["class"] for r in removed if r["reason"] == "near_duplicate_of_suhas")
    assert family == {"Brown Rust": 200, "Healthy": 67}
    assert all(r["source"] == "mubashar" for r in removed
               if r["reason"] == "near_duplicate_of_suhas")


@needs_manifest
def test_real_keeps_the_177_unique_mubashar_images(real_rows):
    kept = Counter(r["class"] for r in real_rows if r["source"] == "mubashar")
    assert kept == {"Brown Rust": 118, "Healthy": 59}
    assert {r["family"] for r in real_rows if r["source"] == "mubashar"} == {"suhas"}


@needs_manifest
def test_real_no_image_in_two_splits(real_rows):
    splits_by_sha = defaultdict(set)
    for r in real_rows:
        splits_by_sha[r["sha256"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_by_sha.values())


@needs_manifest
def test_real_no_near_duplicate_across_splits_in_any_class(real_rows):
    by_split = {s: [r["phash"] for r in real_rows if r["split"] == s] for s in SPLITS}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert near_pairs_between(hash_matrix(by_split[a]), hash_matrix(by_split[b]),
                                  THRESHOLD) == []


@needs_manifest
def test_real_every_group_in_one_split(real_rows):
    splits_by_group = defaultdict(set)
    for r in real_rows:
        splits_by_group[r["group_id"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_by_group.values())


@needs_manifest
def test_real_every_family_in_every_split_of_every_class(real_rows):
    present = Counter((r["class"], r["family"], r["split"]) for r in real_rows)
    for cls, fam in {(r["class"], r["family"]) for r in real_rows}:
        for split in SPLITS:
            assert present[(cls, fam, split)] > 0, (cls, fam, split)


@needs_manifest
@needs_images
def test_real_disk_matches_manifest_and_verification_passes():
    checks = verify(PROJECT_ROOT, THRESHOLD, check_sha=False)
    assert checks["passed"], checks
