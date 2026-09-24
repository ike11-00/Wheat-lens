"""Build the V3 datasets: one clean pool, one group-aware split, two nested variants.

Pipeline
--------
1.  **Collect** every image the registry maps onto a configured class, from the
    source clones, validating and fingerprinting each one (SHA-256, perceptual
    hash, dimensions).
2.  **Exact de-duplication** within each class. The first copy in registry
    order, then path order, is kept; every dropped file is recorded with the
    file kept in its place.
3.  **Label-conflict check.** A byte-identical file carrying two different
    labels cannot be trained on either way; any such file is removed from both
    classes and reported.
4.  **Source-family de-duplication.** A source declared as a member of another
    source's family (``DatasetSource.family``) loses every image that is a
    near-duplicate of an image in that family's primary source, same class.
    The primary source is never trimmed. For V3 this is exactly the approved
    rule: ``mubashar`` loses its near-duplicates of ``suhas`` and keeps the rest.
5.  **Near-duplicate grouping** across the whole pool - all classes, all
    sources - with the thresholds calibrated in ``config.yaml``. Remaining
    near-duplicates are *kept*, as in v1/v2, but always travel together.
6.  **Group-aware split** of the pool, stratified by (class, source family), so
    every family is represented in every split in proportion. Whole groups are
    assigned, never individual images, which is what makes cross-split leakage
    impossible rather than merely unlikely.
7.  **Two nested variants.** ``v3_full`` is the entire pool. ``v3_balanced`` is
    selected *from within each split* of ``v3_full``: every balanced image keeps
    the split it has in the full variant. So the balanced training set is a
    subset of the full training set and the balanced test set is a subset of
    the full test set, and both models can be scored on either test set without
    either having seen a test image.
8.  **Stage** by hard link (no bytes copied, no bytes altered) into
    ``data/v3_raw/``, ``data/v3_full/`` and ``data/v3_balanced/``, and write
    ``data/v3_manifest.csv`` and ``data/v3_removed.csv``.

What this module never touches
------------------------------
``data/external_test/`` is not read, listed or written by ``build()``. It is
outside every path this module uses, and ``tests/test_v3_dataset.py`` spies on
file access during a build to prove it. ``check_external_isolation()`` is a
separate, explicitly-invoked verification that reads those images read-only to
confirm none of them appears in the pool; it cannot change the dataset.

Nothing in ``data/raw``, ``data/train``, ``data/validation``, ``data/test`` or
``models/`` is read or written, so v1 and v2 are unaffected.

Run::

    python -m src.data.build_v3 --sources <clone-dir>            # build + verify
    python -m src.data.build_v3 --sources <clone-dir> --dry-run  # no files written
    python -m src.data.build_v3 --verify-only                    # re-verify on disk
    python -m src.data.build_v3 --verify-only --check-external --check-decode
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.config import Config, load_config
from ..utils.helpers import ensure_dir, get_logger, markdown_table, utc_timestamp, write_json, write_text
from ..utils.image_io import file_sha256, inspect_image, iter_image_files, perceptual_hash
from .augment_backgrounds import AUGMENTED_PREFIX
from .sample_plan import even_allocation, stride_select
from .sources_v3 import DatasetSource, active_sources

LOGGER = get_logger(__name__)

SPLITS = ("train", "validation", "test")
VARIANTS = ("full", "balanced")

#: Every directory this module may create or clean. Anything else is refused.
POOL_DIR = "data/v3_raw"
VARIANT_DIRS = {"full": "data/v3_full", "balanced": "data/v3_balanced"}
MANIFEST_PATH = "data/v3_manifest.csv"
REMOVED_PATH = "data/v3_removed.csv"

MANIFEST_FIELDS = [
    "staged_name", "class", "class_dir", "source", "family", "source_commit",
    "source_path", "sha256", "phash", "width", "height", "group_id",
    "split", "in_balanced",
]
REMOVED_FIELDS = ["class", "source", "family", "source_path", "sha256", "reason", "kept_instead"]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class PoolImage:
    class_name: str
    class_dir: str
    source: str
    family: str
    source_commit: str
    source_path: str          # path inside the source repository
    origin: Path              # absolute path on disk
    sha256: str = ""
    phash: str = ""
    width: int = 0
    height: int = 0
    group_id: int = -1
    split: str = ""
    in_balanced: bool = False
    staged_name: str = ""

    @property
    def uid(self) -> str:
        return f"{self.source}/{self.source_path}"


@dataclass
class BuildResult:
    generated_utc: str
    seed: int
    split_fractions: Dict[str, float]
    hash_size: int
    hash_threshold: int
    balance_ratio: float
    images: List[PoolImage] = field(default_factory=list)
    removed: List[Dict[str, str]] = field(default_factory=list)
    invalid: List[Dict[str, str]] = field(default_factory=list)
    declared: Dict[str, Dict[str, int]] = field(default_factory=dict)
    balanced_cap: int = 0
    near_duplicate_groups: int = 0
    images_in_near_duplicate_groups: int = 0
    cross_class_groups: int = 0
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fingerprinting (cached, parallel)
# ---------------------------------------------------------------------------
def _fingerprint(job: Tuple[str, int, int]) -> dict:
    path_str, min_pixels, hash_size = job
    path = Path(path_str)
    info = inspect_image(path, min_pixels=min_pixels)
    if not info.ok:
        return {"ok": False, "reason": info.reason}
    return {"ok": True, "sha256": file_sha256(path),
            "phash": perceptual_hash(path, hash_size=hash_size),
            "width": info.width, "height": info.height}


def _fingerprints(paths: Sequence[Path], min_pixels: int, hash_size: int, workers: int,
                  cache_path: Optional[Path]) -> Dict[str, dict]:
    """Fingerprint ``paths``, reusing a cache keyed on path, size and mtime."""
    cache: Dict[str, dict] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}

    def key(path: Path) -> str:
        stat = path.stat()
        return f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{hash_size}"

    results: Dict[str, dict] = {}
    todo: List[Path] = []
    for path in paths:
        cached = cache.get(key(path))
        if cached is not None:
            results[str(path)] = cached
        else:
            todo.append(path)

    if todo:
        LOGGER.info("fingerprinting %d image(s) (%d cached) with %d worker(s)",
                    len(todo), len(paths) - len(todo), workers)
        jobs = [(str(p), min_pixels, hash_size) for p in todo]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                computed = list(pool.map(_fingerprint, jobs, chunksize=16))
        else:
            computed = [_fingerprint(job) for job in jobs]
        for path, payload in zip(todo, computed):
            results[str(path)] = payload
            cache[key(path)] = payload
        if cache_path:
            ensure_dir(cache_path.parent)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Perceptual-hash distance, vectorised
# ---------------------------------------------------------------------------
_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def hash_matrix(hexes: Sequence[str]) -> np.ndarray:
    """Hex hashes -> (n, bytes) uint8 matrix."""
    if not hexes:
        return np.zeros((0, 0), dtype=np.uint8)
    rows = [np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hexes]
    return np.vstack(rows)


def near_pairs_within(matrix: np.ndarray, threshold: int) -> List[Tuple[int, int]]:
    """All index pairs (i < j) within ``threshold`` bits of each other."""
    pairs: List[Tuple[int, int]] = []
    for i in range(len(matrix) - 1):
        distance = _POPCOUNT[np.bitwise_xor(matrix[i + 1:], matrix[i])].sum(axis=1)
        for j in np.nonzero(distance <= threshold)[0]:
            pairs.append((i, i + 1 + int(j)))
    return pairs


def near_pairs_between(left: np.ndarray, right: np.ndarray, threshold: int
                       ) -> List[Tuple[int, int]]:
    """All (i, j) with left[i] within ``threshold`` bits of right[j]."""
    pairs: List[Tuple[int, int]] = []
    if not len(left) or not len(right):
        return pairs
    for i in range(len(left)):
        distance = _POPCOUNT[np.bitwise_xor(right, left[i])].sum(axis=1)
        for j in np.nonzero(distance <= threshold)[0]:
            pairs.append((i, int(j)))
    return pairs


def _union_find_groups(n: int, pairs: Iterable[Tuple[int, int]]) -> List[int]:
    parent = list(range(n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in pairs:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    return [find(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def collect(config: Config, clone_root: Path, sources: Sequence[DatasetSource],
            workers: int = 1, cache_path: Optional[Path] = None
            ) -> Tuple[List[PoolImage], List[Dict[str, str]], Dict[str, Dict[str, int]]]:
    """Validate and fingerprint every mapped image. Returns (images, invalid, declared)."""
    extensions = config.supported_extensions
    min_pixels = int(config.get("data", "min_image_pixels", default=32))
    hash_size = int((config.get("data", "deduplication", default={}) or {}).get("hash_size", 16))

    candidates: List[PoolImage] = []
    declared: Dict[str, Dict[str, int]] = defaultdict(dict)
    for source in sources:
        for folder in source.folders:
            if not folder.leaf_lens_class:
                continue
            spec = config.resolve_class(folder.leaf_lens_class)
            if spec is None:
                raise ValueError(f"registry maps {source.key}/{folder.path} onto unknown class "
                                 f"{folder.leaf_lens_class!r}")
            directory = clone_root / source.key / folder.path
            if not directory.is_dir():
                raise FileNotFoundError(f"registry folder missing from the clone: {directory}")
            declared[spec.name][source.key] = (declared[spec.name].get(source.key, 0)
                                               + folder.observed_count)
            for path in sorted(iter_image_files(directory, extensions)):
                candidates.append(PoolImage(
                    class_name=spec.name, class_dir=spec.directory, source=source.key,
                    family=source.family_key, source_commit=source.commit,
                    source_path=path.relative_to(clone_root / source.key).as_posix(),
                    origin=path))

    prints = _fingerprints([c.origin for c in candidates], min_pixels, hash_size,
                           workers, cache_path)
    images: List[PoolImage] = []
    invalid: List[Dict[str, str]] = []
    for image in candidates:
        payload = prints[str(image.origin)]
        if not payload.get("ok"):
            invalid.append({"class": image.class_name, "source": image.source,
                            "source_path": image.source_path, "reason": payload.get("reason", "")})
            continue
        if not payload.get("phash"):
            raise RuntimeError("perceptual hashing unavailable - install 'ImageHash'; V3 "
                               "cannot guarantee leakage-free splits without it")
        image.sha256 = payload["sha256"]
        image.phash = payload["phash"]
        image.width = int(payload["width"])
        image.height = int(payload["height"])
        images.append(image)
    return images, invalid, {k: dict(v) for k, v in declared.items()}


def _removal(image: PoolImage, reason: str, kept: Optional[PoolImage]) -> Dict[str, str]:
    return {"class": image.class_name, "source": image.source, "family": image.family,
            "source_path": image.source_path, "sha256": image.sha256, "reason": reason,
            "kept_instead": kept.uid if kept else ""}


def drop_exact_duplicates(images: List[PoolImage], source_order: Sequence[str]
                          ) -> Tuple[List[PoolImage], List[Dict[str, str]]]:
    """Keep the first copy per (class, sha256), in registry then path order."""
    rank = {key: index for index, key in enumerate(source_order)}
    ordered = sorted(images, key=lambda im: (rank.get(im.source, len(rank)), im.source_path))
    seen: Dict[Tuple[str, str], PoolImage] = {}
    kept: List[PoolImage] = []
    removed: List[Dict[str, str]] = []
    for image in ordered:
        slot = (image.class_name, image.sha256)
        if slot in seen:
            removed.append(_removal(image, "exact_duplicate", seen[slot]))
            continue
        seen[slot] = image
        kept.append(image)
    return kept, removed


def drop_label_conflicts(images: List[PoolImage]) -> Tuple[List[PoolImage], List[Dict[str, str]]]:
    """Remove byte-identical files that appear under more than one class."""
    classes_by_sha: Dict[str, set] = defaultdict(set)
    for image in images:
        classes_by_sha[image.sha256].add(image.class_name)
    conflicted = {sha for sha, classes in classes_by_sha.items() if len(classes) > 1}
    kept = [im for im in images if im.sha256 not in conflicted]
    removed = [_removal(im, "label_conflict_exact", None) for im in images
               if im.sha256 in conflicted]
    return kept, removed


def drop_family_near_duplicates(images: List[PoolImage], sources: Sequence[DatasetSource],
                                threshold: int
                                ) -> Tuple[List[PoolImage], List[Dict[str, str]]]:
    """A family member loses images that near-duplicate its family's primary source."""
    members = {s.key: s.family_key for s in sources if s.family_key != s.key}
    if not members:
        return images, []
    doomed: Dict[int, Tuple[str, PoolImage]] = {}
    by_class: Dict[str, List[int]] = defaultdict(list)
    for index, image in enumerate(images):
        by_class[image.class_name].append(index)
    for class_name, indices in by_class.items():
        for member, primary in members.items():
            mine = [i for i in indices if images[i].source == member]
            theirs = [i for i in indices if images[i].source == primary]
            if not mine or not theirs:
                continue
            pairs = near_pairs_between(hash_matrix([images[i].phash for i in mine]),
                                       hash_matrix([images[i].phash for i in theirs]), threshold)
            for a, b in pairs:
                doomed.setdefault(mine[a], (f"near_duplicate_of_{primary}", images[theirs[b]]))
    kept = [im for i, im in enumerate(images) if i not in doomed]
    removed = [_removal(images[i], reason, match) for i, (reason, match) in sorted(doomed.items())]
    return kept, removed


def assign_groups(images: List[PoolImage], threshold: int) -> Tuple[int, int, int]:
    """Near-duplicate groups across the whole pool. Returns (groups, images, cross-class)."""
    ordered = sorted(range(len(images)), key=lambda i: images[i].uid)
    matrix = hash_matrix([images[i].phash for i in ordered])
    roots = _union_find_groups(len(ordered), near_pairs_within(matrix, threshold))
    members: Dict[int, List[int]] = defaultdict(list)
    for position, root in enumerate(roots):
        members[root].append(ordered[position])
    # Stable ids: groups numbered in order of their first member's uid.
    for group_id, root in enumerate(sorted(members, key=lambda r: min(images[i].uid for i in members[r]))):
        for index in members[root]:
            images[index].group_id = group_id
    multi = [m for m in members.values() if len(m) > 1]
    cross_class = sum(1 for m in multi if len({images[i].class_name for i in m}) > 1)
    return len(multi), sum(len(m) for m in multi), cross_class


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def split_pool(images: List[PoolImage], fractions: Dict[str, float], seed: int) -> List[str]:
    """Group-aware split, stratified by (class, source family). Mutates ``split``."""
    warnings: List[str] = []
    groups: Dict[int, List[PoolImage]] = defaultdict(list)
    for image in images:
        groups[image.group_id].append(image)

    strata: Dict[Tuple[str, str], List[List[PoolImage]]] = defaultdict(list)
    for members in groups.values():
        anchor = min(members, key=lambda im: im.uid)
        strata[(anchor.class_name, anchor.family)].append(members)

    for stratum in sorted(strata):
        rng = random.Random(_stable_seed(seed, *stratum))
        group_list = sorted(strata[stratum], key=lambda g: min(im.uid for im in g))
        rng.shuffle(group_list)
        total = sum(len(g) for g in group_list)
        targets = {"test": fractions["test"] * total,
                   "validation": fractions["validation"] * total}
        assigned = {split: 0 for split in SPLITS}
        for members in group_list:
            if assigned["test"] < targets["test"]:
                chosen = "test"
            elif assigned["validation"] < targets["validation"]:
                chosen = "validation"
            else:
                chosen = "train"
            for image in members:
                image.split = chosen
            assigned[chosen] += len(members)
        for split in SPLITS:
            if assigned[split] == 0:
                warnings.append(f"{stratum[0]} / {stratum[1]}: 0 images in '{split}' "
                                f"({total} images in {len(group_list)} group(s))")
    return warnings


def _largest_remainder(shares: Dict[str, float], total: int) -> Dict[str, int]:
    floors = {k: int(np.floor(v)) for k, v in shares.items()}
    short = total - sum(floors.values())
    order = sorted(shares, key=lambda k: (-(shares[k] - floors[k]), k))
    for key in order[:short]:
        floors[key] += 1
    return floors


def select_balanced(images: List[PoolImage], ratio: float) -> int:
    """Mark the balanced subset. Returns the per-class cap.

    Per class, the cap is split across the three splits in the proportions the
    class already has in the full variant, then across source families within
    each split with ``even_allocation``, then across files with
    ``stride_select``. Split membership is inherited, never reassigned.
    """
    by_class: Dict[str, List[PoolImage]] = defaultdict(list)
    for image in images:
        image.in_balanced = False
        by_class[image.class_name].append(image)
    smallest = min(len(members) for members in by_class.values())
    cap = int(smallest * ratio)

    for class_name, members in by_class.items():
        if len(members) <= cap:
            for image in members:
                image.in_balanced = True
            continue
        per_split = Counter(im.split for im in members)
        targets = _largest_remainder(
            {s: cap * per_split[s] / len(members) for s in SPLITS}, cap)
        for split in SPLITS:
            by_family: Dict[str, List[PoolImage]] = defaultdict(list)
            for image in members:
                if image.split == split:
                    by_family[image.family].append(image)
            allocation = even_allocation({f: len(v) for f, v in by_family.items()},
                                         targets[split])
            for family, count in allocation.items():
                chosen = stride_select(sorted(by_family[family], key=lambda im: im.uid), count)
                for image in chosen:
                    image.in_balanced = True
    return cap


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def staged_name(image: PoolImage) -> str:
    """``<source>__<folder>__<file>``, filesystem-safe, readable, unique per source path."""
    path = Path(image.source_path)
    folder = _UNSAFE.sub("-", path.parent.as_posix().replace("/", "_")).strip("-")
    stem = _UNSAFE.sub("-", path.stem).strip("-")
    return f"{image.source}__{folder}__{stem}{path.suffix.lower()}"


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------
def _guarded(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    allowed = {(root / d).resolve() for d in [POOL_DIR, *VARIANT_DIRS.values()]}
    if target not in allowed:
        raise ValueError(f"refusing to write outside the V3 directories: {target}")
    return target


def _link(source: Path, destination: Path) -> None:
    ensure_dir(destination.parent)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage(images: List[PoolImage], root: Path) -> None:
    """Hard-link the pool and both variants into place. Cleans only V3 directories."""
    for relative in [POOL_DIR, *VARIANT_DIRS.values()]:
        target = _guarded(root, relative)
        if target.exists():
            shutil.rmtree(target)
    pool = _guarded(root, POOL_DIR)
    for image in images:
        pooled = pool / image.class_dir / image.staged_name
        _link(image.origin, pooled)
        _link(pooled, _guarded(root, VARIANT_DIRS["full"]) / image.split / image.class_dir
              / image.staged_name)
        if image.in_balanced:
            _link(pooled, _guarded(root, VARIANT_DIRS["balanced"]) / image.split
                  / image.class_dir / image.staged_name)


def write_manifest(images: List[PoolImage], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for image in sorted(images, key=lambda im: (im.class_name, im.staged_name)):
            writer.writerow({
                "staged_name": image.staged_name, "class": image.class_name,
                "class_dir": image.class_dir, "source": image.source, "family": image.family,
                "source_commit": image.source_commit, "source_path": image.source_path,
                "sha256": image.sha256, "phash": image.phash, "width": image.width,
                "height": image.height, "group_id": image.group_id, "split": image.split,
                "in_balanced": int(image.in_balanced)})


def write_removed(removed: List[Dict[str, str]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REMOVED_FIELDS)
        writer.writeheader()
        for row in sorted(removed, key=lambda r: (r["reason"], r["class"], r["source"],
                                                  r["source_path"])):
            writer.writerow(row)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(config: Config, clone_root: Path, root: Optional[Path] = None,
          sources: Optional[Sequence[DatasetSource]] = None, workers: int = 1,
          cache_path: Optional[Path] = None, dry_run: bool = False) -> BuildResult:
    """Run steps 1-8. ``root`` is the repository root (defaults to the config's)."""
    root = Path(root) if root else config.project_root
    sources = list(sources) if sources is not None else active_sources()
    dedup = config.get("data", "deduplication", default={}) or {}
    threshold = int(dedup.get("hash_threshold", 5))
    result = BuildResult(
        generated_utc=utc_timestamp(), seed=config.seed,
        split_fractions=config.split_fractions(),
        hash_size=int(dedup.get("hash_size", 16)), hash_threshold=threshold,
        balance_ratio=float(config.get("data", "imbalance_warn_ratio", default=3.0)))

    images, result.invalid, result.declared = collect(config, clone_root, sources, workers,
                                                      cache_path)
    images, removed = drop_exact_duplicates(images, [s.key for s in sources])
    result.removed += removed
    images, removed = drop_label_conflicts(images)
    result.removed += removed
    images, removed = drop_family_near_duplicates(images, sources, threshold)
    result.removed += removed

    (result.near_duplicate_groups, result.images_in_near_duplicate_groups,
     result.cross_class_groups) = assign_groups(images, threshold)
    if result.cross_class_groups:
        result.warnings.append(
            f"{result.cross_class_groups} near-duplicate group(s) span more than one class - "
            f"they are kept in one split, but the labels disagree and should be reviewed")
    result.warnings += split_pool(images, result.split_fractions, result.seed)
    result.balanced_cap = select_balanced(images, result.balance_ratio)

    for image in images:
        image.staged_name = staged_name(image)
    clashes = [n for n, c in Counter((im.class_dir, im.staged_name) for im in images).items()
               if c > 1]
    if clashes:
        raise RuntimeError(f"staged-name collision(s): {clashes[:5]}")
    if any(im.staged_name.startswith(AUGMENTED_PREFIX) for im in images):
        raise RuntimeError(f"a staged name starts with the augmentation prefix "
                           f"'{AUGMENTED_PREFIX}' and would be mistaken for a composite")

    result.images = images
    if not dry_run:
        stage(images, root)
        write_manifest(images, root / MANIFEST_PATH)
        write_removed(result.removed, root / REMOVED_PATH)
    return result


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(root: Path, threshold: int, check_sha: bool = True,
           extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
           ) -> Dict[str, object]:
    """Check the on-disk dataset against the manifest. Every check must pass."""
    rows = read_manifest(root / MANIFEST_PATH)
    checks: Dict[str, object] = {}

    # 1. Disk matches manifest, for the pool and both variants.
    def expected(variant: Optional[str]) -> set:
        out = set()
        for row in rows:
            if variant == "balanced" and row["in_balanced"] != "1":
                continue
            if variant is None:
                out.add(f"{row['class_dir']}/{row['staged_name']}")
            else:
                out.add(f"{row['split']}/{row['class_dir']}/{row['staged_name']}")
        return out

    def on_disk(base: Path, with_split: bool) -> Tuple[set, List[str]]:
        found, augmented = set(), []
        if not base.exists():
            return found, augmented
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(base).as_posix()
            if path.name.startswith(AUGMENTED_PREFIX):
                augmented.append(rel)
                continue
            found.add(rel)
        return found, augmented

    mismatches: Dict[str, Dict[str, int]] = {}
    stray_augmented: List[str] = []
    for label, base, variant in [("pool", root / POOL_DIR, None),
                                 ("full", root / VARIANT_DIRS["full"], "full"),
                                 ("balanced", root / VARIANT_DIRS["balanced"], "balanced")]:
        found, augmented = on_disk(base, variant is not None)
        # Background composites are legitimate only in a training split.
        stray_augmented += [f"{label}/{a}" for a in augmented if not a.startswith("train/")]
        want = expected(variant)
        mismatches[label] = {"missing": len(want - found), "unexpected": len(found - want)}
    checks["disk_matches_manifest"] = all(
        v["missing"] == 0 and v["unexpected"] == 0 for v in mismatches.values())
    checks["disk_mismatches"] = mismatches
    checks["augmented_outside_train"] = stray_augmented

    # 2. Content hashes match the manifest.
    if check_sha:
        bad = [row["staged_name"] for row in rows
               if file_sha256(root / POOL_DIR / row["class_dir"] / row["staged_name"])
               != row["sha256"]]
        checks["sha256_matches_manifest"] = not bad
        checks["sha256_mismatches"] = bad[:20]

    # 3. No exact duplicate straddles two splits (any class).
    splits_by_sha: Dict[str, set] = defaultdict(set)
    for row in rows:
        splits_by_sha[row["sha256"]].add(row["split"])
    exact_cross = sum(1 for s in splits_by_sha.values() if len(s) > 1)
    checks["exact_duplicates_across_splits"] = exact_cross

    # 4. No near-duplicate straddles two splits (any class), measured directly.
    by_split = {s: [r for r in rows if r["split"] == s] for s in SPLITS}
    near_cross = 0
    examples: List[List[str]] = []
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        pairs = near_pairs_between(hash_matrix([r["phash"] for r in by_split[a]]),
                                   hash_matrix([r["phash"] for r in by_split[b]]), threshold)
        near_cross += len(pairs)
        for i, j in pairs[:5]:
            examples.append([by_split[a][i]["staged_name"], by_split[b][j]["staged_name"]])
    checks["near_duplicates_across_splits"] = near_cross
    checks["near_duplicate_examples"] = examples

    # 5. Every near-duplicate group lives in exactly one split.
    splits_by_group: Dict[str, set] = defaultdict(set)
    for row in rows:
        splits_by_group[row["group_id"]].add(row["split"])
    checks["groups_spanning_splits"] = sum(1 for s in splits_by_group.values() if len(s) > 1)

    # 6. Balanced is nested in full - guaranteed by construction, checked anyway
    #    against the directories actually on disk.
    nested_violations = 0
    for row in rows:
        if row["in_balanced"] != "1":
            continue
        full_path = root / VARIANT_DIRS["full"] / row["split"] / row["class_dir"] / row["staged_name"]
        if not full_path.exists():
            nested_violations += 1
    checks["balanced_not_nested_in_full"] = nested_violations

    # 7. Nothing in V3 is the same file as anything in the external test set:
    #    no symlink resolving into it, and no hard link sharing an inode with it.
    #    This stats files only; it never opens an external image.
    external = (root / "data" / "external_test").resolve()
    external_inodes = set()
    if external.exists():
        external_inodes = {(p.stat().st_dev, p.stat().st_ino)
                           for p in external.rglob("*") if p.is_file()}
    shared = 0
    for base in [root / POOL_DIR, *(root / d for d in VARIANT_DIRS.values())]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            if (stat.st_dev, stat.st_ino) in external_inodes or external in path.resolve().parents:
                shared += 1
    checks["external_paths_in_v3"] = shared

    checks["passed"] = bool(
        checks["disk_matches_manifest"]
        and checks.get("sha256_matches_manifest", True)
        and not stray_augmented
        and exact_cross == 0 and near_cross == 0
        and checks["groups_spanning_splits"] == 0
        and nested_violations == 0
        and checks["external_paths_in_v3"] == 0)
    return checks


def check_external_isolation(root: Path, config: Config, threshold: int) -> Dict[str, object]:
    """Read-only: confirm no external-test photograph appears anywhere in the pool.

    Deliberately separate from ``build()``. It reads data/external_test/ only to
    compare hashes and reports what it finds; it cannot change the dataset.
    """
    rows = read_manifest(root / MANIFEST_PATH)
    external_dir = root / "data" / "external_test"
    hash_size = int((config.get("data", "deduplication", default={}) or {}).get("hash_size", 16))
    files = sorted(iter_image_files(external_dir, config.supported_extensions))
    ext_sha = {file_sha256(p): p.name for p in files}
    ext_phash = [(p.name, perceptual_hash(p, hash_size=hash_size)) for p in files]
    ext_phash = [(n, h) for n, h in ext_phash if h]
    exact = sorted({ext_sha[r["sha256"]] for r in rows if r["sha256"] in ext_sha})
    near = near_pairs_between(hash_matrix([h for _, h in ext_phash]),
                              hash_matrix([r["phash"] for r in rows]), threshold)
    return {"external_images_checked": len(files),
            "exact_matches_in_pool": exact,
            "near_duplicate_matches_in_pool": sorted(
                {(ext_phash[i][0], rows[j]["staged_name"], rows[j]["split"]) for i, j in near}),
            "isolated": not exact and not near}


def check_tensorflow_decode(root: Path) -> Dict[str, object]:
    """Decode every pooled image exactly as training will, to catch failures now."""
    import tensorflow as tf  # heavy; imported only when asked for

    failures: List[str] = []
    rows = read_manifest(root / MANIFEST_PATH)
    for row in rows:
        path = root / POOL_DIR / row["class_dir"] / row["staged_name"]
        try:
            tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            failures.append(f"{row['staged_name']}: {str(exc).splitlines()[0][:120]}")
    return {"decoded": len(rows) - len(failures), "failures": failures}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def class_weights(counts: Dict[str, int], class_names: Sequence[str]) -> Dict[str, float]:
    """Same formula as ``src.model.datasets.compute_class_weights``, without TensorFlow."""
    present = [c for c in class_names if counts.get(c, 0)]
    total = sum(counts.get(c, 0) for c in class_names)
    raw = {c: (total / (len(present) * counts[c])) if counts.get(c, 0) else 1.0
           for c in class_names}
    mean = float(np.mean(list(raw.values())))
    return {c: v / mean for c, v in raw.items()}


def summarise(rows: List[Dict[str, str]], removed: List[Dict[str, str]],
              class_names: Sequence[str]) -> Dict[str, object]:
    fams = sorted({r["family"] for r in rows})
    srcs = sorted({r["source"] for r in rows})
    summary: Dict[str, object] = {"families": fams, "sources": srcs}
    for variant in VARIANTS:
        subset = [r for r in rows if variant == "full" or r["in_balanced"] == "1"]
        split_counts = {c: {s: 0 for s in SPLITS} for c in class_names}
        family_counts = {c: {f: 0 for f in fams} for c in class_names}
        source_counts = {c: {s: 0 for s in srcs} for c in class_names}
        family_split = {c: {s: {f: 0 for f in fams} for s in SPLITS} for c in class_names}
        for r in subset:
            split_counts[r["class"]][r["split"]] += 1
            family_counts[r["class"]][r["family"]] += 1
            source_counts[r["class"]][r["source"]] += 1
            family_split[r["class"]][r["split"]][r["family"]] += 1
        train = {c: split_counts[c]["train"] for c in class_names}
        summary[variant] = {
            "split_counts": split_counts, "family_counts": family_counts,
            "source_counts": source_counts, "family_split_counts": family_split,
            "class_totals": {c: sum(split_counts[c].values()) for c in class_names},
            "split_totals": {s: sum(split_counts[c][s] for c in class_names) for s in SPLITS},
            "class_weights_from_train": class_weights(train, class_names),
        }
    reasons = Counter((r["reason"], r["class"], r["source"]) for r in removed)
    summary["removed"] = [{"reason": k[0], "class": k[1], "source": k[2], "count": v}
                          for k, v in sorted(reasons.items())]
    return summary


def to_markdown(result: Optional[BuildResult], summary: Dict[str, object],
                checks: Dict[str, object], class_names: Sequence[str],
                external: Optional[Dict[str, object]] = None,
                decode: Optional[Dict[str, object]] = None) -> str:
    lines = ["# V3 dataset build report", ""]
    if result is not None:
        lines += [f"Generated {result.generated_utc}. Seed **{result.seed}**. Split "
                  f"{result.split_fractions}. Perceptual hash {result.hash_size}x"
                  f"{result.hash_size} ({result.hash_size ** 2} bits), near-duplicate "
                  f"distance <= {result.hash_threshold}. Balance ratio {result.balance_ratio:g} "
                  f"-> cap **{result.balanced_cap}**.", ""]
    fams = summary["families"]
    for variant in VARIANTS:
        data = summary[variant]
        lines += [f"## v3_{variant}", ""]
        rows = [[c] + [data["split_counts"][c][s] for s in SPLITS] + [data["class_totals"][c]]
                for c in class_names]
        rows.append(["**Total**"] + [data["split_totals"][s] for s in SPLITS]
                    + [sum(data["split_totals"].values())])
        lines += [markdown_table(["Class", *SPLITS, "Total"], rows), ""]
        rows = []
        for c in class_names:
            total = data["class_totals"][c] or 1
            rows.append([c] + [f"{data['family_counts'][c][f]} ({data['family_counts'][c][f] / total * 100:.0f}%)"
                               if data["family_counts"][c][f] else "-" for f in fams])
        lines += ["Source families per class:", "", markdown_table(["Class", *fams], rows), ""]
        weights = data["class_weights_from_train"]
        lines += ["Class weights (train split only, same formula as training): "
                  + ", ".join(f"{c} {weights[c]:.3f}" for c in class_names), ""]
    lines += ["## Verification", ""]
    for key in ["disk_matches_manifest", "sha256_matches_manifest",
                "exact_duplicates_across_splits", "near_duplicates_across_splits",
                "groups_spanning_splits", "balanced_not_nested_in_full",
                "external_paths_in_v3", "augmented_outside_train", "passed"]:
        if key in checks:
            lines.append(f"- `{key}`: {checks[key]}")
    if external is not None:
        lines += ["", f"- external test isolation: {external['isolated']} "
                      f"({external['external_images_checked']} images checked; exact matches "
                      f"{len(external['exact_matches_in_pool'])}, near-duplicate matches "
                      f"{len(external['near_duplicate_matches_in_pool'])})"]
    if decode is not None:
        lines += [f"- TensorFlow decode: {decode['decoded']} decoded, "
                  f"{len(decode['failures'])} failed"]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build and verify the V3 datasets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sources", help="directory holding the source clones")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", default=None, help="fingerprint cache (JSON)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--check-external", action="store_true",
                        help="also confirm, read-only, that no external-test image is in the pool")
    parser.add_argument("--check-decode", action="store_true",
                        help="also decode every image with TensorFlow, as training will")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    root = config.project_root
    threshold = int((config.get("data", "deduplication", default={}) or {}).get("hash_threshold", 5))

    result: Optional[BuildResult] = None
    if not args.verify_only:
        if not args.sources:
            parser.error("--sources is required unless --verify-only")
        result = build(config, Path(args.sources), root, workers=args.workers,
                       cache_path=Path(args.cache) if args.cache else None,
                       dry_run=args.dry_run)
        for warning in result.warnings:
            LOGGER.warning(warning)
        if args.dry_run:
            counts = Counter((im.class_name, im.split, im.in_balanced) for im in result.images)
            for key, value in sorted(counts.items()):
                LOGGER.info("%s", f"{key}: {value}")
            return 0

    checks = verify(root, threshold)
    external = check_external_isolation(root, config, threshold) if args.check_external else None
    decode = check_tensorflow_decode(root) if args.check_decode else None
    rows = read_manifest(root / MANIFEST_PATH)
    removed = read_manifest(root / REMOVED_PATH) if (root / REMOVED_PATH).exists() else []
    summary = summarise(rows, removed, config.class_names)

    out_dir = ensure_dir(config.path("results_dir") / "dataset")
    payload = {"summary": summary, "checks": checks, "external": external, "decode": decode}
    if result is not None:
        payload["build"] = {
            "generated_utc": result.generated_utc, "seed": result.seed,
            "split_fractions": result.split_fractions, "hash_size": result.hash_size,
            "hash_threshold": result.hash_threshold, "balance_ratio": result.balance_ratio,
            "balanced_cap": result.balanced_cap, "declared": result.declared,
            "invalid": result.invalid, "near_duplicate_groups": result.near_duplicate_groups,
            "images_in_near_duplicate_groups": result.images_in_near_duplicate_groups,
            "cross_class_groups": result.cross_class_groups, "warnings": result.warnings}
    write_json(out_dir / "v3_build_report.json", payload)
    write_text(out_dir / "v3_build_report.md",
               to_markdown(result, summary, checks, config.class_names, external, decode))
    LOGGER.info("wrote %s", out_dir / "v3_build_report.md")

    ok = bool(checks["passed"]) and (external is None or external["isolated"]) and (
        decode is None or not decode["failures"])
    LOGGER.info("verification %s", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
