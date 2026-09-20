"""Turn ``data/raw/<Class>/`` into leakage-free train / validation / test folders.

Pipeline
--------
1.  Resolve each raw folder to a configured class (aliases are honoured, so
    ``leaf_rust/`` is accepted for ``Brown Rust``).
2.  Reject unusable files (corrupt, truncated, wrong format, too small).
3.  Drop byte-identical duplicates, keeping one representative.
4.  Group near-duplicate photographs with a perceptual hash.
5.  Split **whole groups** - never individual images - stratified per class.
    This is what guarantees the same photograph cannot appear in both the
    training and the test set (see docs/dataset.md, "Preventing data leakage").
6.  Copy (or hard-link / symlink) the files into data/train, data/validation
    and data/test, and write a manifest recording exactly which file went where.

Run::

    python -m src.data.prepare_dataset
    python -m src.data.prepare_dataset --raw-dir data/raw --link
    python -m src.data.prepare_dataset --dry-run
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..utils.config import Config, load_config
from ..utils.helpers import (
    ensure_dir,
    get_logger,
    markdown_table,
    relative_to_root,
    set_global_seed,
    utc_timestamp,
    write_json,
    write_text,
)
from ..utils.image_io import (
    file_sha256,
    hamming_distance,
    inspect_image,
    iter_image_files,
    perceptual_hash,
)

LOGGER = get_logger(__name__)
SPLITS = ("train", "validation", "test")


@dataclass
class PreparedItem:
    source: Path
    class_name: str
    class_dir: str
    group_id: int
    split: str = ""
    destination: Optional[Path] = None


@dataclass
class PrepareResult:
    generated_utc: str
    raw_root: str
    seed: int
    requested_split: Dict[str, float]
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    totals: Dict[str, int] = field(default_factory=dict)
    skipped_invalid: List[Dict[str, str]] = field(default_factory=list)
    dropped_exact_duplicates: List[Dict[str, str]] = field(default_factory=list)
    near_duplicate_groups: int = 0
    grouped_images: int = 0
    warnings: List[str] = field(default_factory=list)
    manifest: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_utc": self.generated_utc,
            "raw_root": self.raw_root,
            "random_seed": self.seed,
            "requested_split": self.requested_split,
            "counts_per_class_per_split": self.counts,
            "totals_per_split": self.totals,
            "total_images": sum(self.totals.values()),
            "skipped_invalid": self.skipped_invalid,
            "dropped_exact_duplicates": self.dropped_exact_duplicates,
            "near_duplicate_groups": self.near_duplicate_groups,
            "images_in_near_duplicate_groups": self.grouped_images,
            "warnings": self.warnings,
        }


def discover_raw_items(config: Config, raw_root: Path) -> Tuple[List[PreparedItem], List[Dict[str, str]], List[str]]:
    """Collect usable images from the raw folder, resolving class aliases."""
    invalid: List[Dict[str, str]] = []
    warnings: List[str] = []
    items: List[PreparedItem] = []
    extensions = config.supported_extensions
    min_pixels = int(config.get("data", "min_image_pixels", default=32))

    if not raw_root.exists():
        warnings.append(f"raw directory does not exist: {raw_root}")
        return items, invalid, warnings

    # Map every sub-directory of the raw root onto a configured class.
    class_to_dirs: Dict[str, List[Path]] = defaultdict(list)
    for child in sorted(raw_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        spec = config.resolve_class(child.name)
        if spec is None:
            warnings.append(f"ignoring unrecognised folder '{child.name}' (not a configured class)")
            continue
        class_to_dirs[spec.name].append(child)

    for spec in config.classes:
        directories = class_to_dirs.get(spec.name, [])
        if not directories:
            warnings.append(f"no raw folder found for class '{spec.name}' "
                            f"(expected {raw_root.name}/{spec.directory})")
            continue
        if len(directories) > 1:
            warnings.append(
                f"class '{spec.name}' is spread over {len(directories)} folders "
                f"({', '.join(d.name for d in directories)}); all of them will be merged"
            )
        for directory in directories:
            for path in iter_image_files(directory, extensions):
                info = inspect_image(path, min_pixels=min_pixels, extensions=extensions)
                if not info.ok:
                    invalid.append({
                        "path": relative_to_root(path, config.project_root),
                        "class": spec.name,
                        "reason": info.reason,
                    })
                    continue
                items.append(PreparedItem(source=path, class_name=spec.name,
                                          class_dir=spec.directory, group_id=-1))
    return items, invalid, warnings


def drop_exact_duplicates(items: Sequence[PreparedItem], config: Config
                          ) -> Tuple[List[PreparedItem], List[Dict[str, str]]]:
    """Keep one representative per byte-identical file."""
    seen: Dict[str, PreparedItem] = {}
    kept: List[PreparedItem] = []
    dropped: List[Dict[str, str]] = []
    for item in items:
        try:
            digest = file_sha256(item.source)
        except OSError as exc:
            dropped.append({"path": relative_to_root(item.source, config.project_root),
                            "reason": f"unreadable: {exc}", "kept_instead": ""})
            continue
        if digest in seen:
            dropped.append({
                "path": relative_to_root(item.source, config.project_root),
                "reason": "byte-identical duplicate",
                "kept_instead": relative_to_root(seen[digest].source, config.project_root),
            })
            continue
        seen[digest] = item
        kept.append(item)
    return kept, dropped


def assign_duplicate_groups(items: List[PreparedItem], config: Config) -> Tuple[int, int, List[str]]:
    """Assign a ``group_id`` to each item; near-duplicates share an id.

    Returns ``(number_of_multi_image_groups, images_in_those_groups, warnings)``.
    Items whose hash could not be computed get their own singleton group, so a
    missing ``imagehash`` package degrades to "no grouping" rather than to
    silently merging unrelated pictures.
    """
    warnings: List[str] = []
    dedup_cfg = config.get("data", "deduplication", default={}) or {}
    if not bool(dedup_cfg.get("enabled", True)):
        for index, item in enumerate(items):
            item.group_id = index
        warnings.append("near-duplicate grouping disabled in config "
                        "(data.deduplication.enabled = false)")
        return 0, 0, warnings

    hash_size = int(dedup_cfg.get("hash_size", 8))
    threshold = int(dedup_cfg.get("hash_threshold", 5))

    # Grouping is done *within* a class: two visually similar leaves of
    # different classes are a labelling problem, not a split problem, and are
    # reported by validate_dataset instead.
    next_group = 0
    multi_groups = 0
    grouped_images = 0
    missing_hashes = 0

    by_class: Dict[str, List[PreparedItem]] = defaultdict(list)
    for item in items:
        by_class[item.class_name].append(item)

    for class_name, class_items in by_class.items():
        hashes: List[Tuple[PreparedItem, Optional[str]]] = []
        for item in class_items:
            hashes.append((item, perceptual_hash(item.source, hash_size=hash_size)))

        hashed = [(item, h) for item, h in hashes if h is not None]
        unhashed = [item for item, h in hashes if h is None]
        missing_hashes += len(unhashed)

        parent = list(range(len(hashed)))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for i in range(len(hashed)):
            for j in range(i + 1, len(hashed)):
                try:
                    if hamming_distance(hashed[i][1], hashed[j][1]) <= threshold:
                        root_i, root_j = find(i), find(j)
                        if root_i != root_j:
                            parent[root_j] = root_i
                except ValueError:
                    continue

        buckets: Dict[int, List[PreparedItem]] = defaultdict(list)
        for index, (item, _) in enumerate(hashed):
            buckets[find(index)].append(item)

        for members in buckets.values():
            for member in members:
                member.group_id = next_group
            if len(members) > 1:
                multi_groups += 1
                grouped_images += len(members)
            next_group += 1

        for item in unhashed:
            item.group_id = next_group
            next_group += 1

    if missing_hashes:
        warnings.append(
            f"{missing_hashes} image(s) could not be perceptually hashed and were treated as "
            f"unique; install 'imagehash' for full near-duplicate protection"
        )
    return multi_groups, grouped_images, warnings


def split_groups(items: List[PreparedItem], config: Config, seed: int) -> List[str]:
    """Stratified, group-aware split. Mutates ``item.split`` in place."""
    fractions = config.split_fractions()
    warnings: List[str] = []
    rng = random.Random(seed)

    by_class: Dict[str, Dict[int, List[PreparedItem]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_class[item.class_name][item.group_id].append(item)

    for class_name, groups in by_class.items():
        # Shuffle groups, then fill test and validation to their target image
        # counts before giving the remainder to training.  Group-level
        # assignment is what prevents leakage.
        group_list = list(groups.values())
        rng.shuffle(group_list)
        total_images = sum(len(g) for g in group_list)

        targets = {
            "test": fractions["test"] * total_images,
            "validation": fractions["validation"] * total_images,
        }
        assigned = {"train": 0, "validation": 0, "test": 0}

        for group in group_list:
            # Choose the split that is furthest below its target; training
            # absorbs everything once val/test are satisfied.
            if assigned["test"] < targets["test"]:
                chosen = "test"
            elif assigned["validation"] < targets["validation"]:
                chosen = "validation"
            else:
                chosen = "train"
            for member in group:
                member.split = chosen
            assigned[chosen] += len(group)

        for split in SPLITS:
            if assigned[split] == 0 and total_images > 0:
                warnings.append(
                    f"class '{class_name}' ended up with 0 images in '{split}' "
                    f"({total_images} usable images, "
                    f"{len(group_list)} near-duplicate group(s)) - the dataset is too small "
                    f"or too duplicated for the configured split"
                )
    return warnings


def _place_file(source: Path, destination: Path, mode: str) -> None:
    ensure_dir(destination.parent)
    if destination.exists():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    elif mode == "symlink":
        os.symlink(source.resolve(), destination)
    else:
        raise ValueError(f"unknown placement mode: {mode}")


def prepare(config: Config, raw_root: Optional[Path] = None, mode: str = "copy",
            clean: bool = True, dry_run: bool = False) -> PrepareResult:
    """Run the full preparation pipeline."""
    seed = config.seed
    set_global_seed(seed)
    raw_root = Path(raw_root) if raw_root else config.path("raw_dir")
    if not raw_root.is_absolute():
        raw_root = config.project_root / raw_root

    result = PrepareResult(
        generated_utc=utc_timestamp(),
        raw_root=str(raw_root),
        seed=seed,
        requested_split=config.split_fractions(),
    )

    items, invalid, warnings = discover_raw_items(config, raw_root)
    result.skipped_invalid = invalid
    result.warnings.extend(warnings)

    if not items:
        result.warnings.append(
            "no usable images found - nothing to prepare. See docs/dataset.md for the "
            "expected layout and for how to obtain a dataset."
        )
        return result

    dedup_cfg = config.get("data", "deduplication", default={}) or {}
    if bool(dedup_cfg.get("drop_exact_duplicates", True)):
        items, dropped = drop_exact_duplicates(items, config)
        result.dropped_exact_duplicates = dropped

    multi_groups, grouped_images, group_warnings = assign_duplicate_groups(items, config)
    result.near_duplicate_groups = multi_groups
    result.grouped_images = grouped_images
    result.warnings.extend(group_warnings)

    result.warnings.extend(split_groups(items, config, seed))

    counts: Dict[str, Dict[str, int]] = {
        name: {split: 0 for split in SPLITS} for name in config.class_names
    }
    totals = {split: 0 for split in SPLITS}

    if clean and not dry_run:
        for split in SPLITS:
            split_root = config.path(f"{split}_dir")
            for spec in config.classes:
                target = split_root / spec.directory
                if target.exists():
                    shutil.rmtree(target)

    for item in items:
        split_root = config.path(f"{item.split}_dir")
        # Prefix with the class folder to keep names unique across raw sources.
        destination = split_root / item.class_dir / item.source.name
        counter = 1
        while destination.exists() and not dry_run:
            destination = split_root / item.class_dir / f"{item.source.stem}_{counter}{item.source.suffix}"
            counter += 1
        item.destination = destination
        counts[item.class_name][item.split] += 1
        totals[item.split] += 1
        if not dry_run:
            _place_file(item.source, destination, mode)
        result.manifest.append({
            "source": relative_to_root(item.source, config.project_root),
            "destination": relative_to_root(destination, config.project_root),
            "class": item.class_name,
            "split": item.split,
            "group_id": str(item.group_id),
        })

    result.counts = counts
    result.totals = totals
    return result


def result_to_markdown(result: PrepareResult, config: Config) -> str:
    rows = []
    for class_name in config.class_names:
        per_split = result.counts.get(class_name, {s: 0 for s in SPLITS})
        total = sum(per_split.values())
        rows.append([class_name, per_split["train"], per_split["validation"],
                     per_split["test"], total])
    grand_total = sum(result.totals.values())
    rows.append(["**Total**", result.totals.get("train", 0), result.totals.get("validation", 0),
                 result.totals.get("test", 0), grand_total])

    requested = result.requested_split
    actual = {
        split: (result.totals.get(split, 0) / grand_total) if grand_total else 0.0
        for split in SPLITS
    }

    lines = [
        "# Dataset preparation report",
        "",
        f"* Generated (UTC): {result.generated_utc}",
        f"* Raw root: `{result.raw_root}`",
        f"* Random seed: {result.seed}",
        f"* Requested split: train {requested['train']:.1%} / "
        f"validation {requested['validation']:.1%} / test {requested['test']:.1%}",
        f"* Actual split: train {actual['train']:.1%} / "
        f"validation {actual['validation']:.1%} / test {actual['test']:.1%}",
        f"* Images dropped as byte-identical duplicates: {len(result.dropped_exact_duplicates)}",
        f"* Images skipped as unusable: {len(result.skipped_invalid)}",
        f"* Near-duplicate groups kept together in one split: {result.near_duplicate_groups} "
        f"({result.grouped_images} images)",
        "",
        "## Images per class and split",
        "",
        markdown_table(["Class", "Train", "Validation", "Test", "Total"], rows),
        "",
    ]
    if result.warnings:
        lines += ["## Warnings", ""] + [f"* {w}" for w in result.warnings] + [""]
    lines += [
        "## Leakage protection",
        "",
        "Near-duplicate photographs are grouped with a perceptual hash and the **whole group** "
        "is assigned to a single split, so a picture (or a near-identical re-shot of it) cannot "
        "appear in both training and test data. Verify with:",
        "",
        "```bash",
        "python -m src.data.validate_dataset --split-report",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare train/validation/test splits from data/raw")
    parser.add_argument("--config", default=None)
    parser.add_argument("--raw-dir", default=None, help="override paths.raw_dir")
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy",
                        help="how prepared files are materialised (default: copy)")
    parser.add_argument("--link", action="store_const", const="hardlink", dest="mode",
                        help="shorthand for --mode hardlink (saves disk space)")
    parser.add_argument("--no-clean", action="store_true",
                        help="keep existing files in data/train|validation|test")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen without writing anything")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = prepare(config, raw_root=args.raw_dir, mode=args.mode,
                     clean=not args.no_clean, dry_run=args.dry_run)

    out_dir = ensure_dir(config.path("results_dir") / "dataset")
    write_json(out_dir / "prepare_report.json", result.to_dict())
    write_json(out_dir / "split_manifest.json", result.manifest)
    write_text(out_dir / "prepare_report.md", result_to_markdown(result, config))

    print(result_to_markdown(result, config))
    for warning in result.warnings:
        LOGGER.warning(warning)

    if not result.manifest:
        LOGGER.error("nothing was prepared - data/raw contains no usable images.")
        LOGGER.error("Place images as data/raw/<Class_Name>/<image>.jpg, for example:")
        for spec in config.classes:
            LOGGER.error("  data/raw/%s/", spec.directory)
        return 1

    if args.dry_run:
        LOGGER.info("dry run: no files were written")
    else:
        LOGGER.info("prepared %d images into data/train, data/validation and data/test",
                    sum(result.totals.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
