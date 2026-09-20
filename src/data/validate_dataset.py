"""Dataset validation and statistics for Leaf Lens.

Scans an image folder laid out as ``<root>/<Class_Name>/<image files>`` and
reports everything that has to be true before training is worth starting:

* which of the configured classes are present / missing
* how many usable images each class has
* corrupt, truncated, empty or unsupported files
* byte-identical duplicates and near-duplicate photographs
* class imbalance
* a class-distribution chart

Run::

    python -m src.data.validate_dataset                      # validates data/raw
    python -m src.data.validate_dataset --root data/train
    python -m src.data.validate_dataset --split-report       # train/val/test at once
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.config import Config, load_config
from ..utils.helpers import (
    configure_matplotlib,
    ensure_dir,
    get_logger,
    markdown_table,
    relative_to_root,
    utc_timestamp,
    write_json,
    write_text,
)
from ..utils.image_io import (
    ImageInfo,
    file_sha256,
    hamming_distance,
    inspect_image,
    iter_image_files,
    perceptual_hash,
)

LOGGER = get_logger(__name__)


@dataclass
class ClassReport:
    name: str
    directory: str
    present: bool
    total_files: int = 0
    valid_images: int = 0
    invalid_images: int = 0
    percentage_of_valid: float = 0.0
    mean_width: float = 0.0
    mean_height: float = 0.0
    formats: Dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetReport:
    root: str
    generated_utc: str
    classes_configured: List[str]
    classes_present: List[str]
    classes_missing: List[str]
    unexpected_directories: List[str]
    total_files_scanned: int
    total_valid_images: int
    total_invalid_images: int
    per_class: List[ClassReport]
    invalid_files: List[Dict[str, str]]
    exact_duplicate_groups: List[List[str]]
    near_duplicate_groups: List[List[str]]
    cross_class_duplicate_groups: List[List[str]]
    imbalance_ratio: Optional[float]
    warnings: List[str]
    errors: List[str]

    @property
    def usable(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["usable_for_training"] = self.usable
        return payload


def _group_near_duplicates(
    hashes: Dict[Path, str], threshold: int
) -> List[List[Path]]:
    """Union-find grouping of images whose pHash distance is <= ``threshold``.

    O(n^2) in the number of images.  Acceptable for the dataset sizes this
    project targets (thousands); the caller can disable it in config.yaml.
    """
    paths = list(hashes.keys())
    parent = {p: p for p in paths}

    def find(node: Path) -> Path:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: Path, b: Path) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            try:
                if hamming_distance(hashes[paths[i]], hashes[paths[j]]) <= threshold:
                    union(paths[i], paths[j])
            except ValueError:
                continue

    groups: Dict[Path, List[Path]] = defaultdict(list)
    for path in paths:
        groups[find(path)].append(path)
    return [sorted(members) for members in groups.values() if len(members) > 1]


def validate_dataset(
    config: Config,
    root: Path,
    check_duplicates: bool = True,
    max_images_for_duplicates: int = 6000,
) -> DatasetReport:
    """Scan ``root`` and produce a :class:`DatasetReport`."""
    root = Path(root)
    extensions = config.supported_extensions
    min_pixels = int(config.get("data", "min_image_pixels", default=32))

    warnings: List[str] = []
    errors: List[str] = []
    per_class: List[ClassReport] = []
    invalid_files: List[Dict[str, str]] = []
    valid_by_class: Dict[str, List[ImageInfo]] = {}

    if not root.exists():
        errors.append(f"dataset root does not exist: {root}")
        return DatasetReport(
            root=str(root), generated_utc=utc_timestamp(),
            classes_configured=config.class_names, classes_present=[],
            classes_missing=config.class_names, unexpected_directories=[],
            total_files_scanned=0, total_valid_images=0, total_invalid_images=0,
            per_class=[], invalid_files=[], exact_duplicate_groups=[],
            near_duplicate_groups=[], cross_class_duplicate_groups=[],
            imbalance_ratio=None, warnings=warnings, errors=errors,
        )

    # --- per class scan ---------------------------------------------------
    for spec in config.classes:
        class_dir = root / spec.directory
        if not class_dir.is_dir():
            per_class.append(ClassReport(name=spec.name, directory=spec.directory, present=False))
            valid_by_class[spec.name] = []
            continue

        infos = [
            inspect_image(path, min_pixels=min_pixels, extensions=extensions)
            for path in iter_image_files(class_dir, extensions)
        ]
        # Files with an unsupported extension are not returned by
        # iter_image_files; count them separately so nothing is hidden.
        unsupported = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() not in {e.lower() for e in extensions}
            and p.name != ".gitkeep"
        ]
        for path in unsupported:
            invalid_files.append({
                "path": relative_to_root(path, config.project_root),
                "class": spec.name,
                "reason": f"unsupported file type {path.suffix or '(no extension)'}",
            })

        valid = [i for i in infos if i.ok]
        invalid = [i for i in infos if not i.ok]
        for info in invalid:
            invalid_files.append({
                "path": relative_to_root(info.path, config.project_root),
                "class": spec.name,
                "reason": info.reason,
            })

        formats: Dict[str, int] = defaultdict(int)
        for info in valid:
            formats[info.image_format or "UNKNOWN"] += 1

        per_class.append(ClassReport(
            name=spec.name,
            directory=spec.directory,
            present=True,
            total_files=len(infos) + len(unsupported),
            valid_images=len(valid),
            invalid_images=len(invalid) + len(unsupported),
            mean_width=round(sum(i.width for i in valid) / len(valid), 1) if valid else 0.0,
            mean_height=round(sum(i.height for i in valid) / len(valid), 1) if valid else 0.0,
            formats=dict(sorted(formats.items())),
        ))
        valid_by_class[spec.name] = valid

    total_valid = sum(c.valid_images for c in per_class)
    for report in per_class:
        report.percentage_of_valid = (
            round(100.0 * report.valid_images / total_valid, 2) if total_valid else 0.0
        )

    classes_present = [c.name for c in per_class if c.present and c.valid_images > 0]
    classes_missing = [c.name for c in per_class if c.name not in classes_present]

    known_dirs = {spec.directory for spec in config.classes}
    unexpected = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and d.name not in known_dirs and not d.name.startswith(".")
    )
    for name in unexpected:
        resolved = config.resolve_class(name)
        if resolved is not None:
            warnings.append(
                f"directory '{name}' matches configured class '{resolved.name}' by alias but is "
                f"not named '{resolved.directory}'. Rename it or run prepare_dataset, which "
                f"resolves aliases automatically."
            )
        else:
            warnings.append(f"directory '{name}' is not a configured class and will be ignored")

    # --- duplicate detection ---------------------------------------------
    exact_groups: List[List[str]] = []
    near_groups: List[List[str]] = []
    cross_class_groups: List[List[str]] = []

    dedup_cfg = config.get("data", "deduplication", default={}) or {}
    if check_duplicates and total_valid:
        all_valid: List[Tuple[str, ImageInfo]] = [
            (class_name, info) for class_name, infos in valid_by_class.items() for info in infos
        ]
        class_of: Dict[Path, str] = {info.path: name for name, info in all_valid}

        digests: Dict[str, List[Path]] = defaultdict(list)
        for _, info in all_valid:
            try:
                digests[file_sha256(info.path)].append(info.path)
            except OSError:
                continue
        exact_path_groups = [sorted(v) for v in digests.values() if len(v) > 1]
        exact_groups = [
            [relative_to_root(p, config.project_root) for p in group] for group in exact_path_groups
        ]

        if bool(dedup_cfg.get("enabled", True)):
            if len(all_valid) > max_images_for_duplicates:
                warnings.append(
                    f"near-duplicate scan skipped: {len(all_valid)} images exceeds the "
                    f"{max_images_for_duplicates}-image limit for the O(n^2) comparison "
                    f"(raise --max-duplicate-images to force it)"
                )
            else:
                hash_size = int(dedup_cfg.get("hash_size", 8))
                threshold = int(dedup_cfg.get("hash_threshold", 5))
                hashes: Dict[Path, str] = {}
                for _, info in all_valid:
                    digest = perceptual_hash(info.path, hash_size=hash_size)
                    if digest is not None:
                        hashes[info.path] = digest
                if not hashes:
                    warnings.append(
                        "near-duplicate scan skipped: the 'imagehash' package is not installed"
                    )
                else:
                    path_groups = _group_near_duplicates(hashes, threshold)
                    near_groups = [
                        [relative_to_root(p, config.project_root) for p in group]
                        for group in path_groups
                    ]
                    for group in path_groups:
                        labels = {class_of.get(p) for p in group}
                        if len(labels) > 1:
                            cross_class_groups.append(
                                [relative_to_root(p, config.project_root) for p in group]
                            )

    # --- warnings and errors ---------------------------------------------
    if classes_missing:
        errors.append(
            "no usable images for: " + ", ".join(classes_missing)
            + f". Expected folders under {root}: "
            + ", ".join(f"{root.name}/{config.resolve_class(m).directory}" for m in classes_missing)
        )
    if total_valid == 0:
        errors.append(f"no valid images found under {root}")

    min_per_class = int(config.get("data", "min_images_per_class", default=20))
    for report in per_class:
        if report.present and 0 < report.valid_images < min_per_class:
            warnings.append(
                f"class '{report.name}' has only {report.valid_images} usable images "
                f"(configured minimum: {min_per_class})"
            )

    counts = [c.valid_images for c in per_class if c.valid_images > 0]
    imbalance_ratio = round(max(counts) / min(counts), 2) if len(counts) > 1 else None
    warn_ratio = float(config.get("data", "imbalance_warn_ratio", default=3.0))
    if imbalance_ratio is not None and imbalance_ratio > warn_ratio:
        warnings.append(
            f"class imbalance ratio {imbalance_ratio}:1 exceeds the configured "
            f"threshold of {warn_ratio}:1 (largest vs smallest class)"
        )
    if invalid_files:
        warnings.append(f"{len(invalid_files)} file(s) could not be used - see invalid_files")
    if cross_class_groups:
        warnings.append(
            f"{len(cross_class_groups)} near-duplicate group(s) span more than one class - "
            f"these are labelling conflicts and should be reviewed by hand"
        )

    return DatasetReport(
        root=str(root),
        generated_utc=utc_timestamp(),
        classes_configured=config.class_names,
        classes_present=classes_present,
        classes_missing=classes_missing,
        unexpected_directories=unexpected,
        total_files_scanned=sum(c.total_files for c in per_class),
        total_valid_images=total_valid,
        total_invalid_images=sum(c.invalid_images for c in per_class),
        per_class=per_class,
        invalid_files=invalid_files,
        exact_duplicate_groups=exact_groups,
        near_duplicate_groups=near_groups,
        cross_class_duplicate_groups=cross_class_groups,
        imbalance_ratio=imbalance_ratio,
        warnings=warnings,
        errors=errors,
    )


def plot_class_distribution(report: DatasetReport, output_path: Path,
                            title: str = "Class distribution") -> Optional[Path]:
    """Bar chart of usable images per class."""
    configure_matplotlib()
    import matplotlib.pyplot as plt

    names = [c.name for c in report.per_class]
    counts = [c.valid_images for c in report.per_class]
    if not any(counts):
        return None

    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    bars = axis.bar(names, counts, color="#3f7d3f")
    axis.set_ylabel("Usable images")
    axis.set_title(f"{title}\n({report.total_valid_images} images total)")
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=20, ha="right")
    for bar, count in zip(bars, counts):
        axis.annotate(str(count), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom", fontsize=9)
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def plot_split_distribution(reports: Dict[str, DatasetReport], output_path: Path) -> Optional[Path]:
    """Grouped bar chart comparing train / validation / test counts per class."""
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    usable = {k: v for k, v in reports.items() if v.total_valid_images}
    if not usable:
        return None

    class_names = next(iter(usable.values())).classes_configured
    positions = np.arange(len(class_names))
    width = 0.8 / len(usable)
    colors = {"train": "#3f7d3f", "validation": "#d79a2b", "test": "#3b6ea5"}

    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    for index, (split_name, report) in enumerate(usable.items()):
        lookup = {c.name: c.valid_images for c in report.per_class}
        counts = [lookup.get(name, 0) for name in class_names]
        axis.bar(positions + index * width, counts, width,
                 label=f"{split_name} ({report.total_valid_images})",
                 color=colors.get(split_name, None))
    axis.set_xticks(positions + width * (len(usable) - 1) / 2)
    axis.set_xticklabels(class_names, rotation=20, ha="right")
    axis.set_ylabel("Images")
    axis.set_title("Images per class and split")
    axis.legend()
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def report_to_markdown(report: DatasetReport, config: Config) -> str:
    """Human-readable version of a :class:`DatasetReport`."""
    lines = [
        "# Dataset validation report",
        "",
        f"* Root: `{report.root}`",
        f"* Generated (UTC): {report.generated_utc}",
        f"* Total files scanned: {report.total_files_scanned}",
        f"* Usable images: **{report.total_valid_images}**",
        f"* Unusable files: {report.total_invalid_images}",
        f"* Class imbalance ratio (largest:smallest): "
        f"{report.imbalance_ratio if report.imbalance_ratio is not None else 'n/a'}",
        f"* Usable for training: **{'yes' if report.usable else 'no'}**",
        "",
        "## Images per class",
        "",
        markdown_table(
            ["Class", "Folder", "Present", "Usable", "% of usable", "Unusable", "Mean size (px)"],
            [
                [c.name, f"`{c.directory}`", "yes" if c.present else "**no**", c.valid_images,
                 f"{c.percentage_of_valid:.2f}%", c.invalid_images,
                 f"{c.mean_width:.0f}x{c.mean_height:.0f}" if c.valid_images else "-"]
                for c in report.per_class
            ],
        ),
        "",
    ]

    if report.errors:
        lines += ["## Errors (block training)", ""] + [f"* {e}" for e in report.errors] + [""]
    if report.warnings:
        lines += ["## Warnings", ""] + [f"* {w}" for w in report.warnings] + [""]

    lines += [
        "## Duplicates",
        "",
        f"* Byte-identical groups: {len(report.exact_duplicate_groups)}",
        f"* Near-duplicate groups (perceptual hash): {len(report.near_duplicate_groups)}",
        f"* Near-duplicate groups spanning two or more classes: "
        f"{len(report.cross_class_duplicate_groups)}",
        "",
    ]
    if report.cross_class_duplicate_groups:
        lines += ["Cross-class groups (possible labelling conflicts):", ""]
        for group in report.cross_class_duplicate_groups[:20]:
            lines.append("* " + ", ".join(f"`{p}`" for p in group))
        lines.append("")

    if report.invalid_files:
        lines += ["## Unusable files (first 50)", "",
                  markdown_table(["File", "Class", "Reason"],
                                 [[f"`{f['path']}`", f["class"], f["reason"]]
                                  for f in report.invalid_files[:50]]),
                  ""]
    return "\n".join(lines)


def _write_outputs(report: DatasetReport, config: Config, label: str) -> Dict[str, Path]:
    out_dir = config.path("results_dir") / "dataset"
    ensure_dir(out_dir)
    written: Dict[str, Path] = {}
    written["json"] = write_json(out_dir / f"validation_{label}.json", report.to_dict())
    written["markdown"] = write_text(out_dir / f"validation_{label}.md",
                                     report_to_markdown(report, config))
    chart = plot_class_distribution(
        report,
        config.path("results_dir") / "graphs" / f"class_distribution_{label}.png",
        title=f"Class distribution - {label}",
    )
    if chart:
        written["chart"] = chart
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Leaf Lens image dataset")
    parser.add_argument("--config", default=None, help="alternative config file")
    parser.add_argument("--root", default=None,
                        help="dataset root to validate (default: paths.raw_dir)")
    parser.add_argument("--label", default=None,
                        help="label used in output filenames (default: folder name)")
    parser.add_argument("--split-report", action="store_true",
                        help="validate data/train, data/validation and data/test together")
    parser.add_argument("--no-duplicates", action="store_true",
                        help="skip duplicate detection (much faster on large datasets)")
    parser.add_argument("--max-duplicate-images", type=int, default=6000,
                        help="upper bound for the O(n^2) near-duplicate scan")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.split_report:
        reports: Dict[str, DatasetReport] = {}
        for split in ("train", "validation", "test"):
            root = config.path(f"{split}_dir")
            report = validate_dataset(config, root, check_duplicates=not args.no_duplicates,
                                      max_images_for_duplicates=args.max_duplicate_images)
            reports[split] = report
            _write_outputs(report, config, split)
            LOGGER.info("%-11s %5d usable images (%s)", split, report.total_valid_images,
                        "OK" if report.usable else "; ".join(report.errors))
        chart = plot_split_distribution(
            reports, config.path("results_dir") / "graphs" / "class_distribution_splits.png"
        )
        if chart:
            LOGGER.info("split distribution chart: %s", chart)
        # Leakage check across the prepared splits.
        leak = _cross_split_leakage(config, reports)
        write_json(config.path("results_dir") / "dataset" / "split_leakage_check.json", leak)
        if leak["identical_files_across_splits"] or leak["near_duplicates_across_splits"]:
            LOGGER.error("DATA LEAKAGE DETECTED between splits - see %s",
                         config.path("results_dir") / "dataset" / "split_leakage_check.json")
            return 1
        LOGGER.info("no leakage detected between train/validation/test")
        return 0 if all(r.usable for r in reports.values()) else 1

    root = Path(args.root) if args.root else config.path("raw_dir")
    if not root.is_absolute():
        root = config.project_root / root
    label = args.label or root.name
    report = validate_dataset(config, root, check_duplicates=not args.no_duplicates,
                              max_images_for_duplicates=args.max_duplicate_images)
    written = _write_outputs(report, config, label)

    if not args.quiet:
        print(report_to_markdown(report, config))
    for key, path in written.items():
        LOGGER.info("wrote %s: %s", key, path)

    if not report.usable:
        LOGGER.error("dataset is NOT usable for training yet:")
        for error in report.errors:
            LOGGER.error("  - %s", error)
        LOGGER.error("See docs/dataset.md for how to obtain and lay out the images.")
        return 1
    return 0


def _cross_split_leakage(config: Config, reports: Dict[str, DatasetReport]) -> dict:
    """Check that no photograph appears in more than one prepared split."""
    from ..utils.image_io import file_sha256, perceptual_hash

    dedup_cfg = config.get("data", "deduplication", default={}) or {}
    threshold = int(dedup_cfg.get("hash_threshold", 5))
    hash_size = int(dedup_cfg.get("hash_size", 8))

    per_split_files: Dict[str, List[Path]] = {}
    for split in reports:
        root = config.path(f"{split}_dir")
        per_split_files[split] = list(iter_image_files(root, config.supported_extensions))

    digest_map: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    phash_map: List[Tuple[str, Path, str]] = []
    for split, files in per_split_files.items():
        for path in files:
            try:
                digest_map[file_sha256(path)].append((split, path))
            except OSError:
                continue
            digest = perceptual_hash(path, hash_size=hash_size)
            if digest is not None:
                phash_map.append((split, path, digest))

    identical = [
        [f"{split}:{relative_to_root(p, config.project_root)}" for split, p in entries]
        for entries in digest_map.values()
        if len({split for split, _ in entries}) > 1
    ]

    near: List[List[str]] = []
    for i in range(len(phash_map)):
        split_a, path_a, hash_a = phash_map[i]
        for j in range(i + 1, len(phash_map)):
            split_b, path_b, hash_b = phash_map[j]
            if split_a == split_b:
                continue
            try:
                if hamming_distance(hash_a, hash_b) <= threshold:
                    near.append([
                        f"{split_a}:{relative_to_root(path_a, config.project_root)}",
                        f"{split_b}:{relative_to_root(path_b, config.project_root)}",
                    ])
            except ValueError:
                continue

    return {
        "generated_utc": utc_timestamp(),
        "hash_threshold": threshold,
        "images_per_split": {k: len(v) for k, v in per_split_files.items()},
        "identical_files_across_splits": identical,
        "near_duplicates_across_splits": near,
        "leakage_detected": bool(identical or near),
    }


if __name__ == "__main__":
    sys.exit(main())
