"""Measure how much the v3 sources actually overlap with each other.

The v3 hypothesis is that drawing each class from several independent
collections removes the regularities a single collection imposes. That
argument collapses if the collections are not independent - if two of them
redistribute the same photographs, then "five sources" is one source counted
five times, and the extra images buy nothing but a larger number.

This module measures that before any dataset is built. It reports, per class:

* images that fail validation (corrupt, truncated, wrong format, too small);
* **exact** duplicates by SHA-256, separated into within-source and
  cross-source, because only the cross-source ones threaten the hypothesis;
* **near** duplicates by perceptual hash, grouped with union-find at the
  thresholds already calibrated in ``config.yaml`` (hash_size 16, distance 5),
  again separated into within-source and cross-source;
* a source-pair matrix showing which particular collections share material.

The stop criterion, fixed in advance
------------------------------------
Deciding what counts as "too much overlap" after seeing the number is how
results get rationalised, so the thresholds are declared here instead:

* **CLASS_OVERLAP_LIMIT** - more than 5% of a class's images sitting in
  near-duplicate groups that span sources;
* **PAIR_OVERLAP_LIMIT** - any two sources sharing more than 20% of the
  smaller one's contribution to a class.

Either being exceeded means the sampling plan is reporting diversity it does
not have, and the run stops for a human decision rather than proceeding.
``verdict()`` returns that judgement; it does not decide anything by itself.

Run::

    python -m src.data.overlap_check --sources <clone-dir>
    python -m src.data.overlap_check --sources <clone-dir> --workers 4
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..utils.config import Config, load_config
from ..utils.helpers import ensure_dir, get_logger, markdown_table, utc_timestamp, write_json, write_text
from ..utils.image_io import file_sha256, inspect_image, iter_image_files, perceptual_hash
from .sources_v3 import active_sources

LOGGER = get_logger(__name__)

#: Fraction of a class's images allowed to sit in cross-source near-duplicate groups.
CLASS_OVERLAP_LIMIT = 0.05
#: Fraction of the smaller source's contribution allowed to be shared with another source.
PAIR_OVERLAP_LIMIT = 0.20


@dataclass
class ImageRecord:
    path: Path
    source: str
    class_name: str
    ok: bool = True
    reason: str = ""
    sha256: str = ""
    phash: Optional[str] = None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def _fingerprint(job: Tuple[str, str, str, int, int]) -> dict:
    """Validate and fingerprint one image. Runs in a worker process."""
    path_str, source, class_name, min_pixels, hash_size = job
    path = Path(path_str)
    info = inspect_image(path, min_pixels=min_pixels)
    if not info.ok:
        return {"path": path_str, "source": source, "class_name": class_name,
                "ok": False, "reason": info.reason}
    return {"path": path_str, "source": source, "class_name": class_name, "ok": True,
            "sha256": file_sha256(path), "phash": perceptual_hash(path, hash_size=hash_size)}


def collect_records(config: Config, clone_root: Path, workers: int = 1,
                    limit_per_folder: Optional[int] = None) -> List[ImageRecord]:
    """Validate and fingerprint every mapped image across every active source."""
    extensions = config.supported_extensions
    min_pixels = int(config.get("data", "min_image_pixels", default=32))
    dedup = config.get("data", "deduplication", default={}) or {}
    hash_size = int(dedup.get("hash_size", 8))

    jobs: List[Tuple[str, str, str, int, int]] = []
    for source in active_sources():
        for folder in source.folders:
            if not folder.leaf_lens_class:
                continue
            directory = clone_root / source.key / folder.path
            if not directory.is_dir():
                LOGGER.warning("missing folder (declared in the registry): %s", directory)
                continue
            paths = sorted(iter_image_files(directory, extensions))
            if limit_per_folder is not None:
                paths = paths[:limit_per_folder]
            jobs.extend((str(p), source.key, folder.leaf_lens_class, min_pixels, hash_size)
                        for p in paths)

    LOGGER.info("fingerprinting %d images with %d worker(s)", len(jobs), workers)
    results: List[dict] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, payload in enumerate(pool.map(_fingerprint, jobs, chunksize=16), start=1):
                results.append(payload)
                if index % 1000 == 0:
                    LOGGER.info("  %d / %d", index, len(jobs))
    else:
        for index, job in enumerate(jobs, start=1):
            results.append(_fingerprint(job))
            if index % 1000 == 0:
                LOGGER.info("  %d / %d", index, len(jobs))

    return [ImageRecord(path=Path(r["path"]), source=r["source"], class_name=r["class_name"],
                        ok=r["ok"], reason=r.get("reason", ""), sha256=r.get("sha256", ""),
                        phash=r.get("phash"))
            for r in results]


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def _bit_distance(a: int, b: int) -> int:
    """Hamming distance between two hashes held as integers.

    Identical in meaning to ``image_io.hamming_distance``, which parses hex on
    every call; at ~10 million pairwise comparisons that parsing dominates, so
    the hashes are converted once. ``tests/test_overlap_check.py`` pins that
    the two agree.
    """
    return (a ^ b).bit_count()


def group_near_duplicates(records: List[ImageRecord], threshold: int) -> List[List[int]]:
    """Union-find over perceptual hashes; returns groups as lists of indices.

    Records without a hash become singletons, so a missing ``imagehash``
    degrades to "no grouping" rather than to merging unrelated pictures.
    """
    hashed = [(index, int(record.phash, 16))
              for index, record in enumerate(records) if record.phash]
    parent = {index: index for index, _ in hashed}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(len(hashed)):
        index_i, value_i = hashed[i]
        for j in range(i + 1, len(hashed)):
            index_j, value_j = hashed[j]
            if _bit_distance(value_i, value_j) <= threshold:
                root_i, root_j = find(index_i), find(index_j)
                if root_i != root_j:
                    parent[root_j] = root_i

    buckets: Dict[int, List[int]] = defaultdict(list)
    for index, _ in hashed:
        buckets[find(index)].append(index)
    groups = [sorted(members) for members in buckets.values()]
    unhashed = [index for index, record in enumerate(records) if not record.phash]
    groups.extend([index] for index in unhashed)
    return groups


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
@dataclass
class ClassOverlap:
    class_name: str
    counts_by_source: Dict[str, int] = field(default_factory=dict)
    invalid: List[Dict[str, str]] = field(default_factory=list)
    exact_duplicate_pairs_within: int = 0
    exact_duplicate_pairs_cross: int = 0
    exact_cross_images: int = 0
    near_groups_multi: int = 0
    near_groups_cross: int = 0
    near_cross_images: int = 0
    pair_shared: Dict[str, int] = field(default_factory=dict)
    largest_cross_group: int = 0
    examples: List[List[str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.counts_by_source.values())

    @property
    def cross_source_rate(self) -> float:
        return (self.near_cross_images / self.total) if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "total_valid": self.total,
            "counts_by_source": self.counts_by_source,
            "invalid": self.invalid,
            "exact_duplicate_pairs_within_source": self.exact_duplicate_pairs_within,
            "exact_duplicate_pairs_cross_source": self.exact_duplicate_pairs_cross,
            "exact_cross_source_images": self.exact_cross_images,
            "near_duplicate_groups_multi_image": self.near_groups_multi,
            "near_duplicate_groups_cross_source": self.near_groups_cross,
            "near_cross_source_images": self.near_cross_images,
            "cross_source_rate": round(self.cross_source_rate, 4),
            "largest_cross_source_group": self.largest_cross_group,
            "source_pair_shared_groups": self.pair_shared,
            "examples": self.examples,
        }


def analyse(records: List[ImageRecord], threshold: int) -> List[ClassOverlap]:
    """Per-class exact- and near-duplicate analysis, split by source."""
    by_class: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        by_class[record.class_name].append(record)

    reports: List[ClassOverlap] = []
    for class_name, all_records in sorted(by_class.items()):
        report = ClassOverlap(class_name=class_name)
        report.invalid = [{"path": str(r.path), "source": r.source, "reason": r.reason}
                          for r in all_records if not r.ok]
        valid = [r for r in all_records if r.ok]
        for record in valid:
            report.counts_by_source[record.source] = report.counts_by_source.get(record.source, 0) + 1

        # Exact duplicates.
        by_sha: Dict[str, List[ImageRecord]] = defaultdict(list)
        for record in valid:
            by_sha[record.sha256].append(record)
        for members in by_sha.values():
            if len(members) < 2:
                continue
            sources = {m.source for m in members}
            if len(sources) > 1:
                report.exact_duplicate_pairs_cross += 1
                report.exact_cross_images += len(members)
            else:
                report.exact_duplicate_pairs_within += 1

        # Near duplicates.
        groups = group_near_duplicates(valid, threshold)
        for members in groups:
            if len(members) < 2:
                continue
            report.near_groups_multi += 1
            sources = {valid[i].source for i in members}
            if len(sources) < 2:
                continue
            report.near_groups_cross += 1
            report.near_cross_images += len(members)
            report.largest_cross_group = max(report.largest_cross_group, len(members))
            for a, b in combinations(sorted(sources), 2):
                key = f"{a}|{b}"
                report.pair_shared[key] = report.pair_shared.get(key, 0) + 1
            if len(report.examples) < 10:
                report.examples.append([str(valid[i].path) for i in members[:4]])
        reports.append(report)
    return reports


def verdict(reports: List[ClassOverlap]) -> Dict[str, object]:
    """Apply the thresholds declared at the top of this module.

    Returns the judgement and the reasons for it. It stops nothing by itself -
    the caller decides what to do, and a human decides whether to continue.
    """
    breaches: List[str] = []
    for report in reports:
        if report.cross_source_rate > CLASS_OVERLAP_LIMIT:
            breaches.append(
                f"{report.class_name}: {report.cross_source_rate * 100:.1f}% of images are in "
                f"cross-source near-duplicate groups (limit {CLASS_OVERLAP_LIMIT * 100:.0f}%)")
        for pair, shared in sorted(report.pair_shared.items()):
            left, right = pair.split("|")
            smaller = min(report.counts_by_source.get(left, 0),
                          report.counts_by_source.get(right, 0))
            if smaller and shared / smaller > PAIR_OVERLAP_LIMIT:
                breaches.append(
                    f"{report.class_name}: {left} and {right} share {shared} groups, "
                    f"{shared / smaller * 100:.1f}% of the smaller contribution "
                    f"({smaller} images; limit {PAIR_OVERLAP_LIMIT * 100:.0f}%)")
    return {"substantial_overlap": bool(breaches), "breaches": breaches,
            "class_limit": CLASS_OVERLAP_LIMIT, "pair_limit": PAIR_OVERLAP_LIMIT}


def to_markdown(reports: List[ClassOverlap], judgement: Dict[str, object],
                threshold: int, hash_size: int) -> str:
    lines = ["# v3 cross-source overlap check", "",
             f"Generated {utc_timestamp()}. Perceptual hash size {hash_size} "
             f"(={hash_size * hash_size} bits), grouping distance <= {threshold}.", ""]

    if judgement["substantial_overlap"]:
        lines += ["## Verdict: SUBSTANTIAL OVERLAP - stop and decide", ""]
        lines += [f"* {reason}" for reason in judgement["breaches"]]
    else:
        lines += ["## Verdict: overlap within the declared limits", "",
                  f"No class exceeds {CLASS_OVERLAP_LIMIT * 100:.0f}% of its images in "
                  f"cross-source near-duplicate groups, and no source pair shares more than "
                  f"{PAIR_OVERLAP_LIMIT * 100:.0f}% of the smaller contribution."]
    lines.append("")

    rows = []
    for report in reports:
        rows.append([report.class_name, report.total, len(report.invalid),
                     report.exact_duplicate_pairs_within, report.exact_duplicate_pairs_cross,
                     report.near_groups_multi, report.near_groups_cross,
                     f"{report.cross_source_rate * 100:.2f}%"])
    lines.append(markdown_table(
        ["Class", "Valid", "Invalid", "Exact dup groups (same source)",
         "Exact dup groups (cross)", "Near-dup groups", "Near-dup groups (cross)",
         "Images in cross groups"], rows))
    lines.append("")

    lines.append("## Per-source counts after validation")
    lines.append("")
    sources = sorted({s for r in reports for s in r.counts_by_source})
    lines.append(markdown_table(
        ["Class"] + sources + ["Total"],
        [[r.class_name] + [r.counts_by_source.get(s, 0) or "-" for s in sources] + [r.total]
         for r in reports]))
    lines.append("")

    pair_rows = []
    for report in reports:
        for pair, shared in sorted(report.pair_shared.items()):
            left, right = pair.split("|")
            smaller = min(report.counts_by_source.get(left, 0),
                          report.counts_by_source.get(right, 0))
            pair_rows.append([report.class_name, left, right, shared, smaller,
                              f"{(shared / smaller * 100) if smaller else 0:.1f}%"])
    lines.append("## Source pairs that share near-duplicate groups")
    lines.append("")
    if pair_rows:
        lines.append(markdown_table(
            ["Class", "Source A", "Source B", "Shared groups", "Smaller contribution",
             "Share"], pair_rows))
    else:
        lines.append("None. No near-duplicate group in any class spans two sources.")
    lines.append("")

    invalid = [(r.class_name, item) for r in reports for item in r.invalid]
    lines.append("## Images rejected by validation")
    lines.append("")
    if invalid:
        lines.append(markdown_table(["Class", "Source", "File", "Reason"],
                                    [[cls, item["source"], Path(item["path"]).name, item["reason"]]
                                     for cls, item in invalid[:50]]))
        if len(invalid) > 50:
            lines.append("")
            lines.append(f"...and {len(invalid) - 50} more; the full list is in the JSON report.")
    else:
        lines.append("None. Every image in every mapped folder validated.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure cross-source overlap between v3 sources.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sources", required=True, help="directory holding the source clones")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-per-folder", type=int, default=None,
                        help="debug only: cap images read per folder")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    dedup = config.get("data", "deduplication", default={}) or {}
    hash_size = int(dedup.get("hash_size", 8))
    threshold = int(dedup.get("hash_threshold", 5))

    records = collect_records(config, Path(args.sources), workers=args.workers,
                              limit_per_folder=args.limit_per_folder)
    reports = analyse(records, threshold)
    judgement = verdict(reports)

    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir
                         else config.path("results_dir") / "dataset")
    payload = {"generated_utc": utc_timestamp(), "clone_root": str(args.sources),
               "hash_size": hash_size, "hash_threshold": threshold,
               "verdict": judgement, "classes": [r.to_dict() for r in reports]}
    write_json(out_dir / "v3_overlap.json", payload)
    write_text(out_dir / "v3_overlap.md", to_markdown(reports, judgement, threshold, hash_size))
    LOGGER.info("wrote %s", out_dir / "v3_overlap.md")

    for report in reports:
        LOGGER.info("%-16s valid=%-5d invalid=%-3d cross-source near-dup images=%d (%.2f%%)",
                    report.class_name, report.total, len(report.invalid),
                    report.near_cross_images, report.cross_source_rate * 100)
    if judgement["substantial_overlap"]:
        LOGGER.warning("SUBSTANTIAL OVERLAP - stopping for a human decision")
        for reason in judgement["breaches"]:
            LOGGER.warning("  %s", reason)
        return 2
    LOGGER.info("overlap within the declared limits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
