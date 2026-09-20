"""Condition-based testing with photographs taken in realistic settings.

The test split measures accuracy on images drawn from the same pool as the
training images. This module measures something different and more important:
does accuracy hold up when the photograph is dim, cluttered, taken from far
away, at an angle, or with a poor camera?

Input layout
------------
Images live under ``data/realistic/`` and carry their conditions either in a
manifest or in their folder path.

**Manifest (recommended)** - ``data/realistic/manifest.csv``::

    file,actual_class,lighting,background,distance,angle,quality,notes
    img_001.jpg,Yellow Rust,low,soil,close,straight,high,
    img_002.jpg,Healthy,bright,plain,far,angled,medium,windy day

``actual_class`` may be left empty for images whose true class is unknown;
those rows are still predicted and reported, but excluded from accuracy.

**Folder convention (no manifest)** - any nesting of ``key=value`` folders,
ending in a class folder::

    data/realistic/lighting=low/distance=far/Yellow_Rust/img_003.jpg

Run::

    python -m src.testing.realistic_testing
    python -m src.testing.realistic_testing --manifest data/realistic/manifest.csv
    python -m src.testing.realistic_testing --template     # write a blank manifest
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.config import Config, load_config
from ..utils.helpers import (
    configure_matplotlib,
    ensure_dir,
    format_percent,
    get_logger,
    markdown_table,
    relative_to_root,
    utc_timestamp,
    write_json,
    write_text,
)
from ..utils.image_io import InvalidImageError, iter_image_files
from ..model.predict import Predictor

LOGGER = get_logger(__name__)

# Condition dimensions and the values the reporting expects. Unknown values are
# still accepted and reported - this list only drives the ordering of tables
# and the template manifest.
CONDITIONS: Dict[str, List[str]] = {
    "lighting": ["normal", "low", "bright"],
    "background": ["plain", "soil", "plant"],
    "distance": ["close", "medium", "far"],
    "angle": ["straight", "angled"],
    "quality": ["high", "medium", "low"],
}

MANIFEST_FIELDS = ["file", "actual_class", *CONDITIONS.keys(), "notes"]


def write_manifest_template(config: Config, path: Path) -> Path:
    """Write a blank manifest listing whatever images are already present."""
    ensure_dir(path.parent)
    realistic_dir = config.path("realistic_dir")
    existing = [
        p for p in iter_image_files(realistic_dir, config.supported_extensions)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        if existing:
            for image in existing:
                writer.writerow({
                    "file": str(image.relative_to(realistic_dir)),
                    "actual_class": "",
                    **{key: "" for key in CONDITIONS},
                    "notes": "",
                })
        else:
            writer.writerow({
                "file": "example.jpg",
                "actual_class": config.class_names[0],
                "lighting": "normal", "background": "plain", "distance": "close",
                "angle": "straight", "quality": "high",
                "notes": "replace this row with your own photographs",
            })
    return path


def read_manifest(config: Config, manifest_path: Path) -> List[Dict[str, Any]]:
    """Parse a manifest into normalised test-case dictionaries."""
    realistic_dir = config.path("realistic_dir")
    cases: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            raw_file = (row.get("file") or "").strip()
            if not raw_file:
                continue
            path = Path(raw_file)
            if not path.is_absolute():
                path = realistic_dir / path
            if not path.exists():
                LOGGER.warning("manifest line %d: file not found: %s", line_number, path)
                continue
            actual = (row.get("actual_class") or "").strip()
            spec = config.resolve_class(actual) if actual else None
            if actual and spec is None:
                LOGGER.warning("manifest line %d: '%s' is not a configured class; "
                               "treating the true class as unknown", line_number, actual)
            cases.append({
                "path": path,
                "actual_class": spec.name if spec else None,
                "conditions": {key: (row.get(key) or "").strip().lower() or "unspecified"
                               for key in CONDITIONS},
                "notes": (row.get("notes") or "").strip(),
            })
    return cases


def discover_from_folders(config: Config) -> List[Dict[str, Any]]:
    """Derive conditions from ``key=value`` folder names in the path."""
    realistic_dir = config.path("realistic_dir")
    cases: List[Dict[str, Any]] = []
    for path in iter_image_files(realistic_dir, config.supported_extensions):
        conditions = {key: "unspecified" for key in CONDITIONS}
        actual: Optional[str] = None
        for part in path.relative_to(realistic_dir).parts[:-1]:
            if "=" in part:
                key, _, value = part.partition("=")
                key = key.strip().lower()
                if key in conditions:
                    conditions[key] = value.strip().lower()
                continue
            spec = config.resolve_class(part)
            if spec is not None:
                actual = spec.name
        cases.append({"path": path, "actual_class": actual,
                      "conditions": conditions, "notes": ""})
    return cases


def run_tests(config: Config, predictor: Predictor, cases: List[Dict[str, Any]]
              ) -> List[Dict[str, Any]]:
    """Predict every case and attach the outcome."""
    results: List[Dict[str, Any]] = []
    batch_size = config.batch_size
    for start in range(0, len(cases), batch_size):
        chunk = cases[start:start + batch_size]
        usable: List[Dict[str, Any]] = []
        for case in chunk:
            try:
                # Fail fast on unreadable files so one bad image cannot abort
                # the whole batch.
                from ..utils.image_io import load_image_array
                load_image_array(case["path"], predictor.image_size)
                usable.append(case)
            except InvalidImageError as exc:
                results.append({
                    "file": relative_to_root(case["path"], config.project_root),
                    "filename": case["path"].name,
                    "actual_class": case["actual_class"],
                    "predicted_class": None,
                    "confidence": None,
                    "correct": None,
                    "low_confidence": None,
                    "uncertain": None,
                    "error": str(exc),
                    **{f"condition_{k}": v for k, v in case["conditions"].items()},
                    "notes": case["notes"],
                })
        if not usable:
            continue
        predictions = predictor.predict_many([c["path"] for c in usable])
        for case, prediction in zip(usable, predictions):
            correct: Optional[bool] = None
            if case["actual_class"]:
                correct = prediction["predicted_class"] == case["actual_class"]
            results.append({
                "file": relative_to_root(case["path"], config.project_root),
                "filename": case["path"].name,
                "actual_class": case["actual_class"],
                "predicted_class": prediction["predicted_class"],
                "confidence": prediction["confidence"],
                "correct": correct,
                "low_confidence": prediction["low_confidence"],
                "uncertain": prediction["uncertain"],
                "error": "",
                **{f"condition_{k}": v for k, v in case["conditions"].items()},
                "notes": case["notes"],
            })
    return results


def summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Overall and per-condition breakdowns."""
    labelled = [r for r in results if r["correct"] is not None]
    unlabelled = [r for r in results if r["correct"] is None and not r["error"]]
    failed = [r for r in results if r["error"]]

    def block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0, "accuracy": None, "mean_confidence": None,
                    "low_confidence_rate": None, "uncertain_rate": None}
        return {
            "n": len(rows),
            "accuracy": float(np.mean([bool(r["correct"]) for r in rows])),
            "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
            "low_confidence_rate": float(np.mean([bool(r["low_confidence"]) for r in rows])),
            "uncertain_rate": float(np.mean([bool(r["uncertain"]) for r in rows])),
        }

    per_condition: Dict[str, Dict[str, Any]] = {}
    for key in CONDITIONS:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in labelled:
            grouped[row.get(f"condition_{key}", "unspecified")].append(row)
        per_condition[key] = {value: block(rows) for value, rows in sorted(grouped.items())}

    per_class: Dict[str, Dict[str, Any]] = {}
    grouped_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in labelled:
        grouped_class[row["actual_class"]].append(row)
    for name, rows in sorted(grouped_class.items()):
        per_class[name] = block(rows)

    return {
        "generated_utc": utc_timestamp(),
        "total_images": len(results),
        "images_with_known_class": len(labelled),
        "images_without_known_class": len(unlabelled),
        "images_that_failed_to_load": len(failed),
        "overall": block(labelled),
        "per_condition": per_condition,
        "per_class": per_class,
        "note": (
            "Accuracy is computed only over images whose true class was supplied. With small "
            "numbers of photographs per condition these figures are indicative, not "
            "statistically reliable - the 'n' column shows how many images each number rests on."
        ),
    }


def plot_condition_accuracy(summary: Dict[str, Any], output_path: Path) -> Optional[Path]:
    """One bar group per condition dimension, annotated with sample sizes."""
    configure_matplotlib()
    import matplotlib.pyplot as plt

    dimensions = [(k, v) for k, v in summary["per_condition"].items()
                  if any(entry["accuracy"] is not None for entry in v.values())]
    if not dimensions:
        return None

    figure, axes = plt.subplots(1, len(dimensions),
                                figsize=(3.4 * len(dimensions) + 1.0, 4.0), squeeze=False)
    for axis, (key, values) in zip(axes[0], dimensions):
        labels = list(values.keys())
        accuracies = [values[v]["accuracy"] or 0.0 for v in labels]
        counts = [values[v]["n"] for v in labels]
        bars = axis.bar(labels, accuracies, color="#3b6ea5")
        axis.set_ylim(0, 1.05)
        axis.set_title(key)
        axis.set_ylabel("Accuracy" if axis is axes[0][0] else "")
        plt.setp(axis.get_xticklabels(), rotation=20, ha="right")
        for bar, count in zip(bars, counts):
            axis.annotate(f"n={count}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          ha="center", va="bottom", fontsize=8)
    if summary["overall"]["accuracy"] is not None:
        figure.suptitle(
            f"Accuracy by photographic condition "
            f"(overall {summary['overall']['accuracy'] * 100:.1f}% "
            f"on {summary['overall']['n']} images)")
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def results_to_markdown(results: List[Dict[str, Any]], summary: Dict[str, Any],
                        version: str, chart: Optional[Path] = None) -> str:
    overall = summary["overall"]
    lines = [
        f"# Realistic-condition testing - model {version}",
        "",
        f"* Generated (UTC): {summary['generated_utc']}",
        f"* Photographs tested: {summary['total_images']}",
        f"* With a known true class: {summary['images_with_known_class']}",
        f"* Without a known true class (predicted but not scored): "
        f"{summary['images_without_known_class']}",
        f"* Failed to load: {summary['images_that_failed_to_load']}",
        "",
        "## Overall",
        "",
        markdown_table(
            ["Measure", "Value"],
            [
                ["Images scored", overall["n"]],
                ["Accuracy", format_percent(overall["accuracy"], 1)],
                ["Mean confidence", format_percent(overall["mean_confidence"], 1)],
                ["Below confidence threshold", format_percent(overall["low_confidence_rate"], 1)],
                ["Flagged uncertain", format_percent(overall["uncertain_rate"], 1)],
            ]),
        "",
    ]

    if summary["per_class"]:
        lines += [
            "## Per class",
            "",
            markdown_table(
                ["Class", "Images", "Accuracy", "Mean confidence"],
                [[name, entry["n"], format_percent(entry["accuracy"], 1),
                  format_percent(entry["mean_confidence"], 1)]
                 for name, entry in summary["per_class"].items()]),
            "",
        ]

    lines += ["## By photographic condition", ""]
    any_condition = False
    for key, values in summary["per_condition"].items():
        rows = [[value, entry["n"], format_percent(entry["accuracy"], 1),
                 format_percent(entry["mean_confidence"], 1),
                 format_percent(entry["low_confidence_rate"], 1)]
                for value, entry in values.items() if entry["n"]]
        if not rows:
            continue
        any_condition = True
        lines += [f"### {key.capitalize()}", "",
                  markdown_table(["Value", "Images", "Accuracy", "Mean confidence",
                                  "Below threshold"], rows), ""]
    if not any_condition:
        lines += ["No condition information was supplied for any test image.", ""]

    lines += [
        "## Full results table",
        "",
        markdown_table(
            ["Image", "Condition (lighting/background/distance/angle/quality)", "Actual",
             "Predicted", "Confidence", "Result"],
            [[f"`{r['filename']}`",
              "/".join(r.get(f"condition_{k}", "-") for k in CONDITIONS),
              r["actual_class"] or "unknown",
              r["predicted_class"] or f"ERROR: {r['error']}",
              format_percent(r["confidence"], 1) if r["confidence"] is not None else "-",
              ("correct" if r["correct"] else "incorrect") if r["correct"] is not None
              else ("not scored" if not r["error"] else "failed")]
             for r in results]),
        "",
        "## Interpretation",
        "",
        summary["note"],
        "",
        "A drop in accuracy for one condition value is a *signal to investigate*, not proof "
        "that the condition caused it - the photographs also differ in other ways. To draw a "
        "firm conclusion, photograph the same leaves while varying one condition at a time.",
        "",
    ]
    if chart:
        lines += ["## Chart", "", f"![Accuracy by condition]({chart.as_posix()})", ""]
    return "\n".join(lines)


def write_results_csv(results: List[Dict[str, Any]], output_path: Path) -> Path:
    ensure_dir(Path(output_path).parent)
    if not results:
        Path(output_path).write_text("", encoding="utf-8")
        return Path(output_path)
    fields = list(results[0].keys())
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    return Path(output_path)


def _load_cases(config: Config, manifest: Optional[str]) -> Tuple[List[Dict[str, Any]], str]:
    realistic_dir = config.path("realistic_dir")
    manifest_path = Path(manifest) if manifest else realistic_dir / "manifest.csv"
    if not manifest_path.is_absolute():
        manifest_path = config.project_root / manifest_path
    if manifest_path.exists():
        return read_manifest(config, manifest_path), f"manifest {manifest_path}"
    return discover_from_folders(config), f"folder convention under {realistic_dir}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test Leaf Lens on photographs taken under realistic conditions")
    parser.add_argument("--config", default=None)
    parser.add_argument("--version", default=None, help="model version (default: config)")
    parser.add_argument("--model", default=None, help="path to a .keras model file")
    parser.add_argument("--manifest", default=None,
                        help="CSV manifest (default: data/realistic/manifest.csv)")
    parser.add_argument("--template", action="store_true",
                        help="write a blank manifest template and exit")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    overrides = {"model": {"version": args.version}} if args.version else None
    config = load_config(args.config, overrides=overrides)

    if args.template:
        target = config.path("realistic_dir") / "manifest.csv"
        if target.exists():
            LOGGER.error("%s already exists; delete it first or pass --manifest", target)
            return 1
        write_manifest_template(config, target)
        LOGGER.info("wrote manifest template: %s", target)
        LOGGER.info("Columns: %s", ", ".join(MANIFEST_FIELDS))
        for key, values in CONDITIONS.items():
            LOGGER.info("  %-11s one of: %s", key, ", ".join(values))
        return 0

    cases, source = _load_cases(config, args.manifest)
    if not cases:
        LOGGER.error("no realistic test photographs found (%s)", source)
        LOGGER.error("")
        LOGGER.error("To run realistic testing:")
        LOGGER.error("  1. Put your own field photographs in %s", config.path("realistic_dir"))
        LOGGER.error("  2. python -m src.testing.realistic_testing --template")
        LOGGER.error("  3. Fill in the manifest (true class + conditions per photograph)")
        LOGGER.error("  4. python -m src.testing.realistic_testing")
        return 1

    predictor = Predictor(config, model_path=Path(args.model) if args.model else None)
    if not predictor.is_available():
        LOGGER.error("no trained model found at %s", predictor.model_path)
        LOGGER.error("Train one first: python -m src.model.train")
        return 1

    LOGGER.info("testing %d photographs (source: %s)", len(cases), source)
    results = run_tests(config, predictor, cases)
    summary = summarise(results)

    version = predictor.model_version
    out_dir = ensure_dir(config.path("results_dir") / "realistic_testing")
    write_results_csv(results, out_dir / f"realistic_results_{version}.csv")
    write_json(out_dir / f"realistic_summary_{version}.json",
               {"summary": summary, "results": results})
    chart = plot_condition_accuracy(
        summary, config.path("results_dir") / "graphs" / f"realistic_conditions_{version}.png")
    markdown = results_to_markdown(results, summary, version, chart)
    write_text(out_dir / f"realistic_testing_{version}.md", markdown)

    if not args.quiet:
        print(markdown)
    LOGGER.info("realistic-testing results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
