"""Evaluate several model versions on a held-out set of real photographs.

`data/external_test/` holds photographs supplied by the user, with their true
labels in `manifest.csv`. It is a **final evaluation set**: nothing in the
training pipeline reads it, the images are never modified, and the labels are
used only to score predictions after they have been made.

This exists because internal test accuracy has repeatedly failed to predict
real-world behaviour in this project. v1 scored 100% internally while reading
backgrounds rather than leaves; v2 fixed that confound and reached 95.3%
background robustness, yet scored 45% on the user's 20 real photographs. The
only measurement that has tracked reality is this one.

Run::

    python -m src.testing.external_test --template          # make the manifest
    python -m src.testing.external_test --models v1 v2 v3
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..model.predict import Predictor
from ..utils.config import Config, load_config
from ..utils.helpers import (configure_matplotlib, ensure_dir, format_percent,
                             get_logger, markdown_table, utc_timestamp,
                             write_json, write_text)
from ..utils.image_io import InvalidImageError, iter_image_files, open_image

LOGGER = get_logger(__name__)

MANIFEST_FIELDS = ["file", "true_class", "lighting", "background", "distance",
                   "angle", "quality", "notes"]


def external_dir(config: Config) -> Path:
    """Where the held-out photographs live. Deliberately outside data/raw."""
    return config.project_root / "data" / "external_test"


def write_template(config: Config) -> Path:
    """Create a manifest skeleton listing whatever images are present."""
    root = external_dir(config)
    ensure_dir(root)
    target = root / "manifest.csv"
    images = [p for p in iter_image_files(root, config.supported_extensions)]

    existing: Dict[str, Dict[str, str]] = {}
    if target.exists():
        with target.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("file"):
                    existing[row["file"]] = row

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        if not images:
            writer.writerow({"file": "example.jpg", "true_class": "Yellow Rust",
                             "notes": "replace with your own photographs"})
        for image in images:
            name = str(image.relative_to(root))
            prior = existing.get(name, {})
            writer.writerow({field: prior.get(field, "") for field in MANIFEST_FIELDS}
                            | {"file": name})
    return target


def read_manifest(config: Config) -> List[Dict[str, Any]]:
    """Parse the manifest; rows without a valid true_class are reported."""
    root = external_dir(config)
    path = root / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Put your photographs in {root} and run "
            f"'python -m src.testing.external_test --template' first.")

    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            name = (row.get("file") or "").strip()
            if not name:
                continue
            image_path = root / name
            label = (row.get("true_class") or "").strip()
            spec = config.resolve_class(label) if label else None
            if label and spec is None:
                LOGGER.warning("manifest line %d: '%s' is not a configured class", line_number, label)
            cases.append({
                "file": name,
                "path": image_path,
                "true_class": spec.name if spec else None,
                "exists": image_path.exists(),
                "conditions": {k: (row.get(k) or "").strip()
                               for k in ("lighting", "background", "distance", "angle", "quality")},
                "notes": (row.get("notes") or "").strip(),
            })
    return cases


def evaluate_model(config: Config, version: str, cases: List[Dict[str, Any]]
                   ) -> Optional[Dict[str, Any]]:
    """Run one model version over every usable case."""
    model_config = load_config(overrides={"model": {"version": version}})
    predictor = Predictor(model_config)
    if not predictor.is_available():
        LOGGER.warning("no model at %s - skipping %s", predictor.model_path, version)
        return None

    names = predictor.class_names
    rows: List[Dict[str, Any]] = []
    for case in cases:
        if not case["exists"]:
            continue
        try:
            image = open_image(case["path"])
        except InvalidImageError as exc:
            rows.append({**case, "error": str(exc), "predicted_class": None,
                         "confidence": None, "correct": None})
            continue
        result = predictor.predict(image)
        correct = (result["predicted_class"] == case["true_class"]
                   if case["true_class"] else None)
        rows.append({
            "file": case["file"],
            "true_class": case["true_class"],
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "low_confidence": result["low_confidence"],
            "uncertain": result["uncertain"],
            "correct": correct,
            "conditions": case["conditions"],
            "error": "",
        })

    scored = [r for r in rows if r["correct"] is not None]
    accuracy = float(np.mean([r["correct"] for r in scored])) if scored else None
    right = [r for r in scored if r["correct"]]
    wrong = [r for r in scored if not r["correct"]]

    per_class: Dict[str, Dict[str, Any]] = {}
    for name in names:
        of_class = [r for r in scored if r["true_class"] == name]
        if of_class:
            per_class[name] = {
                "n": len(of_class),
                "accuracy": float(np.mean([r["correct"] for r in of_class])),
                "mean_confidence": float(np.mean([r["confidence"] for r in of_class])),
            }

    matrix = [[0] * len(names) for _ in names]
    for r in scored:
        matrix[names.index(r["true_class"])][names.index(r["predicted_class"])] += 1

    return {
        "version": predictor.model_version,
        "model_path": str(predictor.model_path),
        "class_names": names,
        "images_scored": len(scored),
        "accuracy": accuracy,
        "mean_confidence": float(np.mean([r["confidence"] for r in scored])) if scored else None,
        "mean_confidence_correct": float(np.mean([r["confidence"] for r in right])) if right else None,
        "mean_confidence_incorrect": float(np.mean([r["confidence"] for r in wrong])) if wrong else None,
        "high_confidence_errors": sum(1 for r in wrong if not r["low_confidence"]),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "rows": rows,
    }


def calibration_table(results: Dict[str, Any], bins: int = 5) -> List[Dict[str, Any]]:
    """Accuracy within confidence bands - the basis of a calibration curve."""
    scored = [r for r in results["rows"] if r["correct"] is not None]
    if not scored:
        return []
    edges = np.linspace(0.0, 1.0, bins + 1)
    table = []
    for low, high in zip(edges[:-1], edges[1:]):
        band = [r for r in scored if low <= r["confidence"] < high or
                (high == 1.0 and r["confidence"] == 1.0)]
        if not band:
            continue
        table.append({
            "band": f"{low:.0%}-{high:.0%}",
            "n": len(band),
            "mean_confidence": float(np.mean([r["confidence"] for r in band])),
            "accuracy": float(np.mean([r["correct"] for r in band])),
        })
    return table


def to_markdown(all_results: List[Dict[str, Any]], cases: List[Dict[str, Any]]) -> str:
    missing = [c for c in cases if not c["exists"]]
    unlabelled = [c for c in cases if c["exists"] and not c["true_class"]]

    lines = [
        "# External real-world test",
        "",
        f"* Generated (UTC): {utc_timestamp()}",
        f"* Photographs in the manifest: {len(cases)}",
        f"* Missing from disk: {len(missing)}",
        f"* Without a true label: {len(unlabelled)}",
        "",
        "These images are held out entirely: no training, validation, augmentation "
        "or threshold selection has ever read them.",
        "",
        "## Headline",
        "",
        markdown_table(
            ["Model", "Images scored", "Accuracy", "Mean confidence",
             "Confidence when right", "Confidence when wrong", "Confident errors"],
            [[r["version"], r["images_scored"], format_percent(r["accuracy"], 1),
              format_percent(r["mean_confidence"], 1),
              format_percent(r["mean_confidence_correct"], 1),
              format_percent(r["mean_confidence_incorrect"], 1),
              r["high_confidence_errors"]] for r in all_results]),
        "",
    ]

    if all_results:
        names = all_results[0]["class_names"]
        lines += ["## Per class", "",
                  markdown_table(
                      ["Class"] + [f"{r['version']}" for r in all_results] + ["n"],
                      [[name] + [format_percent(r["per_class"].get(name, {}).get("accuracy"), 0)
                                 for r in all_results]
                       + [all_results[0]["per_class"].get(name, {}).get("n", 0)]
                       for name in names]),
                  ""]

        for r in all_results:
            lines += [f"### Confusion matrix — {r['version']}", "",
                      markdown_table(["Actual \\ Predicted"] + names,
                                     [[names[i]] + [str(v) for v in row]
                                      for i, row in enumerate(r["confusion_matrix"])]),
                      ""]

        lines += ["## Calibration", "",
                  "Within each confidence band, how often the model was actually right. "
                  "A well-calibrated model has accuracy close to mean confidence in every band.",
                  ""]
        for r in all_results:
            table = calibration_table(r)
            if not table:
                continue
            lines += [f"**{r['version']}**", "",
                      markdown_table(
                          ["Confidence band", "n", "Mean confidence", "Actual accuracy", "Gap"],
                          [[t["band"], t["n"], format_percent(t["mean_confidence"], 1),
                            format_percent(t["accuracy"], 1),
                            f"{(t['accuracy'] - t['mean_confidence']) * 100:+.1f} pp"]
                           for t in table]),
                      ""]

        # Per-image comparison across versions.
        lines += ["## Per image", "",
                  markdown_table(
                      ["Image", "True class"]
                      + [f"{r['version']} prediction" for r in all_results],
                      [[f"`{row['file']}`", row["true_class"] or "—"]
                       + [_cell(r, row["file"]) for r in all_results]
                       for row in all_results[0]["rows"]]),
                  ""]

        if len(all_results) >= 2:
            a, b = all_results[-2], all_results[-1]
            changed = _changes(a, b)
            lines += [f"## What changed: {a['version']} → {b['version']}", "",
                      f"{len(changed)} of {len(a['rows'])} predictions changed.", ""]
            if changed:
                lines.append(markdown_table(
                    ["Image", "True", f"{a['version']}", f"{b['version']}", "Effect"],
                    changed))
                lines.append("")
    return "\n".join(lines)


def _cell(results: Dict[str, Any], filename: str) -> str:
    row = next((r for r in results["rows"] if r["file"] == filename), None)
    if row is None or row["predicted_class"] is None:
        return "—"
    mark = "✓" if row["correct"] else ("✗" if row["correct"] is False else "?")
    return f"{mark} {row['predicted_class']} ({row['confidence'] * 100:.0f}%)"


def _changes(a: Dict[str, Any], b: Dict[str, Any]) -> List[List[str]]:
    rows = []
    lookup = {r["file"]: r for r in b["rows"]}
    for old in a["rows"]:
        new = lookup.get(old["file"])
        if not new or old["predicted_class"] == new["predicted_class"]:
            continue
        if old["correct"] is None:
            effect = "not scored"
        elif new["correct"] and not old["correct"]:
            effect = "**fixed**"
        elif old["correct"] and not new["correct"]:
            effect = "**broken**"
        else:
            effect = "still wrong"
        rows.append([f"`{old['file']}`", old["true_class"] or "—",
                     f"{old['predicted_class']} ({old['confidence'] * 100:.0f}%)",
                     f"{new['predicted_class']} ({new['confidence'] * 100:.0f}%)", effect])
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate model versions on the held-out external photographs")
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="+", default=["v1", "v2"],
                        help="model versions to evaluate on the same images")
    parser.add_argument("--template", action="store_true",
                        help="write/refresh the manifest skeleton and exit")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    root = external_dir(config)

    if args.template:
        target = write_template(config)
        images = list(iter_image_files(root, config.supported_extensions))
        LOGGER.info("wrote %s listing %d image(s)", target, len(images))
        if not images:
            LOGGER.warning("no images found in %s yet - copy your photographs there first",
                           root)
        LOGGER.info("fill in 'true_class' for each row: %s",
                    ", ".join(config.class_names))
        return 0

    try:
        cases = read_manifest(config)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1

    present = [c for c in cases if c["exists"]]
    if not present:
        LOGGER.error("no photographs found in %s", root)
        LOGGER.error("Copy your images there, then run --template, then fill in true_class.")
        return 1
    labelled = [c for c in present if c["true_class"]]
    if not labelled:
        LOGGER.error("no row in the manifest has a valid true_class, so nothing can be scored")
        LOGGER.error("Valid classes: %s", ", ".join(config.class_names))
        return 1

    all_results = []
    for version in args.models:
        result = evaluate_model(config, version, cases)
        if result:
            all_results.append(result)
    if not all_results:
        LOGGER.error("none of the requested models could be loaded: %s", args.models)
        return 1

    out_dir = ensure_dir(config.path("results_dir") / "external")
    label = "_vs_".join(r["version"] for r in all_results)
    write_json(out_dir / f"external_test_{label}.json",
               {"generated_utc": utc_timestamp(),
                "results": [{k: v for k, v in r.items() if k != "rows"} for r in all_results],
                "rows": {r["version"]: r["rows"] for r in all_results}})
    markdown = to_markdown(all_results, cases)
    write_text(out_dir / f"external_test_{label}.md", markdown)

    if not args.quiet:
        print(markdown)
    LOGGER.info("written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
