"""Error analysis over the test-split predictions.

For every misclassified test image this records the filename, the actual
class, the predicted class and the confidence, and then summarises the
mistakes: which confusion pairs dominate, how confident the model was when it
was wrong, and whether the wrong images differ measurably from the right ones
on a handful of objective image statistics (brightness, contrast, sharpness,
saturation, aspect ratio, resolution).

Those statistics are reported as *observations with numbers attached*. The
report deliberately states that a difference is a correlation on a small
sample, not a demonstrated cause; a cause would need a controlled experiment
(see ``src/testing/realistic_testing.py``).

Run::

    python -m src.testing.error_analysis
    python -m src.testing.error_analysis --version v2 --max-grid 24
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.config import Config, load_config
from ..utils.helpers import (
    configure_matplotlib,
    ensure_dir,
    format_percent,
    get_logger,
    markdown_table,
    utc_timestamp,
    write_json,
    write_text,
)
from ..utils.image_io import InvalidImageError, basic_quality_metrics, make_thumbnail_grid

LOGGER = get_logger(__name__)

# Objective, cheaply computed image statistics compared between correct and
# incorrect predictions. None of these is used to classify anything.
QUALITY_KEYS = [
    ("mean_brightness", "Mean brightness (0-1)"),
    ("brightness_std", "Brightness spread / contrast (0-1)"),
    ("sharpness_laplacian_var", "Sharpness (variance of Laplacian)"),
    ("mean_saturation", "Mean saturation (0-1)"),
    ("megapixels", "Resolution (megapixels)"),
    ("aspect_ratio", "Aspect ratio (w/h)"),
]


def load_predictions_csv(path: Path) -> List[Dict[str, Any]]:
    """Read the per-image CSV written by ``src.model.evaluate``."""
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["confidence"] = float(row["confidence"])
            row["correct"] = str(row["correct"]).lower() == "true"
            row["low_confidence"] = str(row.get("low_confidence", "")).lower() == "true"
            rows.append(row)
    return rows


def _quality_for(path: Path) -> Optional[Dict[str, float]]:
    try:
        metrics = basic_quality_metrics(path)
    except (InvalidImageError, OSError):
        return None
    metrics["megapixels"] = (metrics["width"] * metrics["height"]) / 1_000_000
    metrics["aspect_ratio"] = metrics["width"] / metrics["height"] if metrics["height"] else 0.0
    return metrics


def analyse(config: Config, rows: List[Dict[str, Any]], compute_quality: bool = True
            ) -> Dict[str, Any]:
    """Build the full error-analysis payload from prediction rows."""
    class_names = config.class_names
    errors = [r for r in rows if not r["correct"]]
    correct = [r for r in rows if r["correct"]]

    pair_counts = Counter((r["actual_class"], r["predicted_class"]) for r in errors)
    per_class_errors: Dict[str, Dict[str, Any]] = {}
    for name in class_names:
        of_class = [r for r in rows if r["actual_class"] == name]
        wrong = [r for r in of_class if not r["correct"]]
        per_class_errors[name] = {
            "support": len(of_class),
            "errors": len(wrong),
            "error_rate": (len(wrong) / len(of_class)) if of_class else None,
            "mean_confidence_when_wrong": (float(np.mean([r["confidence"] for r in wrong]))
                                           if wrong else None),
            "most_common_wrong_prediction": (
                Counter(r["predicted_class"] for r in wrong).most_common(1)[0][0]
                if wrong else None),
        }

    confidences_wrong = [r["confidence"] for r in errors]
    confidences_right = [r["confidence"] for r in correct]

    quality_comparison: List[Dict[str, Any]] = []
    quality_available = False
    if compute_quality and rows:
        groups: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for row in rows:
            path = config.project_root / row["file"]
            metrics = _quality_for(path)
            if metrics is None:
                continue
            groups["correct" if row["correct"] else "incorrect"].append(metrics)
        quality_available = bool(groups.get("correct") or groups.get("incorrect"))
        for key, label in QUALITY_KEYS:
            right_values = [m[key] for m in groups.get("correct", []) if key in m]
            wrong_values = [m[key] for m in groups.get("incorrect", []) if key in m]
            if not right_values and not wrong_values:
                continue
            quality_comparison.append({
                "metric": key,
                "label": label,
                "correct_mean": float(np.mean(right_values)) if right_values else None,
                "correct_std": float(np.std(right_values)) if right_values else None,
                "incorrect_mean": float(np.mean(wrong_values)) if wrong_values else None,
                "incorrect_std": float(np.std(wrong_values)) if wrong_values else None,
                "n_correct": len(right_values),
                "n_incorrect": len(wrong_values),
            })

    # Flag only differences that exceed the correct-group spread; anything
    # smaller is noise at these sample sizes.
    notable: List[str] = []
    for entry in quality_comparison:
        if (entry["correct_mean"] is None or entry["incorrect_mean"] is None
                or entry["n_incorrect"] < 3):
            continue
        spread = entry["correct_std"] or 0.0
        difference = entry["incorrect_mean"] - entry["correct_mean"]
        if spread > 0 and abs(difference) > spread:
            direction = "higher" if difference > 0 else "lower"
            notable.append(
                f"{entry['label']}: misclassified images average "
                f"{entry['incorrect_mean']:.3f} versus {entry['correct_mean']:.3f} for correct "
                f"ones ({direction} by more than one standard deviation of the correct group; "
                f"n={entry['n_incorrect']} errors)"
            )

    return {
        "generated_utc": utc_timestamp(),
        "total_predictions": len(rows),
        "correct": len(correct),
        "incorrect": len(errors),
        "accuracy": (len(correct) / len(rows)) if rows else None,
        "mean_confidence_correct": float(np.mean(confidences_right)) if confidences_right else None,
        "mean_confidence_incorrect": float(np.mean(confidences_wrong)) if confidences_wrong else None,
        "high_confidence_errors": [
            r for r in sorted(errors, key=lambda r: r["confidence"], reverse=True)
            if not r["low_confidence"]
        ][:25],
        "errors": [
            {
                "filename": r["filename"],
                "file": r["file"],
                "actual_class": r["actual_class"],
                "predicted_class": r["predicted_class"],
                "confidence": r["confidence"],
                "low_confidence": r["low_confidence"],
            }
            for r in sorted(errors, key=lambda r: r["confidence"], reverse=True)
        ],
        "confusion_pairs": [
            {"actual": actual, "predicted": predicted, "count": count}
            for (actual, predicted), count in pair_counts.most_common()
        ],
        "per_class": per_class_errors,
        "image_quality_comparison": quality_comparison,
        "image_quality_available": quality_available,
        "notable_quality_differences": notable,
    }


def plot_confidence_distribution(rows: List[Dict[str, Any]], output_path: Path,
                                 threshold: float) -> Optional[Path]:
    """Overlaid confidence histograms for correct and incorrect predictions."""
    configure_matplotlib()
    import matplotlib.pyplot as plt

    right = [r["confidence"] for r in rows if r["correct"]]
    wrong = [r["confidence"] for r in rows if not r["correct"]]
    if not right and not wrong:
        return None

    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(0, 1, 21)
    if right:
        axis.hist(right, bins=bins, alpha=0.7, label=f"correct (n={len(right)})", color="#3f7d3f")
    if wrong:
        axis.hist(wrong, bins=bins, alpha=0.7, label=f"incorrect (n={len(wrong)})", color="#b5453b")
    axis.axvline(threshold, color="#333333", linestyle="--", linewidth=1.2,
                 label=f"threshold {threshold:.2f}")
    axis.set_xlabel("Predicted confidence")
    axis.set_ylabel("Number of test images")
    axis.set_title("Confidence distribution: correct vs incorrect predictions")
    axis.legend()
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def write_errors_csv(analysis: Dict[str, Any], output_path: Path) -> Path:
    ensure_dir(Path(output_path).parent)
    fields = ["filename", "file", "actual_class", "predicted_class", "confidence",
              "low_confidence"]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in analysis["errors"]:
            writer.writerow({k: row.get(k) for k in fields})
    return Path(output_path)


def analysis_to_markdown(analysis: Dict[str, Any], config: Config, version: str,
                         grid_path: Optional[Path] = None,
                         confidence_plot: Optional[Path] = None) -> str:
    lines = [
        f"# Error analysis - model {version} (test split)",
        "",
        f"* Generated (UTC): {analysis['generated_utc']}",
        f"* Test predictions analysed: {analysis['total_predictions']}",
        f"* Correct: {analysis['correct']} | Incorrect: **{analysis['incorrect']}**",
        f"* Test accuracy: {format_percent(analysis['accuracy'], 2)}",
        f"* Mean confidence when correct: {format_percent(analysis['mean_confidence_correct'], 1)}",
        f"* Mean confidence when wrong: {format_percent(analysis['mean_confidence_incorrect'], 1)}",
        "",
    ]

    if analysis["incorrect"] == 0:
        lines += ["No misclassifications on this test split - there is nothing to analyse.", ""]
        return "\n".join(lines)

    lines += [
        "## Error rate per class",
        "",
        markdown_table(
            ["Class", "Test images", "Errors", "Error rate", "Most often mistaken for",
             "Mean confidence when wrong"],
            [[name, entry["support"], entry["errors"], format_percent(entry["error_rate"], 1),
              entry["most_common_wrong_prediction"] or "-",
              format_percent(entry["mean_confidence_when_wrong"], 1)]
             for name, entry in analysis["per_class"].items()]),
        "",
        "## Confusion pairs (observed in this run)",
        "",
        markdown_table(["Actual", "Predicted as", "Count"],
                       [[p["actual"], p["predicted"], p["count"]]
                        for p in analysis["confusion_pairs"]]),
        "",
        "## Every incorrect prediction",
        "",
        markdown_table(
            ["Filename", "Actual", "Predicted", "Confidence", "Below threshold?"],
            [[f"`{e['filename']}`", e["actual_class"], e["predicted_class"],
              format_percent(e["confidence"], 1), "yes" if e["low_confidence"] else "no"]
             for e in analysis["errors"]]),
        "",
    ]

    high_conf = analysis["high_confidence_errors"]
    lines += [
        "## Confident mistakes",
        "",
        f"{len(high_conf)} error(s) were made at or above the confidence threshold. "
        f"These matter most: the interface would not have warned the user about them.",
        "",
    ]
    if high_conf:
        lines.append(markdown_table(
            ["Filename", "Actual", "Predicted", "Confidence"],
            [[f"`{e['filename']}`", e["actual_class"], e["predicted_class"],
              format_percent(e["confidence"], 1)] for e in high_conf]))
        lines.append("")

    lines += ["## Image characteristics: correct vs incorrect", ""]
    if analysis["image_quality_available"]:
        lines += [
            "Objective statistics measured directly from the test images. They are descriptive; "
            "no part of the classifier uses them.",
            "",
            markdown_table(
                ["Measure", "Correct (mean +/- sd)", "Incorrect (mean +/- sd)", "n correct", "n incorrect"],
                [[e["label"],
                  f"{e['correct_mean']:.3f} +/- {e['correct_std']:.3f}" if e["correct_mean"] is not None else "n/a",
                  f"{e['incorrect_mean']:.3f} +/- {e['incorrect_std']:.3f}" if e["incorrect_mean"] is not None else "n/a",
                  e["n_correct"], e["n_incorrect"]]
                 for e in analysis["image_quality_comparison"]]),
            "",
        ]
        if analysis["notable_quality_differences"]:
            lines += ["**Differences larger than the spread of the correct group:**", ""]
            lines += [f"* {note}" for note in analysis["notable_quality_differences"]]
            lines += [
                "",
                "These are correlations on a small sample, not demonstrated causes. To test "
                "whether a factor such as lighting or distance actually degrades the model, "
                "photograph the same leaves under controlled conditions and run "
                "`python -m src.testing.realistic_testing`.",
                "",
            ]
        else:
            lines += [
                "No measured characteristic differed from the correct group by more than one "
                "standard deviation, so this run provides no evidence that brightness, "
                "sharpness, saturation or resolution explains the errors.",
                "",
            ]
    else:
        lines += ["Image statistics could not be computed (the test images were not readable "
                  "from their recorded paths).", ""]

    lines += [
        "## Factors this analysis cannot settle",
        "",
        "The following are plausible contributors that the data above cannot confirm or rule "
        "out. Each needs targeted photographs rather than more analysis of the existing set:",
        "",
        "* background clutter (soil, other plants, hands)",
        "* camera distance and how much of the leaf fills the frame",
        "* leaf angle and partially visible leaves",
        "* overlapping symptoms, or two diseases on one leaf",
        "* genuine visual similarity between classes at early infection stages",
        "",
        "`src/testing/realistic_testing.py` exists to collect exactly this evidence.",
        "",
    ]

    if grid_path:
        lines += ["## Misclassified images", "",
                  f"![Misclassified test images]({grid_path.as_posix()})", ""]
    if confidence_plot:
        lines += ["## Confidence distribution", "",
                  f"![Confidence distribution]({confidence_plot.as_posix()})", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse incorrect Leaf Lens predictions")
    parser.add_argument("--config", default=None)
    parser.add_argument("--version", default=None, help="model version (default: config)")
    parser.add_argument("--predictions", default=None,
                        help="path to a predictions CSV (default: the evaluate.py output)")
    parser.add_argument("--max-grid", type=int, default=16,
                        help="maximum misclassified images in the visual grid")
    parser.add_argument("--no-quality", action="store_true",
                        help="skip the image-statistics comparison (faster)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    overrides = {"model": {"version": args.version}} if args.version else None
    config = load_config(args.config, overrides=overrides)
    version = config.model_version
    results_dir = config.path("results_dir")

    predictions_path = (Path(args.predictions) if args.predictions
                        else results_dir / "predictions" / f"test_predictions_{version}.csv")
    if not predictions_path.is_absolute():
        predictions_path = config.project_root / predictions_path

    if not predictions_path.exists():
        LOGGER.error("no test predictions found at %s", predictions_path)
        LOGGER.error("Run the evaluation first: python -m src.model.evaluate")
        return 1

    rows = load_predictions_csv(predictions_path)
    if not rows:
        LOGGER.error("%s contains no predictions", predictions_path)
        return 1

    analysis = analyse(config, rows, compute_quality=not args.no_quality)
    out_dir = ensure_dir(results_dir / "error_analysis")

    grid_path: Optional[Path] = None
    if analysis["errors"]:
        selected = analysis["errors"][:args.max_grid]
        paths = [config.project_root / e["file"] for e in selected]
        captions = [
            f"{e['filename']}\nactual: {e['actual_class']}\n"
            f"predicted: {e['predicted_class']} ({e['confidence'] * 100:.0f}%)"
            for e in selected
        ]
        grid_path = make_thumbnail_grid(
            paths, captions, out_dir / f"misclassified_grid_{version}.png",
            title=f"Misclassified test images - model {version}")

    confidence_plot = plot_confidence_distribution(
        rows, results_dir / "graphs" / f"confidence_distribution_{version}.png",
        threshold=config.confidence_threshold)

    write_json(out_dir / f"error_analysis_{version}.json", analysis)
    write_errors_csv(analysis, out_dir / f"errors_{version}.csv")
    markdown = analysis_to_markdown(analysis, config, version, grid_path, confidence_plot)
    write_text(out_dir / f"error_analysis_{version}.md", markdown)

    if not args.quiet:
        print(markdown)
    LOGGER.info("error analysis written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
