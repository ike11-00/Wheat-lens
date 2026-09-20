"""Evaluate a trained Leaf Lens model on the held-out test split.

Reports overall accuracy, macro/weighted precision, recall and F1, the same
metrics per class, and a confusion matrix over every configured class.

Outputs
-------
``results/evaluation/evaluation_<version>.json``   machine-readable metrics
``results/evaluation/evaluation_<version>.md``     human-readable report
``results/predictions/test_predictions_<version>.csv``  one row per test image
``results/confusion_matrices/confusion_matrix_<version>.png`` (+ ``_normalised.png``)
``results/confusion_matrices/confusion_matrix_<version>.csv``

The report keeps training, validation and test numbers clearly separated:
training and validation figures are copied from the run metadata and labelled
as such; only the test figures describe performance on unseen images.

Run::

    python -m src.model.evaluate
    python -m src.model.evaluate --split validation --version v1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.config import Config, load_config
from ..utils.helpers import (
    configure_matplotlib,
    ensure_dir,
    environment_report,
    format_percent,
    get_logger,
    markdown_table,
    read_json,
    relative_to_root,
    set_global_seed,
    utc_timestamp,
    write_json,
    write_text,
)
from .predict import ModelNotAvailableError, Predictor

LOGGER = get_logger(__name__)


def evaluate_split(config: Config, predictor: Predictor, split: str = "test"
                   ) -> Dict[str, Any]:
    """Run the model over one split and compute every metric."""
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 precision_recall_fscore_support)

    from .datasets import EmptyDatasetError, list_split_files

    split_dir = config.path(f"{split}_dir")
    paths, labels = list_split_files(config, split_dir)
    if not paths:
        raise EmptyDatasetError(
            f"no images found in {split_dir}. Run 'python -m src.data.prepare_dataset' first."
        )

    class_names = predictor.class_names
    if class_names != config.class_names:
        LOGGER.warning(
            "the model's class order (%s) differs from config.yaml (%s); the model's order "
            "is authoritative", class_names, config.class_names)

    LOGGER.info("evaluating %d images from %s", len(paths), split_dir)
    probabilities = np.zeros((len(paths), len(class_names)), dtype="float64")
    batch_size = config.batch_size
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        probabilities[start:start + len(chunk)] = predictor.predict_proba(chunk)

    y_true = np.asarray(labels, dtype="int64")
    y_pred = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    indices = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=indices)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=indices,
                                            average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=indices,
                                               average="weighted", zero_division=0)

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    threshold = predictor.confidence_threshold
    correct = y_true == y_pred

    per_class = []
    for index, name in enumerate(class_names):
        class_mask = y_true == index
        per_class.append({
            "class": name,
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "accuracy_within_class": float(correct[class_mask].mean()) if class_mask.any() else None,
            "mean_confidence": float(confidences[class_mask].mean()) if class_mask.any() else None,
        })

    rows = []
    for path, true_index, pred_index, row in zip(paths, y_true, y_pred, probabilities):
        entry = {
            "file": relative_to_root(path, config.project_root),
            "filename": Path(path).name,
            "actual_class": class_names[true_index],
            "predicted_class": class_names[pred_index],
            "confidence": float(row[pred_index]),
            "correct": bool(true_index == pred_index),
            "low_confidence": bool(row[pred_index] < threshold),
        }
        for index, name in enumerate(class_names):
            entry[f"p_{name.replace(' ', '_')}"] = float(row[index])
        rows.append(entry)

    high_conf = confidences >= threshold
    report_text = classification_report(y_true, y_pred, labels=indices,
                                        target_names=class_names, zero_division=0)

    return {
        "split": split,
        "generated_utc": utc_timestamp(),
        "model_version": predictor.model_version,
        "model_path": str(predictor.model_path),
        "architecture": predictor.architecture,
        "image_size": predictor.image_size,
        "confidence_threshold": threshold,
        "class_names": class_names,
        "num_images": int(len(paths)),
        "overall": {
            "accuracy": accuracy,
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
            "weighted_precision": float(weighted[0]),
            "weighted_recall": float(weighted[1]),
            "weighted_f1": float(weighted[2]),
            "mean_confidence": float(confidences.mean()) if len(confidences) else None,
            "mean_confidence_correct": float(confidences[correct].mean()) if correct.any() else None,
            "mean_confidence_incorrect": (float(confidences[~correct].mean())
                                          if (~correct).any() else None),
        },
        "confidence_threshold_analysis": {
            "threshold": threshold,
            "predictions_at_or_above_threshold": int(high_conf.sum()),
            "coverage": float(high_conf.mean()) if len(high_conf) else 0.0,
            "accuracy_at_or_above_threshold": (float(correct[high_conf].mean())
                                               if high_conf.any() else None),
            "accuracy_below_threshold": (float(correct[~high_conf].mean())
                                         if (~high_conf).any() else None),
        },
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": class_names,
        "sklearn_classification_report": report_text,
        "predictions": rows,
        "environment": environment_report(),
        "note": (
            "These figures describe the model's behaviour on this split only. Test-split "
            "accuracy is an estimate of performance on unseen images drawn from the SAME "
            "distribution as the training data; it does not predict performance on field "
            "photographs taken under different conditions. See docs/limitations.md."
        ),
    }


def plot_confusion_matrix(matrix: np.ndarray, class_names: List[str], output_path: Path,
                          normalise: bool = False, title: str = "Confusion matrix") -> Path:
    """Annotated heat map with true classes on the rows and predictions on the columns."""
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = np.asarray(matrix, dtype="float64")
    if normalise:
        totals = data.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            data = np.divide(data, totals, out=np.zeros_like(data), where=totals > 0)
        fmt, vmax = ".2f", 1.0
    else:
        fmt, vmax = ".0f", None

    figure, axis = plt.subplots(figsize=(1.5 * len(class_names) + 3.0,
                                         1.2 * len(class_names) + 2.5))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="YlGnBu", vmin=0, vmax=vmax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Proportion of true class" if normalise else "Images"},
                ax=axis, square=True, linewidths=0.5, linecolor="white")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(title)
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    plt.setp(axis.get_yticklabels(), rotation=0)
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def plot_per_class_metrics(per_class: List[Dict[str, Any]], output_path: Path,
                           title: str = "Per-class metrics") -> Optional[Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    if not per_class:
        return None
    names = [entry["class"] for entry in per_class]
    positions = np.arange(len(names))
    width = 0.26

    figure, axis = plt.subplots(figsize=(1.7 * len(names) + 3.0, 4.4))
    axis.bar(positions - width, [e["precision"] for e in per_class], width,
             label="Precision", color="#3b6ea5")
    axis.bar(positions, [e["recall"] for e in per_class], width,
             label="Recall", color="#3f7d3f")
    axis.bar(positions + width, [e["f1"] for e in per_class], width,
             label="F1", color="#d79a2b")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.legend()
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def write_predictions_csv(results: Dict[str, Any], output_path: Path) -> Path:
    rows = results["predictions"]
    ensure_dir(Path(output_path).parent)
    if not rows:
        Path(output_path).write_text("", encoding="utf-8")
        return Path(output_path)
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return Path(output_path)


def _confusion_pairs(matrix: np.ndarray, class_names: List[str], top: int = 5
                     ) -> List[Dict[str, Any]]:
    """The most frequent off-diagonal confusions, largest first."""
    pairs: List[Dict[str, Any]] = []
    totals = np.asarray(matrix).sum(axis=1)
    for i, actual in enumerate(class_names):
        for j, predicted in enumerate(class_names):
            if i == j or matrix[i][j] == 0:
                continue
            pairs.append({
                "actual": actual,
                "predicted": predicted,
                "count": int(matrix[i][j]),
                "share_of_actual_class": (float(matrix[i][j] / totals[i]) if totals[i] else 0.0),
            })
    pairs.sort(key=lambda entry: entry["count"], reverse=True)
    return pairs[:top]


def results_to_markdown(results: Dict[str, Any], config: Config,
                        run_metadata: Optional[Dict[str, Any]] = None) -> str:
    overall = results["overall"]
    threshold_analysis = results["confidence_threshold_analysis"]
    class_names = results["class_names"]

    lines = [
        f"# Evaluation report - model {results['model_version']} ({results['split']} split)",
        "",
        f"* Generated (UTC): {results['generated_utc']}",
        f"* Model: `{results['model_path']}` ({results['architecture']}, "
        f"{results['image_size']}x{results['image_size']} input)",
        f"* Images evaluated: **{results['num_images']}** (from `data/{results['split']}`)",
        "",
        "## Training / validation / test performance",
        "",
        "These three numbers measure different things and must not be confused.",
        "",
    ]

    if run_metadata:
        lines.append(markdown_table(
            ["Stage", "Accuracy", "What it means"],
            [
                ["Training (final epoch)",
                 format_percent(run_metadata.get("final_train_accuracy")),
                 "Fit on images the model learned from. Says nothing about new photographs."],
                ["Validation (best epoch)",
                 format_percent(run_metadata.get("best_val_accuracy")),
                 "Used to pick the checkpoint and stop early, so it is mildly optimistic."],
                [f"**Test ({results['split']} split)**",
                 f"**{format_percent(overall['accuracy'])}**",
                 "**Unseen images, never used for training or model selection.**"],
            ]))
    else:
        lines.append(
            f"Training and validation figures are unavailable (no `run_metadata.json` next to "
            f"the model). Test accuracy on the {results['split']} split: "
            f"**{format_percent(overall['accuracy'])}**.")
    lines.append("")

    lines += [
        "## Overall metrics",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Accuracy", format_percent(overall["accuracy"], 2)],
                ["Macro precision", f"{overall['macro_precision']:.4f}"],
                ["Macro recall", f"{overall['macro_recall']:.4f}"],
                ["Macro F1", f"{overall['macro_f1']:.4f}"],
                ["Weighted precision", f"{overall['weighted_precision']:.4f}"],
                ["Weighted recall", f"{overall['weighted_recall']:.4f}"],
                ["Weighted F1", f"{overall['weighted_f1']:.4f}"],
                ["Mean confidence (all)", format_percent(overall["mean_confidence"], 2)],
                ["Mean confidence (correct)", format_percent(overall["mean_confidence_correct"], 2)],
                ["Mean confidence (incorrect)", format_percent(overall["mean_confidence_incorrect"], 2)],
            ]),
        "",
        "## Per-class metrics",
        "",
        markdown_table(
            ["Class", "Support", "Precision", "Recall", "F1", "Mean confidence"],
            [[e["class"], e["support"], f"{e['precision']:.4f}", f"{e['recall']:.4f}",
              f"{e['f1']:.4f}", format_percent(e["mean_confidence"], 1)]
             for e in results["per_class"]]),
        "",
        "## Confusion matrix",
        "",
        "Rows = actual class, columns = predicted class.",
        "",
        markdown_table(
            ["Actual \\ Predicted"] + class_names,
            [[class_names[i]] + [str(v) for v in row]
             for i, row in enumerate(results["confusion_matrix"])]),
        "",
    ]

    pairs = _confusion_pairs(np.asarray(results["confusion_matrix"]), class_names)
    if pairs:
        lines += [
            "### Most frequent confusions in this test run",
            "",
            markdown_table(
                ["Actual", "Predicted as", "Images", "% of the actual class"],
                [[p["actual"], p["predicted"], p["count"],
                  format_percent(p["share_of_actual_class"], 1)] for p in pairs]),
            "",
            "These are the confusions this run actually produced. No claim is made about "
            "*why* they happen - see the error-analysis report for evidence.",
            "",
        ]
    else:
        lines += ["### Most frequent confusions in this test run", "",
                  "No misclassifications occurred on this split.", ""]

    lines += [
        "## Confidence threshold",
        "",
        f"With the configured threshold of {threshold_analysis['threshold']:.2f}:",
        "",
        markdown_table(
            ["Measure", "Value"],
            [
                ["Predictions at or above threshold",
                 f"{threshold_analysis['predictions_at_or_above_threshold']} of "
                 f"{results['num_images']}"],
                ["Coverage", format_percent(threshold_analysis["coverage"], 1)],
                ["Accuracy at or above threshold",
                 format_percent(threshold_analysis["accuracy_at_or_above_threshold"], 2)],
                ["Accuracy below threshold",
                 format_percent(threshold_analysis["accuracy_below_threshold"], 2)],
            ]),
        "",
        "## scikit-learn classification report",
        "",
        "```",
        results["sklearn_classification_report"],
        "```",
        "",
        "## Note",
        "",
        results["note"],
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained Leaf Lens model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="path to a .keras model file")
    parser.add_argument("--version", default=None, help="model version (default: config)")
    parser.add_argument("--split", default="test", choices=["test", "validation", "train"],
                        help="which split to evaluate (default: test)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    overrides = {"model": {"version": args.version}} if args.version else None
    config = load_config(args.config, overrides=overrides)
    set_global_seed(config.seed)

    predictor = Predictor(config, model_path=Path(args.model) if args.model else None)
    if not predictor.is_available():
        LOGGER.error("no trained model found at %s", predictor.model_path)
        LOGGER.error("Train one first: python -m src.model.train")
        return 1

    from .datasets import EmptyDatasetError
    try:
        predictor.load()
        results = evaluate_split(config, predictor, split=args.split)
    except EmptyDatasetError as exc:
        LOGGER.error("EVALUATION CANNOT PROCEED: %s", exc)
        return 1
    except ModelNotAvailableError as exc:
        LOGGER.error("%s", exc)
        return 1

    version = results["model_version"]
    suffix = version if args.split == "test" else f"{version}_{args.split}"
    results_dir = config.path("results_dir")

    metadata_path = predictor.model_path.parent / "run_metadata.json"
    run_metadata = read_json(metadata_path) if metadata_path.exists() else None

    # JSON report without the per-image rows (those go to CSV).
    json_payload = {k: v for k, v in results.items() if k != "predictions"}
    write_json(results_dir / "evaluation" / f"evaluation_{suffix}.json", json_payload)
    write_text(results_dir / "evaluation" / f"evaluation_{suffix}.md",
               results_to_markdown(results, config, run_metadata))
    csv_path = write_predictions_csv(
        results, results_dir / "predictions" / f"{args.split}_predictions_{version}.csv")

    matrix = np.asarray(results["confusion_matrix"])
    names = results["class_names"]
    cm_dir = results_dir / "confusion_matrices"
    plot_confusion_matrix(matrix, names, cm_dir / f"confusion_matrix_{suffix}.png",
                          normalise=False,
                          title=f"Confusion matrix - {version} ({args.split} split, counts)")
    plot_confusion_matrix(matrix, names, cm_dir / f"confusion_matrix_{suffix}_normalised.png",
                          normalise=True,
                          title=f"Confusion matrix - {version} ({args.split} split, row-normalised)")
    ensure_dir(cm_dir)
    with (cm_dir / f"confusion_matrix_{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted"] + names)
        for name, row in zip(names, matrix.tolist()):
            writer.writerow([name] + row)
    plot_per_class_metrics(results["per_class"],
                           results_dir / "graphs" / f"per_class_metrics_{suffix}.png",
                           title=f"Per-class metrics - {version} ({args.split} split)")

    if not args.quiet:
        print(results_to_markdown(results, config, run_metadata))
    LOGGER.info("evaluation written to %s", results_dir / "evaluation")
    LOGGER.info("per-image predictions: %s", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
