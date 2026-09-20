"""Compare two or more trained Leaf Lens model versions.

Reads the artefacts each version already produced - ``run_metadata.json`` from
``models/<version>/`` and ``evaluation_<version>.json`` from
``results/evaluation/`` - and lays them side by side: overall metrics, per-class
metrics, confusion matrices, realistic-testing results, and what changed in the
configuration between the versions.

The tool reports differences. It does **not** declare a winner: a small
accuracy gain on a small test split is not evidence of a better model, and the
report says so and shows the sample size next to every number.

Run::

    python -m src.model.compare_models v1 v2
    python -m src.model.compare_models --all
"""

from __future__ import annotations

import argparse
import sys
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
    read_json,
    utc_timestamp,
    write_json,
    write_text,
)

LOGGER = get_logger(__name__)

# Configuration keys whose differences are worth reporting between versions.
TRACKED_SETTINGS: List[Tuple[str, Tuple[str, ...]]] = [
    ("architecture", ("model", "architecture")),
    ("backbone weights", ("model", "weights")),
    ("dropout", ("model", "dropout")),
    ("dense units", ("model", "dense_units")),
    ("L2", ("model", "l2")),
    ("label smoothing", ("model", "label_smoothing")),
    ("image size", ("data", "image_size")),
    ("batch size", ("data", "batch_size")),
    ("epochs", ("training", "epochs")),
    ("learning rate", ("training", "learning_rate")),
    ("optimizer", ("training", "optimizer")),
    ("class weights", ("training", "use_class_weights")),
    ("fine-tune enabled", ("training", "fine_tune", "enabled")),
    ("fine-tune layers", ("training", "fine_tune", "unfreeze_layers")),
    ("fine-tune LR", ("training", "fine_tune", "learning_rate")),
    ("augmentation", ("training", "augmentation")),
    ("split", ("data", "split")),
    ("random seed", ("random_seed",)),
]


def discover_versions(config: Config) -> List[str]:
    models_dir = config.path("models_dir")
    if not models_dir.exists():
        return []
    return sorted(
        directory.name for directory in models_dir.iterdir()
        if directory.is_dir() and (directory / "model.keras").exists()
    )


def load_version(config: Config, version: str) -> Dict[str, Any]:
    """Collect every artefact available for one version."""
    model_dir = config.path("models_dir") / version
    results_dir = config.path("results_dir")

    payload: Dict[str, Any] = {"version": version, "available": {}}

    metadata_path = model_dir / "run_metadata.json"
    payload["run_metadata"] = read_json(metadata_path) if metadata_path.exists() else None
    payload["available"]["run_metadata"] = metadata_path.exists()

    evaluation_path = results_dir / "evaluation" / f"evaluation_{version}.json"
    payload["evaluation"] = read_json(evaluation_path) if evaluation_path.exists() else None
    payload["available"]["evaluation"] = evaluation_path.exists()

    error_path = results_dir / "error_analysis" / f"error_analysis_{version}.json"
    payload["error_analysis"] = read_json(error_path) if error_path.exists() else None
    payload["available"]["error_analysis"] = error_path.exists()

    realistic_path = results_dir / "realistic_testing" / f"realistic_summary_{version}.json"
    payload["realistic"] = (read_json(realistic_path).get("summary")
                            if realistic_path.exists() else None)
    payload["available"]["realistic_testing"] = realistic_path.exists()

    config_path = model_dir / "config_snapshot.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as handle:
            payload["config_snapshot"] = yaml.safe_load(handle) or {}
    else:
        payload["config_snapshot"] = None
    payload["available"]["config_snapshot"] = config_path.exists()

    return payload


def _nested(data: Optional[Dict[str, Any]], keys: Tuple[str, ...]) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def configuration_differences(versions: List[Dict[str, Any]]) -> List[List[Any]]:
    """Rows of ``[setting, value_v1, value_v2, ...]`` for settings that differ."""
    rows: List[List[Any]] = []
    for label, keys in TRACKED_SETTINGS:
        values = [_nested(v.get("config_snapshot"), keys) for v in versions]
        if all(value is None for value in values):
            continue
        if len({repr(value) for value in values}) == 1:
            continue  # identical across all versions
        rows.append([label] + [("-" if v is None else repr(v)) for v in values])
    return rows


def plot_comparison(versions: List[Dict[str, Any]], output_path: Path) -> Optional[Path]:
    """Grouped bars: overall test metrics per version."""
    configure_matplotlib()
    import matplotlib.pyplot as plt

    usable = [v for v in versions if v.get("evaluation")]
    if len(usable) < 1:
        return None

    metrics = [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
               ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")]
    positions = np.arange(len(metrics))
    width = 0.8 / len(usable)
    colors = ["#3f7d3f", "#3b6ea5", "#d79a2b", "#8a5fa8", "#b5453b"]

    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    for index, version in enumerate(usable):
        overall = version["evaluation"]["overall"]
        values = [overall.get(key, 0.0) or 0.0 for key, _ in metrics]
        n = version["evaluation"].get("num_images", 0)
        axis.bar(positions + index * width, values, width,
                 label=f"{version['version']} (n={n})", color=colors[index % len(colors)])
    axis.set_xticks(positions + width * (len(usable) - 1) / 2)
    axis.set_xticklabels([label for _, label in metrics])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score on the test split")
    axis.set_title("Model version comparison (test split)")
    axis.legend()
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def plot_per_class_comparison(versions: List[Dict[str, Any]], output_path: Path
                              ) -> Optional[Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    usable = [v for v in versions if v.get("evaluation")]
    if not usable:
        return None
    class_names = usable[0]["evaluation"]["class_names"]
    positions = np.arange(len(class_names))
    width = 0.8 / len(usable)
    colors = ["#3f7d3f", "#3b6ea5", "#d79a2b", "#8a5fa8", "#b5453b"]

    figure, axis = plt.subplots(figsize=(1.8 * len(class_names) + 3.0, 4.4))
    for index, version in enumerate(usable):
        lookup = {e["class"]: e["f1"] for e in version["evaluation"]["per_class"]}
        values = [lookup.get(name, 0.0) for name in class_names]
        axis.bar(positions + index * width, values, width, label=version["version"],
                 color=colors[index % len(colors)])
    axis.set_xticks(positions + width * (len(usable) - 1) / 2)
    axis.set_xticklabels(class_names, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("F1 on the test split")
    axis.set_title("Per-class F1 by model version")
    axis.legend()
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def build_report(config: Config, versions: List[Dict[str, Any]]) -> str:
    names = [v["version"] for v in versions]
    lines = [
        "# Model comparison",
        "",
        f"* Generated (UTC): {utc_timestamp()}",
        f"* Versions compared: {', '.join(names)}",
        "",
        "## Availability",
        "",
        markdown_table(
            ["Artefact"] + names,
            [[label] + ["yes" if v["available"].get(key) else "**missing**" for v in versions]
             for key, label in [
                 ("run_metadata", "Training metadata"),
                 ("config_snapshot", "Configuration snapshot"),
                 ("evaluation", "Test-split evaluation"),
                 ("error_analysis", "Error analysis"),
                 ("realistic_testing", "Realistic-condition testing"),
             ]]),
        "",
    ]

    missing = [v["version"] for v in versions if not v.get("evaluation")]
    if missing:
        lines += [
            f"> Versions without a test-split evaluation ({', '.join(missing)}) cannot be "
            f"compared on test metrics. Run `python -m src.model.evaluate --version <version>` "
            f"for each of them.",
            "",
        ]

    evaluated = [v for v in versions if v.get("evaluation")]
    if evaluated:
        lines += [
            "## Test-split metrics",
            "",
            markdown_table(
                ["Metric"] + [f"{v['version']} (n={v['evaluation'].get('num_images', 0)})"
                              for v in evaluated],
                [
                    ["Accuracy"] + [format_percent(v["evaluation"]["overall"]["accuracy"], 2)
                                    for v in evaluated],
                    ["Macro precision"] + [f"{v['evaluation']['overall']['macro_precision']:.4f}"
                                           for v in evaluated],
                    ["Macro recall"] + [f"{v['evaluation']['overall']['macro_recall']:.4f}"
                                        for v in evaluated],
                    ["Macro F1"] + [f"{v['evaluation']['overall']['macro_f1']:.4f}"
                                    for v in evaluated],
                    ["Weighted F1"] + [f"{v['evaluation']['overall']['weighted_f1']:.4f}"
                                       for v in evaluated],
                    ["Mean confidence"] + [
                        format_percent(v["evaluation"]["overall"]["mean_confidence"], 1)
                        for v in evaluated],
                ]),
            "",
        ]

        class_names = evaluated[0]["evaluation"]["class_names"]
        lines += [
            "### Per-class F1",
            "",
            markdown_table(
                ["Class"] + [v["version"] for v in evaluated] + ["Support"],
                [[name]
                 + [f"{next((e['f1'] for e in v['evaluation']['per_class'] if e['class'] == name), 0.0):.4f}"
                    for v in evaluated]
                 + [next((e["support"] for e in evaluated[0]["evaluation"]["per_class"]
                          if e["class"] == name), 0)]
                 for name in class_names]),
            "",
            "### Per-class recall",
            "",
            markdown_table(
                ["Class"] + [v["version"] for v in evaluated],
                [[name] + [f"{next((e['recall'] for e in v['evaluation']['per_class'] if e['class'] == name), 0.0):.4f}"
                           for v in evaluated]
                 for name in class_names]),
            "",
            "### Confusion matrices",
            "",
        ]
        for version in evaluated:
            matrix = version["evaluation"]["confusion_matrix"]
            lines += [
                f"**{version['version']}** (rows = actual, columns = predicted)",
                "",
                markdown_table(["Actual \\ Predicted"] + class_names,
                               [[class_names[i]] + [str(v) for v in row]
                                for i, row in enumerate(matrix)]),
                "",
            ]

    # Training-time figures.
    trained = [v for v in versions if v.get("run_metadata")]
    if trained:
        lines += [
            "## Training figures",
            "",
            "Training and validation accuracy describe the fit, not performance on new "
            "photographs. They are shown for completeness only.",
            "",
            markdown_table(
                ["Measure"] + [v["version"] for v in trained],
                [
                    ["Training images"] + [str(v["run_metadata"].get("dataset_totals", {}).get("train", "-"))
                                           for v in trained],
                    ["Validation images"] + [str(v["run_metadata"].get("dataset_totals", {}).get("validation", "-"))
                                             for v in trained],
                    ["Test images"] + [str(v["run_metadata"].get("dataset_totals", {}).get("test", "-"))
                                       for v in trained],
                    ["Epochs run (phase 1)"] + [str(v["run_metadata"].get("epochs_run_phase1", "-"))
                                                for v in trained],
                    ["Epochs run (fine-tune)"] + [str(v["run_metadata"].get("epochs_run_finetune", "-"))
                                                  for v in trained],
                    ["Final training accuracy"] + [format_percent(v["run_metadata"].get("final_train_accuracy"), 2)
                                                   for v in trained],
                    ["Best validation accuracy"] + [format_percent(v["run_metadata"].get("best_val_accuracy"), 2)
                                                    for v in trained],
                    ["Trainable parameters"] + [str(v["run_metadata"].get("parameters", {}).get("trainable", "-"))
                                                for v in trained],
                    ["Training time (s)"] + [str(v["run_metadata"].get("duration_seconds", "-"))
                                             for v in trained],
                ]),
            "",
        ]

    # Realistic testing.
    realistic = [v for v in versions if v.get("realistic")]
    if realistic:
        lines += [
            "## Realistic-condition testing",
            "",
            markdown_table(
                ["Measure"] + [v["version"] for v in realistic],
                [
                    ["Photographs scored"] + [str(v["realistic"]["overall"]["n"]) for v in realistic],
                    ["Accuracy"] + [format_percent(v["realistic"]["overall"]["accuracy"], 1)
                                    for v in realistic],
                    ["Mean confidence"] + [format_percent(v["realistic"]["overall"]["mean_confidence"], 1)
                                           for v in realistic],
                    ["Below threshold"] + [format_percent(v["realistic"]["overall"]["low_confidence_rate"], 1)
                                           for v in realistic],
                ]),
            "",
        ]
    else:
        lines += ["## Realistic-condition testing", "",
                  "No realistic-condition results are available for any version. "
                  "Run `python -m src.testing.realistic_testing` once you have field "
                  "photographs.", ""]

    # What changed.
    differences = configuration_differences(versions)
    lines += ["## What changed between the versions", ""]
    if differences:
        lines += [markdown_table(["Setting"] + names, differences), ""]
    elif all(v["available"].get("config_snapshot") for v in versions):
        lines += ["The tracked configuration settings are identical across these versions.", ""]
    else:
        lines += ["Configuration snapshots are missing for at least one version, so the "
                  "settings cannot be diffed.", ""]

    # Deliberately no winner.
    lines += [
        "## How to read this comparison",
        "",
        "This report does not name a best model. Before concluding that one version is "
        "genuinely better:",
        "",
        "* Check the test-set size (`n` above). A difference of a few images on a small "
        "test split is within noise.",
        "* Check whether the gain is spread across classes or comes from one class only - "
        "the per-class tables show this.",
        "* Check whether the versions were evaluated on the *same* test split. Re-running "
        "`prepare_dataset` with a different seed or a changed dataset invalidates the "
        "comparison.",
        "* Prefer the realistic-condition results over the test split when they disagree: "
        "they are closer to how the application is actually used.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Leaf Lens model versions")
    parser.add_argument("versions", nargs="*", help="version names, e.g. v1 v2")
    parser.add_argument("--all", action="store_true", help="compare every trained version")
    parser.add_argument("--config", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    names = args.versions
    if args.all or not names:
        names = discover_versions(config)

    if not names:
        LOGGER.error("no trained model versions found under %s", config.path("models_dir"))
        LOGGER.error("Train at least one: python -m src.model.train --version v1")
        return 1
    if len(names) == 1:
        LOGGER.warning("only one version (%s) is available - the report will describe it "
                       "but there is nothing to compare it against", names[0])

    versions = [load_version(config, name) for name in names]
    report = build_report(config, versions)

    out_dir = ensure_dir(config.path("results_dir") / "comparison")
    label = "_vs_".join(names)
    write_text(out_dir / f"comparison_{label}.md", report)
    write_json(out_dir / f"comparison_{label}.json", {
        "generated_utc": utc_timestamp(),
        "versions": names,
        "availability": {v["version"]: v["available"] for v in versions},
        "test_metrics": {v["version"]: (v["evaluation"] or {}).get("overall") for v in versions},
        "configuration_differences": configuration_differences(versions),
    })
    plot_comparison(versions, config.path("results_dir") / "graphs" / f"comparison_{label}.png")
    plot_per_class_comparison(
        versions, config.path("results_dir") / "graphs" / f"comparison_per_class_{label}.png")

    if not args.quiet:
        print(report)
    LOGGER.info("comparison written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
