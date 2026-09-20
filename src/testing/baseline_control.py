"""Trivial baselines that tell you how hard your dataset actually is.

Why this exists
---------------
A convolutional network reporting high test accuracy tells you the classes are
separable. It does not tell you the network learned anything interesting, or
that the task is hard, or that the result will survive contact with real
photographs.

This script fits deliberately weak models on deliberately crude features and
reports their test accuracy:

* **colour**: per-channel mean and standard deviation plus overall brightness,
  computed on a 32x32 thumbnail. Seven numbers. No texture, no shape, no
  spatial information whatsoever.
* **grayscale**: the same statistics after discarding colour, which isolates
  how much of the signal is brightness and contrast rather than hue.

Read the result as a floor, not a ceiling:

* If a seven-number colour baseline gets close to the CNN, **the dataset is
  nearly separable by average colour**. The CNN's score is not evidence that it
  learned lesion morphology, and the gap between the two is the only part of
  the score that required a network at all.
* If the baselines sit near chance and the CNN is high, the CNN is using
  structure the baselines cannot see. That is the result you want.

Colour is genuinely diagnostic for some diseases, so a high colour baseline is
not proof of a defect in the data. It is proof that the *benchmark* is easy,
which is what matters when deciding how much the headline number is worth.

Run::

    python -m src.testing.baseline_control
    python -m src.testing.baseline_control --thumbnail 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..model.datasets import list_split_files
from ..utils.config import Config, load_config
from ..utils.helpers import (ensure_dir, format_percent, get_logger,
                             markdown_table, read_json, utc_timestamp,
                             write_json, write_text)

LOGGER = get_logger(__name__)


def extract_features(paths: List[Path], thumbnail: int, grayscale: bool) -> np.ndarray:
    """Crude global statistics: channel means, channel sds, overall mean."""
    rows = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("L" if grayscale else "RGB").resize((thumbnail, thumbnail))
            array = np.asarray(image, dtype="float32") / 255.0
        if array.ndim == 2:
            array = array[..., None]
        rows.append(
            [array[..., c].mean() for c in range(array.shape[-1])]
            + [array[..., c].std() for c in range(array.shape[-1])]
            + [float(array.mean())]
        )
    return np.asarray(rows, dtype="float64")


def run(config: Config, thumbnail: int = 32) -> Dict[str, Any]:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_paths, train_labels = list_split_files(config, config.path("train_dir"))
    test_paths, test_labels = list_split_files(config, config.path("test_dir"))
    if not train_paths or not test_paths:
        raise RuntimeError("train or test split is empty - run prepare_dataset first")

    y_train = np.asarray(train_labels)
    y_test = np.asarray(test_labels)
    results: List[Dict[str, Any]] = []

    for feature_name, grayscale in (("colour statistics", False),
                                    ("grayscale statistics", True)):
        LOGGER.info("extracting %s from %d + %d images", feature_name,
                    len(train_paths), len(test_paths))
        x_train = extract_features(train_paths, thumbnail, grayscale)
        x_test = extract_features(test_paths, thumbnail, grayscale)

        for model_name, model in (
            ("logistic regression", LogisticRegression(max_iter=3000)),
            ("random forest", RandomForestClassifier(n_estimators=200,
                                                     random_state=config.seed)),
        ):
            pipeline = make_pipeline(StandardScaler(), model).fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            per_class = {
                name: float((predictions[y_test == index] == index).mean())
                for index, name in enumerate(config.class_names)
                if (y_test == index).any()
            }
            results.append({
                "features": feature_name,
                "n_features": int(x_train.shape[1]),
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "macro_f1": float(f1_score(y_test, predictions, average="macro")),
                "per_class_recall": per_class,
            })

    dummy = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(y_train), 1)), y_train)
    majority = float(accuracy_score(y_test, dummy.predict(np.zeros((len(y_test), 1)))))

    return {
        "generated_utc": utc_timestamp(),
        "thumbnail_px": thumbnail,
        "n_train": len(train_paths),
        "n_test": len(test_paths),
        "chance_accuracy": 1.0 / config.num_classes,
        "majority_class_accuracy": majority,
        "baselines": results,
    }


def to_markdown(results: Dict[str, Any], config: Config,
                cnn_accuracy: Optional[float]) -> str:
    best = max(r["accuracy"] for r in results["baselines"])
    lines = [
        "# Trivial-baseline control",
        "",
        f"* Generated (UTC): {results['generated_utc']}",
        f"* Train / test images: {results['n_train']} / {results['n_test']}",
        f"* Features computed on a {results['thumbnail_px']}x{results['thumbnail_px']} thumbnail",
        f"* Chance accuracy: {format_percent(results['chance_accuracy'], 1)}",
        f"* Majority-class accuracy: {format_percent(results['majority_class_accuracy'], 1)}",
        "",
        "## Baselines",
        "",
        markdown_table(
            ["Features", "How many numbers", "Model", "Test accuracy", "Macro F1"],
            [[r["features"], r["n_features"], r["model"],
              format_percent(r["accuracy"], 2), f"{r['macro_f1']:.4f}"]
             for r in results["baselines"]]),
        "",
    ]

    if cnn_accuracy is not None:
        gap = cnn_accuracy - best
        lines += [
            "## Against the trained network",
            "",
            markdown_table(
                ["Model", "Test accuracy"],
                [["Best trivial baseline", format_percent(best, 2)],
                 ["Trained CNN", format_percent(cnn_accuracy, 2)],
                 ["**Gap attributable to the network**", f"**{gap * 100:+.2f} pp**"]]),
            "",
        ]
        if best >= 0.90:
            lines += [
                f"> **The dataset is nearly separable by {results['baselines'][0]['n_features']} "
                f"global statistics.** A model with no notion of texture, shape or spatial "
                f"arrangement reaches {format_percent(best, 1)}. The network adds "
                f"{gap * 100:+.2f} percentage points on top of that.",
                "",
                "This does not mean the network is broken or the labels are wrong. It means "
                "the **benchmark is easy**: the images are curated closely enough that average "
                "colour and brightness almost determine the class. A headline accuracy from "
                "such a benchmark says very little about performance on field photographs, "
                "where lighting, background and framing vary and the global statistics stop "
                "being informative.",
                "",
                "Treat the CNN's score as an upper bound achieved under ideal conditions, not "
                "as an estimate of real-world accuracy.",
                "",
            ]
        else:
            lines += [
                f"The best trivial baseline reaches {format_percent(best, 1)} against the "
                f"network's {format_percent(cnn_accuracy, 1)}. The network is using structure "
                f"the global statistics cannot see, which is the outcome you want.",
                "",
            ]

    lines += [
        "## Per-class recall from trivial features",
        "",
        markdown_table(
            ["Class"] + [f"{r['features']} / {r['model']}" for r in results["baselines"]],
            [[name] + [format_percent(r["per_class_recall"].get(name), 1)
                       for r in results["baselines"]]
             for name in config.class_names]),
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit trivial baselines to measure how hard the dataset really is")
    parser.add_argument("--config", default=None)
    parser.add_argument("--version", default=None, help="model version to compare against")
    parser.add_argument("--thumbnail", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    overrides = {"model": {"version": args.version}} if args.version else None
    config = load_config(args.config, overrides=overrides)
    results = run(config, thumbnail=args.thumbnail)

    # Compare against the trained network when an evaluation exists.
    cnn_accuracy = None
    evaluation = config.path("results_dir") / "evaluation" / f"evaluation_{config.model_version}.json"
    if evaluation.exists():
        try:
            cnn_accuracy = float(read_json(evaluation)["overall"]["accuracy"])
        except (KeyError, ValueError):
            pass
    results["cnn_accuracy"] = cnn_accuracy

    out_dir = ensure_dir(config.path("results_dir") / "baselines")
    write_json(out_dir / f"baseline_{config.model_version}.json", results)
    markdown = to_markdown(results, config, cnn_accuracy)
    write_text(out_dir / f"baseline_{config.model_version}.md", markdown)

    if not args.quiet:
        print(markdown)
    LOGGER.info("written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
