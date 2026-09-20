"""Measure whether the model is keying on a dataset artefact rather than disease.

Why this exists
---------------
When the classes of a training set come from more than one collection, the
collections usually differ in ways that have nothing to do with the disease:
camera, resolution, background, compression. A network will happily learn those
differences, because they separate the classes perfectly and cost nothing to
learn. Test accuracy then looks excellent while the model has learned
"which dataset is this from".

This script tests that directly. It re-runs the test split twice:

1. **as-is** - the normal evaluation.
2. **resolution-normalised** - every image is first downsampled to a common
   resolution (default: the median resolution of the majority classes) and
   re-encoded as JPEG at a common quality, then classified.

For a class whose images already sit at that resolution, step 2 is very nearly
a no-op. For a class that came from a different collection at a very different
resolution, step 2 removes the scale and sharpness signature that distinguished
it. **A class whose accuracy collapses between run 1 and run 2 was being
recognised partly by its source, not only by its symptoms.**

This is evidence, not proof: normalising resolution also destroys genuine fine
detail, so some drop is expected for any class. Read the per-class deltas
against each other, not in isolation.

Run::

    python -m src.testing.confound_test
    python -m src.testing.confound_test --target-pixels 400 --quality 90
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..model.datasets import list_split_files
from ..model.predict import Predictor
from ..utils.config import Config, load_config
from ..utils.helpers import (configure_matplotlib, ensure_dir, format_percent,
                             get_logger, markdown_table, utc_timestamp,
                             write_json, write_text)
from ..utils.image_io import open_image

LOGGER = get_logger(__name__)


def class_resolutions(config: Config, split: str) -> Dict[str, List[float]]:
    """Megapixel counts per class for one split."""
    paths, labels = list_split_files(config, config.path(f"{split}_dir"))
    names = config.class_names
    out: Dict[str, List[float]] = defaultdict(list)
    for path, label in zip(paths, labels):
        with Image.open(path) as image:
            out[names[label]].append(image.width * image.height / 1e6)
    return dict(out)


def normalise(path: Path, target_pixels: int, quality: int) -> Image.Image:
    """Downsample to ``target_pixels`` on the long edge and re-encode as JPEG.

    The JPEG round-trip matters: it equalises compression artefacts, which are
    themselves a source signature.
    """
    image = open_image(path)
    long_edge = max(image.width, image.height)
    if long_edge > target_pixels:
        scale = target_pixels / long_edge
        image = image.resize((max(1, round(image.width * scale)),
                              max(1, round(image.height * scale))),
                             Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return open_image(buffer)


def run(config: Config, predictor: Predictor, split: str,
        target_pixels: int, quality: int) -> Dict[str, Any]:
    paths, labels = list_split_files(config, config.path(f"{split}_dir"))
    if not paths:
        raise RuntimeError(f"no images in data/{split}")
    names = predictor.class_names
    y_true = np.asarray(labels)

    LOGGER.info("run 1/2: %d images as-is", len(paths))
    plain = predictor.predict_many(paths)
    pred_plain = np.array([names.index(r["predicted_class"]) for r in plain])
    conf_plain = np.array([r["confidence"] for r in plain])

    LOGGER.info("run 2/2: %d images normalised to %dpx / quality %d",
                len(paths), target_pixels, quality)
    normalised_images = [normalise(p, target_pixels, quality) for p in paths]
    norm = predictor.predict_many(normalised_images)
    pred_norm = np.array([names.index(r["predicted_class"]) for r in norm])
    conf_norm = np.array([r["confidence"] for r in norm])

    resolutions = class_resolutions(config, split)
    per_class = []
    for index, name in enumerate(names):
        mask = y_true == index
        if not mask.any():
            continue
        acc_plain = float((pred_plain[mask] == index).mean())
        acc_norm = float((pred_norm[mask] == index).mean())
        megapixels = resolutions.get(name, [0.0])
        per_class.append({
            "class": name,
            "n": int(mask.sum()),
            "median_megapixels": float(np.median(megapixels)),
            "accuracy_as_is": acc_plain,
            "accuracy_normalised": acc_norm,
            "delta": acc_norm - acc_plain,
            "mean_confidence_as_is": float(conf_plain[mask].mean()),
            "mean_confidence_normalised": float(conf_norm[mask].mean()),
            "predictions_changed": int((pred_plain[mask] != pred_norm[mask]).sum()),
        })

    return {
        "generated_utc": utc_timestamp(),
        "split": split,
        "model_version": predictor.model_version,
        "target_pixels": target_pixels,
        "jpeg_quality": quality,
        "overall_accuracy_as_is": float((pred_plain == y_true).mean()),
        "overall_accuracy_normalised": float((pred_norm == y_true).mean()),
        "per_class": per_class,
    }


def plot(results: Dict[str, Any], output_path: Path) -> Optional[Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    rows = results["per_class"]
    if not rows:
        return None
    names = [r["class"] for r in rows]
    positions = np.arange(len(names))
    width = 0.38

    figure, axis = plt.subplots(figsize=(1.9 * len(names) + 3.0, 4.4))
    axis.bar(positions - width / 2, [r["accuracy_as_is"] for r in rows], width,
             label="as-is", color="#3f7d3f")
    axis.bar(positions + width / 2, [r["accuracy_normalised"] for r in rows], width,
             label=f"normalised to {results['target_pixels']}px", color="#b5453b")
    for i, r in enumerate(rows):
        axis.annotate(f"{r['median_megapixels']:.2f} MP",
                      (i, max(r["accuracy_as_is"], r["accuracy_normalised"]) + 0.03),
                      ha="center", fontsize=8, color="#555555")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=20, ha="right")
    axis.set_ylim(0, 1.15)
    axis.set_ylabel("Recall on the test split")
    axis.set_title("Does accuracy survive resolution normalisation?\n"
                   "A class that collapses was partly recognised by its source")
    axis.legend()
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def to_markdown(results: Dict[str, Any], chart: Optional[Path]) -> str:
    rows = results["per_class"]
    lines = [
        f"# Source-confound test - model {results['model_version']}",
        "",
        f"* Generated (UTC): {results['generated_utc']}",
        f"* Split: `data/{results['split']}`",
        f"* Normalisation: long edge <= {results['target_pixels']} px, "
        f"re-encoded as JPEG quality {results['jpeg_quality']}",
        f"* Overall accuracy as-is: **{format_percent(results['overall_accuracy_as_is'], 2)}**",
        f"* Overall accuracy normalised: **{format_percent(results['overall_accuracy_normalised'], 2)}**",
        "",
        "## Per class",
        "",
        markdown_table(
            ["Class", "Test images", "Median resolution", "Recall as-is",
             "Recall normalised", "Change", "Predictions changed"],
            [[r["class"], r["n"], f"{r['median_megapixels']:.2f} MP",
              format_percent(r["accuracy_as_is"], 1),
              format_percent(r["accuracy_normalised"], 1),
              f"{r['delta'] * 100:+.1f} pp", r["predictions_changed"]]
             for r in rows]),
        "",
        "## How to read this",
        "",
        "Normalisation downsamples every image to a common resolution and re-encodes it, "
        "removing the scale and compression signature that distinguishes one source "
        "collection from another. It is close to a no-op for a class already at that "
        "resolution.",
        "",
        "* A class that **holds its recall** is being recognised by something that survives "
        "downsampling - plausibly the symptoms.",
        "* A class that **collapses** was being recognised partly by its source. Its headline "
        "metrics in the evaluation report do not describe disease recognition and should not "
        "be quoted as if they did.",
        "",
        "This is evidence, not proof. Downsampling also destroys genuine fine detail, so some "
        "drop is expected everywhere. Compare the per-class changes against each other.",
        "",
    ]
    if chart:
        lines += [f"![Source confound test]({chart.as_posix()})", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test whether accuracy depends on a dataset artefact")
    parser.add_argument("--config", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--target-pixels", type=int, default=400,
                        help="long-edge resolution every image is reduced to")
    parser.add_argument("--quality", type=int, default=90, help="JPEG re-encode quality")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    overrides = {"model": {"version": args.version}} if args.version else None
    config = load_config(args.config, overrides=overrides)
    predictor = Predictor(config)
    if not predictor.is_available():
        LOGGER.error("no trained model at %s - train one first", predictor.model_path)
        return 1

    results = run(config, predictor, args.split, args.target_pixels, args.quality)
    out_dir = ensure_dir(config.path("results_dir") / "confound")
    version = predictor.model_version
    chart = plot(results, config.path("results_dir") / "graphs" / f"confound_{version}.png")
    write_json(out_dir / f"confound_{version}.json", results)
    markdown = to_markdown(results, chart)
    write_text(out_dir / f"confound_{version}.md", markdown)

    if not args.quiet:
        print(markdown)
    LOGGER.info("written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
