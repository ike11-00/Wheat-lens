"""Measure how much of a model's accuracy depends on the background.

This is the metric the v2 experiment is judged on, and the one v1 fails.

Method
------
Take the test split - images the model never trained on - and classify each one
twice:

1. **unchanged**, exactly as the normal evaluation does;
2. **background replaced**, with the leaf segmented out and composited onto a
   different background at a random scale and position. The leaf, and therefore
   the disease, is identical in both.

Any accuracy lost between the two runs is accuracy that depended on the
background rather than on the leaf.

The backgrounds used here come from the **test** split of the field-photography
class, never the training split. That matters: the v2 training augmentation
harvests from the training split, so evaluating with training backgrounds would
be marking its own homework.

Reference result for v1 (studio classes, field backgrounds transplanted):
accuracy fell from 100% to roughly 16%, with the model 80-87% confident while
wrong. See docs/v2_experiment.md.

Run::

    python -m src.testing.background_robustness
    python -m src.testing.background_robustness --version v2 --copies 3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..data.augment_backgrounds import (AUGMENTED_PREFIX, composite, procedural_background,
                                        resolve_plan)
from ..model.datasets import list_split_files
from ..model.predict import Predictor
from ..utils.config import Config, load_config
from ..utils.helpers import (configure_matplotlib, ensure_dir, format_percent,
                             get_logger, markdown_table, set_global_seed,
                             utc_timestamp, write_json, write_text)
from ..utils.image_io import InvalidImageError, iter_image_files, open_image

LOGGER = get_logger(__name__)


def evaluation_backgrounds(config: Config, source_class: str,
                           max_patches: int = 200) -> List[Image.Image]:
    """Background patches from the TEST split of ``source_class``.

    Held out from training augmentation on purpose, so this evaluation cannot
    be gamed by the augmentation that v2 trains on.
    """
    spec = config.resolve_class(source_class)
    if spec is None:
        return []
    root = config.path("test_dir") / spec.directory
    patches: List[Image.Image] = []
    for path in iter_image_files(root, config.supported_extensions):
        if len(patches) >= max_patches:
            break
        try:
            image = open_image(path)
        except InvalidImageError:
            continue
        width, height = image.size
        for box in ((0, 0, width // 3, height // 3),
                    (2 * width // 3, 0, width, height // 3),
                    (0, 2 * height // 3, width // 3, height),
                    (2 * width // 3, 2 * height // 3, width, height)):
            patches.append(image.crop(box))
    return patches


def run(config: Config, predictor: Predictor, source_class: str,
        copies: int, seed: int, use_procedural: bool) -> Dict[str, Any]:
    plan = resolve_plan(config)
    set_global_seed(seed)
    rng = random.Random(seed)
    names = predictor.class_names

    paths, labels = list_split_files(config, config.path("test_dir"))
    # Exclude the background-source class itself: replacing a field photograph's
    # background with another field background tests nothing.
    source_index = names.index(source_class) if source_class in names else -1
    subset = [(p, y) for p, y in zip(paths, labels) if y != source_index]
    if not subset:
        raise RuntimeError("no test images outside the background-source class")

    backgrounds = evaluation_backgrounds(config, source_class)
    harvested = len(backgrounds)
    if use_procedural or not backgrounds:
        backgrounds += [procedural_background(rng, 384) for _ in range(64)]
    if not backgrounds:
        raise RuntimeError("no backgrounds available for the evaluation")

    LOGGER.info("%d test images, %d background patches (%d from the test split of '%s')",
                len(subset), len(backgrounds), harvested, source_class)

    originals = [p for p, _ in subset]
    truth = np.array([y for _, y in subset])

    baseline = predictor.predict_many(originals)
    base_pred = np.array([names.index(r["predicted_class"]) for r in baseline])
    base_conf = np.array([r["confidence"] for r in baseline])

    swapped_pred, swapped_conf, swapped_truth, skipped = [], [], [], 0
    for (path, label) in subset:
        try:
            leaf = open_image(path)
        except InvalidImageError:
            skipped += 1
            continue
        for _ in range(copies):
            blended = composite(leaf, backgrounds[rng.randrange(len(backgrounds))], rng, plan)
            if blended is None:
                skipped += 1
                continue
            result = predictor.predict(blended)
            swapped_pred.append(names.index(result["predicted_class"]))
            swapped_conf.append(result["confidence"])
            swapped_truth.append(label)

    swapped_pred = np.array(swapped_pred)
    swapped_conf = np.array(swapped_conf)
    swapped_truth = np.array(swapped_truth)

    def per_class(pred, true, conf):
        rows = []
        for index, name in enumerate(names):
            if index == source_index:
                continue
            mask = true == index
            if not mask.any():
                continue
            rows.append({
                "class": name,
                "n": int(mask.sum()),
                "accuracy": float((pred[mask] == index).mean()),
                "mean_confidence": float(conf[mask].mean()),
                "predicted_as_source_class": float((pred[mask] == source_index).mean()),
            })
        return rows

    base_acc = float((base_pred == truth).mean())
    swap_acc = float((swapped_pred == swapped_truth).mean()) if len(swapped_pred) else 0.0
    wrong = swapped_pred != swapped_truth

    return {
        "generated_utc": utc_timestamp(),
        "model_version": predictor.model_version,
        "background_source_class": source_class,
        "background_patches": len(backgrounds),
        "patches_from_test_split": harvested,
        "copies_per_image": copies,
        "seed": seed,
        "images_evaluated": len(subset),
        "composites_evaluated": int(len(swapped_pred)),
        "skipped": skipped,
        "accuracy_original": base_acc,
        "accuracy_background_replaced": swap_acc,
        "accuracy_drop": base_acc - swap_acc,
        "mean_confidence_original": float(base_conf.mean()),
        "mean_confidence_background_replaced": float(swapped_conf.mean()) if len(swapped_conf) else None,
        "mean_confidence_when_wrong": float(swapped_conf[wrong].mean()) if wrong.any() else None,
        "share_predicted_as_source_class": (
            float((swapped_pred == source_index).mean()) if len(swapped_pred) else None),
        "per_class_original": per_class(base_pred, truth, base_conf),
        "per_class_background_replaced": per_class(swapped_pred, swapped_truth, swapped_conf),
    }


def plot(results: Dict[str, Any], output_path: Path) -> Optional[Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    rows_a = {r["class"]: r for r in results["per_class_original"]}
    rows_b = {r["class"]: r for r in results["per_class_background_replaced"]}
    names = [n for n in rows_a if n in rows_b]
    if not names:
        return None
    positions = np.arange(len(names))
    width = 0.38

    figure, axis = plt.subplots(figsize=(1.9 * len(names) + 3.0, 4.4))
    axis.bar(positions - width / 2, [rows_a[n]["accuracy"] for n in names], width,
             label="original background", color="#3f7d3f")
    axis.bar(positions + width / 2, [rows_b[n]["accuracy"] for n in names], width,
             label=f"background replaced (from {results['background_source_class']})",
             color="#b5453b")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Accuracy on the test split")
    axis.set_title(f"Background robustness - model {results['model_version']}\\n"
                   "accuracy lost here was accuracy that depended on the background")
    axis.legend()
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def to_markdown(results: Dict[str, Any], chart: Optional[Path]) -> str:
    drop = results["accuracy_drop"]
    lines = [
        f"# Background robustness - model {results['model_version']}",
        "",
        f"* Generated (UTC): {results['generated_utc']}",
        f"* Test images used: {results['images_evaluated']} "
        f"(excluding the background-source class, {results['background_source_class']})",
        f"* Composites evaluated: {results['composites_evaluated']} "
        f"({results['copies_per_image']} per image, seed {results['seed']})",
        f"* Background patches: {results['background_patches']} "
        f"({results['patches_from_test_split']} cropped from the **test** split of "
        f"{results['background_source_class']}, never the training split)",
        "",
        "## Headline",
        "",
        markdown_table(
            ["Condition", "Accuracy", "Mean confidence"],
            [["Original background",
              format_percent(results["accuracy_original"], 1),
              format_percent(results["mean_confidence_original"], 1)],
             ["**Background replaced**",
              f"**{format_percent(results['accuracy_background_replaced'], 1)}**",
              format_percent(results["mean_confidence_background_replaced"], 1)],
             ["**Accuracy lost to the background**", f"**{drop * 100:.1f} pp**", "-"]]),
        "",
        f"* Confidence while wrong: "
        f"{format_percent(results['mean_confidence_when_wrong'], 1)}",
        f"* Share of composites called '{results['background_source_class']}': "
        f"{format_percent(results['share_predicted_as_source_class'], 1)}",
        "",
        "## Per class",
        "",
        markdown_table(
            ["Class", "n", "Original", "Background replaced", "Change",
             f"Called {results['background_source_class']}"],
            [[a["class"], a["n"], format_percent(a["accuracy"], 1),
              format_percent(b["accuracy"], 1),
              f"{(b['accuracy'] - a['accuracy']) * 100:+.1f} pp",
              format_percent(b["predicted_as_source_class"], 1)]
             for a, b in zip(results["per_class_original"],
                             results["per_class_background_replaced"])]),
        "",
        "## Reading this",
        "",
        "The leaf is identical in both runs; only the background differs. Accuracy "
        "lost between them is accuracy that depended on the background rather than "
        "on the symptoms.",
        "",
        "* A **small drop** means the model is mostly reading the leaf. That is the "
        "outcome the v2 experiment is trying to produce.",
        "* A **large drop**, especially with composites being called the "
        "background-source class, means the model is reading the background. That is "
        "the v1 failure.",
        "",
        "Some drop is expected for any model: compositing introduces segmentation "
        "edges and a scale change, which are themselves distribution shift. Compare "
        "versions against each other rather than against a perfect score.",
        "",
    ]
    if chart:
        lines += [f"![Background robustness]({chart.as_posix()})", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure how much accuracy depends on the background")
    parser.add_argument("--config", default=None)
    parser.add_argument("--version", default=None, help="model version to evaluate")
    parser.add_argument("--source-class", default="Yellow Rust",
                        help="class whose backgrounds are transplanted")
    parser.add_argument("--copies", type=int, default=2,
                        help="composites generated per test image")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--procedural", action="store_true",
                        help="also mix in synthetic backgrounds")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    overrides = {"model": {"version": args.version}} if args.version else None
    config = load_config(args.config, overrides=overrides)
    predictor = Predictor(config)
    if not predictor.is_available():
        LOGGER.error("no trained model at %s", predictor.model_path)
        return 1

    results = run(config, predictor, args.source_class, args.copies, args.seed,
                  args.procedural)
    version = predictor.model_version
    out_dir = ensure_dir(config.path("results_dir") / "robustness")
    chart = plot(results, config.path("results_dir") / "graphs" / f"background_robustness_{version}.png")
    write_json(out_dir / f"background_robustness_{version}.json", results)
    markdown = to_markdown(results, chart)
    write_text(out_dir / f"background_robustness_{version}.md", markdown)

    if not args.quiet:
        print(markdown)
    LOGGER.info("written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
