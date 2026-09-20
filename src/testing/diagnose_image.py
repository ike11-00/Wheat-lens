"""Inspect what the model does with an individual image, and why.

Built to diagnose the gap between benchmark accuracy and behaviour on
photographs from elsewhere. For each image it reports the prediction, the
confidence, the full probability distribution, the uncertainty signals, the
image's own properties, and every preprocessing step applied on the way in -
so a surprising answer can be traced to the input rather than guessed at.

It also compares the image's statistics against the training classes, which is
usually where the explanation lies: an image that resembles no training class
statistically will still receive a confident answer, because a softmax always
produces one.

Run::

    python -m src.testing.diagnose_image photo.jpg
    python -m src.testing.diagnose_image ~/Downloads/*.jpg --json report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..model.predict import Predictor
from ..utils.config import Config, load_config
from ..utils.helpers import format_percent, get_logger, markdown_table, utc_timestamp, write_json
from ..utils.image_io import InvalidImageError, iter_image_files, open_image

LOGGER = get_logger(__name__)


def image_statistics(image: Image.Image) -> Dict[str, float]:
    """Global statistics comparable across images of any size."""
    small = image.convert("RGB").resize((64, 64))
    array = np.asarray(small, dtype="float32") / 255.0
    hsv = np.asarray(small.convert("HSV"), dtype="float32") / 255.0
    border = np.concatenate([array[0, :].ravel(), array[-1, :].ravel(),
                             array[:, 0].ravel(), array[:, -1].ravel()])
    return {
        "mean_red": float(array[..., 0].mean()),
        "mean_green": float(array[..., 1].mean()),
        "mean_blue": float(array[..., 2].mean()),
        "brightness": float(hsv[..., 2].mean()),
        "saturation": float(hsv[..., 1].mean()),
        "border_brightness": float(border.mean()),
    }


def training_profiles(config: Config, per_class: int = 150) -> Dict[str, Dict[str, float]]:
    """Mean statistics of each training class, for comparison."""
    profiles: Dict[str, Dict[str, float]] = {}
    for spec in config.classes:
        files = list(iter_image_files(config.path("train_dir") / spec.directory,
                                      config.supported_extensions))[:per_class]
        if not files:
            continue
        rows = []
        for path in files:
            try:
                rows.append(image_statistics(open_image(path)))
            except InvalidImageError:
                continue
        if rows:
            profiles[spec.name] = {
                key: float(np.mean([r[key] for r in rows])) for key in rows[0]
            }
    return profiles


def nearest_profile(stats: Dict[str, float],
                    profiles: Dict[str, Dict[str, float]]) -> List[tuple]:
    """Rank training classes by statistical distance from this image."""
    keys = ["brightness", "saturation", "border_brightness",
            "mean_red", "mean_green", "mean_blue"]
    scored = []
    for name, profile in profiles.items():
        distance = float(np.sqrt(sum((stats[k] - profile[k]) ** 2 for k in keys)))
        scored.append((name, distance))
    return sorted(scored, key=lambda item: item[1])


def diagnose(predictor: Predictor, path: Path, config: Config,
             profiles: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    original = open_image(path)
    width, height = original.size
    size = predictor.image_size
    stats = image_statistics(original)
    result = predictor.predict(path)

    return {
        "file": str(path),
        "filename": path.name,
        "image": {
            "width": width,
            "height": height,
            "megapixels": round(width * height / 1e6, 3),
            "aspect_ratio": round(width / height, 3) if height else None,
            "mode": original.mode,
            "size_kb": round(path.stat().st_size / 1024, 1),
        },
        "preprocessing": [
            f"decoded to RGB (was {original.mode})",
            f"resized {width}x{height} -> {size}x{size} (bilinear, aspect ratio NOT preserved"
            f"{'; distorted by ' + format(abs(width / height - 1) * 100, '.0f') + '%' if height and abs(width / height - 1) > 0.15 else ''})",
            "kept as float32 in the 0-255 range",
            "inside the model: mobilenet_v2.preprocess_input scales to [-1, 1]",
            "inside the model: augmentation layers inactive (training=False)",
        ],
        "prediction": {
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "low_confidence": result["low_confidence"],
            "uncertain": result["uncertain"],
            "normalised_entropy": result["uncertainty"]["normalised_entropy"],
            "margin": result["uncertainty"]["margin"],
        },
        "statistics": stats,
        "closest_training_classes": nearest_profile(stats, profiles),
    }


def render(record: Dict[str, Any], config: Config) -> str:
    image = record["image"]
    prediction = record["prediction"]
    lines = [
        f"## `{record['filename']}`",
        "",
        f"* Dimensions: **{image['width']} x {image['height']}** "
        f"({image['megapixels']} MP, aspect {image['aspect_ratio']}, {image['size_kb']} KB)",
        "",
        "**Preprocessing applied**",
        "",
    ]
    lines += [f"{i}. {step}" for i, step in enumerate(record["preprocessing"], 1)]
    lines += [
        "",
        f"**Prediction: {prediction['predicted_class']} "
        f"({format_percent(prediction['confidence'], 1)})**",
        "",
        markdown_table(
            ["Class", "Probability"],
            [[name, format_percent(value, 1)]
             for name, value in sorted(prediction["probabilities"].items(),
                                       key=lambda kv: kv[1], reverse=True)]),
        "",
        f"* Below the confidence threshold: {'yes' if prediction['low_confidence'] else 'no'}",
        f"* Flagged uncertain: {'yes' if prediction['uncertain'] else 'no'}",
        f"* Normalised entropy: {prediction['normalised_entropy']:.3f} "
        f"(0 = certain, 1 = uniform) · top-2 margin: {prediction['margin']:.3f}",
        "",
        "**How this image compares with the training classes**",
        "",
        markdown_table(
            ["Measure"] + [f"this image"] ,
            [[k.replace("_", " "), f"{v:.3f}"] for k, v in record["statistics"].items()]),
        "",
        "Closest training classes by global image statistics "
        "(smaller distance = more similar *as a photograph*, which is not the same "
        "as sharing a disease):",
        "",
        markdown_table(
            ["Rank", "Training class", "Statistical distance"],
            [[i, name, f"{distance:.3f}"]
             for i, (name, distance) in enumerate(record["closest_training_classes"], 1)]),
        "",
    ]
    if record["closest_training_classes"]:
        nearest = record["closest_training_classes"][0][0]
        if nearest == prediction["predicted_class"]:
            lines += [
                f"> The prediction matches the statistically nearest training class "
                f"(**{nearest}**). That is a warning sign: it is consistent with the "
                f"model responding to how the photograph was taken rather than to the "
                f"symptoms on the leaf.",
                "",
            ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose what the model does with specific images")
    parser.add_argument("images", nargs="+", help="image files to diagnose")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json", default=None, help="also write the full records to JSON")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    predictor = Predictor(config)
    if not predictor.is_available():
        LOGGER.error("no trained model at %s", predictor.model_path)
        return 1

    LOGGER.info("profiling the training classes for comparison ...")
    profiles = training_profiles(config)

    records = []
    print(f"# Image diagnosis\n\n* Model: `{predictor.model_path}` "
          f"(version {predictor.model_version})\n* Generated: {utc_timestamp()}\n")
    for raw in args.images:
        path = Path(raw)
        if not path.exists():
            LOGGER.error("not found: %s", path)
            continue
        try:
            record = diagnose(predictor, path, config, profiles)
        except InvalidImageError as exc:
            LOGGER.error("%s: %s", path.name, exc)
            continue
        records.append(record)
        print(render(record, config))

    if args.json and records:
        write_json(Path(args.json), records)
        LOGGER.info("wrote %s", args.json)
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
