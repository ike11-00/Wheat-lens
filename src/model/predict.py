"""Prediction engine - the single place where a trained model is applied.

The same :class:`Predictor` backs the CLI, the Flask app, the evaluator, the
error-analysis tool and the realistic-condition tester, so all of them apply
identical preprocessing and identical thresholds.

A prediction is returned as::

    {
        "predicted_class": "<one of the configured classes>",
        "confidence": 0.0-1.0,
        "probabilities": {"<class>": 0.0-1.0, ...},
        "low_confidence": bool,
        "uncertain": bool,
        "uncertainty": {...},
        "confidence_threshold": float,
        "model_version": "...",
        "disclaimer": "..."
    }

The numbers always come from the network. Nothing in this file contains rules
about colours, spots or filenames.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..utils.config import Config, load_config
from ..utils.helpers import get_logger, read_json, write_json
from ..utils.image_io import ImageSource, InvalidImageError, load_batch, load_image_array

LOGGER = get_logger(__name__)

DISCLAIMER = (
    "Experimental prototype. The confidence value is the model's estimated probability "
    "for its own prediction - it is not a guarantee that the prediction is correct."
)


class ModelNotAvailableError(FileNotFoundError):
    """Raised when no trained model file exists yet."""


@dataclass
class PredictorInfo:
    """Everything the UI needs to describe the loaded model."""

    model_path: Path
    model_version: str
    class_names: List[str]
    image_size: int
    confidence_threshold: float
    architecture: str
    trained_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "model_version": self.model_version,
            "class_names": self.class_names,
            "image_size": self.image_size,
            "confidence_threshold": self.confidence_threshold,
            "architecture": self.architecture,
            "trained_utc": self.trained_utc,
        }


class Predictor:
    """Loads a trained Keras model once and classifies images.

    Parameters
    ----------
    config:
        Loaded configuration. Defaults to ``config/config.yaml``.
    model_path:
        Explicit ``.keras`` file. Defaults to ``models/<version>/model.keras``.
    """

    def __init__(self, config: Optional[Config] = None,
                 model_path: Optional[Path] = None) -> None:
        self.config = config or load_config()
        self._model = None
        self._model_path = Path(model_path) if model_path else self.config.model_file
        self._labels_path = self._model_path.parent / "labels.json"

        self.class_names: List[str] = list(self.config.class_names)
        self.image_size: int = self.config.image_size
        self.confidence_threshold: float = self.config.confidence_threshold
        self.model_version: str = self.config.model_version
        self.architecture: str = str(self.config.get("model", "architecture",
                                                     default="MobileNetV2"))
        self.trained_utc: Optional[str] = None

        # labels.json is written next to the model at training time and is the
        # authority on class order: a model trained with a different class list
        # must not be interpreted with today's config.
        if self._labels_path.exists():
            try:
                labels = read_json(self._labels_path)
                self.class_names = list(labels.get("class_names", self.class_names))
                self.image_size = int(labels.get("image_size", self.image_size))
                self.confidence_threshold = float(
                    labels.get("confidence_threshold", self.confidence_threshold))
                self.model_version = str(labels.get("model_version", self.model_version))
                self.architecture = str(labels.get("architecture", self.architecture))
                self.trained_utc = labels.get("trained_utc")
            except (ValueError, OSError) as exc:
                LOGGER.warning("could not read %s (%s); falling back to config.yaml",
                               self._labels_path, exc)

    # -- availability -----------------------------------------------------
    @property
    def model_path(self) -> Path:
        return self._model_path

    def is_available(self) -> bool:
        return self._model_path.exists()

    def info(self) -> PredictorInfo:
        return PredictorInfo(
            model_path=self._model_path,
            model_version=self.model_version,
            class_names=self.class_names,
            image_size=self.image_size,
            confidence_threshold=self.confidence_threshold,
            architecture=self.architecture,
            trained_utc=self.trained_utc,
        )

    def load(self):
        """Load the Keras model (lazily, once)."""
        if self._model is not None:
            return self._model
        if not self._model_path.exists():
            raise ModelNotAvailableError(
                f"no trained model at {self._model_path}. Train one with "
                f"'python -m src.model.train' (which first needs a dataset - see "
                f"docs/dataset.md)."
            )
        import tensorflow as tf  # imported lazily so the CLI starts fast

        # Importing build_model runs the @register_keras_serializable decorator
        # on BackbonePreprocessing, without which the saved graph cannot be
        # deserialised. custom_objects is belt-and-braces for the same layer.
        from .build_model import BackbonePreprocessing

        LOGGER.info("loading model from %s", self._model_path)
        try:
            self._model = tf.keras.models.load_model(
                self._model_path, compile=False,
                custom_objects={"leaf_lens>BackbonePreprocessing": BackbonePreprocessing,
                                "BackbonePreprocessing": BackbonePreprocessing},
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean message
            raise ModelNotAvailableError(
                f"the model at {self._model_path} could not be loaded: {exc}"
            ) from exc
        outputs = int(self._model.output_shape[-1])
        if outputs != len(self.class_names):
            raise ModelNotAvailableError(
                f"model at {self._model_path} has {outputs} outputs but "
                f"{len(self.class_names)} class names are configured "
                f"({', '.join(self.class_names)}). Retrain, or restore the matching "
                f"labels.json."
            )
        return self._model

    # -- prediction -------------------------------------------------------
    def predict_proba(self, sources: Sequence[ImageSource]) -> np.ndarray:
        """Raw probability matrix of shape ``(n, num_classes)``."""
        model = self.load()
        if not sources:
            return np.zeros((0, len(self.class_names)), dtype="float32")
        batch = load_batch(list(sources), self.image_size)
        # training=False keeps the in-graph augmentation layers inactive.
        probabilities = model(batch, training=False).numpy()
        return np.asarray(probabilities, dtype="float64")

    def predict(self, source: ImageSource) -> Dict[str, Any]:
        """Classify one image. Raises :class:`InvalidImageError` for bad input."""
        return self.predict_many([source])[0]

    def predict_many(self, sources: Sequence[ImageSource],
                     batch_size: int = 32) -> List[Dict[str, Any]]:
        """Classify several images, batching the forward passes."""
        results: List[Dict[str, Any]] = []
        sources = list(sources)
        for start in range(0, len(sources), batch_size):
            chunk = sources[start:start + batch_size]
            probabilities = self.predict_proba(chunk)
            for row in probabilities:
                results.append(self._format(row))
        return results

    def _format(self, probabilities: np.ndarray) -> Dict[str, Any]:
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        distribution = {
            name: float(probabilities[i]) for i, name in enumerate(self.class_names)
        }
        uncertainty = self._uncertainty(probabilities)
        return {
            "predicted_class": self.class_names[index],
            "predicted_index": index,
            "confidence": confidence,
            "probabilities": distribution,
            "low_confidence": confidence < self.confidence_threshold,
            "confidence_threshold": self.confidence_threshold,
            "uncertain": uncertainty["uncertain"],
            "uncertainty": uncertainty,
            "model_version": self.model_version,
            "disclaimer": DISCLAIMER,
        }

    def _uncertainty(self, probabilities: np.ndarray) -> Dict[str, Any]:
        """Out-of-distribution / uncertainty heuristic.

        Two cheap signals are computed from the softmax output:

        ``normalised_entropy``
            Shannon entropy divided by ``log(num_classes)``. 0 means the model
            put everything on one class, 1 means a uniform guess.
        ``margin``
            Top-1 probability minus top-2 probability. A small margin means the
            model could not separate its two best candidates.

        An image is flagged ``uncertain`` when entropy is high **or** the margin
        is small **or** confidence is below the configured threshold.

        Limitation (important): this is a *softmax confidence* heuristic, not a
        true novelty detector. A neural network can be confidently wrong on an
        image unlike anything it was trained on - a different crop, a sixth
        wheat disease, or a photograph of something that is not a leaf at all.
        A low flag rate is therefore NOT evidence that an input is in
        distribution. See docs/model.md and docs/limitations.md.
        """
        settings = self.config.get("inference", "ood", default={}) or {}
        enabled = bool(settings.get("enabled", True))
        entropy_limit = float(settings.get("entropy_ratio_threshold", 0.80))
        margin_limit = float(settings.get("margin_threshold", 0.10))

        clipped = np.clip(probabilities, 1e-12, 1.0)
        entropy = float(-(clipped * np.log(clipped)).sum())
        max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
        normalised_entropy = float(entropy / max_entropy) if max_entropy else 0.0

        ordered = np.sort(probabilities)[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0
        confidence = float(ordered[0])

        reasons: List[str] = []
        if enabled and normalised_entropy > entropy_limit:
            reasons.append(
                f"the probability distribution is close to uniform "
                f"(normalised entropy {normalised_entropy:.2f} > {entropy_limit})")
        if enabled and margin < margin_limit:
            reasons.append(
                f"the top two classes are nearly tied (margin {margin:.2f} < {margin_limit})")
        if confidence < self.confidence_threshold:
            reasons.append(
                f"confidence {confidence:.2f} is below the threshold "
                f"{self.confidence_threshold:.2f}")

        return {
            "enabled": enabled,
            "normalised_entropy": normalised_entropy,
            "margin": margin,
            "entropy_ratio_threshold": entropy_limit,
            "margin_threshold": margin_limit,
            "uncertain": bool(reasons),
            "reasons": reasons,
            "method": "max-softmax confidence + entropy + top-2 margin",
            "limitation": (
                "Softmax-based heuristic only. It cannot reliably detect images from outside "
                "the five trained classes; the network can be confidently wrong on unfamiliar "
                "input."
            ),
        }


def format_human_readable(result: Dict[str, Any]) -> str:
    """Render one prediction the way the brief specifies."""
    lines = [
        "Prediction:",
        f"  {result['predicted_class']}",
        "",
        "Confidence:",
        f"  {result['confidence'] * 100:.1f}%",
        "",
        "Class probabilities:",
        "",
    ]
    width = max(len(name) for name in result["probabilities"]) + 2
    for name, probability in sorted(result["probabilities"].items(),
                                    key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {name + ':':<{width}} {probability * 100:>6.1f}%")
    lines.append("")
    if result.get("low_confidence"):
        lines.append("Low-confidence prediction. Consider taking a clearer photograph "
                     "or using another image.")
    if result.get("uncertain") and result.get("uncertainty", {}).get("reasons"):
        lines.append("Uncertainty flags:")
        for reason in result["uncertainty"]["reasons"]:
            lines.append(f"  - {reason}")
    lines += ["", result["disclaimer"]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Classify wheat leaf images with Leaf Lens")
    parser.add_argument("images", nargs="*", help="image files to classify")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="path to a .keras model file")
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    parser.add_argument("--output", default=None, help="also write the results to a JSON file")
    parser.add_argument("--info", action="store_true", help="print model information and exit")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    predictor = Predictor(config, model_path=Path(args.model) if args.model else None)

    if args.info:
        info = predictor.info().to_dict()
        info["available"] = predictor.is_available()
        print(json.dumps(info, indent=2))
        return 0 if predictor.is_available() else 1

    if not args.images:
        parser.error("provide at least one image path, or use --info")

    if not predictor.is_available():
        LOGGER.error("no trained model found at %s", predictor.model_path)
        LOGGER.error("Train one with: python -m src.model.train")
        LOGGER.error("(training needs a dataset in data/raw - see docs/dataset.md)")
        return 1

    payload: List[Dict[str, Any]] = []
    exit_code = 0
    for image_path in args.images:
        try:
            result = predictor.predict(image_path)
        except InvalidImageError as exc:
            LOGGER.error("%s: %s", image_path, exc)
            exit_code = 1
            continue
        except ModelNotAvailableError as exc:
            LOGGER.error("%s", exc)
            return 1
        result["image"] = str(image_path)
        payload.append(result)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n=== {image_path} ===")
            print(format_human_readable(result))

    if args.output and payload:
        write_json(Path(args.output), payload)
        LOGGER.info("wrote %s", args.output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
