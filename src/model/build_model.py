"""Transfer-learning model construction for Leaf Lens.

Design
------
``MobileNetV2`` pretrained on ImageNet is used as a frozen feature extractor
and a small classification head is trained on top. MobileNetV2 was chosen
because it is accurate enough for fine-grained leaf texture while being small
(~3.5M parameters) and fast enough to run inference inside a Flask request on
CPU, which is what the web application needs.

Two properties are deliberate:

* **Preprocessing lives inside the model.** The saved ``.keras`` file starts
  with the architecture's own ``preprocess_input`` layer, so every caller -
  the web app, the CLI, the evaluator - feeds raw 0-255 RGB pixels and cannot
  apply the wrong normalisation.
* **Augmentation lives inside the model but is inference-safe.** Keras
  augmentation layers are no-ops when ``training=False``, so the same object
  can be trained and then used for prediction.

Everything that varies (architecture name, dropout, number of classes,
unfreeze depth) comes from ``config.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..utils.config import Config
from ..utils.helpers import get_logger

LOGGER = get_logger(__name__)

# name -> (keras application constructor, matching preprocess_input)
SUPPORTED_ARCHITECTURES: Dict[str, Tuple[Any, Any]] = {
    "MobileNetV2": (
        keras.applications.MobileNetV2,
        keras.applications.mobilenet_v2.preprocess_input,
    ),
    "MobileNetV3Small": (
        keras.applications.MobileNetV3Small,
        keras.applications.mobilenet_v3.preprocess_input,
    ),
    "EfficientNetB0": (
        keras.applications.EfficientNetB0,
        keras.applications.efficientnet.preprocess_input,
    ),
    "ResNet50": (
        keras.applications.ResNet50,
        keras.applications.resnet.preprocess_input,
    ),
}


# Sentinel distinguishing "caller said nothing" from "caller explicitly said
# None" (which is a legitimate request for a randomly initialised backbone).
_FROM_CONFIG = object()


class ModelBuildError(RuntimeError):
    """Raised when the requested architecture or configuration is unusable."""


def normalise_weights(value: Any) -> Optional[str]:
    """Map the several spellings of "no pretrained weights" onto ``None``."""
    if value in (None, "", "none", "None", "null", False):
        return None
    return str(value)


def keras_cache_dir() -> Path:
    """Directory Keras downloads pretrained weights into."""
    return Path(os.environ.get("KERAS_HOME", Path.home() / ".keras")) / "models"


# Cache per (architecture, size) so the probe runs at most once per process.
_WEIGHTS_PROBE: Dict[Tuple[str, int], bool] = {}


def imagenet_weights_available(config: Config, force: bool = False) -> bool:
    """Whether ImageNet weights can be obtained for the configured backbone.

    Returns ``True`` when the weights are already in the Keras cache **or** can
    be downloaded now. Returns ``False`` when they are absent and unreachable,
    which is the state an offline machine is in.

    This is a genuine probe rather than a filename guess: Keras stores weights
    under architecture-specific names that also vary with input size, so
    attempting the load is the only reliable check. The result is memoised, so
    the cost is paid once per process.
    """
    architecture = str(config.get("model", "architecture", default="MobileNetV2"))
    key = (architecture, config.image_size)
    if not force and key in _WEIGHTS_PROBE:
        return _WEIGHTS_PROBE[key]

    entry = SUPPORTED_ARCHITECTURES.get(architecture)
    if entry is None:
        _WEIGHTS_PROBE[key] = False
        return False

    constructor = entry[0]
    try:
        constructor(include_top=False, weights="imagenet",
                    input_shape=(config.image_size, config.image_size, 3))
        available = True
    except Exception:  # noqa: BLE001 - any failure means "not available"
        available = False
    _WEIGHTS_PROBE[key] = available
    return available


def prefetch_weights(config: Config) -> Path:
    """Download the configured backbone's ImageNet weights into the cache.

    Run this once on a machine that has network access so that later runs -
    including the test suite and any offline training - need none. Raises
    :class:`ModelBuildError` with an actionable message if the download fails.
    """
    architecture = str(config.get("model", "architecture", default="MobileNetV2"))
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ModelBuildError(f"unsupported architecture {architecture!r}")
    constructor = SUPPORTED_ARCHITECTURES[architecture][0]
    try:
        constructor(include_top=False, weights="imagenet",
                    input_shape=(config.image_size, config.image_size, 3))
    except Exception as exc:  # noqa: BLE001
        raise ModelBuildError(
            f"could not download ImageNet weights for {architecture} at "
            f"{config.image_size}x{config.image_size}: {exc}"
        ) from exc
    LOGGER.info("ImageNet weights for %s (%dpx) are cached in %s",
                architecture, config.image_size, keras_cache_dir())
    return keras_cache_dir()


@keras.utils.register_keras_serializable(package="leaf_lens")
class BackbonePreprocessing(layers.Layer):
    """Applies the backbone's own ``preprocess_input`` inside the model graph.

    A plain ``Lambda`` layer cannot be deserialised from a saved ``.keras``
    file (Keras cannot locate the wrapped function), which would break every
    caller that loads the model. This layer stores the *architecture name*
    instead and looks the function up on load, so the saved model round-trips.
    """

    def __init__(self, architecture: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ModelBuildError(f"unsupported architecture {architecture!r}")
        self.architecture = architecture
        self._preprocess = SUPPORTED_ARCHITECTURES[architecture][1]

    def call(self, inputs):
        return self._preprocess(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["architecture"] = self.architecture
        return config


def build_augmentation(config: Config) -> Optional[keras.Sequential]:
    """Training-time augmentation pipeline built from ``training.augmentation``.

    Only transformations that leave the disease appearance intact are offered:
    flips, small rotations, small zoom/translation and mild brightness/contrast
    jitter. Colour-shifting augmentations are intentionally **not** included,
    because hue is a genuine discriminative signal between yellow rust and
    brown rust.
    """
    settings = config.get("training", "augmentation", default={}) or {}
    pipeline: List[layers.Layer] = []

    flip_modes = []
    if settings.get("horizontal_flip", True):
        flip_modes.append("horizontal")
    if settings.get("vertical_flip", False):
        flip_modes.append("vertical")
    if flip_modes:
        pipeline.append(layers.RandomFlip("_and_".join(flip_modes), name="aug_flip"))

    rotation = float(settings.get("rotation", 0.0) or 0.0)
    if rotation > 0:
        pipeline.append(layers.RandomRotation(rotation, fill_mode="reflect", name="aug_rotation"))

    zoom = float(settings.get("zoom", 0.0) or 0.0)
    if zoom > 0:
        pipeline.append(layers.RandomZoom(zoom, zoom, fill_mode="reflect", name="aug_zoom"))

    translation = float(settings.get("translation", 0.0) or 0.0)
    if translation > 0:
        pipeline.append(layers.RandomTranslation(translation, translation,
                                                 fill_mode="reflect", name="aug_translation"))

    brightness = float(settings.get("brightness", 0.0) or 0.0)
    if brightness > 0:
        # value_range matches the raw 0-255 input the model accepts.
        pipeline.append(layers.RandomBrightness(brightness, value_range=(0.0, 255.0),
                                                name="aug_brightness"))

    contrast = float(settings.get("contrast", 0.0) or 0.0)
    if contrast > 0:
        pipeline.append(layers.RandomContrast(contrast, name="aug_contrast"))

    if not pipeline:
        return None
    return keras.Sequential(pipeline, name="augmentation")


def build_model(config: Config, num_classes: Optional[int] = None,
                include_augmentation: bool = True,
                weights: Union[str, None, object] = _FROM_CONFIG) -> keras.Model:
    """Assemble the classifier.

    Parameters
    ----------
    config:
        Supplies the architecture, input size, class count and head settings.
    num_classes:
        Overrides ``config.num_classes``.
    include_augmentation:
        Whether to embed the training-time augmentation pipeline.
    weights:
        Backbone initialisation, injected rather than always read from
        configuration. Omit it and the value comes from ``model.weights``
        (``imagenet`` in production). Pass ``None`` to build the same
        architecture with a randomly initialised backbone, which needs no
        download - this is what the test suite uses, explicitly, so that
        architectural behaviour can be verified without network access.

    Passing ``None`` is never done implicitly: production configuration asks
    for ``imagenet``, and if those weights cannot be obtained this function
    raises rather than quietly degrading to an untrained backbone.

    Returns an **uncompiled** model; use :func:`compile_model`.
    """
    architecture = str(config.get("model", "architecture", default="MobileNetV2"))
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ModelBuildError(
            f"unsupported architecture {architecture!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_ARCHITECTURES))}"
        )
    constructor, _ = SUPPORTED_ARCHITECTURES[architecture]

    size = config.image_size
    classes = int(num_classes if num_classes is not None else config.num_classes)
    if classes < 2:
        raise ModelBuildError(f"need at least 2 classes, configuration defines {classes}")

    if weights is _FROM_CONFIG:
        weights = config.get("model", "weights", default="imagenet")
    weights = normalise_weights(weights)

    inputs = keras.Input(shape=(size, size, 3), name="image", dtype="float32")

    x = inputs
    augmentation = build_augmentation(config) if include_augmentation else None
    if augmentation is not None:
        # Keras augmentation layers are inactive when training=False, so the
        # exported model is safe to call directly for inference.
        x = augmentation(x)

    # Preprocessing is part of the graph: callers always pass raw 0-255 pixels.
    x = BackbonePreprocessing(architecture, name="preprocess")(x)

    try:
        base = constructor(include_top=False, weights=weights,
                           input_shape=(size, size, 3), name=f"{architecture.lower()}_base")
    except Exception as exc:  # usually a failed ImageNet weight download
        raise ModelBuildError(_weights_error_message(architecture, weights, config, exc)) from exc

    base.trainable = False  # phase 1: frozen feature extractor
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="pool")(x)

    dense_units = int(config.get("model", "dense_units", default=0) or 0)
    l2_strength = float(config.get("model", "l2", default=0.0) or 0.0)
    regulariser = keras.regularizers.l2(l2_strength) if l2_strength > 0 else None

    if dense_units > 0:
        x = layers.Dense(dense_units, activation="relu",
                         kernel_regularizer=regulariser, name="dense")(x)

    dropout = float(config.get("model", "dropout", default=0.2) or 0.0)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)

    outputs = layers.Dense(classes, activation="softmax",
                           kernel_regularizer=regulariser, name="predictions")(x)

    model = keras.Model(inputs, outputs, name=f"leaf_lens_{architecture.lower()}")
    LOGGER.info("built %s with %d output classes (%s backbone weights)",
                architecture, classes, weights or "random init")
    return model


def compile_model(model: keras.Model, config: Config,
                  learning_rate: Optional[float] = None) -> keras.Model:
    """Compile with the configured optimiser, loss and metrics."""
    lr = float(learning_rate if learning_rate is not None
               else config.get("training", "learning_rate", default=1e-4))
    optimiser_name = str(config.get("training", "optimizer", default="adam")).lower()
    optimisers = {
        "adam": keras.optimizers.Adam,
        "adamw": keras.optimizers.AdamW,
        "sgd": keras.optimizers.SGD,
        "rmsprop": keras.optimizers.RMSprop,
    }
    if optimiser_name not in optimisers:
        raise ModelBuildError(
            f"unsupported optimizer {optimiser_name!r}. Supported: {', '.join(optimisers)}")
    optimiser = optimisers[optimiser_name](learning_rate=lr)

    label_smoothing = float(config.get("model", "label_smoothing", default=0.0) or 0.0)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=optimiser,
        loss=loss,
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )
    return model


def unfreeze_backbone(model: keras.Model, config: Config,
                      unfreeze_layers: Optional[int] = None) -> int:
    """Unfreeze the top ``unfreeze_layers`` layers of the backbone for fine-tuning.

    BatchNormalization layers are deliberately left frozen: updating their
    running statistics on a small dataset is a well-known cause of unstable
    fine-tuning. Returns the number of layers actually made trainable.
    """
    count = int(unfreeze_layers if unfreeze_layers is not None
                else config.get("training", "fine_tune", "unfreeze_layers", default=0) or 0)
    if count <= 0:
        return 0

    base = _find_backbone(model)
    if base is None:
        LOGGER.warning("no backbone sub-model found; nothing to unfreeze")
        return 0

    base.trainable = True
    trainable = 0
    for index, layer in enumerate(base.layers):
        if index < len(base.layers) - count:
            layer.trainable = False
            continue
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            continue
        layer.trainable = True
        trainable += 1

    LOGGER.info("unfroze %d of the last %d backbone layers (BatchNorm kept frozen)",
                trainable, count)
    return trainable


def _find_backbone(model: keras.Model) -> Optional[keras.Model]:
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name.endswith("_base"):
            return layer
    return None


def model_summary_text(model: keras.Model) -> str:
    lines: List[str] = []
    model.summary(print_fn=lines.append, line_length=100)
    return "\n".join(lines)


def count_parameters(model: keras.Model) -> Dict[str, int]:
    trainable = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    non_trainable = int(sum(tf.size(w).numpy() for w in model.non_trainable_weights))
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }


def _weights_error_message(architecture: str, weights: Optional[str],
                           config: Config, exc: Exception) -> str:
    """Actionable text for a backbone that could not be constructed."""
    if weights is None:
        return (f"could not construct {architecture} with a randomly initialised "
                f"backbone: {exc}")

    cache = keras_cache_dir()
    return (
        f"could not construct {architecture} with weights={weights!r}: {exc}\n"
        f"\n"
        f"The pretrained weights are not in the Keras cache ({cache}) and could not "
        f"be downloaded. This usually means the machine has no internet access.\n"
        f"\n"
        f"Fix it in one of these ways:\n"
        f"  1. On a machine with network access, warm the cache once:\n"
        f"       python -m src.model.build_model --prefetch\n"
        f"     then copy {cache} to this machine. The weights are about 9 MB.\n"
        f"  2. Point KERAS_HOME at a directory that already holds them.\n"
        f"  3. Set model.weights to 'none' in config.yaml to train from a random\n"
        f"     initialisation. Only do this deliberately: transfer learning is the\n"
        f"     reason this project works on a few thousand images, and a random\n"
        f"     backbone needs far more data to reach comparable accuracy.\n"
        f"\n"
        f"Running the tests? They do not need these weights - see tests/conftest.py."
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI for cache management: ``python -m src.model.build_model --prefetch``."""
    import argparse
    import sys

    from ..utils.config import load_config

    parser = argparse.ArgumentParser(
        description="Inspect or warm the pretrained-weights cache")
    parser.add_argument("--config", default=None)
    parser.add_argument("--prefetch", action="store_true",
                        help="download the configured backbone's ImageNet weights")
    parser.add_argument("--check", action="store_true",
                        help="report whether the weights are obtainable, then exit")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    architecture = config.get("model", "architecture", default="MobileNetV2")

    if args.prefetch:
        try:
            cache = prefetch_weights(config)
        except ModelBuildError as exc:
            LOGGER.error("%s", exc)
            return 1
        print(f"ImageNet weights for {architecture} "
              f"({config.image_size}x{config.image_size}) are cached in {cache}")
        return 0

    available = imagenet_weights_available(config)
    print(f"architecture      : {architecture}")
    print(f"input size        : {config.image_size}")
    print(f"keras cache       : {keras_cache_dir()}")
    print(f"imagenet weights  : {'available' if available else 'NOT available'}")
    if not available:
        print("\nRun 'python -m src.model.build_model --prefetch' on a machine with "
              "network access.")
    return 0 if available or args.check else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
