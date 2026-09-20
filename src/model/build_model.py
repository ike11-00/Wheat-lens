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

from typing import Any, Dict, List, Optional, Tuple

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


class ModelBuildError(RuntimeError):
    """Raised when the requested architecture or configuration is unusable."""


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
                include_augmentation: bool = True) -> keras.Model:
    """Assemble the classifier.

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

    weights = config.get("model", "weights", default="imagenet")
    weights = None if weights in (None, "", "none", "None") else weights

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
    except Exception as exc:  # network failure while fetching ImageNet weights
        raise ModelBuildError(
            f"could not construct {architecture} with weights={weights!r}: {exc}. "
            f"If the pretrained weights cannot be downloaded, set model.weights to 'none' "
            f"in config.yaml - but note that training from scratch needs far more data."
        ) from exc

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
