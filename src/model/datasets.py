"""``tf.data`` input pipelines built from the prepared split folders.

The class order is taken from ``config.yaml``, **not** from
``image_dataset_from_directory``'s alphabetical ordering, so label index *i*
always means ``config.class_names[i]`` everywhere in the project: in the model
head, the confusion matrix, the JSON reports and the web UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from ..utils.config import Config
from ..utils.helpers import get_logger
from ..utils.image_io import iter_image_files

LOGGER = get_logger(__name__)
AUTOTUNE = tf.data.AUTOTUNE


class EmptyDatasetError(RuntimeError):
    """Raised when a split directory holds no usable images."""


def list_split_files(config: Config, split_dir: Path) -> Tuple[List[Path], List[int]]:
    """Return ``(paths, label_indices)`` for one split, in config class order."""
    paths: List[Path] = []
    labels: List[int] = []
    for index, spec in enumerate(config.classes):
        class_dir = Path(split_dir) / spec.directory
        for path in iter_image_files(class_dir, config.supported_extensions):
            paths.append(path)
            labels.append(index)
    return paths, labels


def _decode(path: tf.Tensor, label: tf.Tensor, image_size: int, num_classes: int):
    raw = tf.io.read_file(path)
    # expand_animations=False keeps animated GIF/WEBP from yielding a 4-D tensor.
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    image = tf.cast(image, tf.float32)  # stays in the 0-255 range on purpose
    image.set_shape((image_size, image_size, 3))
    return image, tf.one_hot(label, num_classes)


def build_dataset(config: Config, split: str, shuffle: bool = False,
                  batch_size: Optional[int] = None,
                  repeat: bool = False) -> Tuple[tf.data.Dataset, List[Path], List[int]]:
    """Build a batched dataset for ``split`` ("train" / "validation" / "test").

    Images are yielded in the raw 0-255 range; normalisation happens inside the
    model (see ``build_model``).
    """
    split_dir = config.path(f"{split}_dir")
    paths, labels = list_split_files(config, split_dir)
    if not paths:
        raise EmptyDatasetError(
            f"no images found in {split_dir}. Run 'python -m src.data.prepare_dataset' first."
        )

    batch = int(batch_size or config.batch_size)
    image_size = config.image_size
    num_classes = config.num_classes

    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(p) for p in paths], np.asarray(labels, dtype="int32"))
    )
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=config.seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda p, l: _decode(p, l, image_size, num_classes), num_parallel_calls=AUTOTUNE
    )
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch).prefetch(AUTOTUNE)
    return dataset, paths, labels


def class_counts(config: Config, split: str) -> Dict[str, int]:
    _, labels = list_split_files(config, config.path(f"{split}_dir"))
    names = config.class_names
    counts = {name: 0 for name in names}
    for label in labels:
        counts[names[label]] += 1
    return counts


def compute_class_weights(config: Config, labels: List[int]) -> Dict[int, float]:
    """Inverse-frequency class weights, normalised to mean 1.0.

    Classes absent from the training split get weight 1.0 so Keras does not
    raise; ``validate_dataset`` is what reports the missing class.
    """
    counts = np.bincount(np.asarray(labels, dtype="int64"), minlength=config.num_classes)
    total = counts.sum()
    weights: Dict[int, float] = {}
    present = (counts > 0).sum()
    for index, count in enumerate(counts):
        weights[index] = float(total / (present * count)) if count else 1.0
    mean = float(np.mean(list(weights.values())))
    if mean > 0:
        weights = {k: v / mean for k, v in weights.items()}
    return weights
