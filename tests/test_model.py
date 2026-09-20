"""Model construction and the tf.data pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.model.build_model import (ModelBuildError, build_augmentation, build_model,
                                   compile_model, count_parameters, unfreeze_backbone)
from src.model.datasets import (EmptyDatasetError, build_dataset, compute_class_weights,
                                list_split_files)
from src.utils.config import load_config

from .conftest import offline_safe_config


@pytest.fixture(scope="module")
def small_config():
    """A tiny model so the tests stay fast, with no pretrained download.

    `weights=None` is explicit: every assertion below is about how the graph is
    assembled, which the backbone's numeric values cannot change. See the note
    in tests/conftest.py, and `test_imagenet_weights_path_builds` for coverage
    of the production ImageNet path.
    """
    return offline_safe_config(data={"image_size": 96, "batch_size": 4})


def test_model_has_one_output_per_configured_class(small_config):
    model = build_model(small_config)
    assert model.output_shape[-1] == small_config.num_classes
    assert small_config.num_classes >= 2


def test_model_accepts_raw_0_255_pixels_and_returns_probabilities(small_config):
    model = build_model(small_config)
    batch = np.random.randint(0, 256, (3, 96, 96, 3)).astype("float32")
    probabilities = model(batch, training=False).numpy()
    assert probabilities.shape == (3, small_config.num_classes)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert (probabilities >= 0).all()


def test_preprocessing_is_inside_the_graph(small_config):
    model = build_model(small_config)
    assert any(layer.name == "preprocess" for layer in model.layers)


def test_augmentation_is_inactive_at_inference(small_config):
    model = build_model(small_config)
    batch = np.random.randint(0, 256, (2, 96, 96, 3)).astype("float32")
    first = model(batch, training=False).numpy()
    second = model(batch, training=False).numpy()
    assert np.allclose(first, second)


def test_output_scales_with_class_count(small_config):
    assert build_model(small_config, num_classes=7).output_shape[-1] == 7


def test_unknown_architecture_rejected(small_config):
    config = offline_safe_config(model={"architecture": "NotARealNet"})
    with pytest.raises(ModelBuildError):
        build_model(config)


def test_too_few_classes_rejected(small_config):
    with pytest.raises(ModelBuildError):
        build_model(small_config, num_classes=1)


def test_augmentation_pipeline_built_from_config(small_config):
    pipeline = build_augmentation(small_config)
    assert pipeline is not None
    names = [layer.name for layer in pipeline.layers]
    assert "aug_flip" in names and "aug_rotation" in names
    # Vertical flip is off by default: an upside-down leaf is unrealistic.
    assert not any("vertical" in name for name in names)


def test_augmentation_can_be_disabled():
    config = offline_safe_config(training={"augmentation": {
        "horizontal_flip": False, "vertical_flip": False, "rotation": 0,
        "zoom": 0, "translation": 0, "brightness": 0, "contrast": 0}})
    assert build_augmentation(config) is None


def test_backbone_frozen_then_unfrozen(small_config):
    model = compile_model(build_model(small_config), small_config)
    before = count_parameters(model)
    unfrozen = unfreeze_backbone(model, small_config)
    after = count_parameters(model)
    assert unfrozen > 0
    assert after["trainable"] > before["trainable"]
    assert after["total"] == before["total"]


def test_compile_sets_loss_and_metrics(small_config):
    model = compile_model(build_model(small_config), small_config)
    assert model.loss is not None
    assert model.optimizer is not None
    # Keras 3 groups compiled metrics, so check the compile config and the
    # actual values returned for a batch.
    compile_config = model.get_compile_config()
    metric_names = str(compile_config.get("metrics"))
    assert "accuracy" in metric_names and "top2_accuracy" in metric_names

    batch = np.random.randint(0, 256, (2, 96, 96, 3)).astype("float32")
    targets = np.eye(small_config.num_classes)[[0, 1]]
    loss, accuracy, top2 = model.test_on_batch(batch, targets)
    assert loss >= 0.0
    assert 0.0 <= accuracy <= 1.0
    assert accuracy <= top2  # top-2 accuracy can never be below top-1


def test_unsupported_optimizer_rejected(small_config):
    config = offline_safe_config(training={"optimizer": "nope"})
    with pytest.raises(ModelBuildError):
        compile_model(build_model(small_config), config)


def test_dataset_raises_clearly_when_split_is_empty(tmp_path):
    config = offline_safe_config(paths={"train_dir": str(tmp_path / "train")})
    with pytest.raises(EmptyDatasetError) as excinfo:
        build_dataset(config, "train")
    assert "prepare_dataset" in str(excinfo.value)


def test_dataset_labels_follow_config_order(tmp_path):
    config = offline_safe_config(
        paths={"train_dir": str(tmp_path / "train")},
        data={"image_size": 32, "batch_size": 2},
    )
    train_dir = tmp_path / "train"
    for index, directory in enumerate(config.class_dirs):
        folder = train_dir / directory
        folder.mkdir(parents=True)
        Image.new("RGB", (40, 40), (index * 40, 90, 60)).save(folder / "a.jpg")

    paths, labels = list_split_files(config, train_dir)
    assert labels == list(range(config.num_classes))
    # Alphabetical order would put Brown_Rust first; config order puts Healthy first.
    assert paths[0].parent.name == config.class_dirs[0] == "Healthy"

    dataset, _, _ = build_dataset(config, "train")
    images, one_hot = next(iter(dataset))
    assert images.shape[1:] == (32, 32, 3)
    assert one_hot.shape[-1] == config.num_classes
    assert float(images.numpy().max()) > 1.5  # still 0-255, not rescaled


def test_class_weights_balance_a_skewed_split(small_config):
    labels = [0] * 100 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10
    weights = compute_class_weights(small_config, labels)
    assert weights[0] < weights[1]
    assert abs(np.mean(list(weights.values())) - 1.0) < 1e-6


def test_class_weights_handle_absent_class(small_config):
    weights = compute_class_weights(small_config, [0] * 10 + [1] * 10)
    assert all(np.isfinite(value) for value in weights.values())
    assert len(weights) == small_config.num_classes


def test_imagenet_weights_path_builds(config, imagenet_weights_available):
    """The production configuration - MobileNetV2 with ImageNet weights.

    This is the one test that exercises the real pretrained path. It runs
    whenever the weights are cached or downloadable and skips otherwise, so an
    offline machine gets a clean skip rather than 30 errors. Warm the cache
    with `python -m src.model.build_model --prefetch`.
    """
    if not imagenet_weights_available:
        pytest.skip("ImageNet weights are not cached and cannot be downloaded; "
                    "run 'python -m src.model.build_model --prefetch' when online")

    production = load_config(overrides={"data": {"image_size": 96}})
    assert production.get("model", "weights") == "imagenet"

    model = build_model(production)
    assert model.output_shape[-1] == production.num_classes

    # A pretrained backbone must not be all-zero or randomly distributed the way
    # a fresh initialisation is; check it actually carries learned values.
    backbone = next(layer for layer in model.layers
                    if getattr(layer, "name", "").endswith("_base"))
    kernels = [w for w in backbone.weights if "kernel" in w.name]
    assert kernels, "backbone exposes no kernel weights"
    assert float(np.abs(kernels[0].numpy()).sum()) > 0.0

    # And the graph is the same one the offline tests verify.
    batch = np.random.randint(0, 256, (2, 96, 96, 3)).astype("float32")
    probabilities = model(batch, training=False).numpy()
    assert probabilities.shape == (2, production.num_classes)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_random_backbone_differs_from_pretrained(small_config, imagenet_weights_available):
    """weights=None and weights='imagenet' must produce different parameters.

    Guards against the fix degenerating into "always random": if these two ever
    came out identical, the production path would silently be untrained.
    """
    if not imagenet_weights_available:
        pytest.skip("ImageNet weights are not available for comparison")

    random_backbone = build_model(small_config, weights=None)
    pretrained = build_model(small_config, weights="imagenet")

    def first_kernel(model):
        base = next(l for l in model.layers if getattr(l, "name", "").endswith("_base"))
        return next(w for w in base.weights if "kernel" in w.name).numpy()

    assert not np.allclose(first_kernel(random_backbone), first_kernel(pretrained))
