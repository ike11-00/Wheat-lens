"""Train a Leaf Lens classifier.

Two-phase transfer learning:

* **Phase 1 - head training.** The backbone is frozen and only the
  classification head learns, at ``training.learning_rate``.
* **Phase 2 - fine-tuning** (optional, ``training.fine_tune.enabled``). The top
  ``unfreeze_layers`` backbone layers are unfrozen and training continues at a
  much lower learning rate.

Everything produced by a run lands in ``models/<version>/``::

    model.keras            best checkpoint by validation accuracy
    labels.json            class order, image size, threshold - read by predict.py
    history.json           per-epoch losses and accuracies (both phases)
    training_log.csv       raw Keras CSVLogger output
    config_snapshot.yaml   the exact configuration used
    run_metadata.json      seed, versions, dataset counts, duration
    summary.txt            model.summary() of the trained network

Training graphs are written to ``results/graphs/``.

If the prepared dataset is missing the script explains what to do and exits
non-zero. It never invents numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.config import Config, load_config
from ..utils.helpers import (
    configure_matplotlib,
    ensure_dir,
    environment_report,
    get_logger,
    set_global_seed,
    utc_timestamp,
    write_json,
    write_text,
)

LOGGER = get_logger(__name__)


def _merge_history(first: Dict[str, List[float]],
                   second: Optional[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    if not second:
        return {k: list(v) for k, v in first.items()}
    merged = {k: list(v) for k, v in first.items()}
    for key, values in second.items():
        merged.setdefault(key, [])
        merged[key].extend(values)
    return merged


def plot_history(history: Dict[str, List[float]], output_path: Path,
                 phase1_epochs: Optional[int] = None, title: str = "Training history") -> Path:
    """Accuracy and loss curves for training and validation."""
    configure_matplotlib()
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.get("loss", [])) + 1)
    figure, (acc_axis, loss_axis) = plt.subplots(1, 2, figsize=(11, 4.2))

    if "accuracy" in history:
        acc_axis.plot(epochs, history["accuracy"], label="train", color="#3f7d3f")
    if "val_accuracy" in history:
        acc_axis.plot(epochs, history["val_accuracy"], label="validation", color="#3b6ea5")
    acc_axis.set_xlabel("Epoch")
    acc_axis.set_ylabel("Accuracy")
    acc_axis.set_title("Accuracy")
    acc_axis.set_ylim(0, 1.02)
    acc_axis.legend()

    if "loss" in history:
        loss_axis.plot(epochs, history["loss"], label="train", color="#3f7d3f")
    if "val_loss" in history:
        loss_axis.plot(epochs, history["val_loss"], label="validation", color="#3b6ea5")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Loss")
    loss_axis.legend()

    if phase1_epochs and phase1_epochs < len(list(epochs)):
        for axis in (acc_axis, loss_axis):
            axis.axvline(phase1_epochs + 0.5, color="#999999", linestyle="--", linewidth=1)
            axis.annotate("fine-tuning starts", (phase1_epochs + 0.6, axis.get_ylim()[0]),
                          fontsize=7, color="#666666", rotation=90, va="bottom")

    figure.suptitle(title)
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def train(config: Config, epochs: Optional[int] = None,
          fine_tune: Optional[bool] = None,
          output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run the full training procedure. Returns a metadata dictionary."""
    # Imported here so that `--help` and the dataset tools work without TF.
    import tensorflow as tf
    from tensorflow import keras

    from .build_model import (build_model, compile_model, count_parameters,
                              model_summary_text, unfreeze_backbone)
    from .datasets import EmptyDatasetError, build_dataset, class_counts, compute_class_weights

    set_global_seed(config.seed)
    started = time.time()

    model_dir = ensure_dir(Path(output_dir) if output_dir else config.model_dir)
    graphs_dir = ensure_dir(config.path("results_dir") / "graphs")

    LOGGER.info("loading datasets")
    try:
        train_ds, train_paths, train_labels = build_dataset(config, "train", shuffle=True)
        val_ds, val_paths, _ = build_dataset(config, "validation", shuffle=False)
    except EmptyDatasetError as exc:
        raise EmptyDatasetError(str(exc)) from exc

    counts = {split: class_counts(config, split) for split in ("train", "validation", "test")}
    LOGGER.info("train: %d images | validation: %d images", len(train_paths), len(val_paths))
    for name, count in counts["train"].items():
        if count == 0:
            LOGGER.warning("class '%s' has no TRAINING images - the model cannot learn it", name)

    model = compile_model(build_model(config), config)
    LOGGER.info("parameters: %s", count_parameters(model))

    class_weights = None
    if bool(config.get("training", "use_class_weights", default=True)):
        class_weights = compute_class_weights(config, train_labels)
        LOGGER.info("class weights: %s",
                    {config.class_names[k]: round(v, 3) for k, v in class_weights.items()})

    checkpoint_path = model_dir / "model.keras"
    monitor = "val_accuracy" if len(val_paths) else "accuracy"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor=monitor, mode="max",
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max",
            patience=int(config.get("training", "early_stopping_patience", default=5)),
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if len(val_paths) else "loss", mode="min",
            factor=float(config.get("training", "reduce_lr_factor", default=0.5)),
            patience=int(config.get("training", "reduce_lr_patience", default=3)),
            min_lr=float(config.get("training", "min_lr", default=1e-6)), verbose=1),
        keras.callbacks.CSVLogger(str(model_dir / "training_log.csv"), append=False),
    ]

    phase1_epochs = int(epochs if epochs is not None
                        else config.get("training", "epochs", default=20))
    LOGGER.info("phase 1: training the classification head for up to %d epochs", phase1_epochs)
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=phase1_epochs,
        callbacks=callbacks, class_weight=class_weights, verbose=2,
    )
    history = _merge_history(history1.history, None)
    epochs_phase1 = len(history1.history.get("loss", []))

    # --- phase 2: fine-tuning -------------------------------------------
    do_fine_tune = (config.get("training", "fine_tune", "enabled", default=False)
                    if fine_tune is None else fine_tune)
    unfrozen = 0
    epochs_phase2 = 0
    if do_fine_tune:
        unfrozen = unfreeze_backbone(model, config)
        if unfrozen:
            ft_lr = float(config.get("training", "fine_tune", "learning_rate", default=1e-5))
            compile_model(model, config, learning_rate=ft_lr)
            ft_epochs = int(config.get("training", "fine_tune", "epochs", default=10))
            LOGGER.info("phase 2: fine-tuning %d layers at lr=%g for up to %d epochs",
                        unfrozen, ft_lr, ft_epochs)
            callbacks[3] = keras.callbacks.CSVLogger(
                str(model_dir / "training_log_finetune.csv"), append=False)
            history2 = model.fit(
                train_ds, validation_data=val_ds, epochs=ft_epochs,
                callbacks=callbacks, class_weight=class_weights, verbose=2,
            )
            history = _merge_history(history, history2.history)
            epochs_phase2 = len(history2.history.get("loss", []))
        else:
            LOGGER.info("fine-tuning requested but no layers were unfrozen; skipping phase 2")

    # ModelCheckpoint keeps the best weights on disk; EarlyStopping restored
    # them in memory. Save once more so the file always matches the returned
    # model even if training ended without an improvement.
    if not checkpoint_path.exists():
        model.save(checkpoint_path)

    # --- artefacts -------------------------------------------------------
    best_val_accuracy = max(history.get("val_accuracy", [0.0])) if history.get("val_accuracy") else None
    best_val_loss = min(history.get("val_loss", [])) if history.get("val_loss") else None

    labels_payload = {
        "class_names": config.class_names,
        "class_directories": config.class_dirs,
        "image_size": config.image_size,
        "confidence_threshold": config.confidence_threshold,
        "model_version": config.model_version,
        "architecture": config.get("model", "architecture", default="MobileNetV2"),
        "trained_utc": utc_timestamp(),
    }
    write_json(model_dir / "labels.json", labels_payload)
    write_json(model_dir / "history.json", history)
    config.snapshot(model_dir / "config_snapshot.yaml")
    write_text(model_dir / "summary.txt", model_summary_text(model))

    metadata: Dict[str, Any] = {
        "model_version": config.model_version,
        "architecture": config.get("model", "architecture", default="MobileNetV2"),
        "backbone_weights": config.get("model", "weights", default="imagenet"),
        "num_classes": config.num_classes,
        "class_names": config.class_names,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "random_seed": config.seed,
        "learning_rate": config.get("training", "learning_rate"),
        "optimizer": config.get("training", "optimizer"),
        "augmentation": config.get("training", "augmentation"),
        "class_weights_used": bool(class_weights),
        "class_weights": ({config.class_names[k]: round(v, 4) for k, v in class_weights.items()}
                          if class_weights else None),
        "epochs_configured": phase1_epochs,
        "epochs_run_phase1": epochs_phase1,
        "fine_tune_enabled": bool(do_fine_tune and unfrozen),
        "fine_tune_layers_unfrozen": unfrozen,
        "epochs_run_finetune": epochs_phase2,
        "dataset_counts": counts,
        "dataset_totals": {k: sum(v.values()) for k, v in counts.items()},
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "final_train_accuracy": history.get("accuracy", [None])[-1],
        "final_val_accuracy": history.get("val_accuracy", [None])[-1] if history.get("val_accuracy") else None,
        "parameters": count_parameters(model),
        "duration_seconds": round(time.time() - started, 1),
        "environment": environment_report(),
        "model_path": str(checkpoint_path),
        "note": (
            "Validation accuracy measures performance on held-out validation images only. "
            "It is NOT evidence of performance on new photographs - run "
            "'python -m src.model.evaluate' on the untouched test split for that."
        ),
    }
    write_json(model_dir / "run_metadata.json", metadata)

    graph_path = plot_history(
        history, graphs_dir / f"training_history_{config.model_version}.png",
        phase1_epochs=epochs_phase1,
        title=f"Leaf Lens {config.model_version} - training history",
    )
    metadata["training_graph"] = str(graph_path)

    LOGGER.info("saved model to %s", checkpoint_path)
    LOGGER.info("best validation accuracy: %s",
                f"{best_val_accuracy:.4f}" if best_val_accuracy is not None else "n/a")
    LOGGER.info("training graph: %s", graph_path)
    return metadata


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a Leaf Lens classifier")
    parser.add_argument("--config", default=None, help="alternative config file (e.g. config/model_v2.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--version", default=None, help="override model.version (output folder)")
    parser.add_argument("--no-fine-tune", action="store_true", help="skip phase 2")
    parser.add_argument("--fine-tune", action="store_true", help="force phase 2 on")
    parser.add_argument("--output-dir", default=None, help="override the model output directory")
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.version:
        overrides["model"] = {"version": args.version}
    config = load_config(args.config, overrides=overrides or None)

    fine_tune: Optional[bool] = None
    if args.no_fine_tune:
        fine_tune = False
    elif args.fine_tune:
        fine_tune = True

    try:
        from .datasets import EmptyDatasetError
        metadata = train(config, epochs=args.epochs, fine_tune=fine_tune,
                         output_dir=Path(args.output_dir) if args.output_dir else None)
    except EmptyDatasetError as exc:
        LOGGER.error("TRAINING CANNOT PROCEED: %s", exc)
        LOGGER.error("")
        LOGGER.error("Leaf Lens has no dataset yet. To train:")
        LOGGER.error("  1. Put labelled images in data/raw/<Class>/ - one folder per class:")
        for spec in config.classes:
            LOGGER.error("       data/raw/%s/", spec.directory)
        LOGGER.error("  2. python -m src.data.validate_dataset")
        LOGGER.error("  3. python -m src.data.prepare_dataset")
        LOGGER.error("  4. python -m src.model.train")
        LOGGER.error("")
        LOGGER.error("Candidate public datasets: python -m src.data.download_dataset --list")
        LOGGER.error("See docs/dataset.md for details.")
        return 1

    print(json.dumps(
        {k: v for k, v in metadata.items() if k not in ("environment", "augmentation")},
        indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
