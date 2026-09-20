# Reproducibility

## Verified environment

The project was built and tested in this environment:

| Component | Version |
|---|---|
| Python | 3.11.15 |
| TensorFlow (CPU) | 2.17.1 |
| Keras | 3.15.1 |
| NumPy | 1.26.4 |
| scikit-learn | 1.9.1 |
| Pillow | 12.3.0 |
| Flask | 3.x |
| matplotlib / seaborn / pandas / PyYAML / ImageHash | see `requirements.txt` |
| OS | Linux (x86-64), 4 CPU cores, no GPU |

`requirements.txt` pins TensorFlow and NumPy exactly. NumPy is held below 2.0
because TensorFlow 2.17 is not compatible with the 2.x ABI.

Every training and evaluation run records its own environment in
`models/<version>/run_metadata.json` and in the `environment` block of
`results/evaluation/evaluation_<version>.json`, so a report always says which
versions produced it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # confirm the install
```

The first `build_model` call downloads MobileNetV2's ImageNet weights
(~9.4 MB) from `storage.googleapis.com` and caches them in `~/.keras/models/`.
On a machine without that access, set `model.weights: none` in
`config/config.yaml` — but a randomly initialised backbone needs far more data
than a wheat-disease dataset typically provides.

## Fixed parameters

All of these live in `config/config.yaml` and are snapshotted into
`models/<version>/config_snapshot.yaml` at training time:

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs (phase 1 / fine-tune) | 20 / 10 |
| Learning rate (phase 1 / fine-tune) | 1e-4 / 1e-5 |
| Optimizer | Adam |
| Split (train / validation / test) | 0.75 / 0.125 / 0.125 |
| Architecture | MobileNetV2, ImageNet weights |
| Dropout | 0.2 |
| Confidence threshold | 0.60 |
| Near-duplicate pHash threshold | 5 (hash size 8) |

## What the seed controls

`set_global_seed()` seeds Python's `random`, NumPy and
`tf.keras.utils.set_random_seed` (which covers TensorFlow's global seed and
Keras initialisers). Consequences:

* **The train/validation/test split is fully deterministic.** Running
  `prepare_dataset` twice on the same `data/raw` produces byte-identical
  assignments. There is a test for this
  (`test_preparation_is_deterministic`).
* **Weight initialisation and shuffling are deterministic** on the same machine
  with the same versions.

## What the seed does not control

Bit-for-bit identical *training* across machines is not guaranteed:

* CPU kernels (oneDNN) may reorder floating-point reductions depending on the
  instruction sets available, so results can differ slightly between CPUs.
* GPU training is non-deterministic by default; `tf.config.experimental.enable_op_determinism()`
  would be required, at a significant speed cost.
* `tf.data` parallel mapping can vary the order of prefetched elements.

Expect small run-to-run differences in accuracy on the same data. That is
normal and is exactly why `compare_models` refuses to declare a winner on a
small margin.

## Reproducing a run

```bash
# 1. Restore the dataset into data/raw/ (see docs/dataset.md)
# 2. Use the snapshotted configuration from the run you want to reproduce
python -m src.data.prepare_dataset --config models/v1/config_snapshot.yaml
python -m src.model.train          --config models/v1/config_snapshot.yaml
python -m src.model.evaluate       --config models/v1/config_snapshot.yaml
```

The snapshot pins every setting that mattered, including the seed and the split
fractions, so the same raw images give the same splits.

**The dataset is the part git does not carry.** Images are excluded by
`.gitignore` (size and licensing). To make a run reproducible for someone else,
record in `docs/dataset.md`: the source and URL, the exact version or download
date, the number of images per class, and any files you removed by hand. The
manifest at `results/dataset/split_manifest.json` lists every source file, its
destination and its duplicate-group id, which is enough to rebuild the exact
splits from the same raw images.

## What is and is not committed

| Committed | Not committed |
|---|---|
| All source code | `data/**` (images) |
| `config/*.yaml` | `models/**` (weights) |
| `docs/`, `README.md` | `results/**` (generated reports) |
| `tests/` including the fixture generator | `.venv/` |

Generated results are excluded because they are reproducible from the code plus
the dataset. To keep a particular report, add it deliberately:

```bash
git add -f results/evaluation/evaluation_v1.md
```
