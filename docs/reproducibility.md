# Reproducibility

## Verified environment

The project was built and tested on this environment:

| Component | Version |
|---|---|
| Python | 3.11.15 |
| TensorFlow (CPU) | 2.17.1 |
| Keras | 3.15.1 |
| NumPy | 1.26.4 |
| scikit-learn | 1.9.1 |
| Pillow | 12.3.0 |
| OS | Linux x86-64, 4 CPU cores, no GPU |

### Python version support

TensorFlow is the only dependency that constrains the Python version. Every
other package in `requirements.txt` already ships wheels for 3.13 and 3.14.

| Python | Supported | Notes |
|---|---|---|
| 3.10 – 3.12 | Yes | TensorFlow 2.17 – 2.21 |
| 3.13 | Yes | Requires TensorFlow ≥ 2.20 |
| **3.14** | **No** | **No TensorFlow release ships a cp314 wheel** |

Checked against the PyPI package index on 2026-09-20: TensorFlow 2.21.0 is the
newest release and publishes wheels for CPython 3.10–3.13 only. No release,
including pre-releases, publishes a 3.14 wheel.

### Forward-compatibility, verified

The project was re-tested end to end against the newest TensorFlow to confirm
that the pinned version is a floor rather than a requirement:

| Check | TensorFlow 2.17.1 / NumPy 1.26 | TensorFlow 2.21.0 / NumPy 2.4 |
|---|---|---|
| Test suite | 90 passed | 90 passed |
| Trained model loads | Yes | Yes |
| Web application | Works | Works |

A model trained under 2.17 loads under 2.21 and produces **bit-identical**
predictions — maximum absolute probability difference 0.000e+00 across the
comparison images. So the `models/v1/model.keras` artefact is portable across
this range and does not need retraining to move Python version.

`requirements.txt` therefore specifies `tensorflow-cpu>=2.17,<3.0` rather than
an exact pin, and deliberately does **not** pin NumPy: TensorFlow declares its
own compatible NumPy range (2.17 requires `<2.0`, 2.20+ accepts 2.x), so
pinning NumPy separately would silently force an old TensorFlow and block
Python 3.13.

If you need an exact-version reproduction of the original run, use the table at
the top of this section rather than a fresh resolve.

### Getting a specific Python version

python.org keeps every release archived, so no version becomes unavailable.
For per-project versions:

```bash
pyenv install 3.13          # then: pyenv local 3.13
conda create -n leaflens python=3.13
uv venv --python 3.13
```

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
