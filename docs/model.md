# Model

## Status

**No model has been trained on real wheat-leaf data.** `models/` is empty.

Every accuracy, precision, recall and F1 field in the project's reports is
therefore **Not yet tested — requires dataset**. The architecture below has
been built, compiled, trained, saved, reloaded and used for inference against
synthetic fixtures, so the code path is verified; the *task* is not.

## Architecture

```text
Input  (224 x 224 x 3, float32, raw 0-255 pixels)
  │
  ├─ augmentation          (training only; no-op at inference)
  │    RandomFlip(horizontal)
  │    RandomRotation(0.08)        ~ +/- 29 degrees
  │    RandomZoom(0.15)
  │    RandomTranslation(0.10)
  │    RandomBrightness(0.15, value_range=(0,255))
  │    RandomContrast(0.15)
  │
  ├─ BackbonePreprocessing         mobilenet_v2.preprocess_input  ->  [-1, 1]
  │
  ├─ MobileNetV2 backbone          include_top=False, ImageNet weights
  │                                (frozen in phase 1)
  │
  ├─ GlobalAveragePooling2D        -> 1280 features
  ├─ Dropout(0.2)
  └─ Dense(5, softmax)             -> one probability per configured class
```

Parameter counts (224×224 input, five classes):

| Phase | Trainable | Non-trainable | Total |
|---|---|---|---|
| Phase 1 (backbone frozen) | 6,405 | 2,257,984 | 2,264,389 |
| Phase 2 (top 40 layers unfrozen, BatchNorm kept frozen) | 1,669,765 | 594,624 | 2,264,389 |

## Why MobileNetV2

* **Small and fast on CPU.** ~2.3M parameters and depthwise-separable
  convolutions mean a prediction completes inside a Flask request on an
  ordinary machine, with no GPU. The web application is the delivery vehicle
  for this project, so inference cost is a real constraint.
* **Sufficient capacity for leaf texture.** Rust pustules, mildew coating and
  chlorotic striping are texture-and-colour patterns; a 1280-dimensional
  ImageNet feature vector separates them well in the published literature.
* **Transfer learning suits a small dataset.** Wheat-disease datasets are
  small (hundreds to a few thousand images). Training a network from scratch on
  that would overfit badly; reusing ImageNet features and training a 6,405-
  parameter head is the right ratio of parameters to data.

`config/config.yaml` also supports `MobileNetV3Small`, `EfficientNetB0` and
`ResNet50` via `model.architecture` — change the one line and retrain. If you
switch, record why in the comparison report.

## Transfer learning, in two phases

**Phase 1 — head training.** The backbone is frozen and only the classifier
head learns. Learning rate `1e-4` (Adam). This is deliberately the larger
learning rate: the head starts from random weights and has to move a long way.

**Phase 2 — fine-tuning.** The top `training.fine_tune.unfreeze_layers` (40)
backbone layers are unfrozen and training continues at `1e-5` — ten times
lower, because the backbone weights are already good and large updates would
destroy them.

**BatchNormalization layers are kept frozen during fine-tuning.** Updating
their running statistics on a small dataset is a well-known cause of unstable
fine-tuning and of a train/inference mismatch. `unfreeze_backbone()` skips
them explicitly.

Both phases share the callbacks below, and `ModelCheckpoint` keeps the single
best network by validation accuracy across the whole run.

## Preprocessing lives inside the model

`BackbonePreprocessing` is a registered Keras layer that applies the
architecture's own `preprocess_input`. Two consequences:

* Every caller — web app, CLI, evaluator, error analysis — hands the model raw
  0–255 RGB pixels. It is impossible for one of them to normalise differently
  from the others.
* The layer stores the *architecture name* rather than a function reference, so
  the saved `.keras` file round-trips through `load_model`. (A plain
  `Lambda(preprocess_input)` does not: Keras cannot locate the function on
  load. This was caught by the test suite and fixed.)

## Augmentation

| Transformation | Setting | Rationale |
|---|---|---|
| Horizontal flip | on | A leaf has no canonical left/right. |
| Vertical flip | **off** | An upside-down leaf photograph is unrealistic. |
| Rotation | ±0.08 turns (~29°) | Hand-held cameras are not level. |
| Zoom | ±15% | Camera distance varies. |
| Translation | ±10% | The leaf is not always centred. |
| Brightness | ±15% | Sun, shade, overcast. |
| Contrast | ±15% | Camera and exposure differences. |

**Deliberately excluded: hue and channel shifts.** Colour is a genuine
discriminative signal here — yellow rust versus brown rust differs largely by
pustule hue. Randomising hue would destroy the very feature the model needs.

All augmentation layers are inside the model and are no-ops when
`training=False`, which the test suite asserts by checking that two inference
passes over the same batch return identical probabilities.

## Training configuration

| Setting | Value | Config key |
|---|---|---|
| Image size | 224 × 224 | `data.image_size` |
| Batch size | 32 | `data.batch_size` |
| Epochs (phase 1) | 20 | `training.epochs` |
| Epochs (phase 2) | 10 | `training.fine_tune.epochs` |
| Optimizer | Adam | `training.optimizer` |
| Learning rate | 1e-4 → 1e-5 | `training.learning_rate`, `training.fine_tune.learning_rate` |
| Loss | Categorical cross-entropy | fixed (label smoothing via `model.label_smoothing`) |
| Metrics | Categorical accuracy, top-2 accuracy | fixed |
| Class weights | Inverse frequency, mean-normalised | `training.use_class_weights` |
| Random seed | 42 | `random_seed` |

**Loss.** Categorical cross-entropy over a softmax output — the standard choice
for single-label multi-class classification, and the loss whose output can be
read as a probability distribution, which the interface needs.

**Callbacks.**

| Callback | Configuration | Purpose |
|---|---|---|
| `ModelCheckpoint` | monitor `val_accuracy`, `save_best_only` | Keeps the best network, not the last. |
| `EarlyStopping` | patience 5, `restore_best_weights` | Stops when validation stops improving. |
| `ReduceLROnPlateau` | factor 0.5, patience 3, min 1e-6 | Smaller steps when progress stalls. |
| `CSVLogger` | `models/<version>/training_log.csv` | Raw per-epoch record. |

## Output classes

Five softmax units, in the order defined by `config/config.yaml`:

| Index | Class |
|---|---|
| 0 | Healthy |
| 1 | Yellow Rust |
| 2 | Karnal Bunt |
| 3 | Brown Rust |
| 4 | Powdery Mildew |

The class order is written to `models/<version>/labels.json` at training time,
and `Predictor` reads that file rather than today's config — so a model trained
with a different class list can never be misinterpreted. If the number of
output units does not match the number of labels, loading fails with an
explicit error instead of producing silently wrong labels.

The pipeline does **not** use `image_dataset_from_directory`'s alphabetical
class ordering, which would put Brown_Rust at index 0 and desynchronise the
model head from the reports.

## Confidence and uncertainty

`Predictor` returns three signals per image:

* **confidence** — the softmax probability of the predicted class.
* **low_confidence** — true when confidence is below
  `inference.confidence_threshold` (0.60).
* **uncertain** — true when confidence is low **or** the normalised Shannon
  entropy of the distribution exceeds `inference.ood.entropy_ratio_threshold`
  (0.80) **or** the gap between the top two probabilities is below
  `inference.ood.margin_threshold` (0.10).

### How the out-of-distribution check works

Entropy is `-Σ p log p` divided by `log(5)`, so 0 means all the mass is on one
class and 1 means a uniform guess. The margin is `p(top1) - p(top2)`. Together
they catch the two shapes of "the model does not know": mass spread everywhere,
and mass split between two candidates.

### What it cannot do

This is a **max-softmax heuristic, not a novelty detector**. A softmax layer
always produces a distribution over exactly the classes it was trained on, and
neural networks are routinely *confidently wrong* on inputs unlike anything
they have seen. Specifically, this check:

* will not reliably detect a sixth wheat disease,
* will not reliably detect a different crop, a nutrient deficiency or pest
  damage,
* will not reliably detect a photograph that is not a leaf at all.

A low flag rate is **not** evidence that an input is in-distribution. Treating
it as such would be the single most misleading thing this project could do, so
the API returns the limitation text alongside the numbers and the web interface
prints it next to the warning.

Proper out-of-distribution detection needs an explicit mechanism — a background
or "other" class trained on negatives, an ensemble disagreement measure, Monte
Carlo dropout, or a density model over the feature space. None of those is
implemented here.

## Model versions

Each version owns a directory:

```text
models/v1/
├── model.keras            best checkpoint by validation accuracy
├── labels.json            class order, image size, threshold
├── history.json           per-epoch metrics for both phases
├── training_log.csv       raw CSVLogger output
├── config_snapshot.yaml   the exact configuration used
├── run_metadata.json      seed, versions, dataset counts, duration
└── summary.txt            model.summary() of the trained network
```

`config/model_v2.yaml` defines a V2 variant containing only the keys that
differ from the base config (more dropout, label smoothing, more epochs, deeper
unfreeze, stronger geometric augmentation). It is deep-merged onto
`config.yaml`, so shared settings stay in one place.

```bash
python -m src.model.train    --config config/model_v2.yaml
python -m src.model.evaluate --config config/model_v2.yaml
python -m src.model.compare_models v1 v2
```

The comparison report does not declare a winner. See `docs/testing.md`.
