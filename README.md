# Leaf Lens

**Using Machine Learning to Detect Agricultural Diseases**

Leaf Lens is an experimental image-classification project that identifies
selected wheat-leaf conditions from photographs. It contains a complete machine
-learning pipeline — dataset validation, leakage-free splitting, transfer
learning, evaluation, error analysis, condition-based testing and model
comparison — plus a web application for uploading a photograph and seeing the
model's prediction.

> ### Current status
>
> The pipeline is **built and tested**. The model is **not trained**, because
> this repository contains **no dataset**.
>
> Every script runs today; training stops with clear instructions until images
> are placed in `data/raw/`. No accuracy figure is claimed anywhere in this
> project, because none has been measured on real wheat images.
> See [What you need to provide](#what-you-need-to-provide).

---

## Contents

1. [What Leaf Lens is](#what-leaf-lens-is)
2. [Diseases identified](#diseases-identified)
3. [How the system works](#how-the-system-works)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Dataset](#dataset)
7. [Training](#training)
8. [Validation and testing](#validation-and-testing)
9. [Evaluation](#evaluation)
10. [Error analysis](#error-analysis)
11. [Realistic testing](#realistic-testing)
12. [Model improvement and comparison](#model-improvement-and-comparison)
13. [Web application](#web-application)
14. [Command reference](#command-reference)
15. [Project structure](#project-structure)
16. [Configuration](#configuration)
17. [Limitations](#limitations)
18. [Troubleshooting](#troubleshooting)
19. [What you need to provide](#what-you-need-to-provide)

---

## What Leaf Lens is

A convolutional neural network learns to distinguish wheat-leaf conditions from
a labelled image dataset, and a small web application puts that network behind
an upload form.

The classification comes **entirely from the trained network**. There are no
colour rules, no "if yellow spots then yellow rust", no filename matching, no
keyword matching and no image-similarity lookup anywhere in this codebase. If
there is no trained model, the system says so and refuses to guess.

## Diseases identified

| Class | Folder |
|---|---|
| Healthy | `Healthy` |
| Yellow Rust | `Yellow_Rust` |
| Karnal Bunt | `Karnal_Bunt` |
| Brown Rust | `Brown_Rust` |
| Powdery Mildew | `Powdery_Mildew` |

These are defined once, in `config/config.yaml`. Adding a sixth class means
appending an entry there and creating the matching folder — the model head, the
confusion matrix, every report table and the web UI all derive their class list
from that single definition. No code changes.

## How the system works

```text
Wheat leaf image database
        ↓
Dataset validation          corrupt files, missing classes, imbalance, duplicates
        ↓
Dataset preparation         near-duplicate grouping → leakage-free 75/12.5/12.5 split
        ↓
Training dataset
        ↓
Machine learning model      MobileNetV2 (ImageNet) + 5-class head
        ↓                   phase 1: frozen backbone · phase 2: fine-tuning
Validation                  checkpoint selection + early stopping
        ↓
Testing on unseen images    the test split, never used for training or selection
        ↓
Error analysis              every mistake, with evidence — no invented causes
        ↓
Model improvement           versioned configs (v1, v2, …)
        ↓
Model comparison            side by side, no winner declared
        ↓
Final trained model         models/<version>/model.keras
        ↓
Leaf Lens web application
        ↓
User uploads a photograph
        ↓
Prediction + confidence + probability distribution
```

## Installation

Requires **Python 3.11** (3.10–3.12 should work).

```bash
git clone <this repository>
cd Wheat-lens

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                          # confirm the install
```

The first model build downloads MobileNetV2's ImageNet weights (~9.4 MB) and
caches them in `~/.keras/models/`.

## Quick start

```bash
# 1. Put labelled images in data/raw/<Class_Name>/   (see "Dataset" below)
python -m src.data.validate_dataset          # check them
python -m src.data.prepare_dataset           # split them, leakage-free
python -m src.data.validate_dataset --split-report   # prove no leakage

# 2. Train and evaluate
python -m src.model.train
python -m src.model.evaluate
python -m src.testing.error_analysis

# 3. Use it
python app/app.py                            # http://127.0.0.1:5000
```

## Dataset

**Requirements:** labelled photographs, one folder per class:

```text
data/raw/
├── Healthy/
├── Yellow_Rust/
├── Karnal_Bunt/
├── Brown_Rust/
└── Powdery_Mildew/
```

Accepted formats: JPG, JPEG, PNG, WEBP, BMP. Folder names may use any of the
configured aliases (`leaf_rust/` works for Brown Rust, `stripe_rust/` for
Yellow Rust); `prepare_dataset` resolves them.

**Sources.** No dataset ships with this project and none was downloaded while
building it — the build environment's network policy blocks Kaggle, Hugging
Face, Zenodo and Mendeley. Candidate public datasets, with their documented
classes and URLs, are listed in [`docs/dataset.md`](docs/dataset.md) and printed
by:

```bash
python -m src.data.download_dataset --list
```

Note that **Karnal Bunt is rarely present in public wheat-leaf datasets** — it
is a grain disease with little leaf-level signal. `docs/dataset.md` explains
your options honestly.

**Preparation:**

```bash
python -m src.data.prepare_dataset
```

Drops byte-identical duplicates, groups near-duplicate photographs with a
perceptual hash, and assigns **whole groups** to one split — so a photograph
(or a re-saved variant of it) can never appear in both training and test data.
Default split 75% / 12.5% / 12.5%, seeded with `random_seed: 42`, stratified per
class. Verify with `--split-report`, which exits non-zero on any leakage.

## Training

```bash
python -m src.model.train                              # v1, the default config
python -m src.model.train --config config/model_v2.yaml
python -m src.model.train --epochs 30 --version v3
```

Two-phase transfer learning: the MobileNetV2 backbone is frozen while the
5-class head learns at `1e-4`, then the top 40 backbone layers are unfrozen and
training continues at `1e-5` (BatchNorm layers stay frozen). Callbacks:
`ModelCheckpoint` (best validation accuracy), `EarlyStopping`,
`ReduceLROnPlateau`, `CSVLogger`. Inverse-frequency class weights compensate
for imbalance.

Everything a run produces lands in `models/<version>/`: the model, the class
list, the epoch-by-epoch history, the raw training log, a snapshot of the exact
configuration, and run metadata including seed, library versions and dataset
counts. Training curves go to `results/graphs/`.

Without a prepared dataset the script explains exactly what to do and exits
non-zero. It never fabricates a run.

## Validation and testing

Three distinct measurements, deliberately kept apart:

| Stage | Data | What it tells you |
|---|---|---|
| **Training** | `data/train` | How well the model fits what it learned from. Not evidence about new photographs. |
| **Validation** | `data/validation` | Used to pick the checkpoint and stop early, so it is mildly optimistic. |
| **Test** | `data/test` | Unseen images, never used for training or selection. |

The evaluation report prints all three side by side with these captions, so
training accuracy can never be mistaken for a result.

## Evaluation

```bash
python -m src.model.evaluate
```

Produces overall accuracy, macro and weighted precision/recall/F1, the same per
class, the full 5×5 confusion matrix (counts and row-normalised), mean
confidence split by correct and incorrect, and accuracy above versus below the
confidence threshold.

Outputs:

```text
results/evaluation/evaluation_v1.{json,md}
results/confusion_matrices/confusion_matrix_v1{,_normalised}.png + .csv
results/predictions/test_predictions_v1.csv
results/graphs/per_class_metrics_v1.png
```

## Error analysis

```bash
python -m src.testing.error_analysis
```

Records filename, actual class, predicted class and confidence for every
mistake, then reports error rate per class, all observed confusion pairs,
**confident mistakes** (errors above the threshold — the ones the UI would not
have warned about), a confidence histogram, a labelled grid of the
misclassified images, and objective image statistics (brightness, contrast,
sharpness, saturation, resolution) compared between correct and incorrect
predictions.

Differences are flagged only when they exceed one standard deviation of the
correct group, and the report says plainly that these are correlations on a
small sample, not demonstrated causes. Factors it cannot settle — background,
distance, leaf angle, partial leaves — are listed as open questions.

## Realistic testing

The test split shares a distribution with the training images. This step asks
the harder question: does accuracy survive dim light, cluttered backgrounds,
distance and awkward angles?

```bash
python -m src.testing.realistic_testing --template   # writes the manifest
# put your photographs in data/realistic/ and fill in the manifest
python -m src.testing.realistic_testing
```

`data/realistic/manifest.csv`:

```csv
file,actual_class,lighting,background,distance,angle,quality,notes
img_001.jpg,Yellow Rust,low,soil,close,straight,high,
img_002.jpg,Healthy,bright,plain,far,angled,medium,windy day
```

Conditions: lighting (normal/low/bright), background (plain/soil/plant),
distance (close/medium/far), angle (straight/angled), quality
(high/medium/low). `actual_class` may be blank — those images are predicted and
listed but not scored.

The report breaks accuracy down by every condition, always with the sample size
beside it.

## Model improvement and comparison

Model versions are configuration files, not code branches.
`config/model_v2.yaml` contains only the keys that differ from the base config
(more dropout, label smoothing, more epochs, deeper unfreeze, stronger
geometric augmentation) and is deep-merged onto it.

```bash
python -m src.model.train    --config config/model_v2.yaml
python -m src.model.evaluate --config config/model_v2.yaml
python -m src.model.compare_models v1 v2
```

The comparison shows overall metrics, per-class F1 and recall, both confusion
matrices, training figures, realistic-testing results, and a diff of the
configuration settings that changed. **It does not declare a winner** — it ends
with what to check first: the test-set size, whether a gain is spread across
classes or concentrated in one, whether both versions used the same split, and
whether realistic testing agrees.

## Web application

```bash
python app/app.py                       # http://127.0.0.1:5000
python app/app.py --host 0.0.0.0 --port 8080
```

Upload a photograph → preview it → press **PREDICT** → see the predicted
category, the confidence, and the probability across all five classes:

```text
RESULT

Predicted Category:     Yellow Rust
Confidence:             89.2%

Probability Distribution:
  Healthy            2.1%
  Yellow Rust       89.2%
  Karnal Bunt        3.7%
  Brown Rust         4.1%
  Powdery Mildew     0.9%
```

*(Layout example only — the numbers shown by the application come from the
trained model.)*

Features: drag-and-drop or click upload, client- and server-side file
validation, image preview, loading indicator, confidence warning below the
configured threshold, an uncertainty notice with its own stated limitations,
reset / upload-another, and friendly JSON error messages — never a Python stack
trace. Supported formats: JPG, JPEG, PNG, WEBP, BMP, up to 10 MB.

If no model is trained the page still loads, explains the situation and
disables the predict button.

**API**

| Endpoint | Purpose |
|---|---|
| `GET /` | The interface |
| `GET /api/status` | Whether a model is loaded, its version, class list and threshold |
| `POST /api/predict` | Multipart upload (`image`) → prediction JSON |

> **The interface is a prototype, not a diagnostic tool.** Predictions may be
> incorrect. Confidence is the model's estimated probability for its own
> prediction, not a guarantee. See [`docs/limitations.md`](docs/limitations.md).

## Command reference

```bash
# --- install -------------------------------------------------------------
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# --- dataset -------------------------------------------------------------
python -m src.data.download_dataset --list                  # candidate sources
python -m src.data.download_dataset --kaggle <owner/slug>   # needs kaggle CLI + token
python -m src.data.download_dataset --inspect data/downloads/<folder>
python -m src.data.validate_dataset                         # validate data/raw
python -m src.data.prepare_dataset                          # build the splits
python -m src.data.prepare_dataset --dry-run                # preview only
python -m src.data.prepare_dataset --link                   # hard-link, saves disk
python -m src.data.validate_dataset --split-report          # leakage check

# --- training ------------------------------------------------------------
python -m src.model.train
python -m src.model.train --config config/model_v2.yaml
python -m src.model.train --epochs 30 --version v3 --no-fine-tune

# --- evaluation ----------------------------------------------------------
python -m src.model.evaluate
python -m src.model.evaluate --split validation
python -m src.testing.error_analysis
python -m src.model.compare_models v1 v2
python -m src.model.compare_models --all

# --- realistic testing ---------------------------------------------------
python -m src.testing.realistic_testing --template
python -m src.testing.realistic_testing

# --- prediction ----------------------------------------------------------
python -m src.model.predict path/to/leaf.jpg
python -m src.model.predict *.jpg --json --output results/predictions/batch.json
python -m src.model.predict --info

# --- web application -----------------------------------------------------
python app/app.py
python app/app.py --host 0.0.0.0 --port 8080

# --- tests ---------------------------------------------------------------
pytest -q
python tests/make_synthetic_fixture.py --root /tmp/fixture/raw --per-class 40
```

## Project structure

```text
Wheat-lens/
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── config.yaml              single source of truth: classes, paths, hyper-parameters
│   └── model_v2.yaml            V2 variant (only the keys that differ)
│
├── data/
│   ├── raw/                     YOUR IMAGES GO HERE, one folder per class
│   ├── train/ validation/ test/ generated by prepare_dataset
│   └── realistic/               field photographs + manifest.csv
│
├── src/
│   ├── data/
│   │   ├── download_dataset.py  candidate sources; optional Kaggle download
│   │   ├── validate_dataset.py  corruption, counts, duplicates, imbalance, leakage
│   │   └── prepare_dataset.py   near-duplicate grouping + leakage-free split
│   ├── model/
│   │   ├── build_model.py       MobileNetV2 + augmentation + preprocessing + head
│   │   ├── datasets.py          tf.data pipelines in config class order
│   │   ├── train.py             two-phase training, callbacks, artefacts, graphs
│   │   ├── evaluate.py          metrics, confusion matrices, per-image CSV
│   │   ├── predict.py           the Predictor used by CLI, web app and tests
│   │   └── compare_models.py    version comparison, no winner declared
│   ├── testing/
│   │   ├── error_analysis.py    every mistake, with evidence
│   │   └── realistic_testing.py condition-based field testing
│   └── utils/
│       ├── config.py            typed config loader with alias resolution
│       ├── helpers.py           logging, seeding, JSON/markdown, plotting defaults
│       └── image_io.py          validation, preprocessing, hashing, grids
│
├── models/<version>/            model.keras, labels.json, history, config snapshot
├── results/                     evaluation, confusion_matrices, graphs, predictions,
│                                error_analysis, realistic_testing, comparison, dataset
│
├── app/
│   ├── app.py                   Flask API + page
│   ├── templates/index.html
│   └── static/{style.css,script.js}
│
├── tests/                       pytest suite + synthetic fixture generator
└── docs/
    ├── dataset.md   model.md   testing.md   limitations.md   reproducibility.md
```

## Configuration

Everything configurable lives in `config/config.yaml`: class names and folder
aliases, dataset paths, image size, batch size, epochs, learning rates,
optimizer, augmentation strengths, split fractions, duplicate-detection
thresholds, class-imbalance warnings, confidence threshold, uncertainty
thresholds and the random seed.

```yaml
classes:
  - name: "Healthy"
    directory: "Healthy"
  # ... four more

data:
  image_size: 224
  batch_size: 32
  split: { train: 0.75, validation: 0.125, test: 0.125 }

training:
  epochs: 20
  learning_rate: 0.0001

inference:
  confidence_threshold: 0.60

random_seed: 42
```

A copy of the effective configuration is saved with every trained model as
`models/<version>/config_snapshot.yaml`, so a run can always be reproduced.

## Limitations

The short version — the full discussion is in
[`docs/limitations.md`](docs/limitations.md):

* **Five classes only, and no way to say "something else".** Any other
  condition, crop or subject is still forced into one of the five, sometimes
  confidently.
* **The uncertainty check is a softmax heuristic, not a novelty detector.** An
  unflagged prediction is not a verified one.
* **Confidence is not correctness.** Networks trained on small datasets are
  typically over-confident, and no calibration is applied.
* **Test accuracy does not predict field accuracy.** Curated images are easier
  than real photographs; that is what realistic testing exists to measure.
* **Karnal Bunt has little leaf-level signal** — it is a grain disease. Treat
  that class with particular scepticism.
* **Not a diagnostic tool.** Do not use it for treatment, certification or
  quarantine decisions. Karnal Bunt is a regulated pathogen in many
  jurisdictions and requires laboratory confirmation.

## Troubleshooting

**`TRAINING CANNOT PROCEED: no images found in data/train`**
Run `python -m src.data.prepare_dataset` first — and before that, put images in
`data/raw/<Class>/`.

**`no usable images for: Healthy, Yellow Rust, ...`**
`data/raw/` is empty or the folder names do not match. Run
`python -m src.data.validate_dataset` for the exact expected paths.

**`no trained model found at models/v1/model.keras`**
Train one: `python -m src.model.train`. The web app and CLI both degrade
gracefully until then.

**`could not construct MobileNetV2 with weights='imagenet'`**
The ImageNet weights could not be downloaded. Check network access to
`storage.googleapis.com`, or set `model.weights: none` — at a real cost in
accuracy on a small dataset.

**`model at ... has N outputs but M class names are configured`**
The model was trained with a different class list. Retrain, or restore the
`labels.json` that belongs with it.

**Near-duplicate scan skipped**
Either the dataset exceeds the O(n²) limit (raise `--max-duplicate-images`) or
`imagehash` is not installed (`pip install ImageHash`). Without it, leakage
protection falls back to exact-duplicate detection only.

**`class 'X' ended up with 0 images in 'test'`**
That class has too few images, or nearly all of them are near-duplicates of
each other. Add more distinct photographs.

**Training is slow**
Expected on CPU. Reduce `data.image_size` to 160 or 96, lower `epochs`, or set
`training.fine_tune.enabled: false` for a quick baseline.

**`ImportError: numpy.core.multiarray failed to import`**
NumPy 2.x with TensorFlow 2.17. `pip install "numpy<2.0"`.

## What you need to provide

The pipeline is finished; these are the inputs only you can supply.

1. **A labelled wheat-leaf dataset** in `data/raw/<Class_Name>/`, covering all
   five classes. This is the blocking item — nothing downstream can run without
   it. Start with `python -m src.data.download_dataset --list` and read
   `docs/dataset.md`, particularly the section on Karnal Bunt.
2. **Run the training pipeline** once the images are in place (validate →
   prepare → split-report → train → evaluate → error analysis).
3. **Field photographs for realistic testing** — your own pictures under
   varied lighting, backgrounds, distances and angles, described in
   `data/realistic/manifest.csv`. Without these, nothing is known about how the
   model behaves outside curated images.
4. **Fill in the result tables** in `docs/dataset.md`, `docs/model.md` and
   `docs/testing.md` from the generated reports. They currently read *Not yet
   provided* and *Not yet tested* on purpose.

---

**Leaf Lens is an experimental machine-learning prototype. It is not a
professional agricultural diagnostic tool. Predictions may be incorrect.**
