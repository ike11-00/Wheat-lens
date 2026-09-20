# Testing

Two different things are called "testing" in this project. They are kept apart
on purpose.

1. **Software testing** — does the code work? Run with `pytest`. Complete and
   passing.
2. **Model testing** — does the classifier work on wheat leaves? Requires a
   real dataset. **Not yet done.**

## 1. Software testing

```bash
pytest -q                    # whole suite
pytest tests/test_model.py -v
```

| File | Covers |
|---|---|
| `tests/test_config.py` | Class list and order, alias resolution, split fractions, path resolution, variant config merging, extensibility to a sixth class, rejection of invalid configs |
| `tests/test_image_io.py` | Valid/corrupt/truncated/empty/oversmall/unsupported files, RGB conversion, preprocessing shape and range, SHA-256 and perceptual hashing |
| `tests/test_dataset_tools.py` | Missing classes, corrupt-file reporting, duplicate detection, imbalance warning, alias folders, split determinism, dry run — and the leakage guarantee |
| `tests/test_model.py` | Output shape, probability normalisation, in-graph preprocessing, augmentation inactive at inference, freeze/unfreeze, class weights, label ordering, empty-split errors |
| `tests/test_predict.py` | Missing-model handling, `labels.json` authority, result schema, determinism, batch/single agreement, threshold behaviour, uncertainty signals, class-count mismatch |
| `tests/test_app.py` | Page rendering, status endpoint, prediction endpoint, every supported format, rejection of unsupported/corrupt/empty/tiny/oversized uploads, JSON errors with no stack traces |

Two tests deserve special mention.

**`test_near_duplicates_never_straddle_splits`** is the leakage guarantee. It
creates near-identical re-saves of one image, runs the real splitter, and
asserts that every perceptual-hash group appears in exactly one split.

**`test_augmentation_is_inactive_at_inference`** asserts that two forward
passes over the same batch return identical probabilities — if augmentation
leaked into inference, predictions would be non-deterministic.

### Pipeline verification with a synthetic fixture

`tests/make_synthetic_fixture.py` generates procedurally drawn abstract images
— coloured noise with stripes, dots, rings, streaks and blobs. They are **not**
wheat leaves and carry no disease information.

```bash
python tests/make_synthetic_fixture.py --root /tmp/leaf_lens_fixture/raw --per-class 40
```

Their only purpose is to execute the full chain — prepare → train → evaluate →
error analysis → realistic testing → comparison → web prediction — and confirm
every artefact is produced correctly. **Any metric obtained from this fixture
describes the model's ability to tell synthetic textures apart and must never be
quoted as a Leaf Lens result.** The generated folder carries a
`SYNTHETIC_FIXTURE_README.txt` saying exactly that.

## 2. Model testing

### Unseen images

The test split is genuinely unseen: it is never used for training, and
`EarlyStopping`/`ModelCheckpoint` select on the *validation* split, not on it.
Near-duplicate grouping (see `docs/dataset.md`) guarantees that a training
photograph cannot reappear in it in a slightly altered form.

Verify before trusting any test number:

```bash
python -m src.data.validate_dataset --split-report
```

This exits non-zero if any image appears in more than one split.

### Metrics

```bash
python -m src.model.evaluate
```

Computes, on the test split:

* overall accuracy
* macro and weighted precision, recall and F1
* per-class precision, recall, F1 and support
* the full N×N confusion matrix (counts and row-normalised)
* mean confidence, split by correct and incorrect predictions
* accuracy above and below the confidence threshold, with coverage

Outputs land in `results/evaluation/`, `results/confusion_matrices/`,
`results/predictions/` and `results/graphs/`.

The report's first table separates the three numbers explicitly:

| Stage | What it measures |
|---|---|
| Training accuracy | Fit on images the model learned from. Not evidence of anything about new photographs. |
| Validation accuracy | Used to choose the checkpoint and stop early, so it is mildly optimistic. |
| **Test accuracy** | **Unseen images, never used for training or model selection.** |

Never quote training accuracy as the model's performance.

### Confusion matrix

Rows are the actual class, columns the prediction, over every configured class. The
evaluator also lists the most frequent off-diagonal pairs with their share of
the actual class.

Pairs worth *checking* — because they are visually plausible, not because they
have been observed here:

* Yellow Rust vs Brown Rust (both rusts; pustule colour and arrangement differ)
* Healthy vs early-stage disease (few or faint lesions)
* Powdery Mildew vs Healthy (thin mildew coating on a green leaf)

**The reports only state a confusion when the test results actually show it.**
Nothing in this project assumes these errors occur.

### Error analysis

```bash
python -m src.testing.error_analysis
```

For every incorrect test prediction it records filename, actual class,
predicted class and confidence, then reports:

* error rate per class and the class each one is most often mistaken for
* every confusion pair observed
* **confident mistakes** — errors made at or above the threshold, which matter
  most because the interface would not have warned the user
* a confidence histogram for correct versus incorrect predictions
* a labelled grid of the misclassified images
* objective image statistics (brightness, contrast, sharpness, saturation,
  resolution, aspect ratio) compared between correct and incorrect predictions

A statistic is flagged only when the incorrect group differs from the correct
group by more than one standard deviation of the correct group. Even then the
report states plainly that this is a correlation on a small sample, not a
demonstrated cause. Factors it explicitly cannot settle — background clutter,
camera distance, leaf angle, partial leaves, overlapping symptoms — are listed
as open questions with a pointer to realistic testing.

### Realistic-condition testing

The test split measures accuracy on images from the same pool as training.
Realistic testing measures something closer to actual use.

```bash
python -m src.testing.realistic_testing --template   # create the manifest
# ... fill it in ...
python -m src.testing.realistic_testing
```

Put photographs in `data/realistic/` and describe each one in
`data/realistic/manifest.csv`:

```csv
file,actual_class,lighting,background,distance,angle,quality,notes
img_001.jpg,Yellow Rust,low,soil,close,straight,high,
img_002.jpg,Healthy,bright,plain,far,angled,medium,windy day
```

| Dimension | Values |
|---|---|
| lighting | normal, low, bright |
| background | plain, soil, plant |
| distance | close, medium, far |
| angle | straight, angled |
| quality | high, medium, low |

`actual_class` may be left blank; those images are still predicted and listed
but excluded from accuracy. Alternatively, skip the manifest and encode
conditions in folder names: `data/realistic/lighting=low/distance=far/Yellow_Rust/`.

The report gives overall accuracy, per-class accuracy, a breakdown by every
condition dimension with sample sizes, a full results table, and a chart. Every
number carries its `n`, because a "60% accuracy in low light" computed from
five photographs means nothing.

A drop under one condition is a signal to investigate, not proof of cause — the
photographs differ in other ways too. To establish a cause, photograph the same
leaves while varying one condition at a time.

### Model comparison

```bash
python -m src.model.compare_models v1 v2
python -m src.model.compare_models --all
```

Compares overall metrics, per-class F1 and recall, confusion matrices, training
figures, realistic-testing results, and diffs the tracked configuration
settings so you can see exactly what changed.

**It does not name a winner.** The report ends with what to check before
concluding a version is genuinely better: the test-set size, whether the gain
is spread across classes or concentrated in one, whether both versions were
evaluated on the *same* split, and whether the realistic-condition results
agree.

## Current results

| Artefact | Status |
|---|---|
| Test accuracy | Not yet tested — requires dataset |
| Per-class precision / recall / F1 | Not yet tested — requires dataset |
| Confusion matrix | Not yet tested — requires dataset |
| Error analysis | Not yet tested — requires dataset |
| Realistic-condition testing | Not yet tested — requires dataset and field photographs |
| V1 vs V2 comparison | Not yet tested — requires dataset |
| Software test suite | **Passing** |
| Full pipeline execution | **Verified on a synthetic fixture** |
