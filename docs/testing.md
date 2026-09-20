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

### The suite runs offline

Model tests construct MobileNetV2 through `build_model`, whose production
configuration asks for ImageNet weights. Downloading those on every test run
would make the whole suite depend on network access and on a warm cache.

The tests therefore pass `weights=None` explicitly, via `offline_safe_config`
in `tests/conftest.py`. That is a test double, not a weakened assertion: the
properties under test — output shape, probability normalisation, preprocessing
inside the graph, augmentation inactive at inference, freezing and unfreezing,
loss and metrics, label ordering — are all determined by how the graph is
assembled, not by the numbers in the backbone.

The production path keeps its own coverage:

| Test | What it checks | When it runs |
|---|---|---|
| `test_imagenet_weights_path_builds` | The production config builds, the backbone carries non-zero learned weights, and the graph behaves as the offline tests assert | When the weights are cached or downloadable |
| `test_random_backbone_differs_from_pretrained` | `weights=None` and `weights='imagenet'` produce genuinely different parameters — guards against the offline mode silently becoming the only mode | Same |

Both skip with an actionable message when the weights cannot be obtained, so an
air-gapped machine sees `90 passed, 2 skipped` rather than 30 errors.

Two further tests deserve special mention.

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

Model v1, test split (617 unseen images), evaluated 2026-09-20.

| Metric | Value |
|---|---|
| Accuracy | **100.00%** |
| Macro / weighted F1 | 1.0000 |
| Errors | 0 of 617 |
| Mean confidence | 99.8% |

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Healthy | 194 | 1.0000 | 1.0000 | 1.0000 |
| Yellow Rust | 26 | 1.0000 | 1.0000 | 1.0000 |
| Brown Rust | 206 | 1.0000 | 1.0000 | 1.0000 |
| Powdery Mildew | 191 | 1.0000 | 1.0000 | 1.0000 |

The confusion matrix is diagonal. Error analysis has nothing to analyse.

### This is a warning, not an achievement

A perfect confusion matrix on a four-class disease problem means something is
trivially separable. Four checks were run to find out what.

**1. Is it leakage?** No. Two independent checks. `validate_dataset
--split-report` re-hashes every prepared file and found no image in more than
one split. A nearest-neighbour analysis over 256-bit perceptual hashes found the
closest training neighbour of any test image at Hamming distance 10, with a
median of 40 — nothing remotely duplicate-like.

**2. Is it the resolution confound?** **No — this prediction was wrong.**
Yellow Rust images are 24 MP while every other class is 0.16 MP, with no
overlap, so resolution looked like an obvious shortcut. `confound_test.py`
downsampled every test image to a common 400 px and re-encoded it. **Every
class held 100% recall.** Resolution is ruled out as the mechanism. Noted here
because the hypothesis was stated before the test and the test refuted it.

*Caveat on that test:* both paths end at the model's 224×224 input, so
normalising to 400 px first is a milder intervention than it sounds. It rules
out a gross scale artefact, not every possible source signature.

**3. Is the benchmark easy?** **Yes. This is the answer.**
`python -m src.testing.baseline_control` fits deliberately weak models on
deliberately crude features:

| Features | Count | Model | Test accuracy |
|---|---|---|---|
| Colour statistics | 7 | Logistic regression | **98.54%** |
| Colour statistics | 7 | Random forest | 98.22% |
| Grayscale statistics | 3 | Random forest | 69.37% |
| — | — | Chance | 25.0% |

Seven numbers — per-channel mean and standard deviation plus overall
brightness, on a 32×32 thumbnail, with no notion of texture, shape or spatial
arrangement — reach 98.54%. **The network contributes +1.46 percentage points
over that.**

**4. Where does the signal come from?** Colour. Dropping to grayscale takes the
baseline from 98.5% to 69.4%. That is consistent with real disease biology —
rust is brown or yellow, mildew is white, healthy is green — and with images
curated tightly enough that average hue nearly determines the class.

### What this means for the headline number

The 100% is real, honestly obtained, and reproducible on this data. It is also
close to meaningless as a predictor of field performance. The images are single
leaves, cropped, evenly lit, mostly on plain backgrounds. Under those
conditions the problem reduces to "what colour is this image", which a linear
model solves. Field photographs have variable lighting, soil and foliage
backgrounds, multiple leaves at varying distances and partial symptoms — and
global colour statistics stop being informative.

**Treat 100% as an upper bound under ideal conditions, not an accuracy
estimate.** The number that matters is the one from realistic-condition
testing, which requires field photographs nobody has supplied yet.

### Background robustness — the metric that matters

```bash
python -m src.testing.background_robustness
```

Classifies the test split twice: unchanged, and with the background replaced by
a patch from the field-photography class. The leaf is identical in both, so any
accuracy lost depended on the background.

Measured for v1 on 591 test images / 1,170 composites:

| Condition | Accuracy |
|---|---|
| Original background | 100.0% |
| Background replaced | **21.5%** |
| **Lost to the background** | **78.5 pp** |

Backgrounds are taken from the **test** split, so the evaluation cannot be
gamed by the v2 training augmentation, which harvests from the training split.

See [`v2_experiment.md`](v2_experiment.md).

### Still not tested

| Artefact | Status |
|---|---|
| Realistic-condition testing | **Not yet tested** — requires field photographs |
| V1 vs V2 comparison | Not yet run — only v1 exists |
| Any performance claim outside this dataset | Unsupported |

