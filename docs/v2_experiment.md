# The v2 experiment: breaking the background confound

## Status

**Infrastructure is built. No v2 model has been trained.** `models/v1/model.keras`
and every v1 result are unchanged. Every figure below for v1 is measured; every
statement about v2 is a hypothesis awaiting a run.

## The v1 confound

Model v1 scores **100% on its test split** and predicts **Yellow Rust for most
wheat photographs taken from anywhere else**. Both are true, and the second
explains what the first is worth.

The cause is in the data, not the model. The four classes differ by
*photographic setting* as much as by disease:

| Class | Source | What the photographs are |
|---|---|---|
| Healthy | WPLDD | One leaf on a plain pale background, evenly lit, close up |
| Brown Rust | WPLDD | Same — many frames are mostly background with a sliver of leaf |
| Powdery Mildew | WPLDD | Same |
| **Yellow Rust** | a different repository | **Field photographs** — soil, surrounding vegetation, outdoor light, a person's hand in frame |

"Field photograph" therefore predicts "Yellow Rust" almost perfectly in
training, and that is a far easier rule to learn than leaf pathology. An image
from the internet is almost always a field photograph, so it matches Yellow
Rust whatever is actually on the leaf.

## The causal evidence

This is not inferred from the accuracy figure — it was tested directly.
Background patches were cropped from the Yellow Rust images and composited
behind leaves of the other three classes. **The leaf, and therefore the
disease, is identical in both conditions; only the background differs.**

Measured on 591 unseen test images, 1,170 composites, with backgrounds taken
from the **test** split so nothing from training was reused:

| Condition | Accuracy | |
|---|---|---|
| Original background | **100.0%** | |
| Background replaced | **21.5%** | |
| **Accuracy lost to the background** | **78.5 pp** | |

| Class | Original | Background replaced | Change |
|---|---|---|---|
| Healthy | 100.0% | 7.8% | **−92.2 pp** |
| Brown Rust | 100.0% | 33.9% | −66.1 pp |
| Powdery Mildew | 100.0% | 22.1% | −77.9 pp |

* **71.8%** of composites were called Yellow Rust.
* Mean confidence **while wrong: 89.0%** — far above the 0.60 warning
  threshold, so the interface shows no caution at all.

Reproduce with `python -m src.testing.background_robustness`.

### What was ruled out

| Hypothesis | Verdict |
|---|---|
| Label mapping wrong | Ruled out — config, `labels.json`, the predictor and the folders the training loader read all agree index for index |
| Inference preprocessing differs from training | Ruled out — preprocessing is a layer inside the saved graph; `preprocess(0) = −1.0`, `preprocess(255) = +1.0`, augmentation inactive at inference, two passes bit-identical |
| Colour alone | Ruled out — shifting brightness and saturation of studio images across a wide range never flipped a single prediction to Yellow Rust |
| A general Yellow Rust attractor | Ruled out — on noise, flat colours and gradients the model collapses onto **Brown Rust**. The bias is specific to field-photograph structure |
| Class imbalance (7.89:1, 2.86× class weight) | Contributing, not causal — the background experiment isolates the mechanism with imbalance held constant |

## Why retraining on the v1 data would not fix it

The confound is a property of the dataset, not of the architecture or the
hyper-parameters. In the v1 training set, background predicts class almost
perfectly. Gradient descent will find that rule under any architecture, any
learning rate, any amount of regularisation, because it is the lowest-loss
solution available — and on the v1 test split, which shares the confound, it is
also the *correct* solution. Nothing in the training signal penalises it.

Changing the model therefore cannot help. The training distribution has to
change.

## What v2 tests

**Hypothesis:** if the studio classes are given varied backgrounds, background
stops carrying class information, and the network is forced onto the leaf
instead.

**The measurement that matters is not test accuracy.** v1 already scores 100%
there and is still broken. v2 is judged on **background robustness** — accuracy
on test images whose backgrounds have been replaced — where v1 scores 21.5%.

A v2 that scores 95% on the test split and 60% on background robustness is a
better model than v1, despite the lower headline number.

### How the pipeline works

```
data/raw/                    original photographs, untouched
      ↓  prepare_dataset             leakage-safe split, unchanged from v1
data/train  data/validation  data/test
      ↓  augment_backgrounds         ← v2 only; writes into data/train ONLY
data/train + generated copies
      ↓  train --config config/model_v2.yaml
models/v2/
      ↓  evaluate                    same test split as v1, so comparable
      ↓  background_robustness       the metric that decides the experiment
```

For each selected image the augmenter segments the leaf from its pale
background, then composites it onto a different background at a random scale
and position. Backgrounds come from two sources:

* **harvested** — real patches cropped from the field-photography class,
  *training split only*. These are the exact backgrounds the model was keying
  on, so they attack the correlation directly.
* **procedural** — synthetic soil, vegetation, sky and straw textures generated
  from noise. Depends on no external data.

### Leakage prevention

Four mechanisms, all enforced in code and covered by tests:

1. **Augmentation is a post-split step by construction.** It refuses to run if
   `data/train` does not exist, directing you to `prepare_dataset` first. There
   is no code path that augments before splitting.
2. **It writes only into `data/train`.** Validation and test are never opened
   for writing. A test asserts their file counts are unchanged across a run.
3. **Backgrounds are harvested only from the training split.** A test spies on
   every file opened during harvesting and asserts each one is under
   `data/train`.
4. **Generated files are marked.** Every one is named `aug__<copy>__<original>`,
   so `--verify` can assert that no file carrying the marker exists in
   validation or test, and `--clean` can remove generated files without ever
   deleting an original.

```bash
python -m src.data.augment_backgrounds --verify
```

The underlying split is unchanged from v1 — same seed, same near-duplicate
grouping — so v1 and v2 are evaluated on **identical** validation and test
sets, and the comparison is meaningful.

### Reproducibility and provenance

The augmentation seed is fixed in configuration (`seed: 42`). A test regenerates
the whole set twice and asserts the files are byte-identical.

Every generated filename embeds its original, which itself still carries its
source tag: `aug__0__wpldd__zc1005.jpg` is copy 0, derived from
`wpldd__zc1005.jpg`, which came from WPLDD. A manifest at
`results/augmentation/augmentation_manifest.json` records each pairing.

## Realistic-condition testing

Two independent workflows, neither of which shares images with training.

**1. Background robustness** — automated, repeatable, no new data:

```bash
python -m src.testing.background_robustness --version v2
```

Uses the test split with backgrounds replaced. Crucially, the backgrounds come
from the **test** split of the field class, while v2's training augmentation
harvests from the **training** split. The evaluation therefore cannot be gamed
by the augmentation it is measuring.

**2. Field photographs** — the real check, requires your own images:

```bash
python -m src.testing.realistic_testing --template
# put photographs in data/realistic/, fill in the manifest
python -m src.testing.realistic_testing --version v2
```

`data/realistic/` is never read by `prepare_dataset` or by training, so these
images are excluded from training by directory layout rather than by
convention. Conditions are recorded per photograph (lighting, background,
distance, angle, quality) and accuracy is broken down by each, with sample
sizes shown.

## Limitations of synthetic background augmentation

These are real and should temper expectations. v2 is an experiment, not a fix.

**Segmentation is a brightness-and-saturation threshold.** It works on the
studio images because their background is genuinely pale and flat. It will clip
pale diseased tissue in places — chlorotic margins, heavy mildew coating — and
leave halo artefacts at the leaf edge. Those artefacts are themselves a new,
consistent signal the network could learn.

**Composites are not photographs.** A cut-out leaf pasted onto a background has
wrong lighting, no shadow, no depth of field and a hard boundary. The network
may learn "has a pasted-on look" as a proxy, which would help on the robustness
metric while not helping on real photographs at all. This is the most important
limitation: **a good background-robustness score is necessary but not
sufficient.**

**The leaves are still studio leaves.** Augmentation changes the background, not
the subject. Real field photographs differ in focus, motion blur, occlusion,
multiple overlapping leaves, viewing angle and growth stage. None of that is
addressed.

**Yellow Rust remains the only genuinely field-photographed class**, and still
has only 208 images against ~1,500 for the others. The 7.89:1 imbalance is
unchanged.

**The honest fix is real field photographs of the studio classes.** A few
hundred per class of Healthy, Brown Rust and Powdery Mildew photographed
outdoors, in situ, would remove the confound at its root rather than
approximating around it. v2 is worth running because it is cheap and it
measures something real — but it is a mitigation, and its own metric cannot
tell you whether it generalises to photographs.

## Commands

```bash
# one-off: split (unchanged from v1)
python -m src.data.prepare_dataset

# v2 preparation
python -m src.data.augment_backgrounds --config config/model_v2.yaml --dry-run
python -m src.data.augment_backgrounds --config config/model_v2.yaml
python -m src.data.augment_backgrounds --verify          # leakage check
python -m src.data.augment_backgrounds --clean           # undo

# v2 training and assessment  (NOT YET RUN)
python -m src.model.train    --config config/model_v2.yaml
python -m src.model.evaluate --config config/model_v2.yaml
python -m src.testing.background_robustness --version v2
python -m src.model.compare_models v1 v2
```

`--clean` removes only files carrying the generated marker, so it can never
delete an original photograph.
