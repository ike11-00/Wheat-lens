# The v3 experiment: does source diversity fix real-world accuracy?

## Status

**PROPOSAL. Nothing has been trained. No dataset has been acquired for v3 yet.**

This document exists because the dataset decision was required before training,
and it is the thing to approve or change. Every number below is either measured
and labelled as such, or projected from observed folder listings and labelled as
such. Nothing in here is an estimate dressed up as a result.

### v1 and v2 are untouched

Verified at the time of writing:

| Model | File | md5 | Artefacts |
|---|---|---|---|
| v1 | `models/v1/model.keras` | `2ae53135f16d3de4d77dd61833fe2bad` | 8 of 8 present |
| v2 | `models/v2/model.keras` | `8773a3305bebafc7e84487b0790b2989` | 8 of 8 present |

Their results in `results/` are unchanged. The preservation rules v3 will follow
are in [Not breaking v1 and v2](#not-breaking-v1-and-v2) below.

---

## 1. What v3 is asking

v2 fixed the confound it was built to fix, and it is worth being precise about
what that did and did not buy:

| Measurement | v1 | v2 |
|---|---|---|
| Test accuracy (same benchmark) | 100.00% | 99.68% |
| Background robustness | 21.5% | **95.3%** |
| Composites called Yellow Rust | 71.8% | 0.7% |
| **Your 20 real photographs** (reported by you) | — | **9/20 = 45%**, Brown Rust **0/5** |

So the background confound was real, and removing it was not sufficient. A model
can be background-invariant and still wrong, because background was only one of
the things that made the benchmark easy.

The remaining structural weakness is visible in the v1/v2 dataset itself:

| Class | v1/v2 images | Sources | Imagery |
|---|---|---|---|
| Healthy | 1,545 | 1 (WPLDD) | studio |
| Yellow Rust | 208 | 1 (hg3817) | field |
| Brown Rust | 1,642 | 1 (WPLDD) | studio |
| Powdery Mildew | 1,526 | 1 (WPLDD) | studio |

Three of four classes came from **one collection**, photographed one way. Any
regularity of that collection — lens, lighting rig, leaf presentation, JPEG
pipeline, growth stage, cultivar — is perfectly correlated with class, and the
model is free to use it. Background augmentation removed one such regularity. It
could not remove the ones nobody has named.

**The v3 hypothesis:** drawing each class from several independent collections
removes the regularities *without having to identify them first*, because a
feature has to survive across collections to remain predictive. If that is true,
v3 should generalise better than v2 to photographs from neither collection. If it
is false, v3 will match or lose to v2 and that will be reported as the result.

This is a hypothesis with a measurable outcome, not a promise. It may fail.

---

## 2. What is actually obtainable here

### Reachability

Kaggle, Hugging Face, Zenodo and Mendeley all return **403** through this
environment's egress gateway. That rules out most of the large field-photography
wheat datasets described in the literature. GitHub and PyPI are reachable.
The survey below is therefore *exhaustive of GitHub*, not exhaustive of the world.
If the environment's network policy is widened later, this survey should be redone
— that is the single highest-value change available to this project.

### Sources that will be used

All counts were taken by listing the repositories on the stated dates and are
recorded in `src/data/sources_v3.py`, which is the machine-readable version of
this table.

| Key | Repository | Licence | Imagery | Contributes |
|---|---|---|---|---|
| `wpldd` | `cyb-personal/VWLM-for-Wheat-Disease-Identification-` | **none found** | studio | Healthy, Brown Rust, Powdery Mildew |
| `hg3817` | `Himanshu-Gupta3817/Wheat_plant_disease_detection` | **none found** | field | Yellow Rust, Healthy |
| `suhas` | `suhasmaddali/Wheat-Disease-Detection-` | **none found** | mixed | Brown Rust, Healthy |
| `mubashar` | `mubashar1030/WheatDiseaseDataset` | **none found** | mixed | Brown Rust, Healthy |
| `kiran` | `kaliprogramer/Wheat-Plant-Disease-Classification-using-Deep-Learning-ResNet18` | **MIT** (see below) | mixed | all four classes |

**On licensing, plainly:** four of these five repositories carry no LICENSE file
at all, which under default copyright means all rights reserved. You instructed
the project to proceed with unlicensed GitHub imagery, so it does. That decision
is recorded here and in `docs/limitations.md` rather than glossed over, and it is
the reason this model is not distributable.

The `kiran` repository does carry an MIT licence (`Copyright (c) 2026 Kiran K.C`).
That is the only licence in the project. The honest caveat: MIT covers "the
Software", and the repository does not state whether the author intended it to
cover the bundled images, nor where those images originally came from. It is
better than nothing and it is not unambiguous.

**"Imagery" is my own coarse judgement** from looking at the folders — studio
means single leaf on a plain background, field means in-situ photography, mixed
means both appear. It is not metadata supplied by the authors.

### Sources surveyed and deliberately rejected

Rejections matter as much as inclusions, because "five sources" would be a lie if
two of them were the same data.

| Key | Repository | Why not |
|---|---|---|
| `aadium` | `aadium/wheat-disease-detection` | Redistributes the same data as `suhas` — identical folder counts *and* identical filenames (`00011.jpg`, `00021.jpg`, …). Counting both would be duplication presented as diversity, which is precisely what v3 is testing. |
| `yr2223` | `Shant-Thakur/YR-22-23` | Named as a yellow-rust dataset, but the upload is broken: 1,193 images in `train/healthy`, **1** in `train/rust`, **1** in `test/rust`. |
| `swinsharp` | `SWIN-SHARP/SWIN-SHARP` | README advertises a five-class dataset; the repository holds 60 images total — sample figures, not the dataset. |
| `anushrii` | `anushriiverse/Wheat_disease_detection` | Binary `healthy`/`unhealthy` only. "Unhealthy" carries no disease label and cannot be mapped to any configured class. Mixing it in would mean training on unknown labels. |
| `goodhope` | `GoodhopeKD/Wheat-Disease-Detection-…` | 4 images per class. |

### The Yellow Rust investigation you asked for

You asked why Yellow Rust had only 208 usable images, and whether that number
could be raised legitimately.

**Finding:** 208 was not a quality-control loss. `hg3817` holds exactly 208 Yellow
Rust images across its train/valid/test folders, and all 208 survived validation.
The scarcity was in the source, not in the pipeline.

**Search result:** one additional genuine source was found — `kiran`, carrying 239
Yellow Rust images (`Dataset/{train,val,test}/YellowRust`). It also supplies the
only second source of Powdery Mildew in the project. Yellow Rust supply therefore
rises from **208 to 447**, which more than doubles it and drops the worst-case
class imbalance from 15.8:1 to 7.6:1 before any sampling.

**What was not found:** no further Yellow Rust source on GitHub. The large stripe-rust
datasets that exist (e.g. the CGIAR/Kaggle wheat rust sets) are behind the blocked
hosts. So Yellow Rust remains the binding constraint on the whole dataset, and v3
does not solve that — it only widens it from one source to two.

### Availability after the survey

Projected from registry counts, **before** quality control and duplicate removal:

| Class | Total | wpldd | hg3817 | suhas | mubashar | kiran |
|---|---|---|---|---|---|---|
| Healthy | 3,057 | 1,545 | 102 | 1,146 | 142 | 122 |
| Yellow Rust | **447** | — | 208 | — | — | 239 |
| Brown Rust | 3,414 | 1,642 | — | 1,286 | 358 | 128 |
| Powdery Mildew | 1,687 | 1,526 | — | — | — | 161 |
| **Total** | **8,605** | 4,713 | 310 | 2,432 | 500 | 650 |

### On the 12,000-image target

**It is not reachable, and the shortfall is not close.** The maximum obtainable
from documented sources in this environment is **8,605 images before QC**, and
the proposal below deliberately uses fewer than that. Padding to 12,000 would
require duplicating images or importing the `anushrii` "unhealthy" folder under a
guessed label. Neither will be done. The real number is documented instead.

---

## 3. The proposed dataset  ← this is the decision to approve

### The problem with simply using everything

Concatenating all 8,605 images gives 7.6:1 imbalance — barely better than v2's
7.9:1 — and, worse, leaves WPLDD supplying **50% of Healthy, 48% of Brown Rust and
90% of Powdery Mildew**. "More data" would mostly mean "more of the same data",
and the v3 hypothesis would go untested.

### Rule 1 — a cap derived from the scarcest class, not chosen

Yellow Rust cannot be enlarged. So every other class is capped at
`data.imbalance_warn_ratio` × the Yellow Rust count. That knob already exists in
`config/config.yaml` and is currently **3.0**, so:

```
cap = 3.0 × 447 = 1,341 images per class
```

The dataset then sits exactly at the imbalance the project already declares
acceptable. The cap follows from the data; it is not a round number chosen to
look tidy, and if Yellow Rust's post-QC count differs from 447 the cap moves with
it automatically.

### Rule 2 — draw evenly across sources, redistributing the shortfall

Within a class, the cap is split equally between that class's sources. A source
that cannot meet its equal share gives everything it has, and the shortfall is
redistributed among the sources that still have room. Implemented in
`src/data/sample_plan.py::even_allocation`; it is arithmetically incapable of
asking a source for more images than it holds, and `tests/test_sample_plan.py`
pins that.

### Rule 3 — choose *which* images by stride, not by taking the first N

Consecutive filenames in these collections are usually frames from one capture
session. Taking the first 487 of WPLDD's Healthy folder would buy a handful of
sessions; `stride_select` takes evenly spaced files across the sorted listing
instead.

### The resulting plan

**Projected** (registry counts, pre-QC). Cells read `selected / available`:

| Class | wpldd | hg3817 | suhas | mubashar | kiran | Selected | Largest source |
|---|---|---|---|---|---|---|---|
| Healthy | 487/1545 | 102/102 | 488/1146 | 142/142 | 122/122 | **1,341** | 36% |
| Yellow Rust | — | 208/208 | — | — | 239/239 | **447** | 53% |
| Brown Rust | 427/1642 | — | 428/1286 | 358/358 | 128/128 | **1,341** | 32% |
| Powdery Mildew | 1180/1526 | — | — | — | 161/161 | **1,341** | 88% |
| **Total** | | | | | | **4,470** | |

Reproduce with:

```bash
python -m src.data.sample_plan
```

What this buys, stated as the comparison that matters:

| | v1 / v2 | v3 (projected) |
|---|---|---|
| Total images | 4,921 | 4,470 |
| Class imbalance | 7.9:1 | **3.00:1** |
| Yellow Rust images | 208 | **447** |
| Classes with ≥2 sources | 0 of 4 | **4 of 4** |
| Healthy from one source | 100% | **36%** |
| Brown Rust from one source | 100% | **32%** |
| Powdery Mildew from one source | 100% | 88% |

**v3 is smaller than v2 in raw image count.** That is deliberate and follows your
stated priority — *diverse real-world images > clean labels > balanced
representation > raw image count*. It is also the single most arguable decision
in this document, so it is the one to push back on if you disagree. The
alternative (use all 8,605) is trained as variant **v3-c** below precisely so the
choice is settled by measurement rather than by assertion.

### Where this plan is weak, stated up front

1. **Powdery Mildew stays 88% WPLDD.** There is no third source. This class is
   the one place where the v3 hypothesis essentially cannot be tested, and if v3
   improves overall but not on Powdery Mildew, that is the expected shape of the
   result rather than a surprise.
2. **Cross-source overlap is not yet measured.** If `suhas` and `wpldd` turn out
   to share photographs, the diversity is partly illusory. Acquisition will run
   perceptual-hash grouping *across all sources within a class* and report the
   number of cross-source duplicate groups **before** training. If that number is
   large, this plan comes back to you rather than proceeding.
3. **"Independent" is an assumption.** These repositories do not state their
   provenance. Two of them could be re-uploads of one upstream dataset with
   different filenames and mild re-encoding. Cross-source hashing is the only
   check available, and it catches re-encoding but not re-photographing of the
   same plots.
4. **Label quality is unverified.** Labels are taken from each repository's folder
   names. Nobody has checked them against a plant pathologist. A wrong folder in
   one source now propagates into a class that previously had one consistent
   labeller.
5. **Brown Rust vs Yellow Rust remains genuinely hard.** Early-stage rust pustules
   are similar across the two species, and your 0/5 on Brown Rust may reflect that
   rather than a data-volume problem. The error analysis in §6 is designed to tell
   these apart, and it may conclude that v3's dataset change was not the relevant
   lever.

---

## 4. Training variants to compare

You asked for class-balanced sampling, class weights and a balanced subset to be
compared rather than assumed. Four runs, each ~25–40 min at v2's measured rate
(v2: 7,176 training images, 2,123 s, 20+10 epochs):

| Variant | Dataset | Imbalance handling | Background augmentation | Isolates |
|---|---|---|---|---|
| **v3-a** *(proposed primary)* | capped, source-balanced (4,470) | class weights | on, v2 settings | the source-diversity hypothesis |
| **v3-b** | same as v3-a | class weights | **off** | what background augmentation is worth once the dataset is diverse |
| **v3-c** | **all 8,605**, uncapped | class weights | on, v2 settings | whether capping helped or just discarded data |
| **v3-d** *(optional)* | capped, source-balanced | **class-balanced sampler** instead of weights | on | whether resampling beats reweighting |

Notes, so the variants are not over-read:

* **v3-a keeps v2's background-augmentation settings unchanged** (`classes:
  [Healthy, Brown Rust, Powdery Mildew]`, `harvest_from: Yellow Rust`, same seed,
  same scales). One variable — the dataset — separates v2 from v3-a. That is the
  point of holding it fixed.
* **v3-d repeats minority images within an epoch.** You said not to duplicate
  Yellow Rust images, and no file will be duplicated on disk; a balanced sampler
  is repetition in the input pipeline rather than in the data. It is flagged
  optional for exactly that reason — say the word and it is dropped.
* Whichever variant is promoted to `models/v3/` is chosen on **validation** and
  internal test metrics, never on your 20 photographs. The other variants are
  kept under `models/v3_b/`, `models/v3_c/`, `models/v3_d/` with their results, so
  the selection is auditable.

---

## 5. Not breaking v1 and v2

v3 acquisition and preparation must not touch v1/v2 inputs or outputs. The rules:

| Asset | Rule |
|---|---|
| `models/v1/`, `models/v2/` | never written. md5s re-verified before and after. |
| `data/raw/`, `data/train/`, `data/validation/`, `data/test/` | never written. v3 uses `data/v3_raw/` and `data/v3_{train,validation,test}/` via `paths` overrides in `config/model_v3.yaml`. |
| `results/evaluation/`, `results/robustness/`, `results/baselines/`, `results/comparison/` | already version-tagged per file, so v3 adds files rather than replacing any. |
| `results/dataset/`, `results/augmentation/` | **not** version-tagged. These are copied to `results/dataset/v1_v2/` and `results/augmentation/v1_v2/` before v3 writes anything. Copy, not move. |
| `data/external_test/` | never read by acquisition, preparation, augmentation, training or model selection. Enforced in code, covered by `tests/test_external_exclusion.py`. |

Source repositories are cloned to a scratch directory **outside** this repository
and are not committed, exactly as for v1/v2.

Files are renamed `<sourcekey>__<originalname>` on the way in. That prevents
filename collisions between sources (`aadium` and `suhas` proved these happen) and
keeps provenance visible in the split manifest, which is what makes per-source
accuracy reporting possible afterwards.

---

## 6. How v3 will be judged

### The comparison trap, and the way around it

v3 has a **different test split** from v1/v2. "v3 scores X on its test set, v2
scored 99.68% on its test set" compares two different benchmarks and means very
little. Worse, the obvious fix is contaminated in both directions:

* v2 trained on WPLDD images that will appear in v3's test split;
* v3 will train on WPLDD images that appear in v2's test split.

So the primary internal comparison is a **clean cross-evaluation set**: v3 test
images from sources v2 never saw (`suhas`, `mubashar`, `kiran`, `hg3817`), with
any image whose perceptual hash matches anything in v2's training split removed.
Both models are scored on exactly those images. That set is uncontaminated for
both, and its size will be reported — if it comes out too small to carry a
conclusion, that will be said rather than worked around.

### Required outputs

1. Standard evaluation on v3's own test split: accuracy, per-class
   precision/recall/F1, confusion matrix, confidence distribution, top-2 accuracy.
2. **Per-source accuracy** on the v3 test split. If v3 is much better on WPLDD
   images than on the others, it has learned WPLDD again.
3. Clean cross-evaluation, v2 vs v3, as defined above.
4. Background robustness for v3, same protocol as v1/v2 (`background_robustness`
   harvests from the **test** split while training augmentation harvests from
   **train**, so it cannot be gamed).
5. Trivial-baseline control on the v3 dataset — the 7-colour-statistic baseline.
   If it still scores highly, the new benchmark is still too easy and v3's headline
   number means as little as v1's did.
6. Brown Rust error analysis: every Brown Rust error classified as BR→YR, BR→PM,
   BR→Healthy, plus others→BR, cross-tabulated against source, imagery type and
   confidence, with example images.
7. Confidence calibration, including any revised uncertainty threshold — derived
   from **validation** data only.
8. Finally, and only after all of the above is fixed and recorded, the 20 external
   photographs scored for v1, v2 and v3 on identical inputs.

### What would make v3 a failure, stated in advance

So that the result cannot be rationalised after the fact:

* v3 does **not** beat v2 on the clean cross-evaluation set → the source-diversity
  hypothesis is not supported, and that is what gets written down.
* v3's trivial-colour baseline stays near v3's own accuracy → the benchmark is
  still trivial and no accuracy claim from it is worth much.
* v3's per-source accuracy is strongly skewed towards WPLDD → source balancing
  did not do its job.
* v3 improves on the 20 external photographs but loses on cross-evaluation → most
  likely noise on 20 images, and it will be reported as inconclusive rather than
  as a win. Twenty images is a small sample: a 45% → 60% move is three images.

The confidence threshold will not be lowered to make any of this look better.

---

## 7. What happens next

**Blocked on you, in this order:**

1. **Approve or amend the dataset plan in §3.** The most arguable calls are the
   3.0:1 cap (v3 ends up smaller than v2) and including `wpldd` at all.
2. **Copy your 20 real-world photographs into `data/external_test/`**, then:

   ```bash
   python -m src.testing.external_test --template
   # fill in the true_class column in data/external_test/manifest.csv
   ```

   Valid labels: `Healthy`, `Yellow Rust`, `Brown Rust`, `Powdery Mildew`
   (aliases such as `stripe rust` are accepted). Those files are git-ignored and
   are never committed.

Training does not start until both are done.

**Then, in order:** acquire the five sources → QC and cross-source duplicate
report → *pause if cross-source overlap is large* → apply the sampling plan →
leakage-safe split → full test suite → train v3-a … v3-d → evaluate → report.
