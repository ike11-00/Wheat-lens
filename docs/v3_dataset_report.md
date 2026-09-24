# V3 dataset report

**Status: built and structurally verified — NOT yet cleared for training.**
Every leakage, grouping, nesting and manifest check passes. Two pre-training
checks failed and one data-quality problem surfaced; all three need a decision
(§13). No V3 model has been trained. V1 and V2 are unchanged (md5 re-verified
after the build), no external-test label has been touched, and nothing about
Claude Vision has been implemented.

Built by `python -m src.data.build_v3`, 2026-09-24, from the five source
repositories at the commits recorded in `src/data/sources_v3.py`. Every image
is listed in `data/v3_manifest.csv`, and every removal in
`data/v3_removed.csv`.

---

## 1. Unique images per class

| Class | Declared | After exact dedup | After family dedup | **Final** |
|---|---|---|---|---|
| Healthy | 3,057 | 2,455 | 2,388 | **2,388** |
| Yellow Rust | 447 | 447 | 447 | **447** |
| Brown Rust | 3,414 | 2,688 | 2,488 | **2,488** |
| Powdery Mildew | 1,687 | 1,687 | 1,687 | **1,687** |
| **Total** | **8,605** | **7,277** | **7,010** | **7,010** |

Zero images failed validation. No byte-identical file appears under two classes
(the label-conflict check removed nothing).

## 2. Removed as exact duplicates — 1,328

| Class | Source | Removed |
|---|---|---|
| Brown Rust | suhas | 686 |
| Brown Rust | mubashar | 40 |
| Healthy | suhas | 585 |
| Healthy | mubashar | 16 |
| Healthy | hg3817 | 1 |

Each removal records the file kept in its place. The first copy in registry
order, then path order, is kept, so the choice is deterministic.

## 3. Removed as near-duplicates — 267

All 267 are `mubashar` images within 5 bits (of 256) of a `suhas` image in the
same class — exactly the approved rule: 200 Brown Rust, 67 Healthy. `suhas`, the
family's primary source, was not trimmed.

**Near-duplicates elsewhere are kept, not removed**, as in v1/v2: 136 groups
holding 275 images, all within a single class and a single source. They are
bound together so each group lands in one split (§8).

## 4. Retained per source and source family

| Family | Source | Healthy | Yellow Rust | Brown Rust | Powdery Mildew | Total |
|---|---|---|---|---|---|---|
| wpldd | wpldd | 1,545 | — | 1,642 | 1,526 | 4,713 |
| suhas | suhas | 561 | — | 600 | — | 1,161 |
| suhas | mubashar | **59** | — | **118** | — | **177** |
| hg3817 | hg3817 | 101 | 208 | — | — | 309 |
| kiran | kiran | 122 | 239 | 128 | 161 | 650 |
| | **Total** | 2,388 | 447 | 2,488 | 1,687 | **7,010** |

The 177 unique `mubashar` images are kept, as approved.

## 5. Source-family proportions per class

| Class | v3_full | v3_balanced |
|---|---|---|
| Healthy | wpldd 64.7%, suhas 26.0%, kiran 5.1%, hg3817 4.2% | wpldd 41.7%, suhas 41.7%, kiran 9.1%, hg3817 7.5% |
| Yellow Rust | kiran 53.5%, hg3817 46.5% | kiran 53.5%, hg3817 46.5% |
| Brown Rust | wpldd 66.0%, suhas 28.9%, kiran 5.1% | suhas 45.3%, wpldd 45.2%, kiran 9.5% |
| Powdery Mildew | wpldd 90.5%, kiran 9.5% | wpldd 88.0%, kiran 12.0% |

This is the real trade between the two variants. `v3_full` keeps every image
but lets WPLDD supply two-thirds of Healthy and Brown Rust; `v3_balanced` gives
up images to hold WPLDD below 46% everywhere except Powdery Mildew, which has no
third source.

## 6. Train / validation / test counts

**Split 75 / 12.5 / 12.5, seed 42.**

### v3_full — 7,010 images

| Class | Train | Validation | Test | Total |
|---|---|---|---|---|
| Healthy | 1,786 | 301 | 301 | 2,388 |
| Yellow Rust | 335 | 56 | 56 | 447 |
| Brown Rust | 1,864 | 312 | 312 | 2,488 |
| Powdery Mildew | 1,263 | 212 | 212 | 1,687 |
| **Total** | **5,248** | **881** | **881** | **7,010** |

Imbalance 5.6:1, handled in the loss. Class weights computed from the training
split only, with the same formula `src/model/train.py` uses:
Healthy 0.460, Yellow Rust 2.450, Brown Rust 0.440, Powdery Mildew 0.650.

### v3_balanced — 4,470 images

| Class | Train | Validation | Test | Total |
|---|---|---|---|---|
| Healthy | 1,003 | 169 | 169 | **1,341** |
| Yellow Rust | 335 | 56 | 56 | **447** |
| Brown Rust | 1,005 | 168 | 168 | **1,341** |
| Powdery Mildew | 1,004 | 168 | 169 | **1,341** |
| **Total** | **3,347** | **561** | **562** | **4,470** |

Imbalance exactly 3.00:1. Class weights from the training split: Healthy 0.668,
Yellow Rust 1.999, Brown Rust 0.666, Powdery Mildew 0.667.

**Nothing is duplicated in either variant, and no valid image is discarded
from `v3_full`.**

### The two variants are nested

`v3_balanced` was selected *from inside each split* of `v3_full`: every
balanced image has the same split in both variants. So the balanced training
set is a subset of the full training set, and the balanced test set is a subset
of the full test set. **Both models can be scored on the 881-image `v3_full`
test set, and neither will have seen any of it.** That makes the balanced-vs-full
comparison clean — which it would not be with two independently drawn splits.

### Every source family is represented in every split

The split is stratified by (class, source family), not by class alone. `v3_full`
test set, per family:

| Class | wpldd | suhas | hg3817 | kiran |
|---|---|---|---|---|
| Healthy | 194 | 78 | 13 | 16 |
| Yellow Rust | — | — | 26 | 30 |
| Brown Rust | 206 | 90 | — | 16 |
| Powdery Mildew | 191 | — | — | 21 |

This is what makes per-source accuracy measurable after training. The smallest
cells (13, 16) will give wide confidence intervals and should be reported with
them.

## 7. Zero image leakage between splits — confirmed

Measured, not assumed, across **all classes**, from the images on disk:

| Check | Result |
|---|---|
| Byte-identical file in two splits | **0** |
| Near-duplicate pair (≤ 5 bits) straddling two splits — train/val, train/test, val/test | **0** |
| Files on disk vs manifest (pool, full, balanced) | **0 missing, 0 unexpected** |
| SHA-256 of every staged file vs manifest | **all 7,010 match** |

## 8. Near-duplicate groups cannot cross splits — confirmed

Grouping runs across the entire pool — all classes, all sources — before the
split, and the split assigns **whole groups**, never individual images. Result:
**0 of 136 groups span more than one split**, and **0 groups span two classes**.
The direct pairwise measurement in §7 confirms it independently of the grouping.

## 9. `data/external_test/` exclusion

**From dataset preparation and training — confirmed.** `build()` never lists,
reads or writes it (a test spies on every file access during a build to prove
it), no V3 path points into it, and no V3 file shares an inode with, or
resolves to, anything in it. Its images are not in any split directory.

**But the content check found contamination from a different direction.** See
Blocker A in §13 — the external photographs themselves are not in the dataset,
but copies of the same photographs arrived independently via the `suhas` source.

## 10. Manifest

`data/v3_manifest.csv` — one row per image, 7,010 rows: staged filename, class,
source, source family, upstream commit, path inside the source repository,
SHA-256, perceptual hash, dimensions, near-duplicate group, split, and
membership of `v3_balanced`. Tracked in git; the images are not.

`data/v3_removed.csv` — 1,595 rows: every removed image, the reason, and what
was kept in its place.

The build is deterministic: rebuilding from the same sources gives a
byte-identical manifest (tested).

## 11. Seed

`random_seed: 42` from `config.yaml`. Each (class, source family) stratum
shuffles with its own seed, derived by SHA-256 from 42 and the stratum's name —
not Python's `hash()`, which is randomised per process. Adding a stratum later
cannot reshuffle the existing ones.

## 12. Tests

`tests/test_v3_dataset.py` — **36 tests, all passing**. Full suite: **182 passed**.

* **27 synthetic tests** build a planted corpus — exact duplicates, cross-source
  near-copies, a label conflict, a near-duplicate triple — and check every
  guarantee above. That includes **tamper tests**: move one member of a
  near-duplicate group into another split, or delete a staged file, and
  `verify()` must fail. They prove the checks are not vacuous.
* **9 tests on the real manifest**: exact counts per class for both variants,
  1,328 exact and 267 family removals, the 177 kept `mubashar` images, no image
  or near-duplicate in two splits, every group in one split, every family in
  every split, and disk/manifest agreement. Counts and leakage are provable
  from the tracked manifest alone, so these run in any clone; the disk check
  skips when the images are absent.

---

## 13. Blockers — why this dataset is not yet cleared for training

### Blocker A — five external test photographs are in the V3 training pool

`check_external_isolation` compared all 10 received external images with the
pool, read-only.

| External image | Matches in the V3 pool | Split |
|---|---|---|
| ext_01.jpg | 2 suhas images (near) | train |
| **ext_03.jpg** | **suhas `…02701.jpeg` — byte-identical**, plus 1 near | train |
| ext_04.jpg | 2 suhas images (near) | train |
| ext_06.webp | 2 suhas images (near) | train |
| ext_07.jpg | 1 suhas image (near) | train |

9 pool images in total, all `suhas` Brown Rust, all in the training split, 6 of
them in `v3_balanced`. Two of the pairs were checked by eye and are the same
photograph.

**How it happened:** the external photographs were found on the web, and
`suhas` is a web-scraped collection. They share an origin. Nothing leaked from
`data/external_test/` itself.

**Why it matters:** V3 would train on 5 of the 10 images it is later judged
on, inflating its external score and invalidating the V2-vs-V3 comparison for
those images.

**V2's baseline is clean.** All 4,921 v1/v2 training images are byte-identical
members of the V3 pool (WPLDD and hg3817), and none of them matched any external
image — every match came from `suhas`. V2 never trained on those photographs.

### Blocker B — 80 images TensorFlow cannot decode

80 `suhas` Brown Rust files are **WebP images with a `.jpg` extension**
(header `RIFF…WEBP`). Pillow reads them, so they passed validation; TensorFlow's
`decode_image`, which training uses, does not support WebP. **Training would
crash** on the first batch containing one. 60 are in train, 7 in validation,
13 in test; 62 are in `v3_balanced`.

### Problem C — `suhas` "Leaf Rust" contains stripe-rust images

The two matched pairs checked by eye in Blocker A — `…00341.jpg` (the same photo
as `ext_06`) and `…03731.jpg` (the same as `ext_07`) — are **textbook stripe
rust with vein-bounded stripes**, filed in `suhas`'s "Leaf Rust" folder and
therefore labelled **Brown Rust** in V3. That is label noise on precisely the
Brown/Yellow confusion the project is trying to fix.

**The rate is unknown.** These two were found by accident, because they happened
to match external images. The other ~590 `suhas` Brown Rust images have not been
reviewed. No rate is claimed here.

---

## 14. Decisions required

| # | Question | Options | Recommendation |
|---|---|---|---|
| **A** | External-test contamination | **(a)** Remove the 9 matching pool images, and re-check whenever new external images arrive · (b) keep the pool and drop ext_01/03/04/06/07 from the V3 comparison · (c) keep both and caveat the result | **(a)** — standard decontamination. It only removes images, so nothing about the external labels can reach training. It does mean reading the external set during preparation, which your isolation rule forbids, so it needs your explicit approval. Option (b) would halve the external set |
| **B** | 80 WebP-as-`.jpg` files | **(a)** Transcode to PNG — pixel-identical to what Pillow decodes, lossless from here on, both hashes recorded · (b) exclude them · (c) teach the training loader WebP | **(a)** — keeps 80 valid images, in line with "do not throw away valid images". (c) changes the training code v1/v2 share, so no |
| **C** | `suhas` Brown Rust label noise | **(a)** Visual sample audit of ~60 `suhas` Brown Rust images before training, reported as an estimated rate with its uncertainty · (b) proceed and measure it afterwards via per-source accuracy · (c) drop `suhas` Brown Rust | **(a)** — 60 images with no stripe rust would put the rate below ~5% (95% bound). A high rate would change which variant is worth training. Same caveat as the external audit: a non-expert visual judgement |

None of these has been applied. After your decisions: rebuild (seconds —
fingerprints are cached), re-verify, re-run the tests, and report again before
any training.
