# V3 dataset — final pre-training report

**Status: both datasets are built, decontaminated, repaired and verified — and
NOT cleared for training.** The Brown Rust audit found substantial stripe-rust
contamination in the `suhas` source (§6), and by your instruction that stops
the process until you decide what to do with those images. No V3 model has been
trained. V1 and V2 are unchanged. No external-test photograph or label has been
modified. Nothing about Claude Vision has been implemented.

Built by `python -m src.data.build_v3`, 2026-09-24. Every image is in
`data/v3_manifest.csv`; every removal, with its reason, in `data/v3_removed.csv`.

---

## 1. Final usable images per class

| Class | Declared | − exact dup | − family near-dup | − external overlap | **Final** |
|---|---|---|---|---|---|
| Healthy | 3,057 | −602 | −67 | 0 | **2,388** |
| Yellow Rust | 447 | 0 | 0 | 0 | **447** |
| Brown Rust | 3,414 | −726 | −200 | −9 | **2,479** |
| Powdery Mildew | 1,687 | 0 | 0 | 0 | **1,687** |
| **Total** | **8,605** | **−1,328** | **−267** | **−9** | **7,001** |

Zero images failed validation, and no byte-identical file carries two labels.
Near-duplicates are kept and bound together: 136 groups, 275 images, none
spanning a class, source or family.

## 2. V3-full — 7,001 images

Every clean image. Nothing duplicated, nothing valid discarded. Imbalance 5.5:1,
handled with inverse-frequency class weights computed from the training split
only: Healthy 0.459, Yellow Rust 2.449, Brown Rust 0.442, Powdery Mildew 0.650.

## 3. V3-balanced — 4,470 images

Exactly **1,341 / 447 / 1,341 / 1,341**, 3.00:1. Selected from inside each split
of V3-full, so every balanced image has the same split in both variants and both
models can be scored on the V3-full test set without either having seen it.
Class weights from its training split: 0.667 / 1.998 / 0.667 / 0.667.

## 4. Images removed for external-test overlap — 9

External photographs were fingerprinted read-only into
`data/v3_external_fingerprints.csv`: file name, SHA-256 and perceptual hash
only — no labels. The build compared against that file and never opened
`data/external_test/`. Removal ran **after** the split, so no other image
changed split.

| Removed pool image (`suhas`, Brown Rust, train) | Matched external | How |
|---|---|---|
| `…Leaf-Rust__02701.jpeg` | ext_03 | **byte-identical** |
| `…Leaf-Rust__01211.jpg` | ext_03 | near, 2 bits |
| `…Leaf-Rust__02881.jpeg` | ext_01 | near, 2 bits |
| `…Leaf-Rust__07771.jpg` | ext_01 | near, 0 bits |
| `…Leaf-Rust__03581.jpg` | ext_04 | near, 0 bits |
| `…Leaf-Rust__07821.jpg` | ext_04 | near, 2 bits |
| `…Leaf-Rust__00341.jpg` | ext_06 | near, 0 bits |
| `…Leaf-Rust__02711.jpeg` | ext_06 | near, 0 bits |
| `…Leaf-Rust__03731.jpg` | ext_07 | near, 0 bits |

**Exactly the 9 identified earlier — no more, no fewer.** Verified against the
previous manifest: 9 images gone, 0 added, **0 images changed split**, 0 group
IDs changed.

## 5. Images converted from WebP — 80

80 `suhas` Brown Rust files were WebP content with `.jpg` names (60 train, 7
validation, 13 test; 64 in V3-balanced). Each was re-encoded to PNG from exactly
the pixels Pillow decodes: no resize, crop, mode change or colour transform,
ICC profile carried over. **Pixel equality was asserted for every file at write
time** — the build refuses to continue otherwise. The manifest records both
`original_sha256` and the new `sha256`, and marks each row `webp->png`. The
upstream files are untouched. No WebP content remains anywhere in the pool.

## 6. Brown Rust audit — substantial contamination, STOP

Full record: `docs/v3_brown_rust_audit.md` and `data/v3_brown_rust_audit.csv`.

60 of 591 `suhas` Brown Rust images, simple random sample, **seed 42**, with the
threshold fixed before viewing (≥ 6 of 60 = substantial):

| Category | n | % | 95% CI |
|---|---|---|---|
| Likely Brown Rust | 22 | 36.7% | 24.6–50.1% |
| **Likely Yellow/Stripe Rust** | **11** | **18.3%** | **9.5–30.4%** |
| Unclear | 12 | 20.0% | 10.8–32.3% |
| Not a usable wheat-leaf image | 15 | 25.0% | 14.7–37.9% |

11 is nearly twice the threshold. Only about a third of this source is
confidently Brown Rust. It sits in the test and validation splits too, and it is
**worse for V3-balanced**: `suhas` supplies 37.8% of its Brown Rust against
23.8% for V3-full. The options are set out in the audit document. **None has
been applied.**

## 7. Final train / validation / test counts

Split 75 / 12.5 / 12.5, seed 42, stratified by (class, source family), whole
near-duplicate groups assigned together.

### V3-full

| Class | Train | Validation | Test | Total |
|---|---|---|---|---|
| Healthy | 1,786 | 301 | 301 | 2,388 |
| Yellow Rust | 335 | 56 | 56 | 447 |
| Brown Rust | 1,855 | 312 | 312 | 2,479 |
| Powdery Mildew | 1,263 | 212 | 212 | 1,687 |
| **Total** | **5,239** | **881** | **881** | **7,001** |

### V3-balanced

| Class | Train | Validation | Test | Total |
|---|---|---|---|---|
| Healthy | 1,003 | 169 | 169 | 1,341 |
| Yellow Rust | 335 | 56 | 56 | 447 |
| Brown Rust | 1,003 | 169 | 169 | 1,341 |
| Powdery Mildew | 1,004 | 168 | 169 | 1,341 |
| **Total** | **3,345** | **562** | **563** | **4,470** |

### Source families per class

| Class | V3-full | V3-balanced |
|---|---|---|
| Healthy | wpldd 64.7%, suhas 26.0%, kiran 5.1%, hg3817 4.2% | wpldd 41.7%, suhas 41.7%, kiran 9.1%, hg3817 7.5% |
| Yellow Rust | kiran 53.5%, hg3817 46.5% | kiran 53.5%, hg3817 46.5% |
| Brown Rust | wpldd 66.2%, suhas 28.6%, kiran 5.2% | suhas 45.3%, wpldd 45.1%, kiran 9.5% |
| Powdery Mildew | wpldd 90.5%, kiran 9.5% | wpldd 88.0%, kiran 12.0% |

Every family is present in every split of every class it belongs to.

## 8. External-test images are excluded — confirmed four ways

| Check | Result |
|---|---|
| Pool vs recorded fingerprints (exact on both hashes; near ≤ 5 bits) | **0 matches** |
| Direct read-only content check of all 10 external photographs vs the pool | **0 exact, 0 near** |
| Any V3 file sharing an inode with, or resolving into, `data/external_test/` | **0** |
| `build()` file access, spied in tests | never touches the directory |

All 10 photographs, the README and the manifest in `data/external_test/` are
byte-identical to before this work.

## 9. V1 and V2 are untouched — confirmed

| Asset | Check | Result |
|---|---|---|
| `models/v1/model.keras` | md5 `2ae53135…` | OK |
| `models/v2/model.keras` | md5 `8773a330…` | OK |
| v1/v2 dataset reports in `results/dataset/` | md5 of every file | 0 changed |
| `data/raw`, `data/train`, `data/validation`, `data/test` | never opened for writing; the V3 pool hard-links to the source clones, not to these | untouched |

The V3 validator reports went to `results/v3_validation/` specifically so they
could not overwrite v1/v2's.

## 10. Full test suite — 194 passed

| Group | Tests |
|---|---|
| Existing v1/v2 suite | 146 passed |
| V3 synthetic: dedup, families, splits, nesting, decontamination, WebP repair, isolation, tamper detection | 37 passed |
| V3 real-manifest: counts, removals, conversions, fingerprints, leakage, groups, families, disk | 11 passed |

One existing V3 test was re-scoped: *"staged files are byte-identical to the
originals"* now skips the file flagged for WebP repair in its synthetic corpus,
and asserts that every other file — an exact count — is still byte-identical. Converted files are covered by a stricter new test
asserting pixel identity. No assertion was weakened or removed.

### Every check re-run, with results

| Check | V3-full | V3-balanced |
|---|---|---|
| Project validator (`validate_dataset --split-report`): all splits usable | ✅ | ✅ |
| Exact duplicates across splits | 0 | 0 |
| Near-duplicates across splits (all classes) | 0 | 0 |
| Near-duplicate groups spanning splits | 0 of 136 | — |
| Balanced nested in full | 0 violations | |
| External-test leakage | 0 | 0 |
| TensorFlow decode of every image, exactly as training reads it | **7,001 / 7,001** | |
| SHA-256 of every staged file vs manifest | 7,001 match | |

---

## 11. One side effect to know about

Removing the 9 images changed **which** Brown Rust images are in V3-balanced for
**758 images** (382 in, 376 out), 573 of them WPLDD training images, while the
totals stayed exactly 1,341. **No image changed split**, so this is not leakage,
and nothing has been trained on either selection.

The cause: `stride_select` picks evenly spaced files, so when a quota moves by
one, almost every pick shifts. It will happen again at the next rebuild — for
example when your remaining 10 external images arrive and are decontaminated.
**Proposed fix, not applied:** select by a stable per-image priority
(SHA-256 of seed + path) instead. That keeps the spread across capture sessions
that stride was chosen for, but removing one image then changes exactly one
selection. It changes which images are in V3-balanced, so it needs your
approval.

## 12. Decisions required before training

| # | Decision | Where |
|---|---|---|
| 1 | What to do with the `suhas` Brown Rust contamination — options (a)–(d) | `docs/v3_brown_rust_audit.md` §6 |
| 2 | Whether to audit `suhas` Healthy the same way first | same, §5 |
| 3 | Whether to adopt stable priority selection for V3-balanced | §11 above |
| 4 | The 10 missing external images (Healthy, Powdery Mildew) — they must be fingerprinted and decontaminated before training, or V3 may train on them | `docs/external_test_audit.md` |
| 5 | External labels still awaiting confirmation: ext_03, ext_04, ext_05/08 swap, ext_10 | `docs/external_test_audit.md` |

**Training is not started, and will not be until you approve the final dataset.**
