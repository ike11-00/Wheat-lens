# V3 source overlap check — results

**Run:** 2026-09-23, all 8,605 registry-declared images across the five active
sources. Perceptual hash size 16 (256 bits), grouping distance ≤ 5 — the
thresholds already calibrated in `config.yaml`, not new ones.

**Verdict: SUBSTANTIAL OVERLAP. The declared stop criterion fired, so nothing
has been written toward the V3 dataset.** The thresholds were fixed in
`src/data/overlap_check.py` before the run, not chosen after seeing the
numbers.

The full generated report is at `results/dataset/v3_overlap.{json,md}`
(gitignored, like every other results file); this document records the findings
and the post-deduplication analysis the tool does not produce.

---

## 1. Registry verification

Every declared folder was found and **every declared count matched exactly**:
8,605 declared, 8,605 found. **Zero images failed validation** — no corrupt,
truncated, undersized or wrong-format file in any source.

| Class | hg3817 | kiran | mubashar | suhas | wpldd | Total |
|---|---|---|---|---|---|---|
| Healthy | 102 | 122 | 142 | 1146 | 1545 | 3057 |
| Yellow Rust | 208 | 239 | — | — | — | 447 |
| Brown Rust | — | 128 | 358 | 1286 | 1642 | 3414 |
| Powdery Mildew | — | 161 | — | — | 1526 | 1687 |

---

## 2. Finding 1 — `suhas` is about half internal duplicates

Byte-identical (SHA-256) duplicates **within** a single source:

| Class | Source | Files | Distinct | Dropped | Loss |
|---|---|---|---|---|---|
| Brown Rust | suhas | 1286 | 600 | 686 | **53%** |
| Healthy | suhas | 1146 | 561 | 585 | **51%** |
| Brown Rust | mubashar | 358 | 318 | 40 | 11% |
| Healthy | mubashar | 142 | 126 | 16 | 11% |
| Healthy | hg3817 | 102 | 101 | 1 | 1% |
| all | wpldd, kiran | 4,713 / 650 | unchanged | 0 | **0%** |

`suhas` ships each of its Brown Rust and Healthy images roughly twice. This is
not a leakage risk — `prepare_dataset` already drops exact duplicates — but it
means **the 8,605 figure overstated the corpus by 15%**.

**8,605 declared → 7,277 distinct images.**

`wpldd` and `kiran` contain no byte-identical duplicates at all.

## 3. Finding 2 — `mubashar` is largely a re-encoding of `suhas`

**Zero cross-source *exact* duplicates exist**, so no source is a straight
re-upload of another. But near-duplicate grouping found one pair, in both
classes they share, and only that pair:

| Class | Pair | Shared near-dup groups | Share of the smaller source |
|---|---|---|---|
| Brown Rust | mubashar ↔ suhas | 192 | 53.6% |
| Healthy | mubashar ↔ suhas | 67 | 47.2% |

Re-measured after exact deduplication, comparing each remaining `mubashar`
image against every remaining `suhas` image:

| Class | suhas distinct | mubashar distinct | mubashar matching a suhas image | mubashar **unique** |
|---|---|---|---|---|
| Brown Rust | 600 | 318 | 200 (**62.9%**) | **118** |
| Healthy | 561 | 126 | 67 (**53.2%**) | **59** |

Different bytes, same photographs — consistent with a re-encode or a re-upload
of the same upstream collection, which is also what `aadium` (already excluded)
does to `suhas`.

**`mubashar` is not an independent source.** Counting it as one would report
diversity the dataset does not have, which is precisely what V3 exists to test.

## 4. What is clean

| Result | Significance |
|---|---|
| **Yellow Rust: zero overlap** between `hg3817` and `kiran` | The two-source diversity in the scarcest class is real |
| **Powdery Mildew: zero overlap** between `wpldd` and `kiran` | Its only second source is genuine |
| **`wpldd`, `hg3817`, `kiran` overlap nothing** | The three sources V3 most depends on are mutually independent |
| **Zero invalid images** across all 8,605 | No quality-control losses anywhere |
| **Zero cross-source exact duplicates** | No source is a byte-level re-upload |

The contamination is confined to one source pair, in two classes.

---

## 5. Effect on the dataset

| | Before the check | After |
|---|---|---|
| Declared images | 8,605 | 8,605 |
| Distinct after exact dedup | not measured | **7,277** |
| Usable after removing `mubashar`'s duplicates of `suhas` | not measured | **7,010** |
| Independent source families | 5 assumed | **4** (`mubashar` folds into `suhas`) |

**The trainable dataset does not change at all.** Every capped class still has
more than the 1,341-image cap available, so the approved plan produces exactly
the same totals:

| Class | wpldd | suhas family | hg3817 | kiran | Selected |
|---|---|---|---|---|---|
| Healthy | 559/1545 | 559/620 | 101/101 | 122/122 | **1,341** |
| Yellow Rust | — | — | 208/208 | 239/239 | **447** |
| Brown Rust | 606/1642 | 607/718 | — | 128/128 | **1,341** |
| Powdery Mildew | 1180/1526 | — | — | 161/161 | **1,341** |

**Total 4,470 images, imbalance 3.00:1** — identical to the approved proposal.

What does change is the honest diversity figure:

| Class | Largest single source — projected | Largest single source — measured |
|---|---|---|
| Healthy | 36% | **42%** |
| Brown Rust | 32% | **45%** |
| Yellow Rust | 53% | 53% |
| Powdery Mildew | 88% | 88% |

Worse than projected, because `suhas` halved and `mubashar` stopped counting as
its own source. Still far better than V1/V2, where every class came 100% from a
single collection.

---

## 6. Recommendation

**No source needs excluding. Two adjustments are needed instead.**

1. **Treat `mubashar` and `suhas` as one source family** for allocation and for
   per-source accuracy reporting. Keeping them as separate rows would report
   diversity that is not there.
2. **Drop the 267 `mubashar` images that are near-duplicates of `suhas`
   images** (200 Brown Rust, 67 Healthy), keeping its 177 genuinely unique ones
   (118 Brown Rust, 59 Healthy).

Dropping `mubashar` wholesale is the alternative. It costs those 177 unique
images and changes the plan's output by one percentage point of Brown Rust
diversity, so the marginal case for it is weak.

Neither adjustment changes the selected image count, the class balance, or the
approved cap. The registry entry for `mubashar` in `src/data/sources_v3.py`
should record the measured overlap so the finding is not lost.
