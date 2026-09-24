# Audit: stripe rust inside the `suhas` "Brown Rust" source

**Result: substantial label contamination. By the threshold fixed before any
image was viewed, the stop condition is met.** No image has been relabelled,
removed or moved because of this audit. What to do about it is your decision
(§6).

## 1. Method

| | |
|---|---|
| Population | The 591 `suhas` images labelled Brown Rust in the final V3 pool (`data/v3_manifest.csv`, after all removals) |
| Sampling | Simple random sample without replacement |
| Code | `random.Random(42).sample(sorted(staged_names), 60)` |
| Seed | **42** |
| Sample size | **60** (10.2% of the population) |
| Rater | One non-expert visual rater (Claude). No microscopy, no pathologist |
| Display | 2×2 contact sheets (≤600 px per image), then every uncertain image re-examined individually at up to 1,400 px |
| Declared before viewing | Categories, sample and seed, and the threshold: **≥ 6 of 60 (≥ 10%) "likely stripe rust" = substantial contamination** |

**Discriminator used**, the same one as the external-test audit: *arrangement*.
Leaf (brown) rust scatters round-to-oval pustules at random across the blade;
stripe (yellow) rust confines smaller, yellower pustules to long vein-bounded
stripes. Colour was used only as supporting evidence.

**One rule was fixed during the audit, not before it, and is disclosed here:**
an image that does not show a single leaf blade in usable detail — a wheat head,
a stem, a lettered figure, a lawn, a field panorama — is category 4 even when
the rust on it looks attributable. It was needed at image #13, the first head
image. It moves at most one image out of "likely stripe rust" (#13); the
sensitivity figures in §3 include it.

## 2. Results

| Category | Count | % | Exact 95% CI (Clopper–Pearson) |
|---|---|---|---|
| 1. Likely Brown Rust | **22** | 36.7% | 24.6 – 50.1% |
| 2. **Likely Yellow/Stripe Rust** | **11** | **18.3%** | **9.5 – 30.4%** |
| 3. Unclear | **12** | 20.0% | 10.8 – 32.3% |
| 4. Not a usable wheat-leaf image | **15** | 25.0% | 14.7 – 37.9% |

**11 of 60 is almost twice the declared threshold of 6**, and even the lower
end of the confidence interval (9.5%) is near it.

Only **22 of 60 (37%)** images in this source are confidently what their label
says.

### Category 4 breaks down as

| Kind | Images |
|---|---|
| Lettered or multi-panel figures | #3, #23, #24, #32, #43 |
| Field panoramas, lawns, or plants with no visible disease (4 are stock photos) | #7, #8, #12, #31, #38, #42, #51 |
| Not a leaf: a wheat head (#13), a stem with stem-rust presentation (#19) | #13, #19 |
| A different condition — no rust pustules at all | #46 |

Two of those also carry a wrong-disease signal: #19 shows **stem rust**, and #43
shows stem-rust-like pustules on culms. Stem rust is not one of the four
classes.

## 3. How to read the 18%

* **The honest range is roughly 10–30%.** 60 is enough to establish that the
  contamination is substantial; it cannot pin down the rate.
* **The unclear images could move it either way.** If none of the 12 were
  stripe rust, the rate stays at 18.3%. If all of them were, it would be 38.3%
  (23/60). Several unclear images (#2, #15, #45) leaned stripe; none is counted.
* **Among images that could be called at all**, stripe rust is **11 of 33 (33%)**.
* Counting #13 (a head with bright yellow glume pustules, more typical of stripe
  rust) would make it 12/60 = 20%.

## 4. What this means for V3

Scaling the sample rates to the images actually in each variant:

| | `v3_full` | `v3_balanced` |
|---|---|---|
| Brown Rust images | 2,479 | 1,341 |
| …from `suhas` | 591 (**23.8%**) | 507 (**37.8%**) |
| Expected stripe rust labelled Brown Rust | ~108 (CI 56–180) | ~93 (CI 48–154) |
| Expected unusable images | ~148 (CI 87–224) | ~127 (CI 75–192) |
| `suhas` Brown Rust in the test split | 78 | 67 |

Three consequences, stated plainly:

1. **`v3_balanced` is more exposed than `v3_full`.** Balancing Brown Rust away
   from WPLDD necessarily raises the share drawn from `suhas`, the noisiest
   source. The variant built to test source diversity is the one this problem
   hurts most.
2. **The test and validation splits are contaminated too.** Roughly a dozen
   `suhas` Brown Rust test images per variant are probably stripe rust. A model
   that correctly calls them Yellow Rust would be scored wrong, which would
   *understate* exactly the Brown/Yellow discrimination V3 is meant to improve.
3. **It teaches the confusion V3 exists to fix.** Stripe rust labelled Brown
   Rust is direct training signal for "stripes mean Brown Rust".

## 5. What was not audited

* `mubashar` (118 Brown Rust, 59 Healthy) — same family as `suhas`, likely a
  similar origin. Not sampled.
* **`suhas` Healthy (561 images).** Seven of the 15 unusable images above are
  scenes a web scrape would also file under "healthy" — lawns, field
  panoramas, stock photos of green plants. Its quality is unknown and plausibly
  similar.
* WPLDD, `hg3817` and `kiran` Brown Rust. They were not sampled, but none of
  them is a web scrape.

## 6. Options — not decided, not applied

| | Option | Effect |
|---|---|---|
| a | Drop `suhas` + `mubashar` Brown Rust entirely | Removes the problem with certainty. Brown Rust falls to 1,770 (WPLDD 1,642 + kiran 128) — above the 1,341 cap, so `v3_balanced` still reaches 1,341, but WPLDD becomes ~90% of it (1,213 of 1,341). Brown Rust returns to near single-source, like Powdery Mildew |
| b | Full visual review of all 591 (and mubashar's 118), keeping only "likely Brown Rust" | Keeps genuine images. About 700 judgements by a non-expert rater; the sample suggests it would keep roughly a quarter to a half |
| c | Keep them, exclude only the 26 audited images judged stripe or unusable | Removes known-bad images only. Leaves an estimated ~85–100 stripe images in place |
| d | Keep everything and report per-source accuracy | Least work. Accepts that Brown Rust metrics are measured against partly wrong labels |

Whatever is chosen, the same question applies to `suhas` Healthy before
training.

---

## Appendix — all 60 sampled images

Sizes are original pixel dimensions. `.png` names were WebP files converted in
this build.

| # | File | Size | Split | Category | Visual reasoning |
|---|---|---|---|---|---|
| 1 | `suhas__Images_Leaf-Rust__02111.jpg` | 800x1067 | train | likely Brown Rust | Heavy, uniform orange-brown coating on several leaves; no vein-bounded striping (medium confidence) |
| 2 | `suhas__Images_Leaf-Rust__00481.jpg` | 3264x1836 | train | unclear | Dense fine yellow pustules on a chlorotic leaf with a streaky texture; leans stripe rust but not resolvable |
| 3 | `suhas__Images_Leaf-Rust__04401.jpg` | 366x138 | validation | not a usable wheat-leaf image | Composite textbook figure: three labelled panels (leaf rust, stripe rust, tan spot) |
| 4 | `suhas__Images_Leaf-Rust__04051.jpg` | 266x189 | validation | unclear | 266x189 with a stock watermark across the leaf; arrangement not resolvable |
| 5 | `suhas__Images_Leaf-Rust__03801.jpg` | 168x300 | train | likely Brown Rust | Cluster of round brown pustules scattered on a yellowing leaf |
| 6 | `suhas__Images_Leaf-Rust__02751.jpeg` | 223x226 | train | likely Brown Rust | Round orange-brown pustules scattered at random |
| 7 | `suhas__Images_Leaf-Rust__01961.jpg` | 390x280 | train | not a usable wheat-leaf image | Stock field panorama; no leaf-level disease visible |
| 8 | `suhas__Images_Leaf-Rust__08501.jpg` | 195x280 | test | not a usable wheat-leaf image | Stock photo of green plants; no rust visible |
| 9 | `suhas__Images_Leaf-Rust__01701.jpg` | 288x192 | train | likely Brown Rust | Round orange pustules scattered across the blade, with necrotic streaks |
| 10 | `suhas__Images_Leaf-Rust__06671.jpg` | 3264x1836 | validation | likely Brown Rust | Small pustules scattered at random with chlorotic mottling |
| 11 | `suhas__Images_Leaf-Rust__00611.jpg` | 1920x1920 | train | likely Brown Rust | Sparse, scattered pustules on a leaf held in the hand |
| 12 | `suhas__Images_Leaf-Rust__00571.jpg` | 4032x2268 | train | not a usable wheat-leaf image | Field of young wheat in rows; no disease visible |
| 13 | `suhas__Images_Leaf-Rust__01801.jpg` | 5184x3456 | train | not a usable wheat-leaf image | Wheat head, not a leaf: bright yellow pustules on a glume (glume infection is more typical of stripe rust) |
| 14 | `suhas__Images_Leaf-Rust__03741.jpg` | 183x275 | test | unclear | Low resolution, watermark; pale flecks only |
| 15 | `suhas__Images_Leaf-Rust__03911.jpg` | 195x259 | train | unclear | Dense orange-brown coating with a row-like texture; 195x259 |
| 16 | `suhas__Images_Leaf-Rust__08051.png` | 390x280 | train | likely Brown Rust | Round orange pustules scattered at random |
| 17 | `suhas__Images_Leaf-Rust__00511.jpg` | 4272x2848 | train | likely Brown Rust | Sparse, scattered pustules on detached seedling leaves |
| 18 | `suhas__Images_Leaf-Rust__08671.jpg` | 400x268 | train | likely Brown Rust | Pustules with chlorotic halos, scattered |
| 19 | `suhas__Images_Leaf-Rust__03481.jpg` | 184x274 | train | not a usable wheat-leaf image | Stem/culm, not a leaf: erumpent pustules with torn epidermis - stem-rust presentation |
| 20 | `suhas__Images_Leaf-Rust__06591.png` | 430x643 | train | likely Yellow/Stripe Rust | Pustules in stripes along the veins, leaf yellowing |
| 21 | `suhas__Images_Leaf-Rust__03771.jpg` | 275x183 | train | likely Brown Rust | Pustules with chlorotic halos, scattered |
| 22 | `suhas__Images_Leaf-Rust__07141.jpg` | 3264x1836 | train | likely Yellow/Stripe Rust | Long chlorotic stripe with pustules in a single line |
| 23 | `suhas__Images_Leaf-Rust__04501.jpg` | 236x214 | train | not a usable wheat-leaf image | Multi-panel scientific figure with day labels |
| 24 | `suhas__Images_Leaf-Rust__00101.png` | 1050x762 | validation | not a usable wheat-leaf image | Lettered figure of four leaves (a-d) |
| 25 | `suhas__Images_Leaf-Rust__02911.jpg` | 267x189 | train | unclear | Stock watermark across the leaves; low resolution |
| 26 | `suhas__Images_Leaf-Rust__05261.jpg` | 5472x3648 | validation | likely Brown Rust | Small pustules scattered with chlorosis (low confidence) |
| 27 | `suhas__Images_Leaf-Rust__02861.jpg` | 305x165 | test | unclear | Senesced leaf with fine speckling; low resolution |
| 28 | `suhas__Images_Leaf-Rust__03701.jpg` | 165x306 | train | likely Yellow/Stripe Rust | Continuous stripe of pustules confined between veins |
| 29 | `suhas__Images_Leaf-Rust__05221.jpg` | 5472x3648 | test | likely Brown Rust | Small pustules scattered at random |
| 30 | `suhas__Images_Leaf-Rust__01801.jpeg` | 237x354 | train | likely Brown Rust | Pustules with chlorotic halos, scattered |
| 31 | `suhas__Images_Leaf-Rust__05851.jpg` | 2000x1300 | train | not a usable wheat-leaf image | A lawn of turf grass; no wheat leaf, no disease |
| 32 | `suhas__Images_Leaf-Rust__01851.jpg` | 1277x1280 | train | not a usable wheat-leaf image | Lettered figure (A-E) from a leaf-rust (Lr21) experiment |
| 33 | `suhas__Images_Leaf-Rust__05471.jpg` | 5472x3648 | train | likely Brown Rust | Small pustules scattered at random |
| 34 | `suhas__Images_Leaf-Rust__05301.jpg` | 5472x3648 | train | likely Brown Rust | Round pustules scattered, with chlorotic flecks |
| 35 | `suhas__Images_Leaf-Rust__04281.jpg` | 275x183 | train | likely Yellow/Stripe Rust | Canopy of leaves with longitudinal orange stripes (medium confidence) |
| 36 | `suhas__Images_Leaf-Rust__00851.png` | 511x764 | train | likely Brown Rust | Dense round pustules scattered over the whole blade - textbook leaf rust |
| 37 | `suhas__Images_Leaf-Rust__07281.jpg` | 3264x1836 | train | unclear | Dark linear streaks (possibly telia) - neither typical presentation |
| 38 | `suhas__Images_Leaf-Rust__08401.jpg` | 195x280 | validation | not a usable wheat-leaf image | Stock photo of plants; no rust visible |
| 39 | `suhas__Images_Leaf-Rust__02451.jpg` | 1920x1920 | train | likely Yellow/Stripe Rust | Pustules in vein-parallel rows with chlorotic stripes |
| 40 | `suhas__Images_Leaf-Rust__05831.jpg` | 1920x1080 | train | likely Brown Rust | Pustules scattered at random |
| 41 | `suhas__Images_Leaf-Rust__01551.jpg` | 5184x3456 | train | likely Yellow/Stripe Rust | Long orange stripes along the veins |
| 42 | `suhas__Images_Leaf-Rust__08581.jpg` | 195x280 | validation | not a usable wheat-leaf image | Stock photo of grass; no rust visible |
| 43 | `suhas__Images_Leaf-Rust__04721.jpg` | 259x194 | train | not a usable wheat-leaf image | Composite with an inset; stems with stem-rust-like pustules |
| 44 | `suhas__Images_Leaf-Rust__05501.jpg` | 5472x3648 | validation | likely Brown Rust | Small pustules scattered, chlorotic flecks |
| 45 | `suhas__Images_Leaf-Rust__03401.jpg` | 275x184 | train | unclear | Yellow-orange pustules in a patch; arrangement ambiguous |
| 46 | `suhas__Images_Leaf-Rust__01391.png` | 390x280 | validation | not a usable wheat-leaf image | No rust pustules: discoloured leaf with purple/dark spots - a different condition |
| 47 | `suhas__Images_Leaf-Rust__00961.jpg` | 2848x4272 | train | likely Yellow/Stripe Rust | Many leaves with long yellow-orange stripes (high confidence) |
| 48 | `suhas__Images_Leaf-Rust__03861.jpg` | 300x168 | train | unclear | Heavily processed image; uniform yellow coating |
| 49 | `suhas__Images_Leaf-Rust__04631.jpg` | 275x183 | validation | likely Yellow/Stripe Rust | Orange stripes along the veins |
| 50 | `suhas__Images_Leaf-Rust__01561.jpg` | 3264x1836 | train | likely Yellow/Stripe Rust | Long yellow stripe with fine pustules (medium confidence) |
| 51 | `suhas__Images_Leaf-Rust__01941.png` | 462x280 | train | not a usable wheat-leaf image | Stock field panorama; no leaf-level disease |
| 52 | `suhas__Images_Leaf-Rust__07201.jpg` | 390x496 | validation | likely Yellow/Stripe Rust | Strong yellow-orange stripes along the veins (high confidence) |
| 53 | `suhas__Images_Leaf-Rust__05541.jpg` | 5472x3648 | test | likely Brown Rust | Round pustules scattered at random |
| 54 | `suhas__Images_Leaf-Rust__02971.jpg` | 184x274 | train | likely Brown Rust | Pustules with chlorotic halos, scattered (same photo as #30, correctly grouped) |
| 55 | `suhas__Images_Leaf-Rust__05711.png` | 283x213 | train | unclear | Canopy of senescing leaves, 283x213; pustules not resolvable |
| 56 | `suhas__Images_Leaf-Rust__05431.jpg` | 5472x3648 | train | likely Brown Rust | Small pustules scattered at random |
| 57 | `suhas__Images_Leaf-Rust__03631.jpg` | 300x168 | train | unclear | Low resolution; small pustules, arrangement unclear |
| 58 | `suhas__Images_Leaf-Rust__04311.jpg` | 259x195 | train | unclear | Same photograph as #55 (correctly grouped); not resolvable |
| 59 | `suhas__Images_Leaf-Rust__01411.png` | 390x280 | train | likely Yellow/Stripe Rust | Fine yellow-orange stripes along the full leaf (high confidence) |
| 60 | `suhas__Images_Leaf-Rust__03111.jpg` | 259x194 | validation | likely Brown Rust | Four detached leaves; dense scattered pustules |
