# External test set — image audit

**Status: audit only. No image byte was modified, no file was renamed or moved,
and `manifest.csv` was not changed.** Every proposed status below is a
recommendation awaiting your decision.

**Method:** visual inspection of the files on disk against published
descriptions of the four configured diseases, plus the project's own local
image metrics (`basic_quality_metrics`, `leaf_mask`). **No Claude API call, no
external vision service, no network request, and no API key were used.** Where
a species cannot be separated from the image, this document says so instead of
choosing one.

---

## 1. What is present

10 of 20 images. All from the Brown Rust and Yellow Rust batches; the Healthy
and Powdery Mildew batches never reached the environment (Section 5).

| File | sha256 (first 16) | Dimensions | Format | Manifest label |
|---|---|---|---|---|
| ext_01.jpg | `ce958d627ed243ba` | 1123×2000 | JPEG | Brown Rust |
| ext_02.jpg | `d0685c66d1bbed8b` | 168×158 | JPEG | Brown Rust |
| ext_03.jpg | `b3ca380d491dd4f9` | 159×300 | JPEG | Brown Rust |
| ext_04.jpg | `8802cb970e789aaa` | 335×597 | JPEG | Brown Rust |
| ext_05.jpg | `d3b95ace5e1fd1a9` | 554×554 | JPEG | Brown Rust |
| ext_06.webp | `0a147ee919566c38` | 540×810 | WEBP | Yellow Rust |
| ext_07.jpg | `d65306fd5046d302` | 511×391 | JPEG | Yellow Rust |
| ext_08.jpg | `90dcc5895fdcc452` | 597×335 | JPEG | Yellow Rust |
| ext_09.jpg | `a414072a619507a7` | 387×516 | JPEG | Yellow Rust |
| ext_10.jpg | `6b37e8ae46b32575` | 547×365 | JPEG | Yellow Rust |

---

## 2. The discriminators used

These are the published field characteristics the audit is judged against.
They are stated here so you can check the reasoning rather than take it on
trust.

| | **Brown / leaf rust** (*Puccinia triticina*) | **Yellow / stripe rust** (*P. striiformis*) | **Stem rust** (*P. graminis*) — **not a supported class** | **Powdery mildew** (*Blumeria graminis*) |
|---|---|---|---|---|
| Where | Upper leaf blade surface | Leaf blade, later glumes | Stems, leaf sheaths, blades | Leaf blade, sheath |
| Pustule shape | Round to slightly oval, discrete | Small, in narrow linear stripes bounded by veins | Large, elongated, often coalescing | Not pustules — fluffy mycelium |
| Arrangement | Scattered, **not** confined to vein rows | Confined **between veins**, forming stripes | Elongated along the stem axis | Irregular patches |
| Colour | Orange-brown to cinnamon | Yellow-orange | Dark reddish-brown | White/grey, later with black cleistothecia |
| Epidermis | Ruptures neatly, little tearing | Ruptures neatly | **Tears conspicuously**, leaving ragged flaps | Not ruptured |

The single most useful discriminator between the two rusts in this set is
**arrangement**, not colour: leaf rust scatters, stripe rust queues along the
veins.

---

## 3. The audit table

| Image | Current label | Proposed status | Reason | Needs human confirmation? |
|---|---|---|---|---|
| **ext_01.jpg** | Brown Rust | **VALID** | Dense, discrete, round orange-brown pustules distributed across the blade, crossing veins rather than queueing along them. Leaf blade clearly visible, field background. Textbook leaf rust. | No |
| **ext_02.jpg** | Brown Rust | **VALID** *(sub-input resolution — flag, do not exclude)* | 168×158 is below the model's 224 px input and is upscaled ×1.42. **But pixel count is not the same as diagnostic information:** this is a tight macro crop, so individual pustules are still separately resolvable — round, discrete, orange, scattered. Ground truth is establishable; Brown Rust is supported. | No — but report its result separately (see §4) |
| **ext_03.jpg** | Brown Rust | **NEEDS_HUMAN_CONFIRMATION** | 159×300, upscaled ×1.41. Unlike ext_02 this is a whole-blade shot, so each pustule occupies very few pixels and, with JPEG artefacts, individual uredinia **cannot be separated**. The lesion distribution shows a longitudinal, vein-aligned component that would point to Yellow Rust, but the resolution does not permit confirming that. | **Yes** |
| **ext_04.jpg** | Brown Rust | **OUT_OF_SCOPE** | The subject is a **stem/culm**, not a leaf blade — cylindrical, waxy blue-green, with an internode visible. Pustules are elongated, coalescing along the stem axis, dark rusty brown, and have **torn the epidermis into ragged flaps**. That combination is stem rust (*P. graminis*), which is **not one of the four configured classes**. See §4.1. | **Yes** |
| **ext_05.jpg** | Brown Rust | **LABEL QUESTIONABLE** → likely **Yellow Rust** | Pustules sit in well-defined rows running parallel to the veins, each elongated along the vein axis, with green tissue between the rows. That is the stripe-rust arrangement, not leaf rust's random scatter. See §4.2. | **Yes** |
| **ext_06.webp** | Yellow Rust | **VALID** | Long, continuous orange stripes confined between veins, running the length of the blade; a second leaf behind shows the same pattern. Textbook stripe rust. Only WEBP in the set. | No |
| **ext_07.jpg** | Yellow Rust | **VALID** | Clear alternating orange stripes and green tissue, strictly vein-bounded. Textbook stripe rust. | No |
| **ext_08.jpg** | Yellow Rust | **LABEL QUESTIONABLE** → likely **Brown Rust** | Pustules are discrete, round-to-oval, comparatively large and well separated, scattered across the blade **without** vein confinement. That is the leaf-rust arrangement. See §4.2. | **Yes** |
| **ext_09.jpg** | Yellow Rust | **VALID** *(medium confidence)* | Dense orange-yellow uredinia in longitudinal, vein-following rows over a chlorotic half-blade — consistent with stripe rust. Marked medium rather than high because the pustules are dense enough to have partly coalesced, which blurs the arrangement cue. A hand fills much of the frame (a real-world condition absent from training). | No |
| **ext_10.jpg** | Yellow Rust | **NEEDS_HUMAN_CONFIRMATION** | Morphology is intermediate: the pustules are discrete and round (leaf-rust-like) yet show partial vein alignment (stripe-rust-like), which is what early stripe rust looks like before uredinia coalesce. Both readings are defensible from this image. Stock-library macro on a plain black background with a visible watermark. | **Yes** |

### Summary

| Status | Count | Images |
|---|---|---|
| VALID | 5 | ext_01, ext_02, ext_06, ext_07, ext_09 |
| LABEL QUESTIONABLE | 2 | ext_05, ext_08 |
| NEEDS_HUMAN_CONFIRMATION | 2 | ext_03, ext_10 |
| OUT_OF_SCOPE | 1 | ext_04 |
| TOO_LOW_RESOLUTION | 0 | — (see §4.3) |

---

## 4. The specific investigations you asked for

### 4.1 ext_04 — stem rust, not Brown Rust

Three independent features point the same way:

1. **The organ is wrong.** The subject is a cylindrical culm with a visible
   node, not a flat blade. Brown rust is overwhelmingly a leaf-blade disease.
2. **The pustules are wrong for leaf rust.** They are elongated and coalescing
   along the stem axis, forming irregular masses, rather than the small
   discrete ovals leaf rust produces.
3. **The epidermis is shredded.** Torn flaps of tissue surround the pustules.
   Conspicuous epidermal tearing is the classic stem-rust sign and is exactly
   what distinguishes it from the other two rusts, which erupt neatly.

**The honest counter-argument:** leaf rust and stripe rust can both occur on
leaf sheaths, and a sheath wraps the stem, so "it is on a stem-like structure"
is not by itself decisive. It is the *combination* — organ plus elongation plus
epidermal tearing plus dark brown colour — that makes stem rust much the better
explanation. Definitive separation needs a microscope and urediniospore
morphology, which no photograph provides.

**Recommendation:** mark **OUT_OF_SCOPE** and exclude it from the headline
accuracy count, reporting it separately as an out-of-scope probe. Rationale:
none of the four classes is the correct answer, so whatever the model outputs
is wrong by construction. Scoring it inside the accuracy figure means the test
set silently contains an unanswerable question, which makes the headline number
mean something other than what it appears to mean. Keeping it as a labelled
out-of-scope probe is strictly more informative — it becomes a direct test of
whether the system can say "I don't know", which is exactly the capability
under investigation in `docs/claude_vision_investigation.md`.

### 4.2 ext_05 and ext_08 — an apparent swapped pair

| | ext_05 (labelled Brown Rust) | ext_08 (labelled Yellow Rust) |
|---|---|---|
| Arrangement | Rows parallel to the veins, green tissue between | Scattered, not vein-confined |
| Pustule shape | Elongated along the vein axis | Round to oval, discrete |
| Pustule spacing | Dense within rows | Well separated |
| Reads as | **Yellow Rust** | **Brown Rust** |

Each image on its own reads as the *other* class. That they are one image from
each batch, and that each matches the other's label, is the pattern you would
expect from two files being transposed when the batches were assembled.

**This materially changes the interpretation of the V2 result you reported.**
V2 scored Brown Rust 0/5. Of those five images, one (ext_04) has no correct
answer available, and one (ext_05) may carry the wrong label. If V2 called
ext_05 "Yellow Rust", it may have read the image correctly and been marked
wrong. The corrected denominator could be 3, not 5 — but **this cannot be
settled from the images alone and needs your confirmation.** I have not changed
either label.

### 4.3 ext_02 and ext_03 — low resolution, different verdicts

You asked whether these are valid test images or unsuitable. **They are not the
same case, and treating them alike would be wrong.**

The metric that matters is not pixel count but whether an individual pustule is
separately resolvable, because pustule shape and arrangement are the entire
basis for telling the two rusts apart.

| | ext_02 (168×158) | ext_03 (159×300) |
|---|---|---|
| Framing | Tight macro crop | Whole blade in frame |
| Pixels per pustule | Enough — pustules are individually visible | Too few — pustules blur into a mottled mass |
| Ground truth establishable? | **Yes** | **No** |
| Verdict | **VALID**, flagged as sub-input resolution | **NEEDS_HUMAN_CONFIRMATION** |

A note on the project's own blur metric, which is relevant to the quality-gate
design proposed in `docs/claude_vision_investigation.md`:

| File | Dimensions | `sharpness_laplacian_var` |
|---|---|---|
| ext_02.jpg | 168×158 | **0.0896** ← highest in the set |
| ext_03.jpg | 159×300 | 0.0376 |
| ext_07.jpg | 511×391 | 0.0238 |
| ext_09.jpg | 387×516 | 0.0158 |
| ext_01.jpg | 1123×2000 | **0.0043** ← lowest in the set |

**Variance of the Laplacian is not scale-invariant.** The two smallest images
score as the *sharpest* and the largest image as the *blurriest*, because
downscaling concentrates high-frequency energy per pixel. Any blur gate built
on this metric must normalise for resolution first, or it will reject good
large photographs and pass unusable small ones. Recorded here so the gate is
not built on a false assumption.

**`leaf_mask` does not help either.** Leaf-pixel fraction across these ten
images ranges from 82.4% to 97.7%, and **ext_04 — the stem — scores 96.1%**.
The mask finds plant tissue, not leaf blades, so it cannot catch the
out-of-scope case on its own.

---

## 5. Missing images — 10 of 20

| Class | Expected | Received | Status |
|---|---|---|---|
| Brown Rust | 5 | 5 | Complete |
| Yellow Rust | 5 | 5 | Complete |
| **Healthy** | 5 | **0** | **Missing — never reached disk** |
| **Powdery Mildew** | 5 | **0** | **Missing — never reached disk** |

Both missing batches rendered in the conversation but no file was written to
the container. The upload numbering is the evidence: it runs `1`–`5` (Brown
Rust), then jumps straight to `6`–`10` (Yellow Rust), with nothing in between,
despite two batches having been sent in that gap.

**No replacement, substitute, downloaded or generated image has been created
for either class, and none will be.** The set stands at 10 until the real files
arrive.

---

## 6. Isolation guarantees (unchanged)

These images remain excluded from every part of V3 by construction:

* **not** copied into `data/v3_raw/` or any dataset directory;
* **not** used in training, validation, augmentation or background harvesting;
* **not** used for model selection or threshold tuning;
* their labels are **not** used to tune anything.

`data/external_test/` sits outside every path in `config.paths`, and
`tests/test_external_exclusion.py` (7 tests, passing) enforces it — including a
test that spies on image inspection during `prepare()` to prove the directory
is never read.

---

## 7. Decisions required from you

Nothing below has been applied.

| # | Question | My recommendation |
|---|---|---|
| 1 | **ext_04** — is it stem rust? | Mark `OUT_OF_SCOPE`; score separately, not in headline accuracy |
| 2 | **ext_05 / ext_08** — are these two swapped? | Swap the labels, or confirm they are correct as they stand |
| 3 | **ext_03** — species unresolvable at 159×300 | Confirm the label, or drop it from the headline count |
| 4 | **ext_10** — morphology intermediate | Confirm the label, or drop it from the headline count |
| 5 | **ext_02** — keep at 168×158? | Keep as VALID; report separately as a sub-resolution case |
| 6 | **The 10 missing images** | Re-send Healthy and Powdery Mildew, or proceed on 10 and say so in every result |

Until 1–4 are settled, any accuracy figure computed on this set carries a
±2-image ambiguity on a denominator of 10 — that is ±20 percentage points,
which is larger than most of the differences V3 is being built to detect.
