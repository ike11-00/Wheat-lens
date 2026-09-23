# Investigation: a Claude vision/reasoning layer for Wheat Lens

**Status: investigation only. Nothing was implemented, no API key exists in this
project, no image was sent anywhere, and no model, config or dataset was
changed.** Every claim about the Claude API below is taken from the official
vision documentation (fetched 2026-09-23) and the current model/pricing table,
and is cited where it matters. Every claim about Wheat Lens is read from the
code in this repository. Where something is unknown, it says so rather than
estimating.

---

## A. Current Wheat Lens architecture

### A.1 The pipeline as built

```
config/config.yaml  ──────── single source of truth (classes, paths, sizes, thresholds)
        │
data/raw/<Class>/           src/data/validate_dataset.py    corrupt / duplicate / imbalance report
        │                   src/data/prepare_dataset.py     QC → sha256 dedup → pHash grouping →
        ▼                                                   leakage-safe split by whole groups
data/{train,validation,test}/
        │                   src/data/augment_backgrounds.py (v2 only) composite leaves onto
        │                                                   harvested/procedural backgrounds
        ▼
src/model/build_model.py    MobileNetV2(imagenet) + BackbonePreprocessing (in-graph) + GAP →
        │                   Dropout → Dense(4)
src/model/train.py          phase 1 frozen lr 1e-4 (20 ep) → phase 2 unfreeze 40 layers lr 1e-5 (10 ep)
        ▼
models/v2/{model.keras, labels.json, history.json, run_metadata.json, …}
        │
src/model/predict.py        Predictor: load → predict_proba → _format → _uncertainty
        ▼
app/app.py                  Flask: POST /api/predict → JSON
```

### A.2 How V2 actually works

`models/v2/model.keras` (md5 `8773a3305bebafc7e84487b0790b2989`) is MobileNetV2
with ImageNet weights, 2,263,108 parameters, input 224×224×3, four softmax
outputs in the order recorded in `models/v2/labels.json`.

Two details matter for anything built on top of it:

1. **Preprocessing is inside the graph.** `BackbonePreprocessing`
   (`src/model/build_model.py`) is a registered Keras layer, so callers hand the
   network raw 0–255 pixels and cannot get the scaling wrong. Any second
   opinion layer must not try to "pre-process for the model" itself.
2. **`labels.json` is the authority on class order, not `config.yaml`.**
   `Predictor.__init__` reads it and overrides the config's class names, image
   size, confidence threshold and version. A model trained on a different class
   list therefore cannot be silently reinterpreted with today's config. Any new
   layer must read class names from the predictor, never from the config.

What V2 was for, and what it achieved, measured: background robustness rose
from 21.5% (v1) to 95.3%, and the share of background-replaced test images
called Yellow Rust fell from 71.8% to 0.7%. Headline test accuracy 99.68%.
**On 20 real-world photographs it scored 9/20, with Brown Rust 0/5.** That gap
is the problem this investigation is ultimately about.

### A.3 The uncertainty machinery that already exists

`Predictor._uncertainty` computes three signals from the softmax vector and
returns them on every prediction:

| Signal | Definition | Configured threshold |
|---|---|---|
| `confidence` | max softmax | `inference.confidence_threshold` = 0.60 |
| `normalised_entropy` | Shannon entropy ÷ log(4) | `> 0.80` ⇒ uncertain |
| `margin` | top-1 − top-2 probability | `< 0.10` ⇒ uncertain |

An image is flagged `uncertain` if any of the three fires, and
`uncertainty.reasons` carries human-readable explanations. The method's own
docstring already states the limitation plainly: *"this is a softmax confidence
heuristic, not a true novelty detector… a low flag rate is NOT evidence that an
input is in distribution."* That is exactly the hole a second opinion layer
would be filling.

### A.4 Where prediction and inference live

| Concern | File | Entry point |
|---|---|---|
| Model loading, softmax, uncertainty | `src/model/predict.py` | `Predictor.predict(source)` |
| Batch prediction | `src/model/predict.py` | `Predictor.predict_many(...)` |
| CLI prediction | `src/model/predict.py` | `main()` |
| Test-split evaluation | `src/model/evaluate.py` | imports `Predictor` |
| Error analysis | `src/testing/error_analysis.py` | imports `Predictor` |
| Realistic-condition testing | `src/testing/realistic_testing.py` | imports `Predictor` |
| Per-image diagnosis | `src/testing/diagnose_image.py` | imports `Predictor` |
| External 20-image scoring | `src/testing/external_test.py` | imports `Predictor` |
| Web prediction | `app/app.py` | `POST /api/predict` |

**Seven modules import `Predictor`, and five of them are measurement code.**
This is the single most important constraint in this document and is why
Section B puts the Claude layer outside `Predictor` rather than inside it.

### A.5 How the app receives an upload and produces a prediction

`app/app.py::create_app` → `POST /api/predict`, in order:

1. `503` if no trained model is on disk.
2. `400` if no `image` field, or no filename.
3. `415` if the extension is not in `data.supported_extensions`
   (`.jpg .jpeg .png .webp .bmp`).
4. `413` if the body exceeds `data.max_upload_mb` (10 MB, also enforced by
   Flask's `MAX_CONTENT_LENGTH`).
5. `open_image(io.BytesIO(data))` — decodes and converts to RGB; a corrupt file
   becomes a clean `400`, never a traceback.
6. `400` if either edge is below `data.min_image_pixels` (32 px).
7. `predictor.predict(image)` → `load_image_array` resizes to 224×224 bilinear →
   `model(batch, training=False)` → softmax → `_format` → `_uncertainty`.
8. Response JSON is enriched with `filename`, `image_dimensions`,
   `probabilities_sorted`, `probabilities_ordered`, and returned `200`.

Everything runs in-process. There is no queue, no async, no background worker.

### A.6 Existing model/API integrations

**None.** `grep` for `anthropic|openai|requests|httpx|api_key|API_KEY` across
`src/`, `app/`, `config/` and `requirements.txt` returns nothing. The only
network-capable dependency in `requirements.txt` is Flask (inbound) and a
commented-out optional `kaggle` line. There is no HTTP client, no secrets
handling, no `.env` loading, and no outbound call anywhere in the codebase.
Adding a Claude layer would be this project's **first** outbound network
dependency, which is a bigger architectural change than it looks — see
Section J.3.

---

## B. Where a second vision model could be integrated

There are four candidate seams. Only one of them is correct.

| # | Seam | Verdict |
|---|---|---|
| 1 | Inside `Predictor.predict()` | **No.** Seven modules import `Predictor`, five of them measurement code (`evaluate`, `error_analysis`, `realistic_testing`, `diagnose_image`, `external_test`). Putting a network call there means every evaluation run makes API calls, costs money, becomes non-deterministic, and stops measuring *the model*. It would also break the offline test suite. |
| 2 | Inside `app/app.py::predict()` | **No, not directly.** Workable but wrong place: the logic would be untestable without a Flask request context, and the CLI would never get it. |
| 3 | A new `src/review/` package, called *after* `Predictor.predict()` by whichever caller wants it | **Yes.** |
| 4 | A separate microservice | Overkill for a single-process Flask app; revisit only if the app is ever deployed for multiple users. |

### The seam, concretely

```
app/app.py  POST /api/predict
    │
    ├── existing steps 1–7 (unchanged)              ← ML result produced exactly as today
    │
    ├── src/review/quality.py   assess(image)       ← LOCAL, no network
    │        └─ resolution, sharpness, leaf-presence, exposure
    │
    ├── src/review/gate.py      should_review(ml_result, quality)   ← LOCAL, no network
    │        └─ returns (bool, reasons)   ── if False, respond exactly as today
    │
    ├── src/review/claude.py    review(image, ml_result, quality)   ← the ONLY network call
    │        └─ returns a ReviewOpinion, or None on any failure
    │
    └── src/review/merge.py     present(ml_result, quality, opinion)
             └─ produces the user-facing verdict; ML class is never changed
```

Four properties this layout buys, all of which matter:

* `Predictor` is untouched, so every existing measurement stays valid and the
  test suite stays offline.
* `quality.py` and `gate.py` have **no** network dependency, so most of the
  value (Section H) is available with the API switched off entirely.
* `claude.py` returning `None` is a normal outcome, not an error — the app
  degrades to today's behaviour on timeout, rate limit, or no key.
* Everything is unit-testable with a fake review client; no test ever needs a
  network.

---

## C. Does the Claude API support this workflow?

Yes, with four caveats that are genuinely relevant to *this* project.

### C.1 What is supported

| Requirement | Supported? | Detail |
|---|---|---|
| Send an image | Yes | `image` content block, `source.type` = `base64`, `url`, or `file_id` |
| Image formats | **Partly** | JPEG, PNG, GIF, WebP. **BMP is not supported** — and `.bmp` *is* in this project's `supported_extensions` (see C.2) |
| Image size | Yes | ≤ 10 MB base64, ≤ 8000×8000 px on the Claude API |
| Reason about the image *and* the ML output together | Yes | Text blocks alongside the image in the same user turn; docs recommend image-before-text ordering |
| Return machine-readable output | Yes | Structured outputs via `output_config: {format: {...}}` — schema-valid JSON, no parsing of prose |
| Express uncertainty / abstain | Yes | It is a prompt-design question, not an API limitation |
| Deterministic output | **No** | `temperature` is removed on current models (400 if sent). Identical inputs can give differently-worded outputs. This is why the ML model must stay primary |

### C.2 Caveat 1 — the BMP gap

`config.yaml` lists `.bmp` in `data.supported_extensions`, so the app accepts
BMP uploads today. The Claude API does not accept BMP. The layer must re-encode
to JPEG or PNG before sending — which it should be doing anyway (C.3).

### C.3 Caveat 2 — the 10 MB upload limit is not the 10 MB API limit

The app accepts uploads up to 10 MB. Base64 inflates payloads by about 33%, so
a 10 MB JPEG becomes ~13.3 MB of base64 and would be **rejected** by the API,
whose 10 MB limit is measured *after* encoding. Any implementation must
downscale and re-encode before sending; it cannot forward the original bytes.

### C.4 Caveat 3 — Claude's documented weakness overlaps ours exactly

The vision documentation states Claude "might hallucinate or make mistakes when
interpreting low-quality, rotated, or very small images **under 200 pixels**".

Of the ten external test images placed so far, **`ext_02.jpg` (168×158) and
`ext_03.jpg` (159×300) are under 200 px on at least one edge.** Both are Brown
Rust — the class V2 scored 0/5 on. So on the images where the ML model is
weakest, the documentation says the reviewer is also least reliable. This does
not sink the idea, but it does mean the honest job for those images is *"this
photograph is too small to classify"* — which the local quality check
(Section H) can say for free, without an API call at all.

The docs also note Claude **cannot** determine whether an image is
AI-generated. At least two of the supplied external images appear to be
stock-library material; do not expect the reviewer to flag that.

### C.5 Caveat 4 — the agronomy disclaimer stands

The docs' healthcare note ("not a substitute for professional medical advice or
diagnosis") is the same posture this project already takes in `DISCLAIMER`.
Nothing in this design may be presented as a diagnosis. That constraint is
unchanged and non-negotiable.

### C.6 Recommended request shape

```
model            claude-opus-5          (default; see E for the cheaper options)
thinking         {"type": "adaptive"}   (on by default on this model)
output_config    {"format": {...}}      structured JSON, schema below
max_tokens       ~1024
content          [ image block, then text block ]   (docs: image before text)
```

---

## D. What would need to be sent to the API

### D.1 Sent

| Item | Why | Form |
|---|---|---|
| The photograph | The thing being reviewed | Downscaled to a 768 px long edge, re-encoded JPEG q90, **EXIF stripped** |
| The four class names + one-line descriptions | The reviewer must know the closed world it is judging against | Read from `Predictor.class_names`, never hardcoded |
| The ML model's full probability vector | So the reviewer can agree, disagree, or qualify | `{"Healthy": 0.03, "Yellow Rust": 0.11, …}` |
| The ML uncertainty signals | Entropy / margin / threshold, already computed | From `result["uncertainty"]` |
| Locally measured quality metrics | Resolution, Laplacian sharpness, brightness, saturation, leaf-pixel fraction | From `basic_quality_metrics` + `leaf_mask` |

### D.2 Deliberately NOT sent

| Item | Why not |
|---|---|
| The original filename | Users name files after themselves, their farm or their location |
| EXIF / GPS metadata | A field photograph routinely carries precise coordinates. The docs confirm Claude does not parse metadata anyway, so sending it has cost and no benefit |
| Any user identifier, IP, or session | Not needed for the task |
| The original full-resolution bytes | Costs tokens for fidelity the task does not need (E) |
| The true label, when one exists | Would let the reviewer "agree" with the answer key; poisons any evaluation |

> **Bug found while investigating, unrelated to Claude:** nothing in the
> pipeline calls `PIL.ImageOps.exif_transpose`. A phone photograph carrying an
> EXIF orientation flag is fed to the model **sideways** today, and would be
> sent to the API sideways too — and "rotated" is on Claude's documented list
> of failure modes. Worth fixing on its own merits; it is one line in
> `open_image`. Not fixed here, per your instruction not to change anything.

### D.3 Proposed response schema

```json
{
  "leaf_present":        true,
  "image_quality":       "good | marginal | insufficient",
  "quality_reasons":     ["…"],
  "visible_symptoms":    ["…"],
  "consistent_with":     ["Brown Rust", "Yellow Rust"],
  "within_four_classes": true,
  "agrees_with_model":   true,
  "reviewer_confidence": "high | medium | low",
  "explanation":         "one or two sentences for the user"
}
```

Note what is absent: **no predicted class and no probability.** The schema is
deliberately incapable of expressing "the answer is X with N% confidence",
because that is the ML model's job. `consistent_with` is a *set*, which is how
you say "rust of some kind" without picking one.

---

## E. Cost and usage

### E.1 How image cost is calculated

Claude bills images by 28×28-pixel patches: **`⌈width/28⌉ × ⌈height/28⌉` visual
tokens**. Claude 4.7 and later are high-resolution tier — long edge capped at
2576 px, at most 4784 visual tokens; earlier models cap at 1568 px / 1568
tokens. Oversized images are downscaled server-side before billing.

| Send size | Visual tokens |
|---|---|
| 224×224 (the model's own input) | 64 |
| 448×448 | 256 |
| 512×512 | 361 |
| **768×768** | **784** |
| 1024×768 | 1036 |

### E.2 Measured on the actual external test images

Downscaled to a 768 px long edge — the size proposed in D.1:

| File | Original | Sent as | Visual tokens |
|---|---|---|---|
| ext_01.jpg | 1123×2000 | 431×768 | 448 |
| ext_02.jpg | 168×158 | unchanged | 36 |
| ext_03.jpg | 159×300 | unchanged | 66 |
| ext_04.jpg | 335×597 | unchanged | 264 |
| ext_05.jpg | 554×554 | unchanged | 400 |
| ext_06.webp | 540×810 | 512×768 | 532 |
| ext_07.jpg | 511×391 | unchanged | 266 |
| ext_08.jpg | 597×335 | unchanged | 264 |
| ext_09.jpg | 387×516 | unchanged | 266 |
| ext_10.jpg | 547×365 | unchanged | 280 |

**Mean 282 visual tokens per image.** These photographs are small; a modern
phone photograph would hit the 768-px cap and cost nearer 500–780.

### E.3 Cost per review call

Assuming 282 image tokens + ~1,000 text tokens in (system prompt, class
definitions, ML probabilities, quality metrics) and ~350 tokens out:

| Model | Per call | Per 1,000 calls | Per 1,000 uploads if the gate fires on 25% |
|---|---|---|---|
| **claude-opus-5** ($5 / $25 per MTok) | **$0.0152** | **$15.16** | **$3.79** |
| claude-sonnet-5 ($2 / $10) | $0.0061 | $6.06 | $1.52 |
| claude-haiku-4-5 ($1 / $5) | $0.0030 | $3.03 | $0.76 |

For a personal project this is negligible — a few dollars for thousands of
reviews. Two honest caveats:

* **The gate rate is a guess.** 25% is illustrative. The real rate is
  measurable only after the gate exists and sees real uploads, and it is the
  single biggest lever on cost.
* **Prompt caching probably will not help here.** The minimum cacheable prefix
  is model-dependent (512–4096 tokens); a ~1,000-token system prompt may fall
  below it and silently not cache. Verify with `usage.cache_read_input_tokens`
  rather than assuming.

Use `client.messages.count_tokens` on a sample of real uploads before quoting
any of these numbers as a budget.

---

## F. Privacy

This project currently sends **nothing** anywhere. That changes the day a
Claude layer ships, and it is the most consequential change in this document.

### F.1 What Anthropic's documentation states

* Image uploads are **ephemeral** — not stored beyond the duration of the API
  request, and automatically deleted after processing.
* Anthropic **does not use uploaded images to train models**.
* Claude **does not parse or receive image metadata**.

### F.2 What is still on you

| Risk | Mitigation |
|---|---|
| GPS coordinates in EXIF | Re-encode before sending. Pillow drops EXIF by default when saving without `exif=`, but make it explicit rather than relying on a default |
| Faces, vehicles, buildings, documents in frame | Unavoidable in field photography. Consent is the only real control |
| Filenames leaking identity | Never send the filename (D.2) |
| Users not knowing their photo leaves the machine | **Opt-in, per upload, default off.** A checkbox in `index.html`, not a buried setting |
| Retention on your side | Don't log the image. Log the review *outcome* and a hash |
| Someone else's photo | Terms must say the uploader has the right to submit it |

### F.3 The licensing constraint that already applies

Independently of Claude: this model's training imagery comes from GitHub
repositories that, with one exception, carry **no licence at all**
(`docs/limitations.md`). The model is not distributable and must not be
deployed publicly or commercially as things stand. A privacy design for a
public deployment is therefore premature — this layer should be built for a
single-user local app first.

---

## G. Handling disagreement between the ML model and Claude

### G.1 The rule

**The ML model's predicted class is the answer. Claude can qualify it, add a
caveat, or ask for a better photograph. Claude cannot change it.**

This is not diplomacy, it is measurability. The moment the reviewer can
override the class, every metric this project has — test accuracy, per-class
F1, background robustness, the external 20 — stops describing anything you can
reproduce, because the output is no longer a deterministic function of the
model. Keeping the ML class fixed keeps the whole evaluation apparatus valid.

### G.2 The decision table

| ML confidence | Reviewer | User-facing result |
|---|---|---|
| High (≥ 0.60) | agrees | **"Brown Rust — 82%."** Normal result, reviewer note optional |
| High | agrees on family, not species (`consistent_with` = both rusts) | "**Likely Brown Rust (82%).** A second check agrees this is a rust, but could not separate brown from yellow rust from this image." |
| High | disagrees, or says quality is insufficient | "**Possibly Brown Rust (82%) — treat with caution.** A second check found the image quality insufficient for reliable classification. Please upload a clearer photograph." |
| High | says no leaf is present, or outside the four classes | "**This may not be something Wheat Lens can identify.** The model's best guess is Brown Rust, but a second check could not find a wheat leaf blade / suggests a condition outside the four supported diseases." |
| Low (< 0.60) or `uncertain` | agrees | "**Possibly Brown Rust, but confidence is low.** …" — today's low-confidence message, with the reviewer's reason attached |
| Low or `uncertain` | disagrees | "**Wheat Lens could not identify this reliably.**" Show the full probability bars and ask for a better photograph |
| any | call failed / no key / timeout | **Exactly today's output.** No mention of a second check |

### G.3 The part that actually pays for itself

Every disagreement is logged — image hash, ML vector, reviewer JSON, gate
reasons, timestamp. That log is a **labelled queue of the model's suspected
failures**, which is precisely what V3's error analysis has to be built from
today by hand. Even if the user-facing benefit turned out to be zero, the
disagreement log would still be worth having.

---

## H. Detecting unknown / out-of-scope images

Four of the six cases you listed need no API at all. Build those first.

### H.1 Local, no network, deterministic

| Case | Method | Already in the codebase? |
|---|---|---|
| Image too small | `width`/`height` vs. the 224 px model input. Below it, detail is interpolated, not photographed | Partly — the app only rejects below 32 px |
| Image blurry | Variance of the Laplacian | **Yes** — `basic_quality_metrics(...)["sharpness_laplacian_var"]` |
| Over/under-exposed | `mean_brightness`, `brightness_std` | **Yes** — same function |
| No clear leaf blade | `leaf_mask()` + `min_leaf_fraction`, already used to skip frames during background augmentation | **Yes** — `src/data/augment_backgrounds.py` |
| ML low confidence | `confidence < 0.60`, `normalised_entropy > 0.80`, `margin < 0.10` | **Yes** — `Predictor._uncertainty` |
| Substantially unlike training data | Not currently possible. Options: distance to class centroids in the penultimate MobileNetV2 embedding, or a Mahalanobis / k-NN score over training embeddings | **No** — would be new, and is a better long-term answer than asking Claude |

**The pieces for four of the six checks already exist in this repository and
are currently used only by the training and evaluation code.** Wiring them into
the request path is local work with no API, no key, no cost and no privacy
question — and it addresses the two smallest external images directly.

### H.2 What genuinely needs a vision model

* **"Is this a wheat leaf at all?"** — `leaf_mask` finds *leaf-coloured pixels*,
  not wheat. It cannot tell wheat from maize, barley or grass. (Relevant: one
  of the Healthy images you sent looks like it may be a maize leaf.)
* **"Is the visible symptom outside the four classes?"** — septoria, stem rust,
  tan spot, nutrient deficiency, herbicide damage, insect damage. A four-output
  softmax cannot represent "none of these". Note `ext_04.jpg` in your own test
  set appears to show **stem rust on a culm**, which is out of scope by
  definition.
* **"Does the stated symptom match the image?"** — e.g. the model says Powdery
  Mildew but there is no white mycelium anywhere in frame.

### H.3 Proposed gate

Call the API only when at least one fires:

```
ml_confidence < 0.60
OR ml_result["uncertain"]                       # entropy or margin
OR min(width, height) < 224                     # below the model's own input
OR sharpness_laplacian_var < <calibrated>       # blurry
OR leaf_fraction < 0.04                         # config's min_leaf_fraction
OR mean_brightness outside [<lo>, <hi>]         # exposure
OR user explicitly asked for a second opinion
```

Every threshold marked `<calibrated>` must be set from the **validation split**,
never from the 20 external photographs. Rate-limit per session and fail open.

---

## I. Would this help the real-world failure? An honest answer

**It would not fix it.** V2 scores 9/20 because V2's learned features do not
transfer to field photographs. A review layer that cannot change the class
cannot turn a wrong answer into a right one — by design (G.1). Anyone promising
otherwise is promising the reviewer will override the model, which is a
different and worse system.

What it plausibly changes:

| Failure mode | Effect | Confidence in this claim |
|---|---|---|
| Confidently wrong on a bad photograph | Result is qualified or withheld instead of stated flatly | **High** — this follows from the gate, not from Claude's judgement |
| Wrong because the image is out of scope (`ext_04`, stem rust) | Flagged as possibly outside the four classes | **Medium** — depends on Claude's actual accuracy here, which is untested |
| Wrong because the photo is 168×158 (`ext_02`, `ext_03`) | Told the photograph is unusable, which is the truthful answer | **High** — a local resolution check does this without any API |
| Brown Rust ↔ Yellow Rust confusion | "A rust, but I can't separate the species" instead of a wrong species | **Low–Medium** — plausible but entirely unmeasured |
| Wrong on a good, in-scope, in-distribution photograph | **No effect at all** | **High** |

And one thing that cuts the other way: the two lowest-resolution images are
exactly where Claude's own documentation says it is unreliable (C.4). The
reviewer may be wrong there too. The local resolution check is the honest
answer for those, not a second opinion.

**Two of your own images may not be scoring what they appear to score.**
`ext_04` looks like stem rust (out of scope), and `ext_05` and `ext_08` look
like they may be swapped between the rust classes. If so, "Brown Rust 0/5" is
partly a labelling artefact rather than a pure model failure. That would be
worth resolving *before* using the 45% figure to justify architecture.

### The measurement that would settle it

Run V2 unchanged on all 20 images, and separately have a reviewer prompt
evaluate the same 20 (once you approve sending them). Then count:

* how many **wrong** predictions the gate would have caught (value);
* how many **right** predictions it would have needlessly qualified (harm);
* how often the reviewer's `consistent_with` contains the true class.

Those three numbers decide whether this ships. n=20 is small, so treat the
result as directional and not as a metric.

---

## J. Proposed V3/V4 architecture

### J.1 The pipeline

```
          USER UPLOADS WHEAT LEAF PHOTO
                      │
       ┌──────────────▼──────────────┐
       │  1. INTAKE                  │  decode, EXIF-transpose, strip metadata,
       │     app/app.py (as today)   │  extension + size checks
       └──────────────┬──────────────┘
                      │
       ┌──────────────▼──────────────┐
       │  2. IMAGE QUALITY CHECK     │  LOCAL. resolution, sharpness, exposure,
       │     src/review/quality.py   │  leaf-pixel fraction
       └──────────────┬──────────────┘
                      │  "insufficient" ──► stop here, ask for a better photo
                      │                     (no model call, no API call)
       ┌──────────────▼──────────────┐
       │  3. WHEAT LENS ML MODEL     │  UNCHANGED. MobileNetV2 v2 today, v3 later.
       │     src/model/predict.py    │  Always runs. Always the primary classifier
       └──────────────┬──────────────┘
                      │
              DISEASE PROBABILITIES  {Healthy, Yellow Rust, Brown Rust, Powdery Mildew}
                      │              + confidence, entropy, margin
       ┌──────────────▼──────────────┐
       │  4. GATE                    │  LOCAL. H.3. Most uploads stop here and
       │     src/review/gate.py      │  return today's response unchanged
       └──────────────┬──────────────┘
                      │ gate fires
       ┌──────────────▼──────────────┐
       │  5. OPTIONAL CLAUDE REVIEW  │  claude-opus-5, structured output,
       │     src/review/claude.py    │  opt-in, timeout, fail-open → None
       └──────────────┬──────────────┘
                      │
       ┌──────────────▼──────────────┐
       │  6. MERGE + PRESENT         │  G.2 decision table. ML class never changes
       │     src/review/merge.py     │
       └──────────────┬──────────────┘
                      │
              FINAL USER-FACING RESULT   + disagreement log (G.3)
```

### J.2 Staging

| Stage | Contents | Network | Value |
|---|---|---|---|
| **V3.1** | Steps 1, 2, 4, 6 — quality check, gate, better messaging. Reviewer stubbed out | **None** | Catches the small/blurry/no-leaf cases; ships the whole UX; all of it testable offline |
| **V3.2** | Embedding-distance novelty score added to the gate | None | A real in/out-of-distribution signal, which softmax cannot give |
| **V4** | Step 5 — the Claude reviewer | Yes | Wheat-vs-not-wheat, out-of-scope symptoms, symptom/label consistency |

**V3.1 is worth building whether or not V4 ever happens**, and nothing in it
requires an API key, a secret, or a privacy policy.

### J.3 What ships with step 5, beyond the call itself

An outbound API dependency brings a tail this codebase has never carried:
secret management and a `.env` path; `anthropic` added to `requirements.txt`;
timeouts, retries and a rate limiter; a fail-open path on every error; an
opt-in consent control in the UI and a privacy notice; the disagreement log and
its retention policy; a cost ceiling; and a way to run the entire test suite
with the network off. Budget for that, not just for the call.

---

## K. Files that would eventually need modification

### New

| File | Purpose | Network |
|---|---|---|
| `src/review/__init__.py` | Package marker | — |
| `src/review/quality.py` | Resolution, sharpness, exposure, leaf-fraction → `QualityReport` | No |
| `src/review/gate.py` | `should_review(ml_result, quality) -> (bool, reasons)` | No |
| `src/review/schema.py` | `ReviewOpinion` dataclass + the JSON schema from D.3 | No |
| `src/review/claude.py` | The only module that imports `anthropic` | **Yes** |
| `src/review/merge.py` | The G.2 decision table → user-facing verdict | No |
| `src/review/log.py` | Disagreement log writer | No |
| `tests/test_quality.py` | Quality metrics on synthetic images | No |
| `tests/test_gate.py` | Gate fires/doesn't on constructed results | No |
| `tests/test_merge.py` | Every row of the G.2 table | No |
| `tests/test_review_client.py` | Fake client: timeout, malformed JSON, refusal, `None` | No |
| `docs/review_layer.md` | What it does, what it cannot do, thresholds and provenance | — |

### Modified

| File | Change | Risk |
|---|---|---|
| `config/config.yaml` | New `review:` block — `enabled: false`, model id, gate thresholds, timeout, max long edge, opt-in default | Low — additive, defaults off |
| `app/app.py` | Call quality → gate → review → merge after `predictor.predict()`; new response fields | **Medium** — the only request-path change |
| `app/templates/index.html` | Opt-in checkbox, reviewer note area, privacy line | Low |
| `app/static/script.js` | Render the reviewer note and the qualified verdict | Low |
| `app/static/style.css` | Styling for the note | Low |
| `requirements.txt` | `anthropic>=0.40` (V4 only) | Low |
| `docs/limitations.md` | What the reviewer can and cannot do; that it never overrides the class | Low |
| `docs/model.md` | Cross-reference | Low |
| `README.md` | Architecture diagram, opt-in, cost | Low |
| `.gitignore` | `.env` | Low |

### Explicitly NOT modified

`src/model/predict.py`, `src/model/build_model.py`, `src/model/train.py`,
`src/model/evaluate.py`, `src/model/compare_models.py`, everything under
`src/data/`, everything under `src/testing/`, `models/v1/`, `models/v2/`,
`data/external_test/`. **The trained model, the training pipeline and every
measurement path stay exactly as they are.** If a change to `Predictor` ever
looks necessary, that is the signal the design has drifted.

---

## Recommendation

**Build V3.1 now. Defer the Claude call to V4, and gate it behind a measurement.**

Why, in order of weight:

1. **Most of the value you described needs no API.** Four of your six cases —
   low confidence, blurry, too small, no leaf blade — are answerable from
   `basic_quality_metrics`, `leaf_mask` and `Predictor._uncertainty`, all of
   which already exist and are currently used only by training and evaluation
   code. That is local, deterministic, free, private, and testable offline.
2. **It addresses your worst images directly.** Two of the ten external
   photographs placed so far are under 200 px; the truthful response to those is
   "this photograph is too small", and a local check says so without a network
   call — and says it more reliably than a reviewer would, per Claude's own
   documentation.
3. **The remaining two cases genuinely need a vision model** — wheat-vs-not-wheat
   and out-of-scope symptoms such as the suspected stem rust in `ext_04`. Those
   are real, and they are what V4 is for.
4. **Cost is not the obstacle.** At `claude-opus-5`, ~$15 per thousand reviews,
   or under $4 per thousand uploads at a 25% gate rate. Use `claude-opus-5`
   unless you decide otherwise; `claude-sonnet-5` at ~$6 and `claude-haiku-4-5`
   at ~$3 per thousand are your call, not a default I should make for you.
5. **The architectural discipline is the whole game.** The reviewer must live
   outside `Predictor`, must never change the predicted class, and must fail
   open. Seven modules import `Predictor` and five of them are measurement code;
   the moment a network call gets inside it, this project stops being able to
   measure itself.

Two things I would want settled before V4:

* **Resolve the label questions on `ext_04`, `ext_05` and `ext_08`.** If Brown
  Rust 0/5 is partly a labelling artefact, the premise for the whole design
  shifts.
* **Run the A/B in I.** Count wrong predictions caught versus right predictions
  needlessly qualified. If the second number dominates, the reviewer makes the
  product worse and should not ship.

And the honest bottom line: **this layer makes Wheat Lens more trustworthy, not
more accurate.** Accuracy is V3's job — a better dataset and a better model.
Do not let a second opinion layer become a substitute for fixing the classifier.

---

*Nothing in this document has been implemented. No API key exists in this
project. No image has been sent to any external service.*
