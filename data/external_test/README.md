# External real-world test set — HELD OUT

Put your own real-world wheat photographs here, with their true labels in
`manifest.csv`. These images are a **final evaluation set only**.

## Rules this directory operates under

* **Never** read by `prepare_dataset`, `augment_backgrounds`, training,
  validation, or any hyper-parameter choice. Enforced in code and covered by
  tests, not left to convention.
* Images are **not modified** — not resized, renamed, re-encoded or augmented.
  The evaluator reads them where they lie and does its resizing in memory.
* Labels in the manifest are used **only** to score predictions after the fact.

## How to add your 20 images

1. Copy the exact files you used for the V2 trial into this directory.
2. Generate a manifest skeleton listing whatever is present:

   ```bash
   python -m src.testing.external_test --template
   ```

3. Open `data/external_test/manifest.csv` and fill in `true_class` for each
   row. Valid values are exactly:

   `Healthy`, `Yellow Rust`, `Brown Rust`, `Powdery Mildew`

   Optional columns — `notes`, and any of the realistic-condition fields
   (`lighting`, `background`, `distance`, `angle`, `quality`) — are recorded in
   the report if you fill them in, and ignored if you leave them blank.

4. Score every model on exactly the same images:

   ```bash
   python -m src.testing.external_test --models v1 v2 v3
   ```

## What it reports

Overall accuracy, per-class accuracy, per-image predicted class and
confidence, a confusion matrix, confidence-calibration figures, and a
model-to-model diff showing which individual images changed prediction between
versions.
