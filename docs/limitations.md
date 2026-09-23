# Limitations

Read this before using Leaf Lens for anything that matters. It is an
experimental prototype, not a diagnostic instrument.

## The overriding limitation, measured

**The model classifies by background, not by leaf.** This was established
causally, not inferred: transplanting backgrounds behind test images — leaf and
disease unchanged — drops accuracy from 100% to 21.5%, and 71.8% of the
composites are called Yellow Rust at 89% mean confidence.

Three of the four classes are studio photographs on a pale background; the
fourth is field photography. Background predicts class, so the network learned
background. Full evidence, and the v2 experiment intended to break it, are in
[`v2_experiment.md`](v2_experiment.md).

Everything below remains true and is compounded by this.

## The original framing

**The model scores 100% on its test split, and that number is close to
meaningless.**

Model v1 is trained and classifies its four categories perfectly on 617 unseen
images. Leakage was ruled out twice. The reason for the perfect score is that
the benchmark is trivially easy: seven colour statistics — per-channel mean and
standard deviation on a 32×32 thumbnail, no texture, no shape — reach 98.54% on
the same split. The network adds 1.46 percentage points.

The dataset is 4,921 curated images of single leaves, cropped, evenly lit,
mostly on plain backgrounds. Under those conditions the task collapses to
"what colour is this image". Field photographs do not have those properties.

So: the model works on data that looks like its training data, and **nothing is
known about how it behaves on a real photograph taken in a field.** No
realistic-condition testing has been done, because that needs photographs
nobody has supplied. Treat every figure in this project as an upper bound
obtained under ideal conditions.

## Dataset limitations

**Size.** Public wheat-disease datasets run from a few hundred to a few
thousand images. A five-class problem learned from, say, 200 images per class
gives a test split of roughly 25 images per class. A single misclassification
then moves per-class accuracy by four percentage points, so small differences
between models or between conditions are noise. The reports print `n` beside
every number for this reason.

**Diversity.** Most public sets come from a small number of locations,
seasons, cultivars and cameras. A model trained on them learns *those*
conditions. Wheat in another region, another cultivar, at another growth stage
or under different weather may look different enough to break it.

**Class imbalance.** Rust and healthy images are plentiful in public data;
Yellow Rust specifically is much scarcer than Brown Rust. Inverse-frequency
class weights stop the loss from ignoring a rare class, but they cannot create
information that is not there. A class with 20 images will be learned badly
whatever the weighting.

**Karnal Bunt is not offered at all.** It was removed from the original
five-class specification because it is a grain disease (*Tilletia indica*) with
essentially no leaf-level imagery, and no source of it could be found. This is
a *limitation of the product*, not just of the data: a wheat plant with Karnal
Bunt photographed at the leaf will be classified as one of the four categories
the model does know, most likely Healthy. See `docs/dataset.md`.

**Source confound when classes come from different datasets.** The current
data *is* mixed: Yellow Rust comes from a different repository than the other
three classes, and the two collections are perfectly separable on resolution
(24 MP against 0.16 MP, no overlap). A network can learn which collection an
image came from rather than which disease it shows.

This was tested directly rather than assumed. `src/testing/confound_test.py`
re-ran the test split with every image downsampled to a common resolution; every
class held 100% recall, so a gross scale artefact is ruled out as the mechanism.
That does not clear the dataset of every possible source signature — it rules
out the one that was measurable. Yellow Rust is also the smallest class by a
factor of eight, so class weighting amplifies whatever shortcut exists. Treat
its metrics with more suspicion than the rest.

**Laboratory versus field images.** Curated datasets favour a single leaf,
flat, well lit, against a plain background. Field photographs have soil, other
leaves, shadows, wind blur, hands and sky. A model that scores well on curated
images routinely loses substantial accuracy on field ones. This gap is the
single most common reason plant-disease classifiers disappoint in practice,
which is why `src/testing/realistic_testing.py` exists as a separate step.

**Dataset bias.** If diseased images were shot close-up and healthy ones from
further away, the model can learn *distance* rather than *disease*. If one
class was photographed on one day, it can learn that day's lighting. Nothing in
the pipeline can detect this; only deliberately varied photography can.

**Label quality.** The pipeline assumes labels are correct. It flags
near-duplicate images that carry different labels (a detectable conflict) but
cannot detect a consistently mislabelled class.

**Licensing — this model is not distributable.** The imagery comes from GitHub
repositories that, with one exception, carry **no LICENSE file at all**, which
under default copyright means all rights reserved. The project owner instructed
it to proceed on that basis for a personal, non-distributed build, and it does;
the position is recorded rather than glossed. The one exception is
`kaliprogramer/Wheat-Plant-Disease-Classification-using-Deep-Learning-ResNet18`,
which carries an MIT licence — though MIT covers "the Software" and the
repository does not state whether the author meant it to cover the bundled
images, nor where those images originally came from. Consequences: do not
redistribute the dataset, the trained weights, or anything derived from them,
and do not use this model commercially. `docs/dataset.md` and
`src/data/sources_v3.py` record the licence status of every source
individually.

## Model limitations

**Only four categories, and no way to say "something else".** A softmax
classifier distributes probability over exactly the classes it was trained on.
Presented with a sixth wheat disease, a nutrient deficiency, pest damage,
herbicide injury, a different crop, or a photograph of a cat, the model still
returns one of Healthy / Yellow Rust / Brown Rust / Powdery Mildew —
sometimes with high confidence. Karnal Bunt is among the conditions it will
silently misfile.

**The uncertainty check is a heuristic, not a safety net.** Leaf Lens flags a
prediction as uncertain when confidence is low, entropy is high, or the top two
classes are nearly tied. This catches *some* difficult inputs. It does **not**
reliably detect out-of-distribution images: neural networks are well documented
to be confidently wrong on inputs unlike their training data. A prediction that
is *not* flagged is not thereby verified. Real novelty detection would require
an explicit "other" class, an ensemble, MC-dropout, or a density model over the
features — none of which is implemented here.

**Confidence is not correctness.** The number shown is the model's estimated
probability for its own prediction. Modern networks are typically
over-confident, especially when trained on small datasets. 90% confidence does
not mean 90% of such predictions are right. If you want it to mean that, the
model needs calibration (temperature scaling or similar) validated on held-out
data — not done here. The evaluation report prints accuracy above and below the
threshold precisely so you can see how far confidence is from correctness on
*your* data.

**One label per image.** A leaf with two diseases, or a photograph containing
several leaves in different states, gets a single answer. Co-infection is
common in the field.

**No severity, no stage, no location.** The model says which class, not how
much, how advanced, or where on the leaf.

**Test accuracy does not transfer — demonstrably so here.** Test-split accuracy
estimates performance on unseen images *from the same distribution as the
training data*. On this dataset that distribution is unusually narrow: a
seven-number colour model scores 98.5% on it. The gap between that benchmark and
a field photograph is the whole question, and it is unmeasured. Only
realistic-condition testing with your own photographs speaks to it.

## Image and capture limitations

Factors that plausibly degrade performance. Each is a hypothesis to test with
`realistic_testing`, not an established finding:

* **Lighting** — harsh sun, deep shade and artificial light shift colour, and
  colour is how the two rusts are told apart.
* **Background** — soil, other plants, hands and sky give the model irrelevant
  things to latch onto.
* **Distance** — too far and lesions are too few pixels; too close and context
  is lost.
* **Angle** — an oblique leaf foreshortens lesion shape and spacing.
* **Partial leaves** — a leaf cut off by the frame edge may lack the diagnostic
  region.
* **Focus and resolution** — rust pustules are small; blur removes the texture
  that distinguishes them. The 224×224 input means a leaf occupying a small
  part of a large photograph loses most of its detail.
* **Camera differences** — phone processing pipelines apply their own colour
  and sharpening.
* **Growth stage** — early infection looks different from established
  infection, and may look like Healthy.

## Visual similarity between the classes

Some of these categories are genuinely hard to separate from a photograph:

* **Yellow Rust and Brown Rust** are both rusts. They differ in pustule colour
  and arrangement (yellow rust forms stripes along the veins; brown rust is
  scattered), but early lesions and poor colour rendition blur the difference.
* **Healthy and early disease** differ only by a few small lesions.
* **Powdery Mildew and Healthy** differ by a thin whitish coating that faint
  lighting can hide.

A human plant pathologist uses more than one photograph: they look at the whole
plant, the field pattern, the season, the weather history, and often a
laboratory test. The model sees 224×224 pixels.

## Application limitations

* No authentication, rate limiting or upload quota. The Flask development
  server is for local use; put it behind a production WSGI server and a reverse
  proxy before exposing it.
* Uploads are processed in memory and not stored. There is no history, no
  audit trail and no user accounts.
* Single-process inference. Concurrent requests queue.
* No mobile app and no offline mode.

## What this tool must not be used for

* Deciding whether to apply a fungicide, or which one.
* Certifying a crop, a consignment or a field as disease-free.
* Regulatory or quarantine decisions. Karnal Bunt in particular is a
  **quarantine-regulated** pathogen in many jurisdictions — and this model
  cannot detect it at all; its status must be determined by an accredited
  laboratory, never by a photograph.
* Any decision with financial, legal or food-safety consequences.

Use it to learn how image classification works, to explore a dataset, and as a
starting point for something properly validated. If a result matters, have a
qualified agronomist or plant pathologist look at the plant.

## Honest summary

Leaf Lens is a complete, tested machine-learning pipeline with a trained
four-class model that scores 100% on its own test split and whose real-world
accuracy is unknown.

The perfect score is not fraud and not a bug — it was obtained cleanly, with
leakage ruled out twice. It is a property of the benchmark: the data is curated
tightly enough that average colour nearly determines the class, and a linear
model on seven numbers gets 98.5% of the way there.

The project's tooling is built to *measure* that gap rather than hide it. The
baseline control and the confound test exist precisely because a headline
accuracy is the easiest number in machine learning to obtain and the least
informative to report alone. What is still missing is field photographs; until
those exist, the honest statement about this model is that it works on images
that look like its training set, and that nobody has checked anything else.
