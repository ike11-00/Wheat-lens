# Dataset

## Status

**No dataset is present in this repository.** `data/raw/` is empty apart from
`.gitkeep` files, and no images were downloaded while the project was built.

Every tool described below has been written and tested against synthetic
fixtures, so the pipeline runs the moment real images are added. Until then,
the following are all **Not yet provided**:

| Item | Status |
|---|---|
| Dataset name | Not yet provided |
| Dataset source | Not yet provided |
| Dataset URL | Not yet provided |
| Licence | Not yet provided |
| Total images | Not yet provided |
| Images per class | Not yet provided |
| Train / validation / test counts | Not yet provided |
| Duplicates found | Not yet provided |
| Class imbalance | Not yet provided |

Fill this table in from the generated reports once you have added images:
`results/dataset/validation_raw.md` and `results/dataset/prepare_report.md`.

## Why no dataset was downloaded

The environment this project was built in has a restricted outbound network
policy. PyPI and GitHub are reachable; the dataset hosts are not. Concretely,
the egress gateway answered `403` to:

* `www.kaggle.com`
* `huggingface.co`
* `data.mendeley.com`
* `zenodo.org`

Downloading images from elsewhere would have meant scraping arbitrary web
images of unknown provenance and unknown labels, which is worse than having no
dataset: it produces a model that appears to work and cannot be trusted. So
nothing was downloaded.

## Required classes

Leaf Lens classifies into exactly five categories, defined in
`config/config.yaml`:

| Display name | Folder name | Also accepted (aliases) |
|---|---|---|
| Healthy | `Healthy` | `healthy`, `Healthy_Wheat`, `normal` |
| Yellow Rust | `Yellow_Rust` | `yellow_rust`, `stripe_rust`, `YellowRust` |
| Karnal Bunt | `Karnal_Bunt` | `karnal_bunt`, `partial_bunt`, `KarnalBunt` |
| Brown Rust | `Brown_Rust` | `brown_rust`, `leaf_rust`, `Leaf_Rust` |
| Powdery Mildew | `Powdery_Mildew` | `powdery_mildew`, `mildew`, `Mildew` |

Aliases exist so that a downloaded dataset using different folder names can be
dropped in without renaming every folder by hand — `prepare_dataset` resolves
them automatically.

### Adding a sixth class

Append an entry to the `classes:` list in `config/config.yaml` and create the
matching folder under `data/raw/`. Nothing else needs editing: the model head,
the confusion matrix, the report tables and the web UI all read their class
list from that one place.

## Required layout

```text
data/raw/
├── Healthy/
├── Yellow_Rust/
├── Karnal_Bunt/
├── Brown_Rust/
└── Powdery_Mildew/
```

Images may be nested inside those folders; the scanner recurses. Accepted
extensions are `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` (configurable under
`data.supported_extensions`).

`prepare_dataset` then produces:

```text
data/train/<Class>/        data/validation/<Class>/        data/test/<Class>/
```

## Candidate public datasets

These are **pointers only**. None of them was downloaded, inspected or verified
by this project. Check the classes and the licence on the source page before
using any of them.

| Dataset | Documented classes | All five Leaf Lens classes? | URL |
|---|---|---|---|
| Wheat Plant Diseases (Kaggle) | Healthy, Brown Rust, Yellow Rust, Loose Smut, Septoria, Mildew and others | Partial — check for Karnal Bunt | https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases |
| Wheat Leaf dataset (Kaggle, olyadgetch) | Healthy, Leaf/Brown Rust, Powdery Mildew, Scab | No — lacks Yellow Rust and Karnal Bunt | https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset |
| Wheat Disease Dataset – Small (Kaggle) | Healthy, Yellow Rust, Brown Rust, Septoria, Mildew (~999 images) | No — lacks Karnal Bunt | https://www.kaggle.com/datasets/yasserhessein/wheat-disease-dataset-small |
| YELLOW-RUST-19 (Kaggle) | Yellow rust severity grades + healthy | No — top-up source only | https://www.kaggle.com/datasets/tolgahayit/yellowrust19-yellow-rust-disease-in-wheat |
| Wheat Fungi Diseases (WFD) | Healthy, Leaf rust, Powdery mildew, Septoria, Stem rust, Yellow rust (~2414 images) | No — lacks Karnal Bunt | https://wfd.sysbio.ru/ |
| Wheat Disease Dataset – Small (Zenodo 7307816) | Yellow rust, Brown rust, Septoria, Mildew, Healthy | No — lacks Karnal Bunt | https://zenodo.org/records/7307816 |

Print this table from the command line with:

```bash
python -m src.data.download_dataset --list
```

### The Karnal Bunt problem

Karnal Bunt (*Tilletia indica*) is a **grain** disease: the visible signs are on
the kernel, not the leaf. Public *leaf*-image datasets therefore rarely include
it, and the table above reflects that — none of the listed sources is confirmed
to contain it.

You have three honest options:

1. **Source Karnal Bunt images separately** (a plant-pathology collection, a
   local agricultural university, or your own field photographs) and merge them
   into `data/raw/Karnal_Bunt/`.
2. **Photograph it yourself**, if you have access to infected material.
3. **Redefine the class set** — drop Karnal Bunt or replace it with a leaf
   disease that public data actually covers (Septoria and Loose Smut are both
   well represented). This is a one-line change in `config/config.yaml`.

Do not fill `Karnal_Bunt/` with images of some other condition to make the
pipeline run. The model would learn that condition and report it as Karnal Bunt.

## Getting a dataset in

**With network access and a Kaggle account:**

```bash
pip install kaggle
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
python -m src.data.download_dataset --kaggle <owner/dataset-slug>
python -m src.data.download_dataset --inspect data/downloads/<folder>
```

The `--inspect` step lists every image folder it found and tells you which
Leaf Lens class each one maps to, so you can copy them into `data/raw/`.

**Without network access:** download on another machine and copy the folders
into `data/raw/` using the layout above.

## Validation

```bash
python -m src.data.validate_dataset
```

Checks performed:

* every file is opened twice — `verify()` for structural corruption, then a
  full decode to catch truncated payloads
* extension and format are on the supported list
* images below `data.min_image_pixels` (default 32 px) are rejected
* images counted per class, with percentages
* missing classes reported as **errors** (they block training)
* class imbalance above `data.imbalance_warn_ratio` (default 3:1) warned
* byte-identical duplicates found by SHA-256
* near-duplicates found by perceptual hash (pHash)
* near-duplicate groups that span two classes flagged as labelling conflicts

Outputs: `results/dataset/validation_raw.{json,md}` and
`results/graphs/class_distribution_raw.png`.

## Preparation and the split

```bash
python -m src.data.prepare_dataset
```

Default split from `config/config.yaml` — **75% train, 12.5% validation,
12.5% test**, stratified per class, seeded with `random_seed: 42`.

Change it under `data.split`. For a small dataset (a few hundred images per
class) consider 70/15/15 so the test set is large enough to mean anything; for
a large one, 80/10/10 is reasonable. Whatever you choose, record the actual
counts from `results/dataset/prepare_report.md` in the table at the top of this
file.

### Preventing data leakage

This is the part that most often goes wrong in image-classification projects,
so it is enforced mechanically rather than left to discipline.

1. **Byte-identical files** are found by SHA-256 and all but one copy is
   dropped before splitting.
2. **Near-duplicate photographs** — the same leaf re-shot, a crop, a re-save at
   a different JPEG quality — are found with a perceptual hash (pHash,
   8×8, Hamming distance ≤ 5 by default) and grouped with union-find.
3. **Whole groups are assigned to a single split.** The splitter never places
   individual images; it places groups. A photograph and its near-duplicates
   therefore always land in the same split, and the test set cannot contain a
   variant of a training image.
4. Grouping happens **within a class**. Two visually similar images with
   *different* labels are a labelling conflict, not a split problem, and
   `validate_dataset` reports them separately.

Verify after preparing:

```bash
python -m src.data.validate_dataset --split-report
```

This re-hashes every prepared file and exits non-zero if any image (identical
or near-identical) appears in more than one split. Results go to
`results/dataset/split_leakage_check.json`.

**Limitations of this protection.** pHash catches re-saves, rescales, crops and
mild edits. It does *not* catch two genuinely different photographs of the same
leaf taken from different angles, or two leaves from the same plant. If your
source images come in bursts from a single field visit, group them yourself
(for example by putting each visit in its own raw sub-folder and splitting by
visit) before relying on the automatic grouping.

## Class imbalance

Two mechanisms, both configurable:

* `validate_dataset` warns when the largest class exceeds the smallest by more
  than `data.imbalance_warn_ratio` (default 3:1).
* Training applies inverse-frequency class weights when
  `training.use_class_weights` is true (the default), so a rare class is not
  ignored by the loss.

Weights help; they do not manufacture information. A class with 20 images will
be poorly learned no matter how it is weighted. Collect more images instead.

## Preprocessing

* Images are decoded to RGB and resized to `data.image_size` (224×224) with
  bilinear interpolation.
* Pixels stay in the **0–255 range** all the way to the model. Normalisation is
  a layer *inside* the saved network (`BackbonePreprocessing`), so the web app,
  the CLI and the evaluator cannot apply the wrong preprocessing.
* Training-time augmentation is also inside the model and inactive at
  inference. See `docs/model.md`.
