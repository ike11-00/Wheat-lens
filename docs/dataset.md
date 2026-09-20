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
| Licence of the data used | Not yet provided |
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

Leaf Lens classifies into the categories defined in `config/config.yaml`.
There are currently **four**:

| Display name | Folder name | Also accepted (aliases) |
|---|---|---|
| Healthy | `Healthy` | `healthy`, `Healthy_Wheat`, `normal` |
| Yellow Rust | `Yellow_Rust` | `yellow_rust`, `stripe_rust`, `YellowRust` |
| Brown Rust | `Brown_Rust` | `brown_rust`, `leaf_rust`, `Leaf_Rust` |
| Powdery Mildew | `Powdery_Mildew` | `powdery_mildew`, `mildew`, `Mildew` |

Aliases exist so that a downloaded dataset using different folder names can be
dropped in without renaming every folder by hand — `prepare_dataset` resolves
them automatically.

### Adding a class

Append an entry to the `classes:` list in `config/config.yaml` and create the
matching folder under `data/raw/`. Nothing else needs editing: the model head,
the confusion matrix, the report tables and the web UI all read their class
list from that one place. This is also how Karnal Bunt goes back in.

## Required layout

```text
data/raw/
├── Healthy/
├── Yellow_Rust/
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

| Dataset | Documented classes | Covers the four classes? | Licence | URL |
|---|---|---|---|---|
| Wheat Plant Diseases (Kaggle) | Healthy, Brown Rust, Yellow Rust, Loose Smut, Septoria, Mildew and others | Likely all four | **Check the page** | https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases |
| Wheat Leaf dataset (Kaggle, olyadgetch) | Healthy, Leaf/Brown Rust, Powdery Mildew, Scab | No — lacks Yellow Rust | **Check the page** | https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset |
| Wheat Disease Dataset – Small (Kaggle) | Healthy, Yellow Rust, Brown Rust, Septoria, Mildew (~999 images) | Likely all four | **Check the page** | https://www.kaggle.com/datasets/yasserhessein/wheat-disease-dataset-small |
| YELLOW-RUST-19 (Kaggle) | Yellow rust severity grades + healthy | Yellow Rust + Healthy only | **Check the page** | https://www.kaggle.com/datasets/tolgahayit/yellowrust19-yellow-rust-disease-in-wheat |
| Wheat Fungi Diseases (WFD) | Healthy, Leaf rust, Powdery mildew, Septoria, Stem rust, Yellow rust (~2414 images) | Likely all four | **Check the site** | https://wfd.sysbio.ru/ |
| Wheat Disease Dataset – Small (Zenodo 7307816) | Yellow rust, Brown rust, Septoria, Mildew, Healthy | Likely all four | **Check the record** | https://zenodo.org/records/7307816 |

Print this table from the command line with:

```bash
python -m src.data.download_dataset --list
```

### Karnal Bunt: removed from the class list

The project originally specified five classes. **Karnal Bunt has been removed**,
leaving four. The reasons are factual, not convenience:

* Karnal Bunt (*Tilletia indica*) is a **grain** disease. The diagnostic signs
  are on the kernel, not the leaf, so leaf-level imagery barely exists.
* A survey of every reachable public source found **zero** Karnal Bunt leaf
  images — not on Kaggle's documented class lists, not in any GitHub
  repository, not in any plant-pathology image collection reachable from here.
* Training a five-output head with an empty fifth class would produce
  near-random Karnal Bunt probabilities on every image — worse than not
  offering the class at all.

**Reinstating it is a config-only change.** `config/config.yaml` carries the
exact entry to paste back. Add it, put images in `data/raw/Karnal_Bunt/`, and
retrain — no Python changes are needed. If you do source Karnal Bunt imagery,
be certain it shows the disease rather than plants from an infected field;
otherwise the model learns the field, not the pathogen.

Do not fill `Karnal_Bunt/` with images of some other condition to make the
pipeline run. The model would learn that condition and report it as Karnal Bunt.

## What is on GitHub

GitHub is reachable from restricted environments where Kaggle and Zenodo are
not, so it was surveyed directly. Several repositories **do** bundle real
wheat-leaf images:

| Repository | Images | Relevant classes | Licence |
|---|---|---|---|
| `cyb-personal/VWLM-for-Wheat-Disease-Identification-` | 7,750 (Blight 1717, Leaf rust 1642, Healthy 1545, Powdery mildew 1526, Septoria 1320) | Healthy, Brown Rust, Powdery Mildew | **None** |
| `Himanshu-Gupta3817/Wheat_plant_disease_detection` | 408 (stripe rust 208, septoria 97, healthy 102) | Yellow Rust — only source found | **None** |
| `suhasmaddali/Wheat-Disease-Detection-` | 4,383 (Leaf Rust 1286, Healthy 1146, …) | Healthy, Brown Rust | **None** |
| `aadium/wheat-disease-detection` | 3,228 (healthy 1146, leaf rust 131, …) | Healthy, Brown Rust | **None** |
| `mubashar1030/WheatDiseaseDataset` | 876 (stem rust 376, leaf rust 358, healthy 142) | Healthy, Brown Rust | **None** |

**None of these carries a LICENSE file**, which under default copyright means
all rights are reserved. The largest is an unpublished paper's dataset. They
are recorded here so the search does not have to be repeated — **not** as a
recommendation. Using them requires permission from the owner.

A GitHub search for wheat or plant-disease image datasets carrying an explicit
open licence (MIT, Apache-2.0, GPL-3.0, CC0, CC-BY-4.0, CC-BY-SA-4.0) returned
nothing usable: the licensed agricultural datasets on GitHub are rice images,
coffee prices and crop-price tables.

**Conclusion: the practical route to licensed wheat imagery is Kaggle, Zenodo
or the WFD site, downloaded on a machine that can reach them, after reading the
licence on the page.**

### A note on mixing sources

If you assemble classes from different datasets — for example Yellow Rust from
one repository and the rest from another — you introduce a **source confound**.
The datasets differ in camera, resolution, background and capture conditions, so
the network can learn "which dataset is this from" instead of "which disease is
this". That inflates test accuracy while producing a model that fails on your
own photographs.

If you must mix sources, say so in this file, expect the imbalance warning to
fire, and treat the affected class's metrics as unreliable until realistic
testing confirms them.

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
