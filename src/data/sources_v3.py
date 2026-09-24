"""Dataset source registry and acquisition for the V3 experiment.

Every source is declared here with its provenance, so the dataset can be
rebuilt from scratch and audited afterwards. Nothing is acquired that is not
in this table, and nothing in this table is asserted without having been
inspected.

V3's question
-------------
V2 fixed the background confound (background robustness 21.5% -> 95.3%) but
scored 45% on the user's 20 real-world photographs, with Brown Rust at 0/5.
V3 asks whether **source diversity** - the same classes drawn from several
independent collections rather than one - improves real-world behaviour.

What is actually available
--------------------------
Kaggle, Hugging Face, Zenodo and Mendeley are unreachable from this
environment (the egress gateway returns 403), so the large field-photography
datasets that exist in the literature cannot be obtained here. GitHub is
reachable and was surveyed exhaustively. The result is an asymmetric
opportunity:

* **Brown Rust** and **Healthy** can be drawn from three or four independent
  collections.
* **Powdery Mildew** has exactly one source.
* **Yellow Rust** has exactly one source, of 208 images.

That asymmetry is the central constraint on V3 and is documented rather than
worked around. See docs/v3_experiment.md.

Status after the survey and the overlap check (2026-09-23)
---------------------------------------------------------
The paragraph above describes the position before ``kiran`` was found. It
added a second, independent source for Yellow Rust (208 -> 447) and for
Powdery Mildew. The cross-source overlap check (docs/v3_source_overlap.md) then
showed that ``mubashar`` is largely a re-encoding of ``suhas``, so the two are
declared one **source family** below. Source families - not repositories - are
what the V3 sampling balances across and what per-source accuracy reports on.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.helpers import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SourceFolder:
    """One folder inside a source repository, mapped to a Leaf Lens class."""

    path: str                 # path inside the repository
    leaf_lens_class: str      # destination class, or "" to ignore
    observed_count: int       # images seen when the registry was written
    imagery: str              # "studio" | "field" | "mixed" | "unknown"
    note: str = ""


@dataclass(frozen=True)
class DatasetSource:
    """A repository the V3 dataset may draw from."""

    key: str
    name: str
    url: str
    clone_url: str
    acquired: str             # date the registry entry was verified
    licence: str
    imagery: str
    folders: List[SourceFolder] = field(default_factory=list)
    use: bool = True          # False => surveyed but deliberately not used
    exclusion_reason: str = ""
    # Upstream commit the clone was taken at. Every image's SHA-256 is also
    # recorded in data/v3_manifest.csv, so content can be verified on rebuild.
    commit: str = ""
    # Source family. Repositories that redistribute the same underlying
    # photographs belong to one family; "" means the source is its own family.
    # A member source whose family is another key is de-duplicated against
    # that family's primary source (see src/data/build_v3.py).
    family: str = ""
    family_note: str = ""

    @property
    def family_key(self) -> str:
        return self.family or self.key


# ---------------------------------------------------------------------------
# The registry.
#
# Counts are what was observed on the stated date. Licence fields record what
# was found, including "none found", which is the honest answer for every one
# of these repositories - none carries a LICENSE file, so all rights are
# reserved by default. That is recorded, not glossed.
# ---------------------------------------------------------------------------
SOURCES: List[DatasetSource] = [
    DatasetSource(
        key="wpldd",
        commit="e44e1727c6afa25ecb3c969c75700debb8ac893f",
        name="WPLDD (VWLM wheat leaf disease dataset)",
        url="https://github.com/cyb-personal/VWLM-for-Wheat-Disease-Identification-",
        clone_url="https://github.com/cyb-personal/VWLM-for-Wheat-Disease-Identification-.git",
        acquired="2026-09-20",
        licence="None found - no LICENSE file; unpublished paper's dataset",
        imagery="studio",
        folders=[
            SourceFolder("WPLDD/Healthy", "Healthy", 1545, "studio",
                         "single leaf on a pale studio background"),
            SourceFolder("WPLDD/Leaf rust", "Brown Rust", 1642, "studio",
                         "leaf rust = brown rust (Puccinia triticina)"),
            SourceFolder("WPLDD/Powdery mildew", "Powdery Mildew", 1526, "studio",
                         "the only Powdery Mildew source available"),
            SourceFolder("WPLDD/Blight", "", 1717, "studio", "not a configured class"),
            SourceFolder("WPLDD/Septoria", "", 1320, "studio", "not a configured class"),
        ],
    ),
    DatasetSource(
        key="hg3817",
        commit="27c0cdb2a936d2d09abace6f31ad770f4702ad28",
        name="Wheat plant disease detection (Himanshu-Gupta3817)",
        url="https://github.com/Himanshu-Gupta3817/Wheat_plant_disease_detection",
        clone_url="https://github.com/Himanshu-Gupta3817/Wheat_plant_disease_detection.git",
        acquired="2026-09-20",
        licence="None found - no LICENSE file",
        imagery="field",
        folders=[
            SourceFolder("Dataset/train/Stripe_rust", "Yellow Rust", 144, "field",
                         "stripe rust = yellow rust; the ONLY Yellow Rust source"),
            SourceFolder("Dataset/test/stripe_rust", "Yellow Rust", 37, "field"),
            SourceFolder("Dataset/valid/stripe_rust", "Yellow Rust", 27, "field"),
            SourceFolder("Dataset/train/healthy", "Healthy", 63, "field",
                         "field-photographed healthy leaves - rare and valuable"),
            SourceFolder("Dataset/valid/healthy", "Healthy", 27, "field"),
            SourceFolder("Dataset/test/healthy", "Healthy", 12, "field"),
            SourceFolder("Dataset/train/septoria", "", 63, "field", "not a configured class"),
        ],
    ),
    DatasetSource(
        key="suhas",
        commit="3a1d68603916dfee0f283dd3f4a961df7951f3fd",
        name="Wheat Disease Detection (suhasmaddali)",
        url="https://github.com/suhasmaddali/Wheat-Disease-Detection-",
        clone_url="https://github.com/suhasmaddali/Wheat-Disease-Detection-.git",
        acquired="2026-09-23",
        licence="None found - no LICENSE file",
        imagery="mixed",
        folders=[
            SourceFolder("Images/Leaf Rust", "Brown Rust", 1286, "mixed",
                         "independent Brown Rust collection - the main V3 addition"),
            SourceFolder("Images/Healthy Wheat", "Healthy", 1146, "mixed"),
            SourceFolder("Images/Crown and Root Rot", "", 1021, "mixed",
                         "not a configured class"),
            SourceFolder("Images/Wheat Loose Smut", "", 930, "mixed",
                         "not a configured class"),
        ],
    ),
    DatasetSource(
        key="mubashar",
        commit="26a8d37ef0f4cd30128a388fbf6429922b787129",
        family="suhas",
        family_note=(
            "Measured 2026-09-23 (docs/v3_source_overlap.md): after exact de-duplication, "
            "200 of 318 Brown Rust (62.9%) and 67 of 126 Healthy (53.2%) images are "
            "perceptual-hash near-duplicates of suhas images - different bytes, same "
            "photographs. Treated as part of the suhas family: those 267 are removed and "
            "the 177 genuinely unique images (118 Brown Rust, 59 Healthy) are kept. "
            "Decision approved by the project owner."
        ),
        name="WheatDiseaseDataset (mubashar1030)",
        url="https://github.com/mubashar1030/WheatDiseaseDataset",
        clone_url="https://github.com/mubashar1030/WheatDiseaseDataset.git",
        acquired="2026-09-23",
        licence="None found - no LICENSE file",
        imagery="mixed",
        folders=[
            SourceFolder("train/leaf_rust", "Brown Rust", 286, "mixed"),
            SourceFolder("test/leaf_rust", "Brown Rust", 72, "mixed"),
            SourceFolder("train/healthy_wheat", "Healthy", 113, "mixed"),
            SourceFolder("test/healthy_wheat", "Healthy", 29, "mixed"),
            SourceFolder("train/stem_rust", "", 300, "mixed", "stem rust is not a class here"),
            SourceFolder("test/stem_rust", "", 76, "mixed"),
        ],
    ),
    DatasetSource(
        key="kiran",
        commit="f4b7cf7c5ac4be3d1d44afb01d0084806e24230b",
        name="Wheat Plant Disease Classification ResNet18 (kaliprogramer / Kiran K.C)",
        url="https://github.com/kaliprogramer/Wheat-Plant-Disease-Classification-using-Deep-Learning-ResNet18",
        clone_url="https://github.com/kaliprogramer/Wheat-Plant-Disease-Classification-using-Deep-Learning-ResNet18.git",
        acquired="2026-09-23",
        licence=(
            "MIT LICENSE file present (Copyright (c) 2026 Kiran K.C). NOTE: MIT covers "
            "'the Software'; whether the author intended it to cover the bundled image "
            "dataset, and where that dataset originally came from, is not stated in the "
            "repository. This is the only source in the registry carrying any licence at "
            "all, but the position is not unambiguous."
        ),
        imagery="mixed",
        folders=[
            # The most valuable find in the survey: it more than doubles Yellow
            # Rust and gives Powdery Mildew a second, independent source.
            SourceFolder("Dataset/train/YellowRust", "Yellow Rust", 191, "mixed",
                         "second Yellow Rust source - doubles the available supply"),
            SourceFolder("Dataset/val/YellowRust", "Yellow Rust", 23, "mixed"),
            SourceFolder("Dataset/test/YellowRust", "Yellow Rust", 25, "mixed"),
            SourceFolder("Dataset/train/Mildew", "Powdery Mildew", 128, "mixed",
                         "second Powdery Mildew source - previously WPLDD only"),
            SourceFolder("Dataset/val/Mildew", "Powdery Mildew", 16, "mixed"),
            SourceFolder("Dataset/test/Mildew", "Powdery Mildew", 17, "mixed"),
            SourceFolder("Dataset/train/BrownRust", "Brown Rust", 102, "mixed"),
            SourceFolder("Dataset/val/BrownRust", "Brown Rust", 12, "mixed"),
            SourceFolder("Dataset/test/BrownRust", "Brown Rust", 14, "mixed"),
            SourceFolder("Dataset/train/Healthy", "Healthy", 97, "mixed"),
            SourceFolder("Dataset/val/Healthy", "Healthy", 12, "mixed"),
            SourceFolder("Dataset/test/Healthy", "Healthy", 13, "mixed"),
            SourceFolder("Dataset/train/Septoria", "", 279, "mixed", "not a configured class"),
        ],
    ),
    DatasetSource(
        key="yr2223",
        name="YR-22-23 yellow rust dataset (Shant-Thakur)",
        url="https://github.com/Shant-Thakur/YR-22-23",
        clone_url="https://github.com/Shant-Thakur/YR-22-23.git",
        acquired="2026-09-23",
        licence="None found - no LICENSE file",
        imagery="unknown",
        use=False,
        exclusion_reason=(
            "Named as a yellow-rust dataset but the upload is broken: dataset/train/healthy "
            "holds 1193 images while dataset/train/rust holds 1 and dataset/test/rust holds 1. "
            "The rust class is effectively empty, and a 1193-image 'healthy' folder of "
            "unverified provenance is not worth mixing in on its own."
        ),
        folders=[],
    ),
    DatasetSource(
        key="swinsharp",
        name="SWIN-SHARP 5-disease wheat dataset",
        url="https://github.com/SWIN-SHARP/SWIN-SHARP",
        clone_url="https://github.com/SWIN-SHARP/SWIN-SHARP.git",
        acquired="2026-09-23",
        licence="None found - no LICENSE file",
        imagery="unknown",
        use=False,
        exclusion_reason=(
            "The README advertises a five-class wheat dataset including Yellow Rust and "
            "Powdery Mildew, but the repository contains only 60 images in total - sample "
            "figures rather than the dataset itself. The data is not actually published here."
        ),
        folders=[],
    ),
    # ---------------- surveyed and deliberately NOT used ----------------
    DatasetSource(
        key="aadium",
        name="wheat-disease-detection (aadium)",
        url="https://github.com/aadium/wheat-disease-detection",
        clone_url="https://github.com/aadium/wheat-disease-detection.git",
        acquired="2026-09-23",
        licence="None found - no LICENSE file",
        imagery="mixed",
        use=False,
        exclusion_reason=(
            "Redistributes the same underlying dataset as 'suhas': identical folder "
            "counts (healthy 1146, crown/root rot 1021, loose smut 930) AND identical "
            "filenames (00011.jpg, 00021.jpg, 00041.jpg ...). Including both would be "
            "duplication presented as source diversity, which is the opposite of what "
            "V3 is testing."
        ),
        folders=[],
    ),
    DatasetSource(
        key="anushrii",
        name="Wheat_disease_detection (anushriiverse)",
        url="https://github.com/anushriiverse/Wheat_disease_detection",
        clone_url="https://github.com/anushriiverse/Wheat_disease_detection.git",
        acquired="2026-09-23",
        licence="None found - no LICENSE file",
        imagery="unknown",
        use=False,
        exclusion_reason=(
            "Binary dataset only: folders are 'healthy' and 'unhealthy'. The unhealthy "
            "class carries no disease label, so it cannot be mapped to any of the four "
            "configured classes. Its 'healthy' folder could be used, but its provenance "
            "and overlap with other sources are unverified, so it is excluded rather "
            "than mixed in with unknown labels."
        ),
        folders=[],
    ),
    DatasetSource(
        key="goodhope",
        name="Wheat Disease Detection and Recognition (GoodhopeKD)",
        url="https://github.com/GoodhopeKD/Wheat-Disease-Detection-and-Recognition-Based-on-Deep-Learning",
        clone_url="https://github.com/GoodhopeKD/Wheat-Disease-Detection-and-Recognition-Based-on-Deep-Learning.git",
        acquired="2026-09-23",
        licence="None found - no LICENSE file",
        imagery="unknown",
        use=False,
        exclusion_reason=(
            "Only 4 images per class in a demonstration test set - too few to affect "
            "training and too few to evaluate on."
        ),
        folders=[],
    ),
]


def active_sources() -> List[DatasetSource]:
    return [s for s in SOURCES if s.use]


def source_families() -> Dict[str, List[str]]:
    """Family key -> member source keys, primary (key == family) first."""
    families: Dict[str, List[str]] = {}
    for source in active_sources():
        families.setdefault(source.family_key, []).append(source.key)
    for family, members in families.items():
        members.sort(key=lambda key: (key != family, key))
    return families


def excluded_sources() -> List[DatasetSource]:
    return [s for s in SOURCES if not s.use]


def mapped_folders() -> List[tuple]:
    """(source, folder) pairs that map onto a configured class."""
    return [(s, f) for s in active_sources() for f in s.folders if f.leaf_lens_class]


def expected_counts() -> Dict[str, Dict[str, int]]:
    """Registry-declared image counts, per class per source."""
    out: Dict[str, Dict[str, int]] = {}
    for source, folder in mapped_folders():
        out.setdefault(folder.leaf_lens_class, {})
        out[folder.leaf_lens_class][source.key] = (
            out[folder.leaf_lens_class].get(source.key, 0) + folder.observed_count)
    return out


def clone_source(source: DatasetSource, workdir: Path) -> Path:
    """Clone (or reuse) a source repository under ``workdir``."""
    target = workdir / source.key
    if (target / ".git").is_dir():
        LOGGER.info("reusing clone: %s", target)
        return target
    workdir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("cloning %s", source.url)
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", source.clone_url, str(target)],
        check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        for line in (completed.stderr or "").strip().splitlines()[-4:]:
            LOGGER.error("  %s", line)
        raise RuntimeError(f"could not clone {source.url}")
    return target
