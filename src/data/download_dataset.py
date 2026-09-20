"""Dataset acquisition helper.

This project does **not** ship images and does not scrape the open web. This
module does two things:

1.  ``--list`` prints the candidate public wheat-disease datasets recorded in
    docs/dataset.md, with their URLs and the classes they are documented to
    contain, so you can choose one and check its licence yourself.
2.  ``--kaggle <owner/dataset>`` downloads and unpacks a Kaggle dataset using
    the official ``kaggle`` CLI **if** you have installed it and placed your
    API token at ``~/.kaggle/kaggle.json``. It then prints the folders it
    found so you can map them onto the configured classes.

Nothing here invents data. If the download host is unreachable (a restricted
network is the usual reason) the command says so and exits non-zero.

Run::

    python -m src.data.download_dataset --list
    python -m src.data.download_dataset --kaggle <owner/dataset-slug>
    python -m src.data.download_dataset --inspect data/downloads/<folder>
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.config import load_config
from ..utils.helpers import get_logger, markdown_table
from ..utils.image_io import iter_image_files

LOGGER = get_logger(__name__)

# Candidate sources. These are POINTERS ONLY - this project has not downloaded,
# inspected or verified any of them. Class lists are as described by the
# publishers; confirm them (and the licence) yourself before use.
CANDIDATE_SOURCES: List[Dict[str, str]] = [
    {
        "key": "wheat-plant-diseases",
        "name": "Wheat Plant Diseases (Kaggle)",
        "url": "https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases",
        "kaggle_slug": "kushagra3204/wheat-plant-diseases",
        "documented_classes": (
            "Healthy, Brown Rust, Yellow Rust, Loose Smut, Septoria, Mildew, "
            "and further wheat conditions (see the dataset page)"
        ),
        "covers_required": "likely all four - verify Yellow Rust and Powdery Mildew",
        "licence": "CHECK THE DATASET PAGE - Kaggle shows the licence in the sidebar",
    },
    {
        "key": "wheat-leaf-dataset",
        "name": "Wheat Leaf dataset (Kaggle, olyadgetch)",
        "url": "https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset",
        "kaggle_slug": "olyadgetch/wheat-leaf-dataset",
        "documented_classes": "Healthy, Leaf/Brown Rust, Powdery Mildew, Scab (3-4 classes)",
        "covers_required": "no - lacks Yellow Rust",
        "licence": "CHECK THE DATASET PAGE",
    },
    {
        "key": "wheat-disease-small",
        "name": "Wheat Disease Dataset - Small (Kaggle / Zenodo)",
        "url": "https://www.kaggle.com/datasets/yasserhessein/wheat-disease-dataset-small",
        "kaggle_slug": "yasserhessein/wheat-disease-dataset-small",
        "documented_classes": "Healthy, Yellow Rust, Brown Rust, Septoria, Mildew (~999 images)",
        "covers_required": "likely all four",
        "licence": "CHECK THE DATASET PAGE",
    },
    {
        "key": "yellowrust19",
        "name": "YELLOW-RUST-19: Yellow Rust Disease in Wheat (Kaggle)",
        "url": "https://www.kaggle.com/datasets/tolgahayit/yellowrust19-yellow-rust-disease-in-wheat",
        "kaggle_slug": "tolgahayit/yellowrust19-yellow-rust-disease-in-wheat",
        "documented_classes": "Yellow rust severity grades + healthy (single-disease dataset)",
        "covers_required": "no - Yellow Rust and Healthy only; useful as a top-up source",
        "licence": "CHECK THE DATASET PAGE",
    },
    {
        "key": "wfd",
        "name": "Wheat Fungi Diseases (WFD) dataset",
        "url": "https://wfd.sysbio.ru/",
        "kaggle_slug": "",
        "documented_classes": (
            "Healthy, Leaf (brown) rust, Powdery mildew, Septoria, Stem rust, Yellow rust "
            "(~2414 images)"
        ),
        "covers_required": "likely all four",
        "licence": "CHECK THE PROJECT SITE",
    },
    {
        "key": "zenodo-wheat-small",
        "name": "Wheat Disease Dataset - Small (Zenodo record 7307816)",
        "url": "https://zenodo.org/records/7307816",
        "kaggle_slug": "",
        "documented_classes": "Yellow rust, Brown rust, Septoria, Mildew, Healthy",
        "covers_required": "likely all four",
        "licence": "Zenodo states a licence on every record - CHECK THE RECORD PAGE",
    },
]

# Result of a GitHub-wide survey (see docs/dataset.md, "What is on GitHub").
# These repositories DO bundle real wheat-leaf images, but none of them carries
# a LICENSE file, which means all rights are reserved by default. They are
# listed so nobody has to repeat the search - NOT as a recommendation to use
# them. Confirm terms with the repository owner before touching the images.
GITHUB_BUNDLED_DATASETS: List[Dict[str, str]] = [
    {
        "repo": "cyb-personal/VWLM-for-Wheat-Disease-Identification-",
        "url": "https://github.com/cyb-personal/VWLM-for-Wheat-Disease-Identification-",
        "images": "7,750 in WPLDD/: Blight 1717, Leaf rust 1642, Healthy 1545, "
                  "Powdery mildew 1526, Septoria 1320",
        "relevance": "Healthy, Brown Rust (leaf rust), Powdery Mildew",
        "licence": "NONE - unpublished paper's dataset, all rights reserved",
    },
    {
        "repo": "Himanshu-Gupta3817/Wheat_plant_disease_detection",
        "url": "https://github.com/Himanshu-Gupta3817/Wheat_plant_disease_detection",
        "images": "408: stripe rust 208, septoria 97, healthy 102",
        "relevance": "Yellow Rust (stripe rust) - the only source found, and small",
        "licence": "NONE - all rights reserved",
    },
    {
        "repo": "suhasmaddali/Wheat-Disease-Detection-",
        "url": "https://github.com/suhasmaddali/Wheat-Disease-Detection-",
        "images": "4,383: Leaf Rust 1286, Healthy 1146, Crown/Root Rot 1021, Loose Smut 930",
        "relevance": "Healthy, Brown Rust (leaf rust)",
        "licence": "NONE - all rights reserved",
    },
    {
        "repo": "aadium/wheat-disease-detection",
        "url": "https://github.com/aadium/wheat-disease-detection",
        "images": "3,228: healthy 1146, crown/root rot 1021, loose smut 930, leaf rust 131",
        "relevance": "Healthy, Brown Rust (leaf rust)",
        "licence": "NONE - all rights reserved",
    },
    {
        "repo": "mubashar1030/WheatDiseaseDataset",
        "url": "https://github.com/mubashar1030/WheatDiseaseDataset",
        "images": "876: stem rust 376, leaf rust 358, healthy 142",
        "relevance": "Healthy, Brown Rust (leaf rust)",
        "licence": "NONE - all rights reserved",
    },
]


def print_sources() -> None:
    """Print the candidate sources, with an explicit licence column."""
    rows = [
        [s["name"], s["documented_classes"], s["covers_required"], s["licence"], s["url"]]
        for s in CANDIDATE_SOURCES
    ]
    print("Candidate public wheat-disease image datasets")
    print("(pointers only - this project has NOT downloaded or verified any of them)\n")
    print(markdown_table(
        ["Dataset", "Documented classes", "Covers the configured classes?", "Licence", "URL"],
        rows))

    print(
        "\nLICENCE: verify it yourself on each page before downloading. This project does\n"
        "not record licence terms it could not read, and will not guess them.\n"
    )

    print("\nWheat image sets that are bundled in public GitHub repositories")
    print("(found by survey; listed so the search need not be repeated - NOT a "
          "recommendation)\n")
    print(markdown_table(
        ["Repository", "Images", "Relevant classes", "Licence"],
        [[f"{d['repo']}", d["images"], d["relevance"], d["licence"]]
         for d in GITHUB_BUNDLED_DATASETS]))
    print(
        "\nEvery one of these repositories lacks a LICENSE file, so all rights are reserved\n"
        "by default. Do not use them without permission from the owner.\n"
        "\n"
        "A search of GitHub for wheat or plant-disease image datasets carrying an explicit\n"
        "open licence (MIT, Apache-2.0, GPL-3.0, CC0, CC-BY-4.0, CC-BY-SA-4.0) returned no\n"
        "usable result. The practical route to licensed data is the Kaggle / Zenodo table\n"
        "above, downloaded on a machine that can reach those hosts.\n"
    )
    print(
        "Karnal Bunt is NOT in the configured class list. It is a grain disease\n"
        "(Tilletia indica) with almost no leaf-level imagery, and no licensed source was\n"
        "found. config/config.yaml shows how to reinstate it. See docs/dataset.md.\n"
    )


def kaggle_available() -> bool:
    return shutil.which("kaggle") is not None


def download_kaggle(slug: str, destination: Path) -> int:
    """Download and unzip a Kaggle dataset with the official CLI."""
    if not kaggle_available():
        LOGGER.error("the 'kaggle' command is not installed.")
        LOGGER.error("Install it and add your API token, then re-run:")
        LOGGER.error("  pip install kaggle")
        LOGGER.error("  mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
        return 2

    token = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not token.exists() and not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        LOGGER.error("no Kaggle credentials found (%s missing and KAGGLE_USERNAME/KAGGLE_KEY unset)",
                     token)
        return 2

    destination.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", slug, "-p", str(destination), "--unzip"]
    LOGGER.info("running: %s", " ".join(command))
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        LOGGER.error("could not start the kaggle CLI: %s", exc)
        return 2

    if completed.returncode != 0:
        LOGGER.error("kaggle download failed (exit %d)", completed.returncode)
        for line in (completed.stderr or completed.stdout or "").strip().splitlines()[-10:]:
            LOGGER.error("  %s", line)
        LOGGER.error(
            "If this is a network error, the host may be blocked by your environment's egress "
            "policy. Download the dataset on a machine with access and copy it into data/raw/."
        )
        return 1

    LOGGER.info("downloaded into %s", destination)
    inspect_download(destination)
    return 0


def inspect_download(root: Path) -> None:
    """Print the class-like folders found in a downloaded dataset."""
    config = load_config()
    root = Path(root)
    if not root.exists():
        LOGGER.error("not found: %s", root)
        return

    rows = []
    for directory in sorted(p for p in root.rglob("*") if p.is_dir()):
        images = list(iter_image_files(directory, config.supported_extensions))
        # Only report leaf directories that actually hold images.
        if not images:
            continue
        spec = config.resolve_class(directory.name)
        rows.append([
            str(directory.relative_to(root)),
            len(images),
            spec.name if spec else "-",
            f"data/raw/{spec.directory}/" if spec else "(map manually or ignore)",
        ])

    if not rows:
        LOGGER.warning("no image folders found under %s", root)
        return

    print(f"\nImage folders found under {root}:\n")
    print(markdown_table(["Folder", "Images", "Matches Leaf Lens class", "Copy to"], rows))
    print(
        "\nCopy or move the folders that match a Leaf Lens class into data/raw/ using the\n"
        "'Copy to' column, then run:\n"
        "  python -m src.data.validate_dataset\n"
        "  python -m src.data.prepare_dataset\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Leaf Lens dataset acquisition helper")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true",
                       help="print candidate public datasets and their URLs")
    group.add_argument("--kaggle", metavar="OWNER/SLUG",
                       help="download a Kaggle dataset with the official kaggle CLI")
    group.add_argument("--inspect", metavar="DIR",
                       help="list the image folders inside an already-downloaded dataset")
    parser.add_argument("--dest", default="data/downloads",
                        help="download destination (default: data/downloads)")
    args = parser.parse_args(argv)

    config = load_config()

    if args.list:
        print_sources()
        return 0

    if args.inspect:
        path = Path(args.inspect)
        inspect_download(path if path.is_absolute() else config.project_root / path)
        return 0

    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = config.project_root / dest
    return download_kaggle(args.kaggle, dest / args.kaggle.split("/")[-1])


if __name__ == "__main__":
    sys.exit(main())
