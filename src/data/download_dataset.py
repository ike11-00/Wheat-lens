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
        "covers_five_required": "partial - check whether Karnal Bunt is present",
        "licence": "stated on the dataset page - verify before use",
    },
    {
        "key": "wheat-leaf-dataset",
        "name": "Wheat Leaf dataset (Kaggle, olyadgetch)",
        "url": "https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset",
        "kaggle_slug": "olyadgetch/wheat-leaf-dataset",
        "documented_classes": "Healthy, Leaf/Brown Rust, Powdery Mildew, Scab (3-4 classes)",
        "covers_five_required": "no - lacks Yellow Rust and Karnal Bunt",
        "licence": "stated on the dataset page - verify before use",
    },
    {
        "key": "wheat-disease-small",
        "name": "Wheat Disease Dataset - Small (Kaggle / Zenodo)",
        "url": "https://www.kaggle.com/datasets/yasserhessein/wheat-disease-dataset-small",
        "kaggle_slug": "yasserhessein/wheat-disease-dataset-small",
        "documented_classes": "Healthy, Yellow Rust, Brown Rust, Septoria, Mildew (~999 images)",
        "covers_five_required": "no - lacks Karnal Bunt",
        "licence": "stated on the dataset page - verify before use",
    },
    {
        "key": "yellowrust19",
        "name": "YELLOW-RUST-19: Yellow Rust Disease in Wheat (Kaggle)",
        "url": "https://www.kaggle.com/datasets/tolgahayit/yellowrust19-yellow-rust-disease-in-wheat",
        "kaggle_slug": "tolgahayit/yellowrust19-yellow-rust-disease-in-wheat",
        "documented_classes": "Yellow rust severity grades + healthy (single-disease dataset)",
        "covers_five_required": "no - Yellow Rust and Healthy only; useful as a top-up source",
        "licence": "stated on the dataset page - verify before use",
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
        "covers_five_required": "no - lacks Karnal Bunt",
        "licence": "stated on the project site - verify before use",
    },
    {
        "key": "zenodo-wheat-small",
        "name": "Wheat Disease Dataset - Small (Zenodo record 7307816)",
        "url": "https://zenodo.org/records/7307816",
        "kaggle_slug": "",
        "documented_classes": "Yellow rust, Brown rust, Septoria, Mildew, Healthy",
        "covers_five_required": "no - lacks Karnal Bunt",
        "licence": "stated on the Zenodo record - verify before use",
    },
]


def print_sources() -> None:
    rows = [
        [s["name"], s["documented_classes"], s["covers_five_required"], s["url"]]
        for s in CANDIDATE_SOURCES
    ]
    print("Candidate public wheat-disease image datasets")
    print("(pointers only - this project has NOT downloaded or verified any of them)\n")
    print(markdown_table(
        ["Dataset", "Documented classes", "Covers all five Leaf Lens classes?", "URL"], rows))
    print(
        "\nNone of the sources above is confirmed to contain all five required classes, and in\n"
        "particular Karnal Bunt is rare in public leaf-image datasets (it is a grain/kernel\n"
        "disease, so leaf-level imagery is scarce). Expect to combine sources or to collect\n"
        "Karnal Bunt images yourself. See docs/dataset.md."
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
