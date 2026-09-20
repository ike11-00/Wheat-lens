"""Generate a SYNTHETIC image fixture for verifying the pipeline end to end.

WHAT THIS IS
------------
Procedurally drawn abstract images - coloured noise, stripes, blobs and rings.
They are **not** wheat leaves and they are **not** photographs of any disease.
Their only purpose is to prove that the code path

    prepare -> train -> evaluate -> error analysis -> realistic testing
    -> comparison -> web prediction

executes correctly and produces well-formed artefacts.

WHAT THIS IS NOT
----------------
Any accuracy number obtained on this fixture describes the model's ability to
tell synthetic textures apart. It says **nothing** about wheat disease
classification. Never quote a metric produced from this fixture as a result of
the project. Real images are required for that - see docs/dataset.md.

The fixture is written outside ``data/`` by default so it can never be mistaken
for the real dataset.

Run::

    python tests/make_synthetic_fixture.py --root /tmp/leaf_lens_fixture/raw
    python tests/make_synthetic_fixture.py --root <dir> --per-class 40 --size 96
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402

WARNING_TEXT = """\
SYNTHETIC PIPELINE-TEST FIXTURE - NOT REAL DATA
===============================================

The images in this folder were generated procedurally by
tests/make_synthetic_fixture.py. They are abstract patterns, not photographs of
wheat leaves, and they carry no information about any plant disease.

They exist only so that the Leaf Lens pipeline can be executed and verified
without a real dataset.

Any accuracy, precision, recall or confusion matrix produced from these images
is a test of the CODE, not a result of the PROJECT. Do not quote such numbers
anywhere. Replace this fixture with a real labelled dataset before drawing any
conclusion about disease classification.
"""


def _draw_pattern(style: int, size: int, rng: random.Random) -> Image.Image:
    """Draw one abstract image in a per-class visual style."""
    # A distinct base hue and a distinct geometric motif per class, plus noise,
    # so a CNN has something learnable but the images stay obviously synthetic.
    base_hues = [(90, 140, 80), (200, 190, 70), (150, 120, 90), (170, 90, 70), (210, 210, 200)]
    base = base_hues[style % len(base_hues)]

    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 18, (size, size, 3))
    canvas = np.clip(np.asarray(base, dtype="float32") + noise, 0, 255).astype("uint8")
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)

    accent = tuple(min(255, max(0, c + rng.randint(-25, 25))) for c in
                   [(40, 90, 40), (240, 220, 60), (120, 70, 40), (200, 60, 40), (250, 250, 250)][style % 5])

    if style % 5 == 0:                      # vertical bars
        for x in range(0, size, max(6, size // 10)):
            draw.rectangle([x, 0, x + max(2, size // 40), size], fill=accent)
    elif style % 5 == 1:                    # scattered small dots
        for _ in range(rng.randint(30, 60)):
            x, y = rng.randrange(size), rng.randrange(size)
            r = rng.randint(2, max(3, size // 30))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=accent)
    elif style % 5 == 2:                    # concentric rings
        centre = (rng.randrange(size // 3, 2 * size // 3), rng.randrange(size // 3, 2 * size // 3))
        for radius in range(size // 10, size, max(5, size // 10)):
            draw.ellipse([centre[0] - radius, centre[1] - radius,
                          centre[0] + radius, centre[1] + radius],
                         outline=accent, width=max(1, size // 60))
    elif style % 5 == 3:                    # diagonal streaks
        for offset in range(-size, size, max(8, size // 8)):
            draw.line([(offset, 0), (offset + size, size)], fill=accent,
                      width=max(2, size // 45))
    else:                                   # large soft blobs
        for _ in range(rng.randint(3, 6)):
            x, y = rng.randrange(size), rng.randrange(size)
            r = rng.randint(size // 8, size // 4)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=accent)

    # Mild per-image variation so the model cannot memorise a single template.
    if rng.random() < 0.5:
        image = image.rotate(rng.uniform(-12, 12), resample=Image.BILINEAR, fillcolor=base)
    return image


def generate(root: Path, class_dirs: List[str], per_class: int, size: int,
             seed: int = 0) -> Tuple[int, Path]:
    """Write ``per_class`` synthetic images for each class under ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "SYNTHETIC_FIXTURE_README.txt").write_text(WARNING_TEXT, encoding="utf-8")

    rng = random.Random(seed)
    written = 0
    for style, directory in enumerate(class_dirs):
        class_dir = root / directory
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            image = _draw_pattern(style, size, rng)
            image.save(class_dir / f"synthetic_{directory.lower()}_{index:04d}.jpg",
                       quality=88)
            written += 1
    return written, root


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic (NOT real) fixture for pipeline testing")
    parser.add_argument("--root", required=True,
                        help="output directory (use a path outside data/ to avoid confusion)")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    written, root = generate(Path(args.root), config.class_dirs,
                             args.per_class, args.size, args.seed)
    print(f"wrote {written} SYNTHETIC images to {root}")
    print("These are abstract patterns, not wheat leaves. See "
          f"{root / 'SYNTHETIC_FIXTURE_README.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
