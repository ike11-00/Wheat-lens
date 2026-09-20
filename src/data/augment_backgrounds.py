"""Background augmentation for the v2 experiment.

The problem this addresses
--------------------------
In the v1 dataset, three classes (Healthy, Brown Rust, Powdery Mildew) are
studio photographs - one leaf on a plain pale background - and the fourth
(Yellow Rust) is field photography. "Field background" therefore predicts
"Yellow Rust" almost perfectly, and the network learned that instead of the
disease. Transplanting field backgrounds behind the studio classes collapsed
test accuracy from 100% to about 16%, with the model 80-87% confident while
wrong. See docs/v2_experiment.md.

What this module does
---------------------
It breaks that correlation by giving the studio classes varied backgrounds, so
that background no longer carries class information. For each selected source
image it:

1. separates leaf from background (the studio images have a pale, low-saturation
   background, which segments reliably by brightness and saturation),
2. composites the leaf onto a different background at a random scale and
   position,
3. writes the result beside the original with a name that records where both
   the leaf and the background came from.

Backgrounds come from one or both of:

``procedural``
    Generated from noise - soil-like, vegetation-like and sky-like textures.
    Depends on no external data, so it always works.
``harvested``
    Real background patches cropped from the field-photography class. These are
    the actual backgrounds the model was keying on, so they attack the confound
    directly.

Leakage
-------
**This module refuses to write anywhere except the training split.** It is a
post-split step by construction: the split is made first by
``prepare_dataset``, and augmentation only ever adds files to ``data/train``.
Validation and test therefore contain original photographs exclusively, and no
augmented image - nor any image derived from a validation or test photograph -
can reach them. Harvested backgrounds are likewise taken only from the training
split. Both rules are enforced in code and covered by tests.

Run::

    python -m src.data.augment_backgrounds --dry-run
    python -m src.data.augment_backgrounds --config config/model_v2.yaml
    python -m src.data.augment_backgrounds --clean      # remove augmented files
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..utils.config import Config, load_config
from ..utils.helpers import (ensure_dir, get_logger, markdown_table,
                             relative_to_root, set_global_seed, utc_timestamp,
                             write_json, write_text)
from ..utils.image_io import InvalidImageError, iter_image_files, open_image

LOGGER = get_logger(__name__)

#: Marker that identifies a generated file. Originals never carry it, so
#: augmented images can always be found, counted or removed exactly.
AUGMENTED_PREFIX = "aug"


class AugmentationError(RuntimeError):
    """Raised when augmentation is misconfigured or would be unsafe."""


@dataclass
class AugmentationPlan:
    """What the configuration asks for, resolved against the dataset."""

    enabled: bool
    classes: List[str]
    copies_per_image: int
    background_sources: List[str]
    harvest_from: Optional[str]
    seed: int
    leaf_scale: Tuple[float, float]
    brightness_threshold: float
    saturation_threshold: float
    min_leaf_fraction: float
    jpeg_quality: int


@dataclass
class AugmentationResult:
    generated_utc: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    per_class: Dict[str, int] = field(default_factory=dict)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    backgrounds_harvested: int = 0
    backgrounds_procedural: int = 0
    manifest: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.per_class.values())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def resolve_plan(config: Config) -> AugmentationPlan:
    """Read ``data.background_augmentation`` and validate it."""
    settings = config.get("data", "background_augmentation", default={}) or {}
    classes = list(settings.get("classes", []) or [])

    known = set(config.class_names)
    unknown = [name for name in classes if name not in known]
    if unknown:
        raise AugmentationError(
            f"background_augmentation.classes names classes that are not configured: "
            f"{unknown}. Configured classes: {sorted(known)}")

    sources = list(settings.get("background_sources", ["procedural"]) or [])
    allowed = {"procedural", "harvested"}
    bad = [s for s in sources if s not in allowed]
    if bad:
        raise AugmentationError(
            f"background_augmentation.background_sources contains {bad}; "
            f"supported values are {sorted(allowed)}")

    harvest_from = settings.get("harvest_from")
    if "harvested" in sources:
        if not harvest_from:
            raise AugmentationError(
                "background_sources includes 'harvested' but harvest_from names no class")
        if harvest_from not in known:
            raise AugmentationError(
                f"harvest_from={harvest_from!r} is not a configured class")
        if harvest_from in classes:
            raise AugmentationError(
                f"harvest_from={harvest_from!r} is also listed in classes; harvesting a "
                f"class's backgrounds and then pasting them behind that same class would "
                f"not break the correlation")

    scale = settings.get("leaf_scale", [0.5, 0.95]) or [0.5, 0.95]
    return AugmentationPlan(
        enabled=bool(settings.get("enabled", False)),
        classes=classes,
        copies_per_image=int(settings.get("copies_per_image", 1)),
        background_sources=sources,
        harvest_from=harvest_from,
        seed=int(settings.get("seed", config.seed)),
        leaf_scale=(float(scale[0]), float(scale[1])),
        brightness_threshold=float(settings.get("brightness_threshold", 0.70)),
        saturation_threshold=float(settings.get("saturation_threshold", 0.35)),
        min_leaf_fraction=float(settings.get("min_leaf_fraction", 0.04)),
        jpeg_quality=int(settings.get("jpeg_quality", 92)),
    )


# ---------------------------------------------------------------------------
# Leaf / background separation
# ---------------------------------------------------------------------------

def leaf_mask(image: Image.Image, brightness_threshold: float,
              saturation_threshold: float) -> np.ndarray:
    """Boolean mask that is True on leaf pixels.

    The studio images place the leaf against a pale, weakly saturated
    background, so a pixel is background when it is both bright and unsaturated.
    Requiring both conditions keeps pale parts of a diseased leaf - chlorotic
    tissue, mildew coating - on the leaf side of the boundary.
    """
    rgb = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype="float32") / 255.0
    bright = hsv[..., 2] > brightness_threshold
    unsaturated = hsv[..., 1] < saturation_threshold
    return ~(bright & unsaturated)


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------

def procedural_background(rng: random.Random, size: int) -> Image.Image:
    """A synthetic soil / vegetation / sky-like texture.

    Deliberately crude. The aim is to make background uninformative about the
    class, which does not require the backgrounds to be convincing - only
    varied. See the limitations section of docs/v2_experiment.md.
    """
    generator = np.random.default_rng(rng.randrange(1 << 30))
    style = rng.choice(["soil", "vegetation", "sky", "straw"])
    base = {
        "soil": (110, 72, 46),
        "vegetation": (62, 96, 44),
        "sky": (150, 170, 195),
        "straw": (170, 150, 95),
    }[style]

    # Low-frequency variation plus grain, which is enough to look non-uniform.
    coarse = generator.normal(0, 1, (8, 8, 3))
    coarse = np.asarray(Image.fromarray(
        np.clip(coarse * 40 + 128, 0, 255).astype("uint8")).resize((size, size),
                                                                   Image.BICUBIC),
        dtype="float32") - 128.0
    grain = generator.normal(0, 12, (size, size, 3))
    canvas = np.asarray(base, dtype="float32") + coarse * 0.8 + grain

    if style == "vegetation":
        for _ in range(rng.randint(6, 14)):
            x = rng.randrange(size)
            width = rng.randint(2, max(3, size // 40))
            canvas[:, x:x + width] += generator.normal(18, 6, (size, min(width, size - x), 3))
    return Image.fromarray(np.clip(canvas, 0, 255).astype("uint8"))


def harvest_backgrounds(config: Config, plan: AugmentationPlan,
                        max_patches: int = 400) -> List[Image.Image]:
    """Crop background patches from the field-photography class.

    **Training split only.** Taking patches from validation or test would let
    information from those splits reach the training set.
    """
    if not plan.harvest_from:
        return []
    spec = config.resolve_class(plan.harvest_from)
    if spec is None:
        raise AugmentationError(f"harvest_from={plan.harvest_from!r} is not a configured class")

    source_dir = config.path("train_dir") / spec.directory
    files = list(iter_image_files(source_dir, config.supported_extensions))
    if not files:
        raise AugmentationError(
            f"no images in {source_dir} to harvest backgrounds from. Run "
            f"'python -m src.data.prepare_dataset' first.")

    patches: List[Image.Image] = []
    for path in files:
        if len(patches) >= max_patches:
            break
        try:
            image = open_image(path)
        except InvalidImageError:
            continue
        width, height = image.size
        # Corner crops are mostly surroundings rather than the subject leaf.
        for box in ((0, 0, width // 3, height // 3),
                    (2 * width // 3, 0, width, height // 3),
                    (0, 2 * height // 3, width // 3, height),
                    (2 * width // 3, 2 * height // 3, width, height)):
            patches.append(image.crop(box))
    return patches


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def composite(leaf_image: Image.Image, background: Image.Image, rng: random.Random,
              plan: AugmentationPlan) -> Optional[Image.Image]:
    """Place the segmented leaf onto ``background``.

    Returns ``None`` when segmentation found too little leaf to be worth using,
    which happens on frames that are almost entirely background.
    """
    size = max(leaf_image.size)
    size = min(max(size, 224), 512)
    leaf_image = leaf_image.convert("RGB").resize((size, size), Image.BILINEAR)
    mask = leaf_mask(leaf_image, plan.brightness_threshold, plan.saturation_threshold)

    if mask.mean() < plan.min_leaf_fraction:
        return None

    scale = rng.uniform(*plan.leaf_scale)
    inner = max(32, int(size * scale))
    leaf_small = leaf_image.resize((inner, inner), Image.BILINEAR)
    mask_small = np.asarray(
        Image.fromarray((mask * 255).astype("uint8")).resize((inner, inner), Image.NEAREST)
    ) > 127

    canvas = np.asarray(background.convert("RGB").resize((size, size), Image.BILINEAR),
                        dtype="float32")
    max_offset = size - inner
    ox = rng.randint(0, max_offset) if max_offset > 0 else 0
    oy = rng.randint(0, max_offset) if max_offset > 0 else 0

    region = canvas[oy:oy + inner, ox:ox + inner]
    leaf_array = np.asarray(leaf_small, dtype="float32")
    region[mask_small] = leaf_array[mask_small]
    canvas[oy:oy + inner, ox:ox + inner] = region
    return Image.fromarray(np.clip(canvas, 0, 255).astype("uint8"))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def augment_training_split(config: Config, plan: Optional[AugmentationPlan] = None,
                           dry_run: bool = False) -> AugmentationResult:
    """Generate background-varied copies of the studio classes in ``data/train``."""
    plan = plan or resolve_plan(config)
    result = AugmentationResult(generated_utc=utc_timestamp(), plan=plan.__dict__.copy())

    if not plan.enabled:
        result.warnings.append(
            "background augmentation is disabled (data.background_augmentation.enabled "
            "is false); nothing was generated")
        return result
    if not plan.classes:
        result.warnings.append("background_augmentation.classes is empty; nothing to do")
        return result

    train_root = config.path("train_dir")
    if not train_root.is_dir():
        raise AugmentationError(
            f"{train_root} does not exist. Run 'python -m src.data.prepare_dataset' first - "
            f"augmentation is a post-split step and must never run before the split.")

    set_global_seed(plan.seed)
    rng = random.Random(plan.seed)

    backgrounds: List[Image.Image] = []
    if "harvested" in plan.background_sources:
        harvested = harvest_backgrounds(config, plan)
        backgrounds.extend(harvested)
        result.backgrounds_harvested = len(harvested)
        LOGGER.info("harvested %d background patches from the training split of '%s'",
                    len(harvested), plan.harvest_from)

    procedural_count = 0
    if "procedural" in plan.background_sources:
        procedural_count = max(64, len(backgrounds) // 2 or 128)
        backgrounds.extend(procedural_background(rng, 384) for _ in range(procedural_count))
        result.backgrounds_procedural = procedural_count
        LOGGER.info("generated %d procedural backgrounds", procedural_count)

    if not backgrounds:
        raise AugmentationError("no backgrounds available - check background_sources")

    for class_name in plan.classes:
        spec = config.resolve_class(class_name)
        if spec is None:
            continue
        class_dir = train_root / spec.directory
        originals = [p for p in iter_image_files(class_dir, config.supported_extensions)
                     if not p.name.startswith(f"{AUGMENTED_PREFIX}__")]
        if not originals:
            result.warnings.append(f"no original images in {class_dir}")
            continue

        made = 0
        for source in originals:
            try:
                leaf_image = open_image(source)
            except InvalidImageError as exc:
                result.skipped.append({"file": relative_to_root(source, config.project_root),
                                       "reason": str(exc)})
                continue
            for copy_index in range(plan.copies_per_image):
                background = backgrounds[rng.randrange(len(backgrounds))]
                blended = composite(leaf_image, background, rng, plan)
                if blended is None:
                    result.skipped.append({
                        "file": relative_to_root(source, config.project_root),
                        "reason": "segmentation found too little leaf"})
                    continue
                # Provenance in the name: generated marker, copy index, and the
                # original filename (which still carries its own source tag).
                target = class_dir / f"{AUGMENTED_PREFIX}__{copy_index}__{source.stem}.jpg"
                if not dry_run:
                    blended.save(target, format="JPEG", quality=plan.jpeg_quality)
                result.manifest.append({
                    "augmented": relative_to_root(target, config.project_root),
                    "original": relative_to_root(source, config.project_root),
                    "class": class_name,
                    "copy_index": str(copy_index),
                })
                made += 1
        result.per_class[class_name] = made
        LOGGER.info("%-18s %4d originals -> %4d augmented", class_name, len(originals), made)

    return result


def count_augmented(config: Config, split: str = "train") -> Dict[str, int]:
    """How many generated files each class of ``split`` currently holds."""
    root = config.path(f"{split}_dir")
    counts: Dict[str, int] = {}
    for spec in config.classes:
        files = iter_image_files(root / spec.directory, config.supported_extensions)
        counts[spec.name] = sum(1 for p in files if p.name.startswith(f"{AUGMENTED_PREFIX}__"))
    return counts


def remove_augmented(config: Config, split: str = "train", dry_run: bool = False) -> int:
    """Delete generated files, leaving originals untouched.

    Only files carrying the generated marker are removed, so an original
    photograph can never be deleted by this function.
    """
    root = config.path(f"{split}_dir")
    removed = 0
    for spec in config.classes:
        for path in list(iter_image_files(root / spec.directory, config.supported_extensions)):
            if path.name.startswith(f"{AUGMENTED_PREFIX}__"):
                if not dry_run:
                    path.unlink()
                removed += 1
    return removed


def verify_no_augmented_outside_train(config: Config) -> Dict[str, int]:
    """Assert validation and test hold no generated files. Returns the counts."""
    found = {}
    for split in ("validation", "test"):
        counts = count_augmented(config, split)
        total = sum(counts.values())
        found[split] = total
    return found


def result_to_markdown(result: AugmentationResult, config: Config) -> str:
    plan = result.plan
    lines = [
        "# Background augmentation report",
        "",
        f"* Generated (UTC): {result.generated_utc}",
        f"* Enabled: {plan.get('enabled')}",
        f"* Seed: {plan.get('seed')}",
        f"* Classes augmented: {', '.join(plan.get('classes') or []) or 'none'}",
        f"* Copies per image: {plan.get('copies_per_image')}",
        f"* Background sources: {', '.join(plan.get('background_sources') or [])}",
        f"* Harvested from: {plan.get('harvest_from') or '-'} "
        f"({result.backgrounds_harvested} patches, training split only)",
        f"* Procedural backgrounds: {result.backgrounds_procedural}",
        f"* Images generated: **{result.total}**",
        f"* Images skipped: {len(result.skipped)}",
        "",
        "## Generated per class",
        "",
        markdown_table(["Class", "Augmented images"],
                       [[name, count] for name, count in result.per_class.items()]
                       + [["**Total**", result.total]]),
        "",
        "## Leakage",
        "",
        "Augmentation writes only into `data/train`. Validation and test contain "
        "original photographs exclusively. Backgrounds are harvested only from the "
        "training split. Verify with:",
        "",
        "```bash",
        "python -m src.data.augment_backgrounds --verify",
        "```",
        "",
    ]
    if result.warnings:
        lines += ["## Warnings", ""] + [f"* {w}" for w in result.warnings] + [""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate background-varied training images for the v2 experiment")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be generated without writing")
    parser.add_argument("--clean", action="store_true",
                        help="remove previously generated images from data/train")
    parser.add_argument("--verify", action="store_true",
                        help="check that no generated image sits in validation or test")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.verify:
        found = verify_no_augmented_outside_train(config)
        train_counts = count_augmented(config, "train")
        print(markdown_table(
            ["Split", "Generated images", "Expected"],
            [["train", sum(train_counts.values()), "any"],
             ["validation", found["validation"], "0"],
             ["test", found["test"], "0"]]))
        if found["validation"] or found["test"]:
            LOGGER.error("LEAKAGE: generated images found outside the training split")
            return 1
        LOGGER.info("no generated images in validation or test")
        return 0

    if args.clean:
        removed = remove_augmented(config, "train", dry_run=args.dry_run)
        LOGGER.info("%s %d generated file(s) from data/train",
                    "would remove" if args.dry_run else "removed", removed)
        return 0

    try:
        result = augment_training_split(config, dry_run=args.dry_run)
    except AugmentationError as exc:
        LOGGER.error("%s", exc)
        return 1

    out_dir = ensure_dir(config.path("results_dir") / "augmentation")
    if not args.dry_run:
        write_json(out_dir / "augmentation_report.json", {
            **{k: v for k, v in result.__dict__.items() if k != "manifest"},
            "total": result.total,
        })
        write_json(out_dir / "augmentation_manifest.json", result.manifest)
        write_text(out_dir / "augmentation_report.md", result_to_markdown(result, config))

    if not args.quiet:
        print(result_to_markdown(result, config))
    for warning in result.warnings:
        LOGGER.warning(warning)
    if args.dry_run:
        LOGGER.info("dry run: nothing was written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
