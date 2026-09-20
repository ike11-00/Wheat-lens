"""Image loading, validation and preprocessing shared by every entry point.

Keeping this in one module guarantees that the web application, the CLI
predictor and the evaluation scripts feed the network *identically*
preprocessed pixels.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

# Pillow refuses truncated files by default; we want to *detect* them rather
# than crash, so loading stays strict and validation reports the problem.
ImageFile.LOAD_TRUNCATED_IMAGES = False

ImageSource = Union[str, Path, bytes, bytearray, io.BytesIO, Image.Image]


class InvalidImageError(ValueError):
    """Raised when a file is not a usable image."""


@dataclass
class ImageInfo:
    """Result of inspecting a single file."""

    path: Path
    ok: bool
    reason: str = ""
    width: int = 0
    height: int = 0
    mode: str = ""
    image_format: str = ""
    size_bytes: int = 0

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000


# Files and folders that operating systems scatter through image directories.
# They are not data and must not be reported as corrupt images: a dataset
# copied on macOS carries a .DS_Store in every folder, and one unzipped there
# also carries an AppleDouble "._name" sidecar next to every file.
METADATA_FILENAMES = {".DS_Store", "Thumbs.db", "desktop.ini", ".gitkeep"}
METADATA_DIRECTORIES = {"__MACOSX", ".AppleDouble", ".Spotlight-V100", ".Trashes",
                        "$RECYCLE.BIN", "System Volume Information"}


def is_metadata_path(path: Path) -> bool:
    """True for OS bookkeeping files that should be skipped silently."""
    path = Path(path)
    if path.name in METADATA_FILENAMES:
        return True
    # AppleDouble resource forks: "._original-name.jpg"
    if path.name.startswith("._"):
        return True
    return any(part in METADATA_DIRECTORIES for part in path.parts)


def iter_image_files(directory: Path, extensions: Sequence[str]) -> Iterator[Path]:
    """Yield candidate image files under ``directory`` (recursively, sorted).

    Operating-system metadata is skipped silently - see :func:`is_metadata_path`.
    """
    directory = Path(directory)
    if not directory.exists():
        return
    allowed = {e.lower() for e in extensions}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or is_metadata_path(path):
            continue
        if path.suffix.lower() in allowed:
            yield path


def inspect_image(path: Path, min_pixels: int = 32,
                  extensions: Optional[Sequence[str]] = None) -> ImageInfo:
    """Validate a single image file without raising.

    Two passes are needed: ``Image.verify()`` detects structural corruption but
    leaves the file unusable afterwards, so the file is re-opened and a real
    decode is forced to catch truncated payloads.
    """
    path = Path(path)
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return ImageInfo(path=path, ok=False, reason=f"unreadable: {exc}")

    if size_bytes == 0:
        return ImageInfo(path=path, ok=False, reason="empty file", size_bytes=0)

    if extensions is not None and path.suffix.lower() not in {e.lower() for e in extensions}:
        return ImageInfo(path=path, ok=False, reason=f"unsupported extension {path.suffix}",
                         size_bytes=size_bytes)

    try:
        with Image.open(path) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return ImageInfo(path=path, ok=False, reason=f"corrupt or unreadable: {exc}",
                         size_bytes=size_bytes)

    try:
        with Image.open(path) as image:
            image_format = image.format or ""
            mode = image.mode
            width, height = image.size
            image.load()  # forces a full decode - catches truncated files
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return ImageInfo(path=path, ok=False, reason=f"truncated or undecodable: {exc}",
                         size_bytes=size_bytes)

    if width < min_pixels or height < min_pixels:
        return ImageInfo(path=path, ok=False,
                         reason=f"too small ({width}x{height}, minimum {min_pixels})",
                         width=width, height=height, mode=mode,
                         image_format=image_format, size_bytes=size_bytes)

    return ImageInfo(path=path, ok=True, width=width, height=height, mode=mode,
                     image_format=image_format, size_bytes=size_bytes)


def open_image(source: ImageSource) -> Image.Image:
    """Open any supported source as an RGB :class:`PIL.Image.Image`.

    Raises :class:`InvalidImageError` with a user-safe message - the web app
    surfaces this text directly instead of a stack trace.
    """
    try:
        if isinstance(source, Image.Image):
            image = source
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(bytes(source)))
            image.load()
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            image = Image.open(source)
            image.load()
        else:
            path = Path(source)
            if not path.exists():
                raise InvalidImageError(f"image not found: {path}")
            image = Image.open(path)
            image.load()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"the file could not be read as an image ({exc})") from exc

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def load_image_array(source: ImageSource, image_size: int) -> np.ndarray:
    """Return a ``(image_size, image_size, 3)`` float32 array in the 0-255 range.

    Rescaling to the model's input range is done *inside* the Keras model (see
    ``src/model/build_model.py``), so every caller can hand the network raw
    0-255 pixels and cannot get the preprocessing wrong.
    """
    image = open_image(source)
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype="float32")
    if array.ndim == 2:  # defensive: grayscale that escaped the RGB convert
        array = np.stack([array] * 3, axis=-1)
    return array[..., :3]


def load_batch(sources: Sequence[ImageSource], image_size: int) -> np.ndarray:
    """Stack several images into a ``(n, size, size, 3)`` batch."""
    if not sources:
        return np.zeros((0, image_size, image_size, 3), dtype="float32")
    return np.stack([load_image_array(src, image_size) for src in sources], axis=0)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Content hash used to detect byte-identical duplicate files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path, hash_size: int = 8) -> Optional[str]:
    """Perceptual hash (pHash) as a hex string, or ``None`` if unavailable.

    Used to group *near*-duplicate photographs so they never straddle the
    train/test boundary.  Returns ``None`` when ``imagehash`` is not installed
    or the file cannot be decoded; callers must treat that as "no group".
    """
    try:
        import imagehash
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            return str(imagehash.phash(image.convert("RGB"), hash_size=hash_size))
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def hamming_distance(hex_a: str, hex_b: str) -> int:
    """Bit distance between two hex-encoded hashes of equal length."""
    if len(hex_a) != len(hex_b):
        raise ValueError("hashes must have the same length")
    return bin(int(hex_a, 16) ^ int(hex_b, 16)).count("1")


def basic_quality_metrics(source: ImageSource) -> dict:
    """Cheap, interpretable image statistics used during error analysis.

    These are *descriptive* only - they are never used to make a prediction.
    """
    image = open_image(source)
    array = np.asarray(image, dtype="float32") / 255.0
    gray = array.mean(axis=-1)
    # Variance of the Laplacian is a standard, simple sharpness proxy.
    laplacian = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    ) if gray.shape[0] > 2 and gray.shape[1] > 2 else np.zeros((1, 1))
    return {
        "width": image.width,
        "height": image.height,
        "mean_brightness": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "sharpness_laplacian_var": float(np.var(laplacian)),
        "mean_saturation": float(np.asarray(image.convert("HSV"), dtype="float32")[..., 1].mean() / 255.0),
    }


def make_thumbnail_grid(paths: Sequence[Path], captions: Sequence[str],
                        output_path: Path, columns: int = 4,
                        thumb_size: int = 220, title: str = "") -> Optional[Path]:
    """Render a labelled grid of images (used for misclassification grids)."""
    from .helpers import configure_matplotlib, ensure_dir

    if not paths:
        return None
    configure_matplotlib()
    import matplotlib.pyplot as plt

    count = len(paths)
    columns = max(1, min(columns, count))
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3.0, rows * 3.4))
    axes = np.atleast_1d(axes).ravel()

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= count:
            continue
        try:
            image = open_image(paths[index])
            image.thumbnail((thumb_size, thumb_size))
            axis.imshow(image)
        except InvalidImageError:
            axis.text(0.5, 0.5, "unreadable", ha="center", va="center")
        axis.set_title(captions[index], fontsize=7, wrap=True)

    if title:
        figure.suptitle(title, fontsize=11)
    ensure_dir(Path(output_path).parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)
