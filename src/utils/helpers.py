"""Small shared helpers: logging, seeding, JSON/IO, plotting defaults."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout exactly once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible runs.

    Full bit-for-bit determinism on GPU is not guaranteed; this makes shuffles,
    splits and weight initialisation repeatable, which is what the project
    documents in docs/reproducibility.md.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        # Dataset tooling must remain usable without TensorFlow installed.
        pass


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any, indent: int = 2) -> Path:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False, default=_json_default)
        handle.write("\n")
    return Path(path)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    if hasattr(value, "tolist"):  # numpy arrays
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


def write_text(path: Path, text: str) -> Path:
    ensure_dir(Path(path).parent)
    Path(path).write_text(text, encoding="utf-8")
    return Path(path)


def relative_to_root(path: Path, root: Path) -> str:
    """Best-effort repo-relative path for human-readable reports."""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    """Render a GitHub-flavoured markdown table."""
    headers = list(headers)
    body = [list(row) for row in rows]
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in body:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def format_percent(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def configure_matplotlib() -> None:
    """Headless-safe matplotlib defaults used by every plotting script."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "figure.autolayout": True,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def environment_report() -> Dict[str, str]:
    """Versions recorded alongside every training / evaluation run."""
    report: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "timestamp_utc": utc_timestamp(),
    }
    for module_name, key in (
        ("tensorflow", "tensorflow"),
        ("numpy", "numpy"),
        ("sklearn", "scikit_learn"),
        ("PIL", "pillow"),
    ):
        try:
            module = __import__(module_name)
            report[key] = getattr(module, "__version__", "unknown")
        except ImportError:
            report[key] = "not installed"
    return report
