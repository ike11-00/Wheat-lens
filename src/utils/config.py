"""Configuration loading for Leaf Lens.

All paths, class names and hyper-parameters live in ``config/config.yaml``.
This module turns that file into a small typed object so the rest of the code
never has to touch raw dictionaries or hard-code a class name.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

# Repository root = two levels above this file (src/utils/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or inconsistent."""


@dataclass(frozen=True)
class ClassSpec:
    """One classification category."""

    name: str          # human readable, e.g. "Yellow Rust"
    directory: str     # folder on disk, e.g. "Yellow_Rust"
    aliases: List[str] = field(default_factory=list)

    def matches(self, candidate: str) -> bool:
        """True if ``candidate`` is a plausible folder name for this class."""
        norm = _normalise(candidate)
        options = [self.name, self.directory, *self.aliases]
        return any(norm == _normalise(option) for option in options)


def _normalise(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


@dataclass
class Config:
    """Parsed ``config.yaml`` with convenience accessors."""

    raw: Dict[str, Any]
    source_path: Optional[Path] = None

    # -- classes ----------------------------------------------------------
    @property
    def classes(self) -> List[ClassSpec]:
        specs: List[ClassSpec] = []
        for entry in self.raw.get("classes", []):
            if isinstance(entry, str):
                specs.append(ClassSpec(name=entry, directory=entry.replace(" ", "_")))
            else:
                name = entry["name"]
                specs.append(
                    ClassSpec(
                        name=name,
                        directory=entry.get("directory", name.replace(" ", "_")),
                        aliases=list(entry.get("aliases", [])),
                    )
                )
        if not specs:
            raise ConfigError("config.yaml defines no classes")
        return specs

    @property
    def class_names(self) -> List[str]:
        """Display names, in the canonical (label-index) order."""
        return [c.name for c in self.classes]

    @property
    def class_dirs(self) -> List[str]:
        """On-disk folder names, in the canonical (label-index) order."""
        return [c.directory for c in self.classes]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def display_name(self, directory: str) -> str:
        """Map a folder name back to the display name."""
        for spec in self.classes:
            if spec.matches(directory):
                return spec.name
        return directory

    def class_index(self, name_or_dir: str) -> int:
        for index, spec in enumerate(self.classes):
            if spec.matches(name_or_dir):
                return index
        raise ConfigError(f"unknown class: {name_or_dir!r}")

    def resolve_class(self, candidate: str) -> Optional[ClassSpec]:
        """Return the ClassSpec matching an arbitrary folder name, or None."""
        for spec in self.classes:
            if spec.matches(candidate):
                return spec
        return None

    # -- generic access ---------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup: ``cfg.get("training", "epochs", default=10)``."""
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def require(self, *keys: str) -> Any:
        sentinel = object()
        value = self.get(*keys, default=sentinel)
        if value is sentinel:
            raise ConfigError(f"missing required config key: {'.'.join(keys)}")
        return value

    # -- paths ------------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a ``paths.<key>`` entry against the repository root."""
        value = self.get("paths", key)
        if value is None:
            raise ConfigError(f"missing required config key: paths.{key}")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    # -- frequently used scalars -----------------------------------------
    @property
    def image_size(self) -> int:
        return int(self.get("data", "image_size", default=224))

    @property
    def batch_size(self) -> int:
        return int(self.get("data", "batch_size", default=32))

    @property
    def seed(self) -> int:
        return int(self.get("random_seed", default=42))

    @property
    def supported_extensions(self) -> List[str]:
        exts = self.get("data", "supported_extensions", default=[".jpg", ".jpeg", ".png", ".webp"])
        return [e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts]

    @property
    def confidence_threshold(self) -> float:
        return float(self.get("inference", "confidence_threshold", default=0.60))

    @property
    def model_version(self) -> str:
        return str(self.get("model", "version", default="v1"))

    @property
    def model_dir(self) -> Path:
        """Directory holding the artefacts of the configured model version."""
        return self.path("models_dir") / self.model_version

    @property
    def model_file(self) -> Path:
        return self.model_dir / "model.keras"

    def split_fractions(self) -> Dict[str, float]:
        split = self.get("data", "split", default={}) or {}
        fractions = {
            "train": float(split.get("train", 0.75)),
            "validation": float(split.get("validation", 0.125)),
            "test": float(split.get("test", 0.125)),
        }
        total = sum(fractions.values())
        if total <= 0:
            raise ConfigError("data.split fractions must be positive")
        if abs(total - 1.0) > 1e-6:
            # Normalise rather than fail: users often write 70/15/15.
            fractions = {k: v / total for k, v in fractions.items()}
        return fractions

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)

    def snapshot(self, destination: Path) -> Path:
        """Write a copy of the effective configuration next to a run's output."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)
        return destination


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    path: Optional[os.PathLike | str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load ``config.yaml``.

    Parameters
    ----------
    path:
        Optional alternative config file.  A file containing only the keys that
        differ (e.g. ``config/model_v2.yaml``) is deep-merged on top of the
        default configuration, so variant files stay short.
    overrides:
        Optional dictionary deep-merged last, used by tests and CLI flags.
    """
    base_path = DEFAULT_CONFIG_PATH
    if not base_path.exists():
        raise ConfigError(f"default configuration not found at {base_path}")

    with base_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    resolved_path = base_path
    if path is not None:
        variant_path = Path(path)
        if not variant_path.is_absolute():
            variant_path = PROJECT_ROOT / variant_path
        if not variant_path.exists():
            raise ConfigError(f"configuration file not found: {variant_path}")
        if variant_path != base_path:
            with variant_path.open("r", encoding="utf-8") as handle:
                variant = yaml.safe_load(handle) or {}
            # A variant that redefines `classes` replaces the list outright.
            data = _deep_merge(data, variant)
            resolved_path = variant_path

    if overrides:
        data = _deep_merge(data, overrides)

    config = Config(raw=data, source_path=resolved_path)
    _validate(config)
    return config


def _validate(config: Config) -> None:
    names = config.class_names
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate class names in configuration: {names}")
    dirs = config.class_dirs
    if len(set(dirs)) != len(dirs):
        raise ConfigError(f"duplicate class directories in configuration: {dirs}")
    if config.image_size < 32:
        raise ConfigError("data.image_size must be at least 32")
    if config.batch_size < 1:
        raise ConfigError("data.batch_size must be at least 1")
    config.split_fractions()  # raises if unusable


def iter_class_dirs(config: Config, root: Path) -> Iterable[tuple[ClassSpec, Path]]:
    """Yield ``(spec, directory)`` pairs for each configured class under ``root``."""
    for spec in config.classes:
        yield spec, root / spec.directory
