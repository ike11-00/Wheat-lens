"""Source-balanced sampling plan for the V3 dataset.

Why this module exists
----------------------
V3 draws the same four classes from several independent collections. Simply
concatenating every folder would reproduce V2's problem in a new form: the
largest collection (WPLDD, studio photography) would still supply most of
three classes, so "more data" would mostly mean "more of the same data". The
V3 hypothesis is about **source diversity**, so the sampling has to protect
diversity explicitly rather than hope for it.

Two rules do that, and nothing else in here is discretionary:

1.  **A derived cap, not a chosen one.** Yellow Rust is the scarcest class and
    cannot be enlarged - there are only two sources for it. Every other class
    is capped at ``imbalance_warn_ratio`` (config/config.yaml, currently 3.0)
    times the Yellow Rust count, so the finished dataset sits exactly at the
    imbalance the project already declares acceptable. The cap follows from
    the data; it is not a round number picked to look tidy.

2.  **Even draw across sources, with redistribution.** Within a class the cap
    is split equally between that class's sources. A source holding less than
    its equal share contributes everything it has, and the shortfall is
    redistributed among the sources that still have room. This maximises the
    number of images coming from the smaller, more varied collections without
    ever duplicating an image to fill a quota.

Neither rule invents images. The plan can only ever select a subset of what
was actually acquired, and every number it produces is traceable to a count
somebody can re-measure.

Selection order inside a source
-------------------------------
``stride_select`` takes evenly spaced items from the sorted file list rather
than the first N. In these datasets consecutive filenames are usually frames
from one capture session, so taking the first N would quietly buy a handful of
sessions instead of a spread across the whole collection.

Run::

    python -m src.data.sample_plan            # projection from the registry
    python -m src.data.sample_plan --counts counts.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TypeVar

from ..utils.config import Config, load_config
from ..utils.helpers import get_logger, markdown_table
from .sources_v3 import expected_counts

LOGGER = get_logger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
def even_allocation(available: Dict[str, int], budget: int) -> Dict[str, int]:
    """Spread ``budget`` across sources as evenly as their supply allows.

    Sources that cannot meet an equal share give everything they have; the
    shortfall is redistributed over the sources that still have room, and the
    process repeats until either the budget or the supply runs out.

    The result always sums to ``min(budget, sum(available))`` and never asks a
    source for more than it holds.
    """
    if budget < 0:
        raise ValueError("budget must not be negative")
    allocation = {key: 0 for key in available}
    pending = {key: count for key, count in available.items() if count > 0}
    remaining = min(budget, sum(pending.values()))

    while pending and remaining > 0:
        share = remaining / len(pending)
        exhausted = [key for key, count in pending.items() if count <= share]
        if exhausted:
            for key in exhausted:
                allocation[key] = pending.pop(key)
                remaining -= allocation[key]
            continue
        # Every remaining source can supply at least an equal share. Split the
        # remainder by largest-remainder over sorted keys, so the outcome is
        # deterministic rather than dependent on dict ordering.
        keys = sorted(pending)
        base, extra = divmod(remaining, len(keys))
        for index, key in enumerate(keys):
            allocation[key] += base + (1 if index < extra else 0)
        remaining = 0
        pending = {}
    return allocation


def stride_select(items: Sequence[T], count: int) -> List[T]:
    """Take ``count`` evenly spaced items, preserving the original order."""
    if count <= 0:
        return []
    total = len(items)
    if count >= total:
        return list(items)
    # Midpoint sampling: positions are spread across the whole sequence
    # instead of clustering at either end.
    picked = sorted({int((index + 0.5) * total / count) for index in range(count)})
    # Rounding collisions can leave the set short; backfill deterministically.
    if len(picked) < count:
        chosen = set(picked)
        for index in range(total):
            if len(chosen) == count:
                break
            chosen.add(index)
        picked = sorted(chosen)
    return [items[index] for index in picked[:count]]


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
@dataclass
class SamplingPlan:
    """What the V3 dataset would contain, per class and per source."""

    available: Dict[str, Dict[str, int]]
    selected: Dict[str, Dict[str, int]]
    cap: int
    ratio: float
    binding_class: str
    capped_classes: List[str] = field(default_factory=list)
    measured: bool = False          # True only when built from real counts

    @property
    def available_totals(self) -> Dict[str, int]:
        return {cls: sum(by_source.values()) for cls, by_source in self.available.items()}

    @property
    def selected_totals(self) -> Dict[str, int]:
        return {cls: sum(by_source.values()) for cls, by_source in self.selected.items()}

    @property
    def total_available(self) -> int:
        return sum(self.available_totals.values())

    @property
    def total_selected(self) -> int:
        return sum(self.selected_totals.values())

    @property
    def imbalance(self) -> float:
        totals = [n for n in self.selected_totals.values() if n > 0]
        if not totals:
            return 0.0
        return max(totals) / min(totals)

    def largest_source_share(self, class_name: str) -> float:
        """Fraction of a selected class coming from its single biggest source."""
        by_source = self.selected.get(class_name, {})
        total = sum(by_source.values())
        if not total:
            return 0.0
        return max(by_source.values()) / total

    def to_dict(self) -> dict:
        return {
            "measured": self.measured,
            "ratio": self.ratio,
            "cap": self.cap,
            "binding_class": self.binding_class,
            "capped_classes": self.capped_classes,
            "available": self.available,
            "selected": self.selected,
            "available_totals": self.available_totals,
            "selected_totals": self.selected_totals,
            "total_available": self.total_available,
            "total_selected": self.total_selected,
            "imbalance_ratio": round(self.imbalance, 3),
            "largest_source_share": {
                cls: round(self.largest_source_share(cls), 3) for cls in self.selected
            },
        }


def build_plan(available: Dict[str, Dict[str, int]], ratio: float,
               measured: bool = False) -> SamplingPlan:
    """Derive the cap from the scarcest class and allocate every other class."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    totals = {cls: sum(by_source.values()) for cls, by_source in available.items()}
    present = {cls: n for cls, n in totals.items() if n > 0}
    if not present:
        raise ValueError("no images available in any class")

    binding_class = min(sorted(present), key=lambda cls: present[cls])
    cap = int(present[binding_class] * ratio)

    selected: Dict[str, Dict[str, int]] = {}
    capped: List[str] = []
    for cls in available:
        selected[cls] = even_allocation(available[cls], cap)
        if totals[cls] > cap:
            capped.append(cls)
    return SamplingPlan(available=available, selected=selected, cap=cap,
                        ratio=ratio, binding_class=binding_class,
                        capped_classes=capped, measured=measured)


def plan_from_registry(config: Config) -> SamplingPlan:
    """A *projection* built from the registry's observed folder counts.

    These counts were taken by listing the source repositories, before any
    quality control or duplicate removal, so the real plan will be smaller.
    ``measured`` is False to keep that distinction visible downstream.
    """
    ratio = float(config.get("data", "imbalance_warn_ratio", default=3.0))
    return build_plan(expected_counts(), ratio, measured=False)


def plan_to_markdown(plan: SamplingPlan, config: Optional[Config] = None) -> str:
    """Render the plan, per class and per source."""
    class_order = list(config.class_names) if config else sorted(plan.available)
    class_order = [c for c in class_order if c in plan.available]
    class_order += [c for c in sorted(plan.available) if c not in class_order]

    sources = sorted({src for by_source in plan.available.values() for src in by_source})

    kind = "Measured" if plan.measured else "Projected (registry counts, before QC)"
    lines = [f"{kind} sampling plan",
             "",
             f"- Cap per class: **{plan.cap}** = {plan.ratio:g} x "
             f"{plan.available_totals[plan.binding_class]} "
             f"({plan.binding_class}, the scarcest class)",
             f"- Selected: **{plan.total_selected}** of {plan.total_available} available",
             f"- Imbalance after sampling: **{plan.imbalance:.2f}:1**",
             ""]

    rows = []
    for cls in class_order:
        row = [cls]
        for src in sources:
            available = plan.available[cls].get(src, 0)
            chosen = plan.selected[cls].get(src, 0)
            row.append("-" if not available else f"{chosen}/{available}")
        row.append(str(sum(plan.selected[cls].values())))
        row.append(f"{plan.largest_source_share(cls) * 100:.0f}%")
        rows.append(row)
    lines.append(markdown_table(["Class"] + sources + ["Selected", "Largest source"], rows))
    lines.append("")
    lines.append("Cells read `selected/available`. \"Largest source\" is the share of the "
                 "selected class coming from its single biggest source - the lower it is, "
                 "the more the class actually tests source diversity.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Show the V3 source-balanced sampling plan.")
    parser.add_argument("--config", default=None, help="configuration file")
    parser.add_argument("--counts", default=None,
                        help="JSON file of measured {class: {source: count}} counts; "
                             "without it the registry's pre-QC counts are used")
    parser.add_argument("--ratio", type=float, default=None,
                        help="override the class-imbalance cap multiplier")
    parser.add_argument("--json", default=None, help="write the plan to this JSON file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    ratio = args.ratio if args.ratio is not None else float(
        config.get("data", "imbalance_warn_ratio", default=3.0))

    if args.counts:
        counts = json.loads(Path(args.counts).read_text(encoding="utf-8"))
        plan = build_plan(counts, ratio, measured=True)
    else:
        plan = build_plan(expected_counts(), ratio, measured=False)
        LOGGER.warning("using registry counts: these are pre-QC projections, not measurements")

    print(plan_to_markdown(plan, config))
    if args.json:
        Path(args.json).write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
