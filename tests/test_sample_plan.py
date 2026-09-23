"""The V3 source-balanced sampling plan.

These tests pin the two properties the V3 dataset decision rests on: the plan
can never ask for more images than a source holds (no silent duplication to
fill a quota), and the cap is derived from the scarcest class rather than
chosen.
"""

from __future__ import annotations

import pytest

from src.data.sample_plan import (
    build_plan,
    even_allocation,
    plan_from_registry,
    plan_to_markdown,
    stride_select,
)
from src.data.sources_v3 import expected_counts
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# even_allocation
# ---------------------------------------------------------------------------
def test_allocation_never_exceeds_supply():
    available = {"a": 5, "b": 100, "c": 3}
    allocation = even_allocation(available, 60)
    for key, count in allocation.items():
        assert count <= available[key], f"{key} was asked for more than it holds"


def test_allocation_spends_the_whole_budget_when_supply_allows():
    allocation = even_allocation({"a": 100, "b": 100, "c": 100}, 61)
    assert sum(allocation.values()) == 61


def test_allocation_is_capped_by_total_supply():
    allocation = even_allocation({"a": 4, "b": 6}, 1000)
    assert allocation == {"a": 4, "b": 6}


def test_small_sources_are_drained_before_large_ones_are_trimmed():
    # 'small' cannot meet an equal share of 50, so it gives everything it has
    # and the remainder goes to the two sources that still have room.
    allocation = even_allocation({"small": 4, "big": 500, "mid": 500}, 50)
    assert allocation["small"] == 4
    assert allocation["big"] + allocation["mid"] == 46
    assert abs(allocation["big"] - allocation["mid"]) <= 1


def test_allocation_is_deterministic_and_order_independent():
    forward = even_allocation({"a": 10, "b": 10, "c": 10}, 20)
    backward = even_allocation({"c": 10, "b": 10, "a": 10}, 20)
    assert forward == backward


def test_zero_budget_allocates_nothing():
    assert even_allocation({"a": 10}, 0) == {"a": 0}


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        even_allocation({"a": 10}, -1)


# ---------------------------------------------------------------------------
# stride_select
# ---------------------------------------------------------------------------
def test_stride_select_spreads_across_the_sequence():
    items = list(range(100))
    picked = stride_select(items, 5)
    assert len(picked) == 5
    assert picked == sorted(picked)
    # Not simply the first five - that is the failure mode this guards against.
    assert picked != items[:5]
    assert picked[-1] > 50


def test_stride_select_returns_everything_when_asked_for_too_much():
    items = [1, 2, 3]
    assert stride_select(items, 10) == items


def test_stride_select_exact_count_even_with_rounding_collisions():
    for total in range(1, 40):
        for count in range(0, total + 1):
            picked = stride_select(list(range(total)), count)
            assert len(picked) == count
            assert len(set(picked)) == count


# ---------------------------------------------------------------------------
# build_plan
# ---------------------------------------------------------------------------
def test_cap_is_derived_from_the_scarcest_class():
    available = {
        "Rare": {"x": 100},
        "Common": {"x": 5000, "y": 5000},
    }
    plan = build_plan(available, ratio=3.0)
    assert plan.binding_class == "Rare"
    assert plan.cap == 300
    assert plan.selected_totals["Rare"] == 100      # never padded up to the cap
    assert plan.selected_totals["Common"] == 300


def test_plan_respects_the_configured_imbalance_ratio():
    available = {"Rare": {"x": 100}, "Common": {"x": 5000, "y": 5000}}
    plan = build_plan(available, ratio=3.0)
    assert plan.imbalance == pytest.approx(3.0)


def test_plan_never_selects_more_than_is_available_anywhere():
    plan = build_plan(expected_counts(), ratio=3.0)
    for class_name, by_source in plan.selected.items():
        for source, count in by_source.items():
            assert count <= plan.available[class_name].get(source, 0)


def test_registry_plan_uses_every_yellow_rust_image():
    config = load_config()
    plan = plan_from_registry(config)
    assert plan.binding_class == "Yellow Rust"
    assert plan.selected_totals["Yellow Rust"] == plan.available_totals["Yellow Rust"]
    assert "Yellow Rust" not in plan.capped_classes


def test_registry_plan_is_flagged_as_a_projection():
    # Registry counts were taken before quality control, so anything built
    # from them must not be presented as measured.
    plan = plan_from_registry(load_config())
    assert plan.measured is False
    assert plan.to_dict()["measured"] is False
    assert "Projected" in plan_to_markdown(plan, load_config())


def test_source_balancing_reduces_the_dominant_source_share():
    # The point of the exercise: WPLDD must stop supplying most of a class
    # wherever an alternative source exists.
    plan = build_plan(expected_counts(), ratio=3.0)
    before = {
        cls: max(by_source.values()) / sum(by_source.values())
        for cls, by_source in plan.available.items()
    }
    for cls in ("Healthy", "Brown Rust"):
        assert plan.largest_source_share(cls) < before[cls]


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        build_plan({"Healthy": {}}, ratio=3.0)
