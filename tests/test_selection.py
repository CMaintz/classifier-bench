"""Domain filtering and seeded, label-stratified sampling for cost control."""

from __future__ import annotations

from collections import Counter

import pytest

from classifier_bench.corpus import load_corpus
from classifier_bench.selection import DOMAINS, _allocate, domain_of, sample_task, select

AUTHORED = load_corpus("authored")
TIER = next(t for t in AUTHORED if t.name == "tier_routing")


def test_every_authored_and_public_task_has_a_domain() -> None:
    assert all(domain_of(t.name) != "other" for t in load_corpus("all"))
    assert domain_of("brand_new_task") == "other"


def test_domain_filter() -> None:
    safety = select(AUTHORED, ["safety"])
    assert {t.name for t in safety} == {t.name for t in AUTHORED} & set(DOMAINS["safety"])
    with pytest.raises(ValueError, match="unknown domain"):
        select(AUTHORED, ["nope"])
    with pytest.raises(ValueError, match="no tasks"):
        select(AUTHORED, ["other"])


def test_sample_is_stratified_seeded_and_ordered() -> None:
    half = sample_task(TIER, 0.5, None, seed=7)
    assert len(half.cases) == (len(TIER.cases) + 1) // 2
    full = Counter(c.labels["tier"] for c in TIER.cases)
    got = Counter(c.labels["tier"] for c in half.cases)
    assert all(abs(got[k] - full[k] / 2) <= 1 for k in full)
    assert [c.id for c in half.cases] == [c.id for c in sample_task(TIER, 0.5, None, seed=7).cases]
    order = [c.id for c in TIER.cases]
    assert [order.index(c.id) for c in half.cases] == sorted(order.index(c.id) for c in half.cases)
    assert sample_task(TIER, 1.0, None, 0) is TIER


def test_max_cases_and_bounds() -> None:
    assert len(sample_task(TIER, 1.0, 5, seed=0).cases) == 5
    assert len(sample_task(TIER, 0.001, None, seed=0).cases) == 1
    with pytest.raises(ValueError, match="fraction"):
        select(AUTHORED, fraction=0)
    assert _allocate({"a": 1, "b": 9}, 5) == {"a": 1, "b": 4} or sum(_allocate({"a": 1, "b": 9}, 5).values()) == 5
