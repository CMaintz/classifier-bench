"""Stdlib statistics for the bench: linear-interpolated quantiles, clustered bootstrap, agreement tests."""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")
QUANTILES = {"p25": 0.25, "p50": 0.5, "p75": 0.75, "p90": 0.9, "p95": 0.95, "p99": 0.99}


def quantile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default, 'type 7')."""
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(values: Sequence[float]) -> dict[str, float]:
    """n, mean, stdev, min, max and the reporting quantiles."""
    if not values:
        return {"n": 0}
    out = {"n": float(len(values)), "mean": statistics.fmean(values), "min": min(values), "max": max(values)}
    out["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    out |= {name: quantile(values, q) for name, q in QUANTILES.items()}
    return out


def interval(samples: Sequence[float], level: float = 0.95) -> tuple[float, float] | None:
    clean = [s for s in samples if not math.isnan(s)]
    if not clean:
        return None
    tail = (1 - level) / 2
    return quantile(clean, tail), quantile(clean, 1 - tail)


def cluster_bootstrap(
    clusters: Sequence[T], stat: Callable[[Sequence[T]], float], resamples: int, seed: int
) -> tuple[float, float] | None:
    """Percentile interval of `stat`, resampling whole clusters (a case with all its repeats)."""
    if not clusters or resamples <= 0:
        return None
    rng = random.Random(seed)
    n = len(clusters)
    draws = [stat([clusters[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)]
    return interval(draws)


def cohen_kappa(pairs: Sequence[tuple[Any, Any]]) -> float:
    """Agreement beyond chance between two raters' labels."""
    if not pairs:
        return math.nan
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right.get(k, 0) for k in left) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value over discordant pairs (A right/B wrong vs the reverse)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan
