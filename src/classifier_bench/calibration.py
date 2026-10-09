"""Per-answer outcomes and calibration error (ECE/MCE over equal-width confidence bins)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scored:
    """One (case, question) outcome.

    `confidence`/`correct` are the top-label view. `prob`/`outcome` feed the reliability bins: for
    choice/score they equal confidence/correct; for noul they are the raw P(yes) against the gold
    yes, so a noul bin compares p to the yes-rate. `distance` is the score level gap.
    """

    row: int
    kind: str
    confidence: float
    correct: bool
    prob: float
    outcome: bool
    predicted: Any
    gold: Any
    distance: int | None = None


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    mean_prob: float
    rate: float

    @property
    def gap(self) -> float:
        return abs(self.mean_prob - self.rate)


def nearest_level(value: float, levels: list[str], legend: Any = None) -> int:
    """Index of the level nearest the score: by a numeric `legend` ({label: position}) or as an index."""
    if isinstance(legend, dict) and all(isinstance(legend.get(lv), (int, float)) for lv in levels):
        return min(range(len(levels)), key=lambda i: abs(float(legend[levels[i]]) - value))
    return min(max(round(value), 0), len(levels) - 1)


def score_noul(row: int, answer: dict[str, Any], gold: bool) -> Scored:
    p = float(answer["noul"])
    predicted = p >= 0.5
    return Scored(row, "noul", abs(p - 0.5) * 2, predicted == gold, p, bool(gold), predicted, gold)


def score_score(row: int, answer: dict[str, Any], gold: Any, levels: list[str]) -> Scored:
    index = nearest_level(float(answer["score"]), levels, answer.get("legend"))
    distance = abs(index - levels.index(gold))
    conf = float(answer.get("confidence") or 0.0)
    return Scored(row, "score", conf, distance == 0, conf, distance == 0, levels[index], gold, distance)


def bin_by_confidence(items: Sequence[Scored], n: int = 10) -> list[Bin]:
    """Equal-width bins over [0, 1] of `prob` vs the empirical rate of `outcome` (empty bins kept)."""
    groups: list[list[Scored]] = [[] for _ in range(n)]
    for item in items:
        groups[min(max(int(item.prob * n), 0), n - 1)].append(item)
    return [_make_bin(i, n, group) for i, group in enumerate(groups)]


def _make_bin(i: int, n: int, group: list[Scored]) -> Bin:
    if not group:
        return Bin(i / n, (i + 1) / n, 0, 0.0, 0.0)
    mean = sum(item.prob for item in group) / len(group)
    rate = sum(1 for item in group if item.outcome) / len(group)
    return Bin(i / n, (i + 1) / n, len(group), mean, rate)


def ece(bins: Sequence[Bin]) -> float:
    """Expected Calibration Error: the count-weighted mean |confidence - empirical rate|."""
    total = sum(b.count for b in bins)
    return sum(b.count * b.gap for b in bins) / total if total else 0.0


def mce(bins: Sequence[Bin]) -> float:
    """Max Calibration Error: the worst non-empty bin's gap."""
    return max((b.gap for b in bins if b.count), default=0.0)
