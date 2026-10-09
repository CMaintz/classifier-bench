"""Head-to-head between two classifiers on the same calls: ratios with clustered CIs, agreement, cascade."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .analyze import Attempt, group
from .scoring import Grade
from .stats import cluster_bootstrap, cohen_kappa, mcnemar_exact, quantile, safe_ratio, summarize

CASCADE_GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)


@dataclass(frozen=True)
class Cluster:
    """Everything one authored case contributes, with all its repeats kept together."""

    lat_a: tuple[float, ...]
    lat_b: tuple[float, ...]
    cost_a: float
    cost_b: float
    paired: tuple[tuple[bool, bool, bool], ...]  # (a correct, b correct, a == b) per question x repeat


def _paired(grades_a: Sequence[Grade], grades_b: Sequence[Grade]) -> list[tuple[Grade, Grade]]:
    index = {(g.task, g.case, g.repeat, g.question): g for g in grades_b}
    return [(g, index[k]) for g in grades_a if (k := (g.task, g.case, g.repeat, g.question)) in index]


def clusters(
    atts_a: Sequence[Attempt], atts_b: Sequence[Attempt], pairs: Sequence[tuple[Grade, Grade]]
) -> list[Cluster]:
    by_a, by_b = group(atts_a, _case_key), group(atts_b, _case_key)
    by_pair = group(pairs, lambda p: (p[0].task, p[0].case))
    return [
        Cluster(
            tuple(a["elapsed_ms"] for a in by_a[key]),
            tuple(b["elapsed_ms"] for b in by_b.get(key, [])),
            sum(a["cost_usd"] for a in by_a[key]),
            sum(b["cost_usd"] for b in by_b.get(key, [])),
            tuple((x.correct, y.correct, str(x.predicted) == str(y.predicted)) for x, y in by_pair.get(key, [])),
        )
        for key in sorted(by_a)
    ]


def _case_key(a: Attempt) -> tuple[str, str]:
    return (a["task"], a["case"])


def _flat(cs: Sequence[Cluster], field: str) -> list[float]:
    return [v for c in cs for v in getattr(c, field)]


def _ratio_at(q: float | None) -> Any:
    def stat(cs: Sequence[Cluster]) -> float:
        a, b = _flat(cs, "lat_a"), _flat(cs, "lat_b")
        if q is None:
            return safe_ratio(sum(b) / max(len(b), 1), sum(a) / max(len(a), 1))
        return safe_ratio(quantile(b, q), quantile(a, q))

    return stat


def _savings(cs: Sequence[Cluster]) -> float:
    return 100 * (1 - safe_ratio(sum(c.cost_a for c in cs), sum(c.cost_b for c in cs)))


def _diff(cs: Sequence[Cluster]) -> float:
    rows = [p for c in cs for p in c.paired]
    return 100 * safe_ratio(sum(a for a, _, _ in rows) - sum(b for _, b, _ in rows), len(rows))


def _agree(cs: Sequence[Cluster]) -> float:
    rows = [p for c in cs for p in c.paired]
    return 100 * safe_ratio(sum(s for _, _, s in rows), len(rows))


STATS = {
    "p50_speed_ratio": _ratio_at(0.5),
    "p95_speed_ratio": _ratio_at(0.95),
    "p99_speed_ratio": _ratio_at(0.99),
    "mean_speed_ratio": _ratio_at(None),
    "cost_savings_pct": _savings,
    "match_rate_diff_points": _diff,
    "agreement_pct": _agree,
}


def head_to_head(
    atts_a: Sequence[Attempt], atts_b: Sequence[Attempt], grades_a: Sequence[Grade], grades_b: Sequence[Grade],
    resamples: int, seed: int,
) -> dict[str, Any]:  # fmt: skip
    """B relative to A: a ratio above 1 means A is faster; savings are A's cost below B's."""
    pairs = _paired(grades_a, grades_b)
    cs = clusters(atts_a, atts_b, pairs)
    estimates = {
        name: {"estimate": fn(cs), "ci95": cluster_bootstrap(cs, fn, resamples, seed)} for name, fn in STATS.items()
    }
    only_a = sum(x.correct and not y.correct for x, y in pairs)
    only_b = sum(y.correct and not x.correct for x, y in pairs)
    return {
        "clusters": len(cs),
        "paired_answers": len(pairs),
        "estimates": estimates,
        "cohen_kappa": cohen_kappa([(str(x.predicted), str(y.predicted)) for x, y in pairs]),
        "mcnemar": {"a_right_b_wrong": only_a, "b_right_a_wrong": only_b, "p_value": mcnemar_exact(only_a, only_b)},
        "agreement_by_task": {
            t: safe_ratio(sum(str(x.predicted) == str(y.predicted) for x, y in v), len(v))
            for t, v in group(pairs, lambda p: p[0].task).items()
        },
    }


def _call_confidence(grades: Sequence[Grade]) -> float:
    """A call is as confident as its least confident answer; a failed answer counts as 0."""
    return min((g.scored.confidence if g.scored is not None else 0.0) for g in grades)


def cascade(
    atts_a: Sequence[Attempt], atts_b: Sequence[Attempt], grades_a: Sequence[Grade], grades_b: Sequence[Grade]
) -> list[dict[str, Any]]:
    """Run A first; escalate the call to B when A's confidence is below t. One row per t."""
    calls_a = group(grades_a, lambda g: (g.task, g.case, g.repeat))
    calls_b = group(grades_b, lambda g: (g.task, g.case, g.repeat))
    att_a = {(a["task"], a["case"], a["repeat"]): a for a in atts_a}
    att_b = {(b["task"], b["case"], b["repeat"]): b for b in atts_b}
    keys = [k for k in calls_a if k in calls_b and k in att_a and k in att_b]
    return [_cascade_row(t, keys, calls_a, calls_b, att_a, att_b) for t in CASCADE_GRID]


def _cascade_row(
    t: float, keys: Sequence[Any], calls_a: dict[Any, Any], calls_b: dict[Any, Any], att_a: dict[Any, Any],
    att_b: dict[Any, Any],
) -> dict[str, Any]:  # fmt: skip
    escalated = {k for k in keys if _call_confidence(calls_a[k]) < t}
    answers = [g for k in keys for g in (calls_b[k] if k in escalated else calls_a[k])]
    latency = [att_a[k]["elapsed_ms"] + (att_b[k]["elapsed_ms"] if k in escalated else 0.0) for k in keys]
    spent = sum(att_a[k]["cost_usd"] + (att_b[k]["cost_usd"] if k in escalated else 0.0) for k in keys)
    lat = summarize(latency)
    return {
        "threshold": t,
        "escalation_rate": safe_ratio(len(escalated), len(keys)),
        "match_rate": safe_ratio(sum(g.correct for g in answers), len(answers)),
        "cost_usd": spent,
        "latency_ms": {k: lat.get(k) for k in ("mean", "p50", "p95")},
    }
