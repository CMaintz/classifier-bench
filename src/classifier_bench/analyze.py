"""Per-classifier metrics over a run: reliability, latency, cost, tokens, accuracy, calibration."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .calibration import bin_by_confidence, ece, mce
from .corpus import Task
from .scoring import Grade, grade_attempt
from .stats import safe_ratio, summarize

Attempt = dict[str, Any]
FAILED = ("http_error", "timeout", "network_error", "unparseable")


def grade_all(attempts: Sequence[Attempt], tasks: Sequence[Task]) -> list[Grade]:
    cases = {(t.name, c.id): (t, c) for t in tasks for c in t.cases}
    grades: list[Grade] = []
    for a in attempts:
        if (a["task"], a["case"]) in cases:
            grades += grade_attempt(a, *cases[(a["task"], a["case"])])
    return grades


def group(items: Iterable[Any], key: Callable[[Any], Any]) -> dict[Any, list[Any]]:
    out: dict[Any, list[Any]] = defaultdict(list)
    for item in items:
        out[key(item)].append(item)
    return dict(out)


def reliability(attempts: Sequence[Attempt]) -> dict[str, Any]:
    outcomes = Counter(a["outcome"] for a in attempts)
    statuses = Counter(str(a["status"]) for a in attempts)
    return {
        "attempts": len(attempts),
        "http_200": statuses.get("200", 0),
        "outcomes": dict(outcomes),
        "statuses": dict(statuses),
        "provider_errors": outcomes.get("http_error", 0),
        "rate_limited_429": statuses.get("429", 0),
        "overloaded_529": statuses.get("529", 0),
        "timeouts": outcomes.get("timeout", 0),
        "network_errors": outcomes.get("network_error", 0),
        "unparseable": outcomes.get("unparseable", 0),
        "fallback_decisions": sum(outcomes.get(k, 0) for k in FAILED),
        "models_reported": dict(Counter(str(a.get("model")) for a in attempts if a.get("model"))),
    }


def latency(attempts: Sequence[Attempt]) -> dict[str, Any]:
    ok = [a["elapsed_ms"] for a in attempts if a["outcome"] == "ok"]
    span = max((a["started_at"] + a["elapsed_ms"] / 1000 for a in attempts), default=0.0) - min(
        (a["started_at"] for a in attempts), default=0.0
    )
    by_position = group(attempts, lambda a: a["position"])
    return {
        "all_attempts_ms": summarize([a["elapsed_ms"] for a in attempts]),
        "successful_ms": summarize(ok),
        "throughput_calls_per_s": safe_ratio(len(attempts), span),
        "p50_by_pair_position_ms": {
            str(k): summarize([a["elapsed_ms"] for a in v]).get("p50") for k, v in by_position.items()
        },
    }


def _usage_sum(attempts: Sequence[Attempt], key: str) -> int:
    return sum(int((a.get("usage") or {}).get(key) or 0) for a in attempts)


def spend(attempts: Sequence[Attempt]) -> dict[str, Any]:
    n = max(len(attempts), 1)
    tokens = {k: _usage_sum(attempts, k) for k in ("input_tokens", "output_tokens")}
    tokens |= {k: _usage_sum(attempts, k) for k in ("cache_read_input_tokens", "cache_creation_input_tokens")}
    total = sum(a["cost_usd"] for a in attempts)
    return {
        "tokens": tokens,
        "mean_input_tokens": tokens["input_tokens"] / n,
        "mean_output_tokens": tokens["output_tokens"] / n,
        "cost_usd": total,
        "mean_cost_usd": total / n,
        "cost_per_1k_calls_usd": 1000 * total / n,
        "cost_per_correct_answer_usd": None,
    }


def accuracy(grades: Sequence[Grade]) -> dict[str, Any]:
    answered = [g for g in grades if not g.fallback]
    return {
        "n": len(grades),
        "match_rate": safe_ratio(sum(g.correct for g in grades), len(grades)),
        "match_rate_answered_only": safe_ratio(sum(g.correct for g in answered), len(answered)),
        "fallback_answers": len(grades) - len(answered),
    }


def decidable(grades: Sequence[Grade]) -> dict[str, Any]:
    """Match on cases with an agreed ground truth (not contested by the blind second annotator)."""
    keep = [g for g in grades if not g.meta.get("contested")]
    return {"n": len(keep), "match_rate": safe_ratio(sum(g.correct for g in keep), len(keep))}


def call_exact_match(grades: Sequence[Grade]) -> float:
    calls = group(grades, lambda g: (g.task, g.case, g.repeat, g.pad_tokens))
    return safe_ratio(sum(all(g.correct for g in v) for v in calls.values()), len(calls))


def calibration(grades: Sequence[Grade], bins: int = 10) -> dict[str, Any]:
    scored = [g.scored for g in grades if g.scored is not None]
    briers = [g.brier for g in grades if g.brier is not None]
    table = bin_by_confidence(scored, bins) if scored else []
    return {
        "n": len(scored),
        "ece": ece(table) if table else None,
        "mce": mce(table) if table else None,
        "brier": statistics.fmean(briers) if briers else None,
        "mean_confidence": statistics.fmean(s.confidence for s in scored) if scored else None,
        "reliability": [
            {"lo": b.lo, "hi": b.hi, "n": b.count, "mean_prob": b.mean_prob, "rate": b.rate} for b in table
        ],
    }


def macro_f1(grades: Sequence[Grade]) -> float:
    labels = {str(g.gold) for g in grades} | {str(g.predicted) for g in grades}
    scores = []
    for label in labels:
        tp = sum(str(g.gold) == label and str(g.predicted) == label for g in grades)
        fp = sum(str(g.gold) != label and str(g.predicted) == label for g in grades)
        fn = sum(str(g.gold) == label and str(g.predicted) != label for g in grades)
        scores.append(safe_ratio(2 * tp, 2 * tp + fp + fn) if tp + fp + fn else 1.0)
    return statistics.fmean(scores) if scores else float("nan")


def question_metrics(grades: Sequence[Grade]) -> dict[str, Any]:
    confusion = Counter(f"{g.gold} -> {g.predicted}" for g in grades if not g.correct)
    per_gold = group(grades, lambda g: str(g.gold))
    out = accuracy(grades) | {"macro_f1": macro_f1(grades), "calibration": calibration(grades)}
    out["by_gold_label"] = {k: _label_cell(v) for k, v in sorted(per_gold.items())}
    out["mismatches"] = dict(confusion.most_common())
    distances = [g.scored.distance for g in grades if g.scored is not None and g.scored.distance is not None]
    if distances:
        out["within_one_level"] = safe_ratio(sum(d <= 1 for d in distances), len(distances))
        out["mean_level_distance"] = statistics.fmean(distances)
    return out


def _label_cell(grades: Sequence[Grade]) -> dict[str, Any]:
    lat = summarize([float(g.meta.get("elapsed_ms") or 0.0) for g in grades])
    return {
        "n": len(grades),
        "recall": safe_ratio(sum(g.correct for g in grades), len(grades)),
        "p50_ms": lat.get("p50"),
        "p95_ms": lat.get("p95"),
    }


def self_consistency(grades: Sequence[Grade]) -> dict[str, Any]:
    """Across repeats of the same (case, question): does the classifier give the same answer?"""
    runs = [v for v in group(grades, lambda g: (g.task, g.case, g.pad_tokens, g.question)).values() if len(v) > 1]
    stable = sum(len({str(g.predicted) for g in v}) == 1 for v in runs)
    return {"items": len(runs), "identical_across_repeats": safe_ratio(stable, len(runs))}


def sliced(grades: Sequence[Grade], attempts: Sequence[Attempt], key: str) -> dict[str, Any]:
    """Accuracy and latency per meta slice (subset, lang) of the authored cases."""
    out: dict[str, Any] = {}
    case_meta = {(g.task, g.case): g.meta for g in grades}
    for name, items in group(grades, lambda g: str(g.meta.get(key))).items():
        lat = [a["elapsed_ms"] for a in attempts if str(case_meta.get((a["task"], a["case"]), {}).get(key)) == name]
        out[name] = accuracy(items) | {"latency_ms": {k: summarize(lat).get(k) for k in ("p50", "p95")}}
    return out


def per_task(grades: Sequence[Grade], attempts: Sequence[Attempt]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task, items in group(grades, lambda g: g.task).items():
        atts = [a for a in attempts if a["task"] == task]
        out[task] = accuracy(items) | {
            "call_exact_match": call_exact_match(items),
            "latency_ms": {k: summarize([a["elapsed_ms"] for a in atts]).get(k) for k in ("mean", "p50", "p95", "max")},
            "cost_usd": sum(a["cost_usd"] for a in atts),
            "questions": {q: question_metrics(v) for q, v in group(items, lambda g: g.question).items()},
        }
    return out


def per_pad(grades: Sequence[Grade], attempts: Sequence[Attempt]) -> dict[str, Any]:
    """Scale sweep: accuracy, latency and input tokens as background padding grows."""
    out: dict[str, Any] = {}
    for pad, items in sorted(group(grades, lambda g: int(g.pad_tokens)).items()):
        atts = [a for a in attempts if a["pad_tokens"] == pad]
        lat = summarize([a["elapsed_ms"] for a in atts])
        out[str(pad)] = accuracy(items) | {
            "latency_ms": {k: lat.get(k) for k in ("p50", "p95", "p99")},
            "mean_input_tokens": spend(atts)["mean_input_tokens"],
            "failures": sum(a["outcome"] != "ok" for a in atts),
        }
    return out


def classifier_metrics(attempts: Sequence[Attempt], grades: Sequence[Grade]) -> dict[str, Any]:
    base_atts = [a for a in attempts if a["pad_tokens"] == 0]
    base = [g for g in grades if g.pad_tokens == 0]
    money = spend(base_atts)
    correct = sum(g.correct for g in base)
    money["cost_per_correct_answer_usd"] = safe_ratio(money["cost_usd"], correct)
    return {
        "reliability": reliability(attempts),
        "latency": latency(base_atts),
        "spend": money,
        "accuracy": accuracy(base) | {"call_exact_match": call_exact_match(base), "decidable": decidable(base)},
        "calibration": calibration(base),
        "self_consistency": self_consistency(base),
        "by_task": per_task(base, base_atts),
        "by_subset": sliced(base, base_atts, "subset"),
        "by_language": sliced(base, base_atts, "lang"),
        "by_pad_tokens": per_pad(grades, attempts),
    }
