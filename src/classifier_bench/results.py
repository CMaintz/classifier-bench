"""Assemble metrics.json from a sealed run directory (offline; re-run it freely)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analyze import classifier_metrics, grade_all, group
from .compare import cascade, head_to_head
from .corpus import Task, parse_task
from .session import load_attempts

DISCLOSURES = [
    "Expected labels were authored with the cases by one author (an LLM-assisted pass), "
    "without independent review or blind adjudication.",
    "Match rate measures agreement with those frozen labels on this corpus, "
    "not general classification accuracy or downstream answer quality.",
    "Claude probabilities are verbalized (the model states them in JSON); Jev probabilities come from the model. "
    "Calibration compares them as reported.",
    "Repeats of a case are correlated; intervals resample whole cases (clusters), not individual calls.",
    "Timings include the local client, network and provider; client and provider regions are not controlled.",
    "Cost is observed tokens x the frozen registry list prices; invoices, discounts and taxes are not checked.",
    "A timed-out or failed call may still be billed by the provider; cost here counts only usage a response returned.",
    "Jev and Claude count tokens with different tokenizers: compare registry-priced cost, not raw token counts.",
    "Authored and public-dataset results are reported separately and never pooled. Public test sets are old and "
    "likely in LLM training data, which can favor the LLM there.",
    "For choice questions with more than 12 options, Claude states one confidence instead of a full distribution; "
    "its Brier score there spreads the remainder evenly over the other options.",
]


def load_run(path: Path) -> tuple[dict[str, Any], list[Task], list[dict[str, Any]]]:
    protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
    tasks = [parse_task(doc) for doc in json.loads((path / "cases.json").read_text(encoding="utf-8"))]
    return protocol, tasks, load_attempts(path)


def _comparisons(
    names: list[str], by_clf: dict[str, Any], grades: dict[str, Any], resamples: int, seed: int
) -> dict[str, Any]:
    """Every later classifier against the first one listed (the baseline, usually Jev)."""
    base = names[0]
    out: dict[str, Any] = {}
    for other in names[1:]:
        args = (by_clf[base], by_clf[other], grades[base], grades[other])
        out[f"{base} vs {other}"] = head_to_head(*args, resamples=resamples, seed=seed) | {"cascade": cascade(*args)}
    return out


def build_metrics(path: Path, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    protocol, tasks, attempts = load_run(path)
    measured = [a for a in attempts if not a.get("warmup")]
    names = [c["name"] for c in protocol["classifiers"]]
    groups = {g: [t for t in tasks if t.group == g] for g in ("authored", "public")}
    blocks = {
        g: _block(names, [a for a in measured if a["task"] in {t.name for t in ts}], ts, resamples, seed)
        for g, ts in groups.items()
        if ts
    }
    return {
        "protocol": protocol,
        "disclosures": DISCLOSURES + (["DRY RUN: every number below is synthetic."] if protocol.get("dry_run") else []),
        "bootstrap": {"resamples": resamples, "seed": seed, "unit": "case (all repeats kept together)"},
        "excluded_warmup_cost_usd": sum(a["cost_usd"] for a in attempts if a.get("warmup")),
        "sources": {t.name: t.source for t in groups["public"]},
        "groups": blocks,
    }


def _block(
    names: list[str], measured: list[dict[str, Any]], tasks: list[Task], resamples: int, seed: int
) -> dict[str, Any]:
    """Metrics for one group of tasks (authored or public), never pooled with the other group."""
    by_clf = group(measured, lambda a: a["classifier"])
    grades = {n: grade_all(by_clf.get(n, []), tasks) for n in names}
    base_atts = {n: [a for a in by_clf.get(n, []) if a["pad_tokens"] == 0] for n in names}
    base_grades = {n: [g for g in grades[n] if g.pad_tokens == 0] for n in names}
    return {
        "classifiers": {n: classifier_metrics(by_clf.get(n, []), grades[n]) for n in names},
        "comparisons": _comparisons(names, base_atts, base_grades, resamples, seed),
    }
