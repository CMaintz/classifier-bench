"""Cost control: pick task domains and run a seeded, label-stratified share of each task's cases."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import replace

from .corpus import Case, Task

DOMAINS: dict[str, tuple[str, ...]] = {
    "routing": (
        "tier_routing",
        "department_routing",
        "banking_intent",
        "banking77",
        "clinc_bank_scope",
        "massive_scenario_da",
    ),
    "support": ("support_ticket", "danish_support", "issue_triage", "issue_kind_nlbse", "duplicate_issue"),
    "safety": (
        "toxicity",
        "civil_comments",
        "dkhate",
        "prompt_injection",
        "prompt_injection_deepset",
        "jailbreak",
        "tool_call_risk",
        "pii",
        "spam",
        "sms_spam",
    ),  # fmt: skip
    "grounding": ("citation_support", "vitaminc", "paws_paraphrase"),
    "sentiment": ("review_sentiment", "sst5", "goemotions_ekman", "isarcasm"),
    "language": ("language_id", "language_id_public"),
    "code": ("code_review_severity",),
}


def domain_of(task: str) -> str:
    return next((d for d, names in DOMAINS.items() if task in names), "other")


def by_domain(tasks: Sequence[Task], domains: Sequence[str]) -> list[Task]:
    unknown = set(domains) - set(DOMAINS) - {"other"}
    if unknown:
        raise ValueError(f"unknown domain(s): {', '.join(sorted(unknown))}; known: {', '.join(DOMAINS)}, other")
    return [t for t in tasks if domain_of(t.name) in domains]


def _stratum(case: Case) -> str:
    return "|".join(f"{k}={case.labels[k]}" for k in sorted(case.labels))


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split n across strata in proportion to their sizes (largest remainder), at most each stratum's size."""
    total = sum(sizes.values())
    exact = {k: n * v / total for k, v in sizes.items()}
    take = {k: min(sizes[k], math.floor(x)) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - take[k], reverse=True):
        if sum(take.values()) >= n:
            break
        take[k] += int(take[k] < sizes[k])
    return take


def sample_task(task: Task, fraction: float, max_cases: int | None, seed: int) -> Task:
    """A seeded, label-stratified subset of a task's cases, kept in their original order."""
    n = max(1, math.ceil(len(task.cases) * fraction))
    n = min(n, max_cases) if max_cases else n
    if n >= len(task.cases):
        return task
    rng = random.Random(f"{task.name}:{seed}")
    strata: dict[str, list[Case]] = {}
    for case in task.cases:
        strata.setdefault(_stratum(case), []).append(case)
    for cases in strata.values():
        rng.shuffle(cases)
    take = _allocate({k: len(v) for k, v in strata.items()}, n)
    keep = {c.id for k, cases in strata.items() for c in cases[: take[k]]}
    return replace(task, cases=tuple(c for c in task.cases if c.id in keep))


def select(
    tasks: Sequence[Task],
    domains: Sequence[str] = (),
    fraction: float = 1.0,
    max_cases: int | None = None,
    seed: int = 0,
) -> list[Task]:
    if not 0 < fraction <= 1:
        raise ValueError("--sample must be a fraction in (0, 1]")
    chosen = by_domain(tasks, domains) if domains else list(tasks)
    if not chosen:
        raise ValueError("no tasks left after the domain filter")
    return [sample_task(t, fraction, max_cases, seed) for t in chosen]
