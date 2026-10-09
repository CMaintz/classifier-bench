"""Grade each attempt's answers against the authored labels, one Grade per (attempt, question)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .calibration import Scored, score_noul, score_score
from .classifiers import options
from .corpus import Case, Task


@dataclass(frozen=True)
class Grade:
    classifier: str
    task: str
    case: str
    repeat: int
    pad_tokens: int
    question: str
    kind: str
    gold: Any
    predicted: Any
    correct: bool
    fallback: bool
    scored: Scored | None
    brier: float | None
    meta: dict[str, Any]


def _probabilities(question: dict[str, Any], answer: dict[str, Any]) -> dict[str, float] | None:
    """Option -> probability, normalized; Score keys may be level indices ("0", "1", ...)."""
    raw = answer.get("probabilities")
    if not isinstance(raw, dict) or not raw:
        return None
    opts = options(question)
    mapped = {(opts[int(k)] if str(k).isdigit() and int(k) < len(opts) else str(k)): float(v) for k, v in raw.items()}
    total = sum(mapped.get(o, 0.0) for o in opts)
    return {o: mapped.get(o, 0.0) / total for o in opts} if total > 0 else None


def brier(question: dict[str, Any], answer: dict[str, Any], gold: Any) -> float | None:
    """Squared error of the probability vector against the one-hot gold (0 is perfect, 2 is worst)."""
    if question["type"] == "noul":
        return (float(answer["noul"]) - (1.0 if gold else 0.0)) ** 2
    probs = _probabilities(question, answer)
    if probs is None:
        return None
    return sum((p - (1.0 if o == gold else 0.0)) ** 2 for o, p in probs.items())


def score_choice(answer: dict[str, Any], question: dict[str, Any], gold: Any) -> Scored:
    """The stated choice (what a router consumes), else the argmax; confidence as reported."""
    opts = options(question)
    probs = _probabilities(question, answer) or {}
    stated = answer.get("choice")
    label = stated if stated in opts else (max(probs, key=lambda o: probs[o]) if probs else None)
    conf = answer.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else probs.get(str(label), 0.0)
    ok = label == gold
    return Scored(0, "choice", conf, ok, conf, ok, label, gold)


def _scored(question: dict[str, Any], answer: Any, gold: Any) -> Scored | None:
    if not isinstance(answer, dict):
        return None
    try:
        if question["type"] == "choice":
            return score_choice(answer, question, gold)
        if question["type"] == "noul":
            return score_noul(0, answer, bool(gold))
        return score_score(0, answer, gold, options(question))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _grade(attempt: dict[str, Any], task: Task, case: Case, qid: str) -> Grade:
    question, gold = task.questions[qid], case.labels[qid]
    answer = (attempt.get("answers") or {}).get(qid)
    scored = _scored(question, answer, gold)
    if scored is None:
        predicted = task.fallback.get(qid)
        correct, b = predicted == gold, None
    else:
        predicted, correct = scored.predicted, scored.correct
        b = brier(question, answer, gold) if isinstance(answer, dict) else None
    meta = {**case.meta, "outcome": attempt["outcome"], "elapsed_ms": attempt["elapsed_ms"]}
    return Grade(
        attempt["classifier"], task.name, case.id, attempt["repeat"], attempt["pad_tokens"], qid,
        question["type"], gold, predicted, correct, scored is None, scored, b, meta,
    )  # fmt: skip


def grade_attempt(attempt: dict[str, Any], task: Task, case: Case) -> list[Grade]:
    return [_grade(attempt, task, case, qid) for qid in task.questions if qid in case.labels]
