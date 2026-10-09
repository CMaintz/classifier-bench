"""Question shapes: choice (one of a set), noul (yes/no probability) and score (ordered levels)."""

from __future__ import annotations

from typing import Any


def choice(options: dict[str, str], instructions: str) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def noul(instructions: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def score(levels: list[str], instructions: str) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def normalize_question(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Accept the config shape (kind/options/levels) or the wire shape (type/criteria)."""
    kind = raw.get("type") or raw.get("kind")
    instructions = str(raw.get("instructions") or name)
    if kind == "noul":
        return noul(instructions)
    if kind == "choice":
        options = raw.get("criteria") or raw.get("options") or {}
        return choice(options if isinstance(options, dict) else {o: o for o in options}, instructions)
    if kind == "score":
        return score(list(raw.get("criteria") or raw.get("levels") or []), instructions)
    raise ValueError(f'question "{name}": unknown type {kind!r}')


def gold_problem(question: dict[str, Any], gold: Any) -> str | None:
    """Why a gold label cannot be scored against this question, or None if it is fine."""
    kind = question["type"]
    if kind == "noul" and not isinstance(gold, bool):
        return f"noul gold must be true/false, got {gold!r}"
    if kind in ("choice", "score") and gold not in question["criteria"]:
        return f"{kind} gold {gold!r} is not one of {list(question['criteria'])}"
    return None
