"""The bench corpus: authored tasks (questions + labeled cases) shipped as JSON next to this module."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .hashing import stable_hash
from .questions import gold_problem, normalize_question

TASKS_DIR = Path(__file__).parent / "tasks"
PUBLIC_DIR = TASKS_DIR / "public"
SUITES = ("authored", "public", "all")
WORDS_PER_TOKEN = 0.8  # the filler vocabulary averages ~1.25 tokens per word
_FILLER_WORDS = (
    "the",
    "quarterly",
    "logistics",
    "review",
    "noted",
    "shipment",
    "volumes",
    "warehouse",
    "staffing",
    "schedule",
    "archive",
    "weather",
    "sensor",
    "calibration",
    "maintenance",
    "window",
    "committee",
    "minutes",
    "appendix",
    "inventory",
    "audit",
    "regional",
    "office",
    "parking",
    "policy",
    "cafeteria",
    "menu",
    "printer",
    "firmware",
    "badge",
    "reader",
    "elevator",
    "facilities",
    "newsletter",
    "recycling",
    "program",
    "training",
    "session",
    "holiday",
    "calendar",
)


@dataclass(frozen=True)
class Case:
    task: str
    id: str
    state: Any
    labels: dict[str, Any]
    meta: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    questions: dict[str, dict[str, Any]]
    fallback: dict[str, Any]
    cases: tuple[Case, ...]
    source: dict[str, Any] | None = None  # provenance of an imported public dataset; None when authored

    @property
    def group(self) -> str:
        return "public" if self.source else "authored"


def _questions(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize to the wire shape, keeping any authored noul criteria."""
    out = {}
    for qid, raw in doc["questions"].items():
        question = normalize_question(qid, raw)
        if not isinstance(raw.get("instructions"), str) and raw.get("instructions"):
            question["instructions"] = raw["instructions"]  # structured: level definitions etc.
        if raw.get("type") == "noul" and raw.get("criteria"):
            question["criteria"] = raw["criteria"]
        out[qid] = question
    return out


def _case(task: str, raw: dict[str, Any], questions: dict[str, dict[str, Any]]) -> Case:
    for qid, gold in raw["labels"].items():
        problem = gold_problem(questions[qid], gold)
        if problem:
            raise ValueError(f"{task}/{raw['id']} {qid}: {problem}")
    return Case(
        task, raw["id"], raw["state"], dict(raw["labels"]), dict(raw.get("meta") or {}), raw.get("rationale", "")
    )


def parse_task(doc: dict[str, Any]) -> Task:
    name = doc["task"]
    questions = _questions(doc)
    cases = tuple(_case(name, raw, questions) for raw in doc["cases"])
    fallback = dict(doc.get("fallback") or {})
    for qid, gold in fallback.items():
        if gold_problem(questions[qid], gold):
            raise ValueError(f"{name}: bad fallback for {qid}: {gold!r}")
    return Task(name, doc.get("description", ""), questions, fallback, cases, doc.get("source"))


def load_tasks(names: Sequence[str] | None = None, directory: Path = TASKS_DIR) -> list[Task]:
    """All tasks in `directory` (sorted by name), or just the named ones."""
    tasks = [parse_task(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(directory.glob("*.json"))]
    return _select(tasks, names)


def load_corpus(suite: str = "authored", names: Sequence[str] | None = None, extra: Sequence[Path] = ()) -> list[Task]:
    """The authored tasks, the committed public imports, or both, plus any extra import directories."""
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {', '.join(SUITES)}")
    dirs = ([TASKS_DIR] if suite != "public" else []) + ([PUBLIC_DIR, *extra] if suite != "authored" else [])
    return _select(_unique([t for d in dirs if d.is_dir() for t in load_tasks(directory=d)]), names)


def _unique(tasks: list[Task]) -> list[Task]:
    """Drop exact re-imports of a task already loaded; two different tasks with one name is an error."""
    seen: dict[str, Task] = {}
    for t in tasks:
        if t.name in seen and task_dict(seen[t.name]) != task_dict(t):
            raise ValueError(f"two different task files are both named {t.name!r}")
        seen.setdefault(t.name, t)
    return list(seen.values())


def _select(tasks: list[Task], names: Sequence[str] | None) -> list[Task]:
    if not names:
        return tasks
    unknown = set(names) - {t.name for t in tasks}
    if unknown:
        raise ValueError(f"unknown task(s): {', '.join(sorted(unknown))}")
    return [t for t in tasks if t.name in names]


def task_dict(t: Task) -> dict[str, Any]:
    doc = {"task": t.name, "description": t.description, "questions": t.questions, "fallback": t.fallback}
    if t.source:
        doc["source"] = t.source
    return doc | {"cases": [{k: v for k, v in c.__dict__.items() if k != "task"} for c in t.cases]}


def corpus_document(tasks: Sequence[Task]) -> list[dict[str, Any]]:
    """A plain, hashable snapshot of the tasks (what a run freezes to cases.json)."""
    return [task_dict(t) for t in tasks]


def dump_task(doc: dict[str, Any]) -> str:
    """Task JSON with readable headers and one case per line (diff-friendly)."""
    head = [f"  {json.dumps(k)}: {_indent(v)}," for k, v in doc.items() if k != "cases"]
    cases = ",\n".join("    " + json.dumps(c, ensure_ascii=False) for c in doc["cases"])
    return "{\n" + "\n".join(head) + '\n  "cases": [\n' + cases + "\n  ]\n}\n"


def _indent(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2).replace("\n", "\n  ")


def corpus_hash(tasks: Sequence[Task]) -> str:
    return stable_hash(corpus_document(tasks))


@lru_cache(maxsize=4096)
def filler(tokens: int, seed: str) -> str:
    """Deterministic, topic-neutral distractor of about `tokens` tokens (nominal; usage has the real count)."""
    rng = random.Random(seed)
    return " ".join(rng.choice(_FILLER_WORDS) for _ in range(int(tokens * WORDS_PER_TOKEN)))


def pad_state(state: Any, tokens: int, seed: str) -> Any:
    """Wrap a state with unrelated background text, to test accuracy as inputs grow."""
    if tokens <= 0:
        return state
    body = state if isinstance(state, dict) else {"text": state}
    return {"unrelated_background_log": filler(tokens, seed), **body}
