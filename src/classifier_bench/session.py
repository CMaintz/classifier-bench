"""A run directory: frozen protocol, corpus and registry first, then attempts, then a hash manifest."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .classifiers import SYSTEM_PROMPT, Classifier, ClaudeClassifier, JevClassifier, auth_headers, model_settings
from .corpus import Task, corpus_hash, pad_state
from .hashing import stable_hash
from .registry import cost, price
from .runner import Context, Job

ALIASES = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
ATTEMPTS = "attempts.jsonl"


def parse_spec(spec: str) -> tuple[str, str]:
    """'jev' | 'jev:jev-1.13.0' | 'haiku' | 'sonnet' | 'claude:<model id>' -> (provider, model)."""
    provider, _, model = spec.partition(":")
    if provider == "jev":
        return "jev", model or "jev-latest"
    if provider in ALIASES and not model:
        return "claude", ALIASES[provider]
    if provider == "claude" and model:
        return "claude", model
    raise ValueError(f"unknown classifier {spec!r}: use jev[:model], haiku, sonnet, opus or claude:<model>")


def build_classifiers(specs: Sequence[str], dry_run: bool, effort: str | None = None) -> list[Classifier]:
    out: list[Classifier] = []
    for spec in specs:
        provider, model = parse_spec(spec)
        if provider == "jev":
            key = "dry-run" if dry_run else os.environ.get("JEV_API_KEY", "")
            if not key:
                raise RuntimeError("JEV_API_KEY is not set (or pass --dry-run)")
            base = os.environ.get("TYPESAFE_AI_BASE_URL") or "https://api.typesafe.ai/v1"
            out.append(JevClassifier(key, model, base.rstrip("/"), name="jev" if model == "jev-latest" else model))
        else:
            auth = {"x-api-key": "dry-run"} if dry_run else auth_headers()
            out.append(ClaudeClassifier(model, auth, effort))
    if len({c.name for c in out}) != len(out):
        raise ValueError("each classifier may appear once")
    return out


def _describe(clf: Classifier) -> dict[str, Any]:
    if isinstance(clf, ClaudeClassifier):
        settings = model_settings(clf.model, clf.effort)
        return {
            "name": clf.name,
            "provider": "anthropic",
            "model": clf.model,
            "endpoint": clf.base_url,
            "settings": settings,
        }
    return {"name": clf.name, "provider": "typesafe", "model": clf.model, "endpoint": getattr(clf, "base_url", "")}


def protocol_document(
    settings: dict[str, Any], classifiers: Sequence[Classifier], tasks: Sequence[Task], registry: dict[str, Any]
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": f"classifier-bench {__version__}",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "client": "Python stdlib urllib, one HTTPS request per call, zero retries, no prompt caching",
        "timing": "wall clock (perf_counter) around the single HTTP call; the deadline is urllib's per-socket "
        "timeout, so a slowly trickling response can run past it",
        "classifiers": [_describe(c) for c in classifiers],
        "claude_system_prompt": SYSTEM_PROMPT,
        "tasks": [t.name for t in tasks],
        "cases": sum(len(t.cases) for t in tasks),
        "corpus_sha256": corpus_hash(tasks),
        "registry_sha256": stable_hash(registry),
        **settings,
    }


@dataclass
class RunDir:
    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self) -> None:
        """Refuse to touch a directory that already holds measured attempts."""
        if (self.path / ATTEMPTS).exists():
            raise RuntimeError(f"{self.path} already has {ATTEMPTS}; use a fresh run directory")
        self.path.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, doc: Any) -> None:
        (self.path / name).write_text(
            json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )

    def appender(self, name: str) -> Any:
        target = self.path / name

        def append(record: dict[str, Any]) -> None:
            with self._lock, target.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        return append

    def seal(self) -> None:
        """Write SHA256SUMS over every file in the run directory."""
        lines = []
        for p in sorted(self.path.iterdir()):
            if p.is_file() and p.name != "SHA256SUMS":
                lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
        (self.path / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_attempts(path: Path) -> list[dict[str, Any]]:
    lines = (path / ATTEMPTS).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _output_guess(questions: dict[str, Any]) -> int:
    return sum(12 + 8 * len(q.get("criteria") or []) for q in questions.values())


def estimate(jobs: Sequence[Job], ctx: Context) -> dict[str, dict[str, float]]:
    """Registry-priced cost guess before any paid call (chars / 4 for tokens)."""
    out: dict[str, dict[str, float]] = {}
    for job in jobs:
        clf, task = ctx.classifiers[job.classifier], ctx.tasks[job.task]
        body = clf.request(
            pad_state(job.case.state, job.pad_tokens, f"{job.case.id}:{job.pad_tokens}"), task.questions
        ).body
        usage = {
            "input_tokens": len(json.dumps(body, ensure_ascii=False)) // 4,
            "output_tokens": _output_guess(task.questions),
        }
        row = out.setdefault(clf.name, {"calls": 0, "input_tokens": 0, "cost_usd": 0.0})
        row["calls"] += 1
        row["input_tokens"] += usage["input_tokens"]
        row["cost_usd"] += cost(price(ctx.registry, clf.model), usage)
    return out
