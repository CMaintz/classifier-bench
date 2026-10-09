"""Two more typed-classification APIs: OpenAI Decisions and Cloudflare Clef, as request builders and parsers.

Both get the same questions as Jev. Where an API takes plain-text instructions, the structured parts
(Score level definitions, Noul yes/no criteria) are rendered into the text by one deterministic rule,
so nothing a question says is dropped. Answers are reshaped into Jev's answer shape for one scorer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .classifiers import Parsed, Request, UnparseableError, options

OPENAI_BASE_URL = "https://api.openai.com/v1"
CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4"


def level_text(question: dict[str, Any]) -> str:
    """A Score question's instructions as one string: the question, then each level's definition."""
    raw = question["instructions"]
    if not isinstance(raw, dict):
        return str(raw)
    levels = raw.get("levels") or {}
    lines = [f"- {name}: {levels[name]}" for name in options(question) if name in levels]
    return "\n".join([str(raw.get("question", "")), "", "Levels, lowest first:", *lines]).strip()


def noul_text(question: dict[str, Any]) -> str:
    """A Noul question's instructions with its yes/no criteria appended."""
    criteria = question.get("criteria") or {}
    parts = [str(question["instructions"])]
    if "true" in criteria:
        parts.append(f"Yes means: {criteria['true']}")
    if "false" in criteria:
        parts.append(f"No means: {criteria['false']}")
    return "\n".join(parts)


def _levels(question: dict[str, Any]) -> list[dict[str, Any]]:
    raw = question["instructions"]
    defs = (raw.get("levels") if isinstance(raw, dict) else None) or {}
    return [{"label": name, **({"description": defs[name]} if name in defs else {})} for name in options(question)]


def decisions_question(qid: str, question: dict[str, Any]) -> dict[str, Any]:
    """One corpus question in the Decisions API's shape (predicate | choice | score)."""
    kind = question["type"]
    if kind == "noul":
        return {"type": "predicate", "name": qid, "instructions": noul_text(question)}
    if kind == "choice":
        choices = [{"value": o, "description": str(d)} for o, d in question["criteria"].items()]
        return {"type": "choice", "name": qid, "instructions": str(question["instructions"]), "choices": choices}
    return {"type": "score", "name": qid, "instructions": level_text(question), "levels": _levels(question)}


def _decisions_answer(question: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any] | None:
    kind = raw.get("type")
    if kind == "predicate" and isinstance(raw.get("probability"), (int, float)):
        return {"type": "noul", "noul": float(raw["probability"])}
    probs = {str(p.get("label", p.get("value"))): float(p["probability"]) for p in raw.get("probabilities") or []}
    if kind == "choice" and raw.get("choice") in options(question):
        return {"type": "choice", "choice": raw["choice"], "probabilities": probs, "confidence": raw.get("confidence")}
    if kind == "score" and isinstance(raw.get("score"), (int, float)):
        return {
            "type": "score",
            "score": float(raw["score"]),
            "probabilities": probs,
            "confidence": raw.get("confidence"),
        }
    return None  # a refusal or an answer that does not fit the question


@dataclass
class DecisionsClassifier:
    api_key: str
    model: str = "gpt-6-luna"
    base_url: str = OPENAI_BASE_URL
    name: str = "decisions"

    def request(self, state: Any, questions: dict[str, Any]) -> Request:
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        body = {
            "model": self.model,
            "input": text,
            "questions": [decisions_question(qid, q) for qid, q in questions.items()],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return Request(f"{self.base_url}/decisions", headers, body)

    def parse(self, body: dict[str, Any], questions: dict[str, Any]) -> Parsed:
        answers = body.get("answers")
        if not isinstance(answers, list):
            raise UnparseableError("no answers list")
        by_name = {a.get("name"): a for a in answers if isinstance(a, dict)}
        out = {qid: _decisions_answer(q, by_name[qid]) for qid, q in questions.items() if qid in by_name}
        return Parsed({k: v for k, v in out.items() if v is not None}, dict(body.get("usage") or {}), body.get("model"))


def clef_question(question: dict[str, Any]) -> dict[str, Any]:
    """Clef follows Jev's question shape; Score level definitions go into the instruction text."""
    if question["type"] == "score":
        return {**question, "instructions": level_text(question)}
    return question


@dataclass
class ClefClassifier:
    api_token: str
    account_id: str
    model: str = "clef"
    base_url: str = CLOUDFLARE_BASE_URL
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or self.model

    def request(self, state: Any, questions: dict[str, Any]) -> Request:
        body = {"model": self.model, "state": state, "questions": {k: clef_question(q) for k, q in questions.items()}}
        headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        return Request(f"{self.base_url}/accounts/{self.account_id}/ai/run/@cf/cloudflare/{self.model}", headers, body)

    def parse(self, body: dict[str, Any], questions: dict[str, Any]) -> Parsed:
        result = body.get("result")
        inner: dict[str, Any] = result if isinstance(result, dict) else body  # REST wraps the answer
        answers = inner.get("answers")
        if not isinstance(answers, dict):
            raise UnparseableError("no answers map")
        return Parsed(answers, dict(inner.get("usage") or {}), inner.get("model") or self.model)
