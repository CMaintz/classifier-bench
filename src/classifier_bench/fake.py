"""An offline stand-in for both providers (`--dry-run`): synthetic answers, latency and failures.

It proves the run -> analyze -> report pipeline end to end without a key or a cent. The numbers
it produces are made up and are labeled as such in every report.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from .classifiers import Request, options
from .hashing import stable_hash
from .transport import Response, TransportTimeout


@dataclass(frozen=True)
class Profile:
    """Synthetic behavior for one provider."""

    median_ms: float
    sigma: float
    accuracy: float
    error_rate: float = 0.004
    timeout_rate: float = 0.002


JEV_PROFILE = Profile(median_ms=125.0, sigma=0.25, accuracy=0.9)
CLAUDE_PROFILE = Profile(median_ms=690.0, sigma=0.2, accuracy=0.8)


@dataclass
class FakeClock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass
class FakeTransport:
    """Answers from an oracle (state hash -> gold labels), advancing a fake clock by synthetic latency."""

    oracle: dict[str, dict[str, Any]]
    clock: FakeClock
    seed: int = 0
    profiles: dict[str, Profile] = field(default_factory=lambda: {"jev": JEV_PROFILE, "claude": CLAUDE_PROFILE})

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def __call__(self, request: Request, timeout_s: float) -> Response:
        kind = "claude" if request.url.endswith("/messages") else "jev"
        profile = self.profiles[kind]
        elapsed = self._rng.lognormvariate(math.log(profile.median_ms), profile.sigma) / 1000
        self.clock.now += min(elapsed, timeout_s)
        roll = self._rng.random()
        if elapsed >= timeout_s or roll < profile.timeout_rate:
            raise TransportTimeout("synthetic timeout")
        if roll < profile.timeout_rate + profile.error_rate:
            return Response(529, {}, b'{"error": "overloaded"}')
        return Response(
            200, {"request-id": f"fake-{self._rng.getrandbits(32):08x}"}, self._body(kind, request, profile)
        )

    def _body(self, kind: str, request: Request, profile: Profile) -> bytes:
        state = request.body["state"] if kind == "jev" else json.loads(request.body["messages"][0]["content"])["state"]
        questions = (
            request.body["questions"]
            if kind == "jev"
            else json.loads(request.body["messages"][0]["content"])["questions"]
        )
        gold = self.oracle.get(stable_hash(state), {})
        raw = {qid: self._answer(q, gold.get(qid), profile.accuracy) for qid, q in questions.items()}
        if kind == "jev":
            usage = {"input_tokens": len(json.dumps(request.body)) // 4, "output_tokens": 8 * len(raw)}
            return json.dumps({"model": "jev-1.13.0", "answers": _jev_shape(raw, questions), "usage": usage}).encode()
        usage = {"input_tokens": len(json.dumps(request.body)) // 4 + 300, "output_tokens": 40 * len(raw)}
        usage |= {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        content = [{"type": "text", "text": json.dumps(_claude_shape(raw, questions))}]
        model = request.body["model"]
        return json.dumps({"model": model, "content": content, "stop_reason": "end_turn", "usage": usage}).encode()

    def _answer(self, question: dict[str, Any], gold: Any, accuracy: float) -> tuple[Any, float]:
        right = self._rng.random() < accuracy
        conf = self._rng.uniform(0.7, 0.99) if right else self._rng.uniform(0.35, 0.8)
        if question["type"] == "noul":
            truth = bool(gold) if gold is not None else self._rng.random() < 0.5
            return (truth if right else not truth), conf
        opts = options(question)
        wrong = [o for o in opts if o != gold] or opts
        return (gold if right and gold in opts else self._rng.choice(wrong)), conf


def _spread(opts: list[str], pick: str, conf: float) -> dict[str, float]:
    rest = (1 - conf) / max(len(opts) - 1, 1)
    return {o: (conf if o == pick else rest) for o in opts}


def _jev_shape(raw: dict[str, tuple[Any, float]], questions: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qid, (pick, conf) in raw.items():
        q = questions[qid]
        if q["type"] == "noul":
            out[qid] = {"type": "noul", "noul": 0.5 + conf / 2 if pick else 0.5 - conf / 2}
        elif q["type"] == "choice":
            out[qid] = {
                "type": "choice",
                "choice": pick,
                "probabilities": _spread(options(q), pick, conf),
                "confidence": conf,
            }
        else:
            idx = options(q).index(pick)
            probs = {str(i): p for i, p in enumerate(_spread(options(q), pick, conf).values())}
            out[qid] = {"type": "score", "score": float(idx), "probabilities": probs, "confidence": conf}
    return out


def _claude_shape(raw: dict[str, tuple[Any, float]], questions: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qid, (pick, conf) in raw.items():
        q = questions[qid]
        if q["type"] == "noul":
            out[qid] = {"p_yes": 0.5 + conf / 2 if pick else 0.5 - conf / 2}
        else:
            out[qid] = {"answer": pick, "probabilities": _spread(options(q), pick, conf)}
    return out
