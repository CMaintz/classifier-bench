"""Unit tests for the bench: corpus, classifiers, runner outcomes, scoring and statistics."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from classifier_bench import classifiers as cl
from classifier_bench.corpus import load_tasks, pad_state, parse_task
from classifier_bench.registry import cost, load_registry, price, resolve
from classifier_bench.runner import Context, Sweep, attempt, plan, warmups
from classifier_bench.scoring import brier, grade_attempt
from classifier_bench.stats import cluster_bootstrap, cohen_kappa, mcnemar_exact, quantile, summarize
from classifier_bench.transport import Response, TransportTimeout

TASKS = load_tasks()
TICKET = next(t for t in TASKS if t.name == "support_ticket")


def test_corpus_is_large_varied_and_valid() -> None:
    assert len(TASKS) >= 15
    assert sum(len(t.cases) for t in TASKS) >= 250
    kinds = {q["type"] for t in TASKS for q in t.questions.values()}
    assert kinds == {"choice", "noul", "score"}
    ids = [c.id for t in TASKS for c in t.cases]
    assert len(ids) == len(set(ids))
    assert all(t.fallback.keys() >= t.questions.keys() for t in TASKS)


def test_bad_gold_and_unknown_task_fail() -> None:
    doc = {
        "task": "x",
        "questions": {"q": {"type": "choice", "criteria": {"a": "A"}}},
        "cases": [{"id": "1", "state": "s", "labels": {"q": "b"}}],
    }
    with pytest.raises(ValueError):
        parse_task(doc)
    with pytest.raises(ValueError, match="unknown task"):
        load_tasks(["nope"])
    assert [t.name for t in load_tasks(["spam"])] == ["spam"]


def test_pad_state_wraps_and_grows() -> None:
    assert pad_state("hi", 0, "s") == "hi"
    padded = pad_state("hi", 1000, "s")
    assert padded["text"] == "hi" and len(padded["unrelated_background_log"].split()) == 800
    assert pad_state({"a": 1}, 10, "s")["a"] == 1


def test_plan_rotates_who_goes_first() -> None:
    jobs = plan(TASKS[:2], 2, 2, seed=1)
    n_cases = sum(len(t.cases) for t in TASKS[:2])
    assert len(jobs) == n_cases * 2 * 2
    firsts = [j.classifier for j in jobs if j.position == 0]
    assert set(firsts) == {0, 1}
    scaled = plan(TASKS[:2], 2, 1, seed=1, sweep=Sweep((0, 500), 2))
    assert sum(j.pad_tokens == 500 for j in scaled) == 2 * 2 * 2
    assert all(j.warmup for j in warmups(TASKS, 2, 3)) and len(warmups(TASKS, 2, 3)) == 6


def test_registry_prices_and_resolves() -> None:
    reg = load_registry()
    assert resolve(reg, "jev-latest") == "jev-1.13.0"
    assert resolve(reg, "claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000}
    assert cost(price(reg, "claude-haiku-4-5"), usage) == pytest.approx(6.1)
    assert cost(price(reg, "jev-latest"), {"input_tokens": 1_000_000}) == pytest.approx(0.042)
    with pytest.raises(ValueError):
        price(reg, "gpt-9")


def test_claude_request_settings_per_model() -> None:
    haiku = cl.ClaudeClassifier("claude-haiku-4-5", {"x-api-key": "k"}).request("s", TICKET.questions).body
    assert "thinking" not in haiku and "effort" not in haiku["output_config"]
    assert haiku["output_config"]["format"]["schema"]["required"] == ["team", "urgent", "sentiment"]
    sonnet = cl.ClaudeClassifier("claude-sonnet-5-5", {"x-api-key": "k"}).request("s", TICKET.questions).body
    assert sonnet["thinking"] == {"type": "between_tools"} and sonnet["output_config"]["effort"] == "low"
    assert cl.auth_headers({"ANTHROPIC_AUTH_TOKEN": "t"})["anthropic-beta"] == "oauth-2025-04-20"
    with pytest.raises(RuntimeError):
        cl.auth_headers({})


def _claude_body(payload: Any, stop: str = "end_turn") -> dict[str, Any]:
    return {
        "model": "m",
        "stop_reason": stop,
        "usage": {"input_tokens": 5},
        "content": [{"type": "text", "text": json.dumps(payload)}],
    }


def test_claude_parse_reshapes_to_jev_answers() -> None:
    clf = cl.ClaudeClassifier("claude-haiku-4-5", {})
    payload = {
        "team": {"answer": "billing", "probabilities": {"billing": 3, "technical": 1, "account": 0, "sales": 0}},
        "urgent": {"p_yes": 1.4},
        "sentiment": {
            "answer": "angry",
            "probabilities": {"angry": 0.9, "frustrated": 0.1, "neutral": 0, "positive": 0},
        },
    }
    parsed = clf.parse(_claude_body(payload), TICKET.questions)
    assert parsed.answers["team"]["confidence"] == pytest.approx(0.75)
    assert parsed.answers["urgent"]["noul"] == 1.0
    assert parsed.answers["sentiment"]["score"] == 0.0
    for bad in (_claude_body(payload, "refusal"), {"content": [{"type": "text", "text": "nope"}]}):
        with pytest.raises(cl.UnparseableError):
            clf.parse(bad, TICKET.questions)
    assert cl.to_jev_answer(TICKET.questions["team"], {"answer": "nope", "probabilities": {}}) is None
    assert cl.to_jev_answer(TICKET.questions["urgent"], "x") is None


def test_jev_request_and_parse() -> None:
    clf = cl.JevClassifier("key", base_url="https://x/v1")
    req = clf.request({"a": 1}, TICKET.questions)
    assert req.url == "https://x/v1/systemone" and req.headers["Authorization"] == "Bearer key"
    assert clf.parse({"answers": {}, "model": "jev-1.13.0"}, {}).model == "jev-1.13.0"
    with pytest.raises(cl.UnparseableError):
        clf.parse({}, {})


def _ctx(transport: Any) -> Context:
    clock = iter(range(0, 10_000, 1))
    jev = cl.JevClassifier("k")
    return Context({TICKET.name: TICKET}, [jev], transport, load_registry(), 1.0, clock=lambda: float(next(clock)))


def _run(transport: Any) -> dict[str, Any]:
    job = plan([TICKET], 1, 1, seed=0)[0]
    return attempt(job, _ctx(transport))


def test_attempt_outcomes() -> None:
    ok_body = {
        "model": "jev-1.13.0",
        "answers": {"team": {"choice": "billing", "confidence": 0.9}},
        "usage": {"input_tokens": 100},
    }
    ok = _run(lambda r, t: Response(200, {"X-Request-Id": "abc"}, json.dumps(ok_body).encode()))
    assert ok["outcome"] == "ok" and ok["request_id"] == "abc" and ok["elapsed_ms"] == 1000.0
    assert ok["cost_usd"] == pytest.approx(100 * 0.042 / 1e6)
    assert _run(lambda r, t: Response(529, {}, b"busy"))["outcome"] == "http_error"
    assert _run(lambda r, t: Response(200, {}, b"not json"))["outcome"] == "unparseable"

    def boom(exc: Exception) -> Any:
        def transport(r: Any, t: float) -> Response:
            raise exc

        return transport

    assert _run(boom(TransportTimeout("slow")))["outcome"] == "timeout"
    assert _run(boom(ConnectionResetError("reset")))["outcome"] == "network_error"


def test_grading_uses_fallback_for_failed_calls() -> None:
    case = TICKET.cases[1]
    failed = {
        "classifier": "jev",
        "repeat": 0,
        "pad_tokens": 0,
        "outcome": "timeout",
        "elapsed_ms": 9.0,
        "answers": None,
    }
    grades = {g.question: g for g in grade_attempt(failed, TICKET, case)}
    assert all(g.fallback for g in grades.values())
    assert grades["team"].predicted == TICKET.fallback["team"]
    answers = {
        "team": {"choice": "technical", "probabilities": {"technical": 1.0}, "confidence": 1.0},
        "urgent": {"noul": 0.8},
    }
    good = grade_attempt(failed | {"outcome": "ok", "answers": answers}, TICKET, case)
    by_q = {g.question: g for g in good}
    assert by_q["team"].correct and by_q["team"].brier == 0.0
    assert by_q["urgent"].brier == pytest.approx(0.04)
    assert by_q["sentiment"].fallback


def test_brier_handles_score_index_keys() -> None:
    q = TICKET.questions["sentiment"]
    assert brier(q, {"probabilities": {"0": 1.0}}, "angry") == 0.0
    assert brier(q, {"probabilities": {}}, "angry") is None


def test_stats() -> None:
    assert quantile([1, 2, 3, 4], 0.5) == 2.5 and math.isnan(quantile([], 0.5))
    s = summarize([1.0, 2.0, 3.0])
    assert s["p50"] == 2.0 and s["min"] == 1.0 and s["stdev"] == 1.0
    assert summarize([]) == {"n": 0} and summarize([5.0])["stdev"] == 0.0
    assert cohen_kappa([("a", "a"), ("b", "b")]) == 1.0 and math.isnan(cohen_kappa([]))
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(10, 0) < 0.01
    lo, hi = cluster_bootstrap([1.0, 2.0, 3.0, 4.0], lambda xs: sum(xs) / len(xs), 200, 0) or (0, 0)
    assert 1.0 <= lo <= hi <= 4.0
    assert cluster_bootstrap([], sum, 10, 0) is None
