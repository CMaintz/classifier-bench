"""OpenAI Decisions and Cloudflare Clef: wire shapes, answer reshaping, spec parsing and credentials."""

from __future__ import annotations

from typing import Any

import pytest

from classifier_bench import session
from classifier_bench.corpus import load_corpus
from classifier_bench.typed_apis import (
    ClefClassifier,
    DecisionsClassifier,
    decisions_question,
    level_text,
    noul_text,
)

TICKET = next(t for t in load_corpus("authored") if t.name == "support_ticket")
QUESTIONS = TICKET.questions


def test_score_and_noul_text_keep_every_definition() -> None:
    text = level_text(QUESTIONS["sentiment"])
    assert text.startswith("How does the customer sound")
    assert all(f"- {level}:" in text for level in QUESTIONS["sentiment"]["criteria"])
    assert text.index("- angry:") < text.index("- positive:")
    urgent = noul_text(QUESTIONS["urgent"])
    assert "Yes means: Blocked now" in urgent and "No means: Can wait" in urgent
    assert level_text({"type": "score", "instructions": "plain", "criteria": ["a", "b"]}) == "plain"


def test_decisions_request_shape() -> None:
    req = DecisionsClassifier("k").request({"subject": "x", "body": "y"}, QUESTIONS)
    assert req.url == "https://api.openai.com/v1/decisions"
    assert req.headers["Authorization"] == "Bearer k"
    assert req.body["model"] == "gpt-6-luna" and req.body["input"] == '{"subject": "x", "body": "y"}'
    by_name = {q["name"]: q for q in req.body["questions"]}
    assert by_name["urgent"]["type"] == "predicate"
    assert {c["value"] for c in by_name["team"]["choices"]} == set(QUESTIONS["team"]["criteria"])
    assert [lv["label"] for lv in by_name["sentiment"]["levels"]] == QUESTIONS["sentiment"]["criteria"]
    assert all("description" in lv for lv in by_name["sentiment"]["levels"])
    assert decisions_question("q", {"type": "score", "instructions": "s", "criteria": ["a"]})["levels"] == [
        {"label": "a"}
    ]


def _decisions_body(*answers: dict[str, Any]) -> dict[str, Any]:
    return {"model": "gpt-6-luna", "answers": list(answers), "usage": {"input_tokens": 42, "output_tokens": 0}}


def test_decisions_answers_become_jev_shaped() -> None:
    team = [{"value": "billing", "probability": 0.8}]
    sentiment = [{"value": 2, "label": "neutral", "probability": 0.7}]
    body = _decisions_body(
        {"type": "predicate", "name": "urgent", "probability": 0.9},
        {"type": "choice", "name": "team", "choice": "billing", "probabilities": team, "confidence": 0.8},
        {"type": "score", "name": "sentiment", "score": 2.2, "probabilities": sentiment, "confidence": 0.7},
    )
    parsed = DecisionsClassifier("k").parse(body, QUESTIONS)
    assert parsed.answers["urgent"] == {"type": "noul", "noul": 0.9}
    assert parsed.answers["team"]["choice"] == "billing" and parsed.answers["team"]["probabilities"] == {"billing": 0.8}
    assert parsed.answers["sentiment"]["score"] == 2.2 and parsed.answers["sentiment"]["probabilities"] == {
        "neutral": 0.7
    }
    assert parsed.model == "gpt-6-luna" and parsed.usage["input_tokens"] == 42


def test_decisions_refusal_and_bad_answers_are_dropped() -> None:
    body = _decisions_body(
        {"type": "refusal", "name": "urgent"},
        {"type": "choice", "name": "team", "choice": "not-an-option", "probabilities": []},
    )
    assert DecisionsClassifier("k").parse(body, QUESTIONS).answers == {}
    with pytest.raises(ValueError, match="no answers list"):
        DecisionsClassifier("k").parse({"answers": {}}, QUESTIONS)


def test_clef_request_and_wrapped_response() -> None:
    clf = ClefClassifier("t", "acct", "clef-flash")
    req = clf.request("hello", QUESTIONS)
    assert req.url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-flash"
    assert req.body["model"] == "clef-flash" and req.body["state"] == "hello"
    assert isinstance(req.body["questions"]["sentiment"]["instructions"], str)
    assert req.body["questions"]["urgent"] == QUESTIONS["urgent"]
    answers = {"urgent": {"type": "noul", "noul": 0.2}}
    wrapped = {"result": {"answers": answers, "usage": {"input_tokens": 9}}, "success": True}
    parsed = clf.parse(wrapped, QUESTIONS)
    assert parsed.answers == answers and parsed.model == "clef-flash" and parsed.usage == {"input_tokens": 9}
    assert clf.parse({"answers": answers, "model": "clef"}, QUESTIONS).model == "clef"
    with pytest.raises(ValueError, match="no answers map"):
        clf.parse({"result": {}}, QUESTIONS)


def test_specs_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    assert session.parse_spec("decisions") == ("decisions", "gpt-6-luna")
    assert session.parse_spec("decisions:gpt-6-luna-2026-10-01") == ("decisions", "gpt-6-luna-2026-10-01")
    assert session.parse_spec("clef") == ("clef", "clef")
    assert session.parse_spec("clef-flash") == ("clef", "clef-flash")
    for name in ("OPENAI_API_KEY", "CLOUDFLARE_AUTH_TOKEN", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        session.build_classifiers(["decisions"], dry_run=False)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    with pytest.raises(RuntimeError, match="CLOUDFLARE_ACCOUNT_ID"):
        session.build_classifiers(["clef"], dry_run=False)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    built = session.build_classifiers(["decisions", "clef", "clef-flash"], dry_run=False)
    assert [c.name for c in built] == ["decisions", "clef", "clef-flash"]
    assert {session._describe(c)["provider"] for c in built} == {"openai", "cloudflare"}
