"""End-to-end bench CLI on the offline fake providers: run -> analyze -> report, plus guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from classifier_bench.cli import main
from classifier_bench.corpus import load_corpus
from classifier_bench.session import build_classifiers, parse_spec

BASE = ["run", "-c", "jev", "-c", "haiku", "--dry-run", "--tasks", "support_ticket,tier_routing,spam"]


def test_tasks_and_estimate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tasks"]) == 0
    assert "tier_routing" in capsys.readouterr().out
    assert main(["estimate", "-c", "jev", "-c", "sonnet", "--tasks", "spam", "--repeats", "1"]) == 0
    out = capsys.readouterr().out
    assert "claude-sonnet-5-5" in out and "Total estimated" in out


def test_dry_run_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert main([*BASE, "--out", str(out), "--pad", "0,500", "--scale-cases", "2", "--wire"]) == 0
    names = {p.name for p in out.iterdir()}
    assert {"protocol.json", "cases.json", "registry.json", "attempts.jsonl", "wire.jsonl", "SHA256SUMS"} <= names
    assert main(["analyze", str(out), "--resamples", "50"]) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    jev = metrics["groups"]["authored"]["classifiers"]["jev"]
    n_cases = sum(len(t.cases) for t in load_corpus("authored", ["support_ticket", "tier_routing", "spam"]))
    assert jev["reliability"]["attempts"] == (n_cases * 3 + 6 * 3) and jev["latency"]["all_attempts_ms"]["p50"] > 0
    comp = metrics["groups"]["authored"]["comparisons"]["jev vs claude-haiku-4-5"]
    assert comp["estimates"]["p50_speed_ratio"]["estimate"] > 1 and len(comp["cascade"]) == 8
    report = (out / "report.md").read_text(encoding="utf-8")
    for heading in (
        "## Headline",
        "## Head to head",
        "## Scale sweep",
        "## Per expected label",
        "## Cascade",
        "DRY RUN",
    ):
        assert heading in report
    assert chr(0x2014) not in report and chr(0x2013) not in report


def test_run_refuses_existing_attempts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run"
    args = [*BASE, "--out", str(out), "--repeats", "1", "--warmup", "0"]
    assert main(args) == 0
    assert main(args) == 2
    assert "fresh run directory" in capsys.readouterr().err


def test_concurrency_sweep_writes_load_table(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    assert main([*BASE, "--out", str(out), "--repeats", "1", "--concurrency", "1,4"]) == 0
    assert (out / "c1" / "attempts.jsonl").exists() and (out / "c4" / "attempts.jsonl").exists()
    assert main(["analyze", str(out), "--resamples", "20"]) == 0
    assert "Concurrency sweep" in (out / "load.md").read_text(encoding="utf-8")


def test_paid_run_needs_keys_and_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert main(["run", "-c", "jev", "--out", str(tmp_path / "x")]) == 2
    assert "JEV_API_KEY" in capsys.readouterr().err
    monkeypatch.setenv("JEV_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert main(["run", "-c", "jev", "-c", "sonnet", "--out", str(tmp_path / "y"), "--max-cost", "0.0001"]) == 2
    assert "exceeds --max-cost" in capsys.readouterr().err
    assert not (tmp_path / "y").exists()


def test_specs() -> None:
    assert parse_spec("jev") == ("jev", "jev-latest")
    assert parse_spec("jev:jev-1.13.0") == ("jev", "jev-1.13.0")
    assert parse_spec("claude:claude-opus-5-5") == ("claude", "claude-opus-5-5")
    with pytest.raises(ValueError):
        parse_spec("gpt")
    with pytest.raises(ValueError, match="once"):
        build_classifiers(["haiku", "claude:claude-haiku-4-5"], dry_run=True)
    assert build_classifiers(["jev:jev-1.13.0"], dry_run=True)[0].name == "jev-1.13.0"


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "classifier-bench 0.1.0" in capsys.readouterr().out


def test_dry_run_with_every_provider(tmp_path: Path) -> None:
    out = tmp_path / "all"
    specs = ["-c", "jev", "-c", "decisions", "-c", "clef", "-c", "clef-flash", "-c", "haiku"]
    assert main(["run", *specs, "--dry-run", "--tasks", "support_ticket,review_sentiment", "--repeats", "1",
                 "--out", str(out)]) == 0  # fmt: skip
    assert main(["analyze", str(out), "--resamples", "20"]) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    clfs = metrics["groups"]["authored"]["classifiers"]
    assert set(clfs) == {"jev", "decisions", "clef", "clef-flash", "claude-haiku-4-5"}
    assert all(c["accuracy"]["match_rate"] > 0.5 for c in clfs.values())
    assert "jev vs decisions" in metrics["groups"]["authored"]["comparisons"]
