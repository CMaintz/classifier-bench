"""classifier-bench CLI: tasks | estimate | run | import | analyze. Only `run` calls paid APIs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

from . import __version__
from .catalog import select
from .corpus import Task, corpus_document, load_corpus, pad_state
from .fake import FakeClock, FakeTransport
from .hashing import stable_hash
from .hub import Hub
from .importer import run_import
from .registry import load_registry
from .report import ms, num, pct, render, table
from .results import build_metrics
from .runner import Context, Job, Sweep, execute, plan, warmups
from .selection import DOMAINS, domain_of
from .selection import select as select_cases
from .session import RunDir, build_classifiers, estimate, protocol_document
from .transport import urllib_transport

Sub: TypeAlias = "argparse._SubParsersAction[argparse.ArgumentParser]"
DEFAULT_MAX_COST = 5.0
DEFAULT_IMPORT_DIR = Path.home() / ".cache" / "classifier-bench"


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _add_plan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-c", "--classifier", action="append", required=True, help="jev[:model] | haiku | sonnet | opus | claude:<id>"
    )
    p.add_argument(
        "--suite", default="authored", choices=("authored", "public", "all"), help="authored, public or both"
    )
    p.add_argument("--corpus-dir", action="append", default=[], help="extra directory of imported task files")
    p.add_argument("--tasks", help="comma-separated task names (default: all in the suite)")
    p.add_argument("--domain", help=f"comma-separated domains: {', '.join(DOMAINS)}, other")
    p.add_argument(
        "--sample", type=float, default=1.0, help="run this fraction of each task, label-stratified (e.g. 0.25)"
    )
    p.add_argument("--max-cases", type=int, help="at most this many cases per task (stratified)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--warmup", type=int, default=4, help="warmup pairs, run first and excluded from metrics")
    p.add_argument("--pad", default="0", help="padding token buckets for the scale sweep, e.g. 0,2000,8000,16000,28000")
    p.add_argument("--scale-cases", type=int, default=4, help="cases per task at each non-zero pad")
    p.add_argument("--effort", help="Claude effort for models that take it (default low)")
    p.add_argument("--dry-run", action="store_true", help="synthetic offline providers; no keys, no cost")


def _add_import(inner: Sub) -> None:
    imp = inner.add_parser("import", help="sample public labeled datasets into task files (free; no model calls)")
    imp.add_argument("--sources", default="runtime", help="'all', 'committable', 'runtime' or a comma-separated list")
    imp.add_argument("--out", default=str(DEFAULT_IMPORT_DIR), help="directory for the task files")
    imp.add_argument("--seed", type=int, default=20261007)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classifier-bench", description="Benchmark typed classifiers head to head against LLMs."
    )
    parser.add_argument("--version", action="version", version=f"classifier-bench {__version__}")
    inner = parser.add_subparsers(dest="command", required=True)
    tasks = inner.add_parser("tasks", help="list the corpus tasks with their domain and size")
    tasks.add_argument("--corpus-dir", action="append", default=[], help="also list imported task files here")
    est = inner.add_parser("estimate", help="registry-priced cost of a planned run (no calls)")
    _add_plan_args(est)
    _add_run(inner)
    _add_import(inner)
    an = inner.add_parser("analyze", help="metrics.json + report.md for a run directory (offline)")
    an.add_argument("run_dir")
    an.add_argument("--resamples", type=int, default=2000)
    an.add_argument("--seed", type=int, default=0)
    return parser


def _add_run(inner: Sub) -> None:
    run = inner.add_parser("run", help="execute a run into a fresh directory")
    _add_plan_args(run)
    run.add_argument("--out", required=True, help="run directory (must not hold attempts yet)")
    run.add_argument("--concurrency", default="1", help="parallel calls; a list (1,4,16) runs one sub-run per level")
    run.add_argument("--timeout-ms", type=int, default=10000)
    run.add_argument("--wire", action="store_true", help="also keep request/response bodies (no auth headers)")
    run.add_argument(
        "--max-cost", type=float, default=DEFAULT_MAX_COST, help="refuse a paid run whose estimate exceeds this (USD)"
    )


def _setup(
    args: argparse.Namespace, timeout_s: float = 10.0, keys: bool = True
) -> tuple[list[Task], Context, list[Job], list[Job]]:
    extra = [Path(d) for d in args.corpus_dir]
    every = load_corpus(args.suite, args.tasks.split(",") if args.tasks else None, extra)
    domains = args.domain.split(",") if args.domain else []
    tasks = select_cases(every, domains, args.sample, args.max_cases, args.seed)
    classifiers = build_classifiers(args.classifier, args.dry_run or not keys, args.effort)
    jobs = plan(tasks, len(classifiers), args.repeats, args.seed, Sweep(_ints(args.pad), args.scale_cases))
    warm = warmups(tasks, len(classifiers), args.warmup)
    ctx = Context({t.name: t for t in tasks}, classifiers, urllib_transport, load_registry(), timeout_s)
    if args.dry_run:
        clock = FakeClock()
        ctx.clock, ctx.transport = clock, FakeTransport(_oracle(jobs + warm, ctx), clock, seed=args.seed)
    return tasks, ctx, jobs, warm


def _oracle(jobs: Sequence[Job], ctx: Context) -> dict[str, dict[str, Any]]:
    return {
        stable_hash(pad_state(j.case.state, j.pad_tokens, f"{j.case.id}:{j.pad_tokens}")): j.case.labels for j in jobs
    }


def _print_estimate(est: dict[str, dict[str, float]]) -> float:
    rows = [[n, f"{int(r['calls']):,}", f"{int(r['input_tokens']):,}", f"${r['cost_usd']:.4f}"] for n, r in est.items()]
    print(table(["Classifier", "Calls", "Est. input tokens", "Est. cost"], rows))
    return sum(r["cost_usd"] for r in est.values())


def cmd_tasks(args: argparse.Namespace) -> int:
    rows = [
        [t.name, t.group, domain_of(t.name), str(len(t.cases)), str(sum(len(c.labels) for c in t.cases))]
        for t in load_corpus("all", extra=[Path(d) for d in args.corpus_dir])
    ]
    print(table(["Task", "Group", "Domain", "Cases", "Decisions"], rows))
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    _, ctx, jobs, warm = _setup(args, keys=False)
    total = _print_estimate(estimate(jobs + warm, ctx))
    print(f"\nTotal estimated: ${total:.4f} for {len(jobs) + len(warm):,} calls (chars/4 token guess; output guessed).")
    return 0


def _run_one(args: argparse.Namespace, out: Path, concurrency: int) -> None:
    tasks, ctx, jobs, warm = _setup(args, args.timeout_ms / 1000)
    run_dir = RunDir(out)
    run_dir.create()
    settings = {
        "repeats": args.repeats,
        "seed": args.seed,
        "concurrency": concurrency,
        "timeout_ms": args.timeout_ms,
        "retries": 0,
        "warmup_pairs": args.warmup,
        "pad_tokens": _ints(args.pad),
        "scale_cases_per_task": args.scale_cases,
        "selection": {"suite": args.suite, "domains": args.domain, "sample": args.sample, "max_cases": args.max_cases},
        "dry_run": args.dry_run,
        "ordering": "seeded shuffle per repeat; the classifier going first rotates per case",
    }
    run_dir.write_json("protocol.json", protocol_document(settings, ctx.classifiers, tasks, ctx.registry))
    run_dir.write_json("cases.json", corpus_document(tasks))
    run_dir.write_json("registry.json", ctx.registry)
    ctx.wire = run_dir.appender("wire.jsonl") if args.wire else None
    sink = run_dir.appender("attempts.jsonl")
    execute(warm, ctx, sink, 1)
    execute(jobs, ctx, sink, concurrency)
    run_dir.seal()
    print(f"wrote {out} ({len(jobs):,} measured + {len(warm):,} warmup attempts)")


def cmd_run(args: argparse.Namespace) -> int:
    levels = _ints(args.concurrency)
    _, ctx, jobs, warm = _setup(args)
    total = _print_estimate(estimate(jobs + warm, ctx)) * len(levels)
    if not args.dry_run and total > args.max_cost:
        raise RuntimeError(f"estimated ${total:.2f} exceeds --max-cost {args.max_cost:.2f}; raise it to proceed")
    for level in levels:
        out = Path(args.out) if len(levels) == 1 else Path(args.out) / f"c{level}"
        _run_one(args, out, level)
    return 0


def _load_rows(paths: Sequence[Path]) -> list[list[str]]:
    rows = []
    for p in paths:
        doc = json.loads((p / "metrics.json").read_text(encoding="utf-8"))
        for (group_name, name), clf in _per_classifier(doc).items():
            lat, rel = clf["latency"]["all_attempts_ms"], clf["reliability"]
            rows.append(
                [
                    str(doc["protocol"]["concurrency"]),
                    f"{name} ({group_name})",
                    ms(lat.get("p50")),
                    ms(lat.get("p95")),
                    ms(lat.get("p99")),
                    num(clf["latency"]["throughput_calls_per_s"], 2),
                    str(rel["rate_limited_429"]),
                    str(rel["timeouts"]),
                    pct(clf["accuracy"]["match_rate"]),
                ]
            )
    return rows


def _per_classifier(doc: dict[str, Any]) -> dict[tuple[str, str], Any]:
    return {(g, n): clf for g, block in doc["groups"].items() for n, clf in block["classifiers"].items()}


def _analyze_one(path: Path, resamples: int, seed: int) -> None:
    metrics = build_metrics(path, resamples, seed)
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    (path / "report.md").write_text(render(metrics), encoding="utf-8")
    print(f"wrote {path / 'metrics.json'} and {path / 'report.md'}")


def cmd_analyze(args: argparse.Namespace) -> int:
    root = Path(args.run_dir)
    subs = sorted((p for p in root.glob("c*") if (p / "protocol.json").exists()), key=lambda p: int(p.name[1:]))
    for path in subs or [root]:
        _analyze_one(path, args.resamples, args.seed)
    if subs:
        header = ["Concurrency", "Classifier", "p50", "p95", "p99", "Calls/s", "429s", "Timeouts", "Match"]
        (root / "load.md").write_text(
            "# Concurrency sweep\n\n" + table(header, _load_rows(subs)) + "\n", encoding="utf-8"
        )
        print(f"wrote {root / 'load.md'}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    hub = Hub()
    run_import(select(hub, args.sources), hub, args.seed, Path(args.out))
    print(f"wrote {args.out}; run them with --suite public --corpus-dir {args.out}")
    return 0


COMMANDS = {
    "import": cmd_import,
    "tasks": cmd_tasks,
    "estimate": cmd_estimate,
    "run": cmd_run,
    "analyze": cmd_analyze,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ValueError, RuntimeError, OSError) as err:
        print(f"classifier-bench: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
