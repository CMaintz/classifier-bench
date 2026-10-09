"""Plan and execute measured calls; one JSONL record per attempt, success or not.

Timing is a wall clock around the single HTTP call (no retries), so latency includes client,
network and provider work. Failed attempts keep their latency and are scored with the task's
fallback answer, the way a router would.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from .classifiers import Classifier, Parsed, Request, UnparseableError
from .corpus import Case, Task, pad_state
from .registry import cost, price
from .transport import Response, Transport, TransportTimeout

Clock = Callable[[], float]
Sink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class Sweep:
    """The padding sweep: token buckets to wrap cases in, and how many cases per task at each non-zero pad."""

    pads: Sequence[int] = (0,)
    cases_per_task: int = 4


NO_SWEEP = Sweep()


@dataclass(frozen=True)
class Job:
    task: str
    case: Case
    repeat: int
    pad_tokens: int
    classifier: int
    slot: int
    position: int
    warmup: bool = False


@dataclass
class Context:
    tasks: dict[str, Task]
    classifiers: Sequence[Classifier]
    transport: Transport
    registry: dict[str, Any]
    timeout_s: float
    clock: Clock = time.perf_counter
    wire: Sink | None = None


def _pairs(cases: Sequence[tuple[Task, Case]], n: int, repeat: int, pad: int, rng: random.Random) -> list[Job]:
    """One shuffled pass; the classifier that goes first rotates from slot to slot."""
    order = list(cases)
    rng.shuffle(order)
    jobs = []
    for slot, (task, case) in enumerate(order):
        first = (slot + repeat) % n
        jobs += [Job(task.name, case, repeat, pad, (first + k) % n, slot, k) for k in range(n)]
    return jobs


def scale_subset(tasks: Sequence[Task], per_task: int) -> list[tuple[Task, Case]]:
    return [(t, c) for t in tasks for c in t.cases[:per_task]]


def plan(tasks: Sequence[Task], n: int, repeats: int, seed: int, sweep: Sweep = NO_SWEEP) -> list[Job]:
    """Every case x repeat x classifier at pad 0; a per-task subset at each larger pad."""
    rng = random.Random(seed)
    jobs: list[Job] = []
    for pad in sweep.pads:
        cases = [(t, c) for t in tasks for c in t.cases] if pad == 0 else scale_subset(tasks, sweep.cases_per_task)
        for repeat in range(repeats):
            jobs += _pairs(cases, n, repeat, pad, rng)
    return jobs


def warmups(tasks: Sequence[Task], n: int, count: int) -> list[Job]:
    cases = [(t, c) for t in tasks for c in t.cases][:count]
    return [Job(t.name, c, -1, 0, k, slot, k, warmup=True) for slot, (t, c) in enumerate(cases) for k in range(n)]


def _send(ctx: Context, request: Request) -> tuple[Response | None, str | None, str, float]:
    start = ctx.clock()
    try:
        response: Response | None = ctx.transport(request, ctx.timeout_s)
        outcome, error = None, ""
    except TransportTimeout as err:
        response, outcome, error = None, "timeout", str(err)
    except OSError as err:
        response, outcome, error = None, "network_error", f"{type(err).__name__}: {err}"
    return response, outcome, error, (ctx.clock() - start) * 1000


def _text(response: Response | None) -> str | None:
    return None if response is None else response.body.decode("utf-8", errors="replace")


def _interpret(clf: Classifier, response: Response, task: Task) -> tuple[str, Parsed | None, str]:
    if response.status != 200:
        return "http_error", None, _text(response) or ""
    try:
        return "ok", clf.parse(json.loads(response.body), task.questions), ""
    except (UnparseableError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as err:
        return "unparseable", None, str(err)


def _request_id(response: Response | None) -> str | None:
    if response is None:
        return None
    headers = {k.lower(): v for k, v in response.headers.items()}
    return headers.get("request-id") or headers.get("x-request-id")


def attempt(job: Job, ctx: Context) -> dict[str, Any]:
    clf, task = ctx.classifiers[job.classifier], ctx.tasks[job.task]
    state = pad_state(job.case.state, job.pad_tokens, f"{job.case.id}:{job.pad_tokens}")
    request = clf.request(state, task.questions)
    started_at = time.time()
    response, outcome, error, elapsed_ms = _send(ctx, request)
    if ctx.wire is not None:
        ctx.wire(
            {
                "classifier": clf.name,
                "case": job.case.id,
                "repeat": job.repeat,
                "request": request.body,
                "response": _text(response),
            }
        )
    parsed = None
    if response is not None:
        outcome, parsed, error = _interpret(clf, response, task)
    usage = parsed.usage if parsed else {}
    return {
        "classifier": clf.name,
        "requested_model": clf.model,
        "model": parsed.model if parsed else None,
        "task": job.task,
        "case": job.case.id,
        "repeat": job.repeat,
        "pad_tokens": job.pad_tokens,
        "slot": job.slot,
        "position": job.position,
        "warmup": job.warmup,
        "started_at": started_at,
        "elapsed_ms": elapsed_ms,
        "status": response.status if response else None,
        "outcome": outcome,
        "error": error[:500],
        "request_id": _request_id(response),
        "usage": usage,
        "cost_usd": cost(price(ctx.registry, clf.model), usage),
        "stop_reason": parsed.stop_reason if parsed else None,
        "answers": parsed.answers if parsed else None,
    }


def execute(jobs: Sequence[Job], ctx: Context, sink: Sink, concurrency: int = 1) -> None:
    """Run jobs in plan order (concurrency 1) or through a bounded thread pool."""
    if concurrency <= 1:
        for job in jobs:
            sink(attempt(job, ctx))
        return
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for future in as_completed([pool.submit(attempt, job, ctx) for job in jobs]):
            sink(future.result())
