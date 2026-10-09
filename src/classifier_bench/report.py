"""Render metrics.json as a Markdown report (the article-style tables)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

Row = list[str]


def pct(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.2f}%"


def ms(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.2f} ms"


def num(x: Any, digits: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{digits}f}"


def usd(x: Any) -> str:
    return "n/a" if x is None else f"${x:,.9f}".rstrip("0").rstrip(".")


def table(header: Sequence[str], rows: Sequence[Row]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def _get(doc: dict[str, Any], path: str) -> Any:
    for part in path.split("."):
        doc = doc.get(part) if isinstance(doc, dict) else None  # type: ignore[assignment]
    return doc


HEADLINE: list[tuple[str, str, Callable[[Any], str]]] = [
    ("Match with expected labels", "accuracy.match_rate", pct),
    ("Match, answered calls only", "accuracy.match_rate_answered_only", pct),
    ("Every answer in the call right", "accuracy.call_exact_match", pct),
    ("Match, agreed ground truth only (contested cases excluded)", "accuracy.decidable.match_rate", pct),
    ("Mean latency", "latency.all_attempts_ms.mean", ms),
    ("p50 latency", "latency.all_attempts_ms.p50", ms),
    ("p90 latency", "latency.all_attempts_ms.p90", ms),
    ("p95 latency", "latency.all_attempts_ms.p95", ms),
    ("p99 latency", "latency.all_attempts_ms.p99", ms),
    ("Minimum measured latency", "latency.all_attempts_ms.min", ms),
    ("Maximum measured latency", "latency.all_attempts_ms.max", ms),
    ("Latency stdev", "latency.all_attempts_ms.stdev", ms),
    ("p50 / p95, successful calls only", "latency.successful_ms", lambda s: f"{ms(s.get('p50'))} / {ms(s.get('p95'))}"),
    ("Throughput (calls/s over the run's wall clock)", "latency.throughput_calls_per_s", lambda x: num(x, 2)),
    ("Registry-priced cost, all base calls", "spend.cost_usd", usd),
    ("Mean cost per call", "spend.mean_cost_usd", usd),
    ("Cost per 1,000 calls", "spend.cost_per_1k_calls_usd", usd),
    ("Cost per correct answer", "spend.cost_per_correct_answer_usd", usd),
    ("Input / output tokens", "spend.tokens", lambda t: f"{t['input_tokens']:,} / {t['output_tokens']:,}"),
    (
        "Cache read / creation tokens",
        "spend.tokens",
        lambda t: f"{t['cache_read_input_tokens']} / {t['cache_creation_input_tokens']}",
    ),
    (
        "Mean input / output tokens per call",
        "spend",
        lambda s: f"{s['mean_input_tokens']:.1f} / {s['mean_output_tokens']:.1f}",
    ),
    ("ECE / MCE (10 bins)", "calibration", lambda c: f"{num(c['ece'])} / {num(c['mce'])}"),
    ("Brier score (lower is better)", "calibration.brier", num),
    ("Mean stated confidence", "calibration.mean_confidence", pct),
    ("Same answer on every repeat", "self_consistency.identical_across_repeats", pct),
]

RELIABILITY: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
    ("HTTP 200 responses", lambda r: f"{r['http_200']}/{r['attempts']}"),
    (
        "Provider errors / timeouts / fallback decisions",
        lambda r: f"{r['provider_errors']} / {r['timeouts']} / {r['fallback_decisions']}",
    ),
    ("Unparseable 200s / network errors", lambda r: f"{r['unparseable']} / {r['network_errors']}"),
    ("HTTP 429 / 529", lambda r: f"{r['rate_limited_429']} / {r['overloaded_529']}"),
    ("Models reported", lambda r: ", ".join(f"{k} x{v}" for k, v in r["models_reported"].items()) or "n/a"),
]


def headline(metrics: dict[str, Any]) -> str:
    names = list(metrics["classifiers"])
    clfs = metrics["classifiers"]
    rows = [
        [label] + [fmt(_get(clfs[n], path)) if _get(clfs[n], path) is not None else "n/a" for n in names]
        for label, path, fmt in HEADLINE
    ]
    rows += [[label] + [fmt(clfs[n]["reliability"]) for n in names] for label, fmt in RELIABILITY]
    return table(["Metric"] + names, rows)


def _ci(est: dict[str, Any], suffix: str = "") -> str:
    ci = est.get("ci95")
    bounds = f"{num(ci[0], 2)}{suffix} to {num(ci[1], 2)}{suffix}" if ci else "n/a"
    return f"{num(est['estimate'], 2)}{suffix} [{bounds}]"


def comparison(name: str, comp: dict[str, Any]) -> str:
    e = comp["estimates"]
    rows = [
        ["p50 speed ratio", _ci(e["p50_speed_ratio"], "x")],
        ["p95 speed ratio", _ci(e["p95_speed_ratio"], "x")],
        ["p99 speed ratio", _ci(e["p99_speed_ratio"], "x")],
        ["Mean speed ratio", _ci(e["mean_speed_ratio"], "x")],
        ["Registry-priced cost savings", _ci(e["cost_savings_pct"], "%")],
        ["Paired match-rate difference (points)", _ci(e["match_rate_diff_points"])],
        ["Agreement between classifiers", _ci(e["agreement_pct"], "%")],
        ["Cohen's kappa", num(comp["cohen_kappa"], 3)],
        [
            "McNemar (A right & B wrong / B right & A wrong, p)",
            "{a_right_b_wrong} / {b_right_a_wrong}, p={p_value:.4g}".format(**comp["mcnemar"]),
        ],
    ]
    intro = (
        f"### {name}\n\n{comp['clusters']} case clusters, {comp['paired_answers']} paired answers. "
        "Ratios are B over A (above 1 means A is faster).\n\n"
    )
    return intro + table(["Comparison", "Estimate [95% clustered-bootstrap interval]"], rows)


def cascade_table(rows: Sequence[dict[str, Any]]) -> str:
    body = [
        [
            f"{r['threshold']:.2f}",
            pct(r["escalation_rate"]),
            pct(r["match_rate"]),
            usd(r["cost_usd"]),
            ms(r["latency_ms"]["mean"]),
            ms(r["latency_ms"]["p95"]),
        ]
        for r in rows
    ]
    return table(["Escalate below", "Escalated", "Match", "Cost", "Mean latency", "p95 latency"], body)


def per_task(metrics: dict[str, Any]) -> str:
    names = list(metrics["classifiers"])
    tasks = sorted({t for n in names for t in metrics["classifiers"][n]["by_task"]})
    rows = []
    for t in tasks:
        cells = [metrics["classifiers"][n]["by_task"].get(t) for n in names]
        rows.append(
            [t]
            + [
                f"{pct(c['match_rate'])}, p50 {ms(c['latency_ms']['p50'])}, p95 {ms(c['latency_ms']['p95'])}"
                if c
                else "n/a"
                for c in cells
            ]
        )
    return table(["Task"] + names, rows)


def sliced(metrics: dict[str, Any], key: str, title: str) -> str:
    names = list(metrics["classifiers"])
    slices = sorted({s for n in names for s in metrics["classifiers"][n][key]})
    rows = [[s] + [_slice_cell(metrics["classifiers"][n][key].get(s)) for n in names] for s in slices]
    return table([title] + names, rows)


def _slice_cell(cell: dict[str, Any] | None) -> str:
    if not cell:
        return "n/a"
    return f"{pct(cell['match_rate'])} of {cell['n']}, p50 {ms(cell['latency_ms']['p50'])}"


def questions(metrics: dict[str, Any]) -> str:
    rows = []
    for n, clf in metrics["classifiers"].items():
        for task, tm in sorted(clf["by_task"].items()):
            for q, qm in tm["questions"].items():
                cal = qm["calibration"]
                worst = ", ".join(f"{k} x{v}" for k, v in list(qm["mismatches"].items())[:3]) or "none"
                rows.append(
                    [
                        f"{task}.{q}",
                        n,
                        pct(qm["match_rate"]),
                        num(qm["macro_f1"], 3),
                        num(cal["ece"], 3),
                        num(cal["brier"], 3),
                        worst,
                    ]
                )
    return table(["Question", "Classifier", "Match", "Macro F1", "ECE", "Brier", "Top mismatches"], rows)


def scale(metrics: dict[str, Any]) -> str:
    names = list(metrics["classifiers"])
    pads = sorted({int(s) for n in names for s in metrics["classifiers"][n]["by_pad_tokens"]})
    rows = []
    for pad in pads:
        for n in names:
            c = metrics["classifiers"][n]["by_pad_tokens"].get(str(pad))
            if c:
                lat = c["latency_ms"]
                rows.append(
                    [
                        f"{pad:,}",
                        n,
                        f"{c['mean_input_tokens']:,.0f}",
                        pct(c["match_rate"]),
                        ms(lat["p50"]),
                        ms(lat["p95"]),
                        ms(lat["p99"]),
                        str(c["failures"]),
                    ]
                )
    return table(["Padding", "Classifier", "Mean input tokens", "Match", "p50", "p95", "p99", "Failed calls"], rows)


def labels(metrics: dict[str, Any]) -> str:
    names = list(metrics["classifiers"])
    first = metrics["classifiers"][names[0]]["by_task"]
    rows = []
    for task, tm in sorted(first.items()):
        for q, qm in tm["questions"].items():
            if qm.get("mean_level_distance") is None and set(qm["by_gold_label"]) <= {"True", "False"}:
                continue
            for label in qm["by_gold_label"]:
                cells = [
                    metrics["classifiers"][n]["by_task"]
                    .get(task, {})
                    .get("questions", {})
                    .get(q, {})
                    .get("by_gold_label", {})
                    .get(label)
                    for n in names
                ]
                rows.append(
                    [f"{task}.{q}", label]
                    + [
                        f"{pct(c['recall'])} of {c['n']}, p50 {ms(c['p50_ms'])}, p95 {ms(c['p95_ms'])}" if c else "n/a"
                        for c in cells
                    ]
                )
    return table(["Question", "Expected"] + names, rows)


GROUP_TITLES = {
    "authored": "Authored corpus (hand-written cases, rule-based labels)",
    "public": "Public datasets (published human labels; likely seen in LLM training)",
}


def group_sections(name: str, block: dict[str, Any]) -> list[str]:
    parts = [
        f"# {GROUP_TITLES.get(name, name)}",
        "## Headline (no padding)\n\n" + headline(block),
        "## Head to head\n\n" + "\n\n".join(comparison(k, v) for k, v in block["comparisons"].items()),
        "## Per task\n\n" + per_task(block),
        "## Per subset\n\n" + sliced(block, "by_subset", "Subset"),
        "## Per language\n\n" + sliced(block, "by_language", "Language"),
        "## Scale sweep\n\nPadding is nominal unrelated background text; the measured mean input tokens are shown.\n\n"
        + scale(block),
        "## Per question\n\n" + questions(block),
        "## Per expected label (choice and score questions)\n\n" + labels(block),
    ]
    for k, v in block["comparisons"].items():
        parts.append(
            f"## Cascade: {k}\n\nRun A on every call; send the call to B when A's least confident "
            "answer is below the threshold.\n\n" + cascade_table(v["cascade"])
        )
    return parts


def sources_table(sources: dict[str, Any]) -> str:
    rows = [
        [task, s["dataset"].split("/")[-1] if s["dataset"].startswith("http") else s["dataset"], s["split"],
         s["revision"][:12], s["license"], s["label_basis"]]
        for task, s in sources.items()
    ]  # fmt: skip
    return table(["Task", "Dataset", "Split", "Revision", "License", "How the gold was obtained"], rows)


def render(metrics: dict[str, Any]) -> str:
    p = metrics["protocol"]
    parts = [
        f"# Jev bench report\n\nRun created {p['created_at']} with {p['tool']}. Classifiers: "
        + ", ".join(f"`{c['name']}` ({c['model']})" for c in p["classifiers"])
        + f". {p['cases']} cases x {p['repeats']} repeats, seed {p['seed']}, concurrency {p['concurrency']}, "
        f"deadline {p['timeout_ms']} ms, zero retries.",
        "## Read this first\n\n" + "\n".join(f"- {d}" for d in metrics["disclosures"]),
    ]
    for name, block in metrics["groups"].items():
        parts += group_sections(name, block)
    if metrics.get("sources"):
        parts.append("## Public sources\n\n" + sources_table(metrics["sources"]))
    parts.append(f"Excluded warmup cost: {usd(metrics['excluded_warmup_cost_usd'])}.")
    return "\n\n".join(parts) + "\n"
