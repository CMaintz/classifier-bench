"""Import public, human-labeled datasets as bench tasks: seeded, stratified, with provenance.

Each imported task records where every case came from (dataset, revision sha, split, row) and
how its gold was obtained, so a result can always be traced back to a published label.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .corpus import dump_task
from .hub import PAGE, Hub


@dataclass(frozen=True)
class Mapped:
    """One dataset row turned into a bench case; `stratum` drives the per-label quota."""

    stratum: str
    state: Any
    labels: dict[str, Any]
    lang: str = "en"


Mapper = Callable[[dict[str, Any]], "Mapped | None"]


@dataclass(frozen=True)
class Picked:
    row: int
    mapped: Mapped


@dataclass(frozen=True)
class Source:
    name: str
    prefix: str
    description: str
    questions: dict[str, Any]
    fallback: dict[str, Any]
    license: str
    attribution: str
    label_basis: str
    redistributable: bool
    dataset: str
    split: str
    quotas: dict[str, int]
    mapper: Mapper
    config: str = "default"
    kind: str = "hf"  # "hf" (rows API) or "csv" (pinned raw URL in `dataset`)
    sampler: Callable[[Hub, int], list[Picked]] | None = field(default=None)
    decode: tuple[str, ...] = ()  # ClassLabel features whose integer values are mapped to class names


def _decoder(src: Source, hub: Hub) -> Mapper:
    names = {f: hub.label_names(src.dataset, src.config, f) for f in src.decode}

    def name(feature: str, value: Any) -> Any:
        table = names[feature]
        return [table[v] for v in value] if isinstance(value, list) else table[value]

    return lambda row: src.mapper({k: (name(k, v) if k in names else v) for k, v in row.items()})


def take(
    rows: Sequence[dict[str, Any]], base: int, mapper: Mapper, need: dict[str, int], rng: random.Random
) -> list[Picked]:
    """Pick rows in a seeded order until each stratum's quota is met."""
    picked = []
    order = list(range(len(rows)))
    rng.shuffle(order)
    for i in order:
        mapped = mapper(rows[i])
        if mapped is not None and need.get(mapped.stratum, 0) > 0:
            need[mapped.stratum] -= 1
            picked.append(Picked(base + i, mapped))
    return picked


def sample_hf(src: Source, hub: Hub, seed: int, max_pages: int = 200) -> list[Picked]:
    rng = random.Random(f"{src.name}:{seed}")
    _, total = hub.page(src.dataset, src.config, src.split, 0)
    pages = list(range(math.ceil(total / PAGE)))
    rng.shuffle(pages)
    need, picked, mapper = dict(src.quotas), [], _decoder(src, hub)
    for page in pages[:max_pages]:
        rows, _ = hub.page(src.dataset, src.config, src.split, page * PAGE)
        picked += take(rows, page * PAGE, mapper, need, rng)
        if not any(need.values()):
            break
    return picked


def sample(src: Source, hub: Hub, seed: int) -> list[Picked]:
    if src.sampler is not None:
        return src.sampler(hub, seed)
    if src.kind == "csv":
        return take(hub.csv_rows(src.dataset), 0, src.mapper, dict(src.quotas), random.Random(f"{src.name}:{seed}"))
    return sample_hf(src, hub, seed)


def _shortfall(src: Source, picked: Sequence[Picked]) -> dict[str, int]:
    got: dict[str, int] = {}
    for p in picked:
        got[p.mapped.stratum] = got.get(p.mapped.stratum, 0) + 1
    return {k: q - got.get(k, 0) for k, q in src.quotas.items() if got.get(k, 0) < q}


def _case(src: Source, p: Picked) -> dict[str, Any]:
    return {
        "id": f"{src.prefix}-{p.row:06d}",
        "state": p.mapped.state,
        "labels": p.mapped.labels,
        "meta": {"subset": "public", "lang": p.mapped.lang, "row": p.row, "stratum": p.mapped.stratum},
        "rationale": f"Gold from {src.dataset} ({src.split}, row {p.row}): {src.label_basis}",
    }


def task_document(src: Source, picked: Sequence[Picked], sha: str, seed: int) -> dict[str, Any]:
    ordered = sorted(picked, key=lambda p: (p.mapped.stratum, p.row))
    source = {
        "dataset": src.dataset, "config": src.config, "split": src.split, "revision": sha, "license": src.license,
        "attribution": src.attribution, "label_basis": src.label_basis, "redistributable": src.redistributable,
        "seed": seed, "quotas": src.quotas, "shortfall": _shortfall(src, picked),
    }  # fmt: skip
    return {
        "task": src.name,
        "description": src.description,
        "source": source,
        "fallback": src.fallback,
        "questions": src.questions,
        "cases": [_case(src, p) for p in ordered],
    }


def _revision(src: Source, hub: Hub) -> str:
    if src.kind == "csv":
        return src.dataset.split("/")[5] if src.dataset.count("/") > 5 else ""
    return hub.sha(src.dataset)


def licenses_markdown(docs: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Imported datasets",
        "",
        "Sampled cases in this directory come from these sources, under their own licenses.",
        "",
    ]
    for d in docs:
        s = d["source"]
        lines.append(
            f"- `{d['task']}`: {s['dataset']} ({s['split']}, revision {s['revision'][:12]}), "
            f"{s['license']}. {s['attribution']}"
        )  # noqa: E501
    return "\n".join(lines) + "\n"


def run_import(sources: Sequence[Source], hub: Hub, seed: int, out_dir: Path, log: Callable[[str], None] = print) -> list[dict[str, Any]]:  # fmt: skip  # noqa: E501
    """Write one task file per source plus LICENSES.md; returns the task documents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    for src in sources:
        doc = task_document(src, sample(src, hub, seed), _revision(src, hub), seed)
        (out_dir / f"{src.name}.json").write_text(dump_task(doc), encoding="utf-8")
        short = doc["source"]["shortfall"]
        log(f"{src.name}: {len(doc['cases'])} cases" + (f" (short of quota: {json.dumps(short)})" if short else ""))
        docs.append(doc)
    every = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*.json"))]
    (out_dir / "LICENSES.md").write_text(licenses_markdown([d for d in every if "source" in d]), encoding="utf-8")
    return docs
