"""Frozen price registry: cost = observed tokens x captured list price (never provider invoices)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).parent / "registry.json"
PER_MILLION = 1_000_000


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(Path(path or REGISTRY_PATH).read_text(encoding="utf-8"))
    if not isinstance(doc.get("models"), dict):
        raise ValueError("registry needs a 'models' map")
    return doc


def resolve(registry: dict[str, Any], model: str) -> str:
    """Strip a date suffix and follow aliases, e.g. jev-latest -> jev-1.13.0."""
    aliases = registry.get("aliases") or {}
    name = aliases.get(model, model)
    if name not in registry["models"] and name[-9:-8] == "-" and name[-8:].isdigit():
        name = name[:-9]
    return str(name)


def price(registry: dict[str, Any], model: str) -> Price:
    entry = registry["models"].get(resolve(registry, model))
    if entry is None:
        raise ValueError(f"no registry price for model {model!r}; add it to the registry file")
    return Price(**{k: float(v) for k, v in entry.items()})


def cost(p: Price, usage: dict[str, Any]) -> float:
    """USD for one call's usage (Anthropic counts cached tokens outside input_tokens)."""
    tokens = (
        (usage.get("input_tokens") or 0) * p.input
        + (usage.get("output_tokens") or 0) * p.output
        + (usage.get("cache_read_input_tokens") or 0) * p.cache_read
        + (usage.get("cache_creation_input_tokens") or 0) * p.cache_write
    )
    return float(tokens) / PER_MILLION
