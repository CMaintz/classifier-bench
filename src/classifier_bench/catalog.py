"""Every importable public source by name; BANKING77 needs its class names from the hub."""

from __future__ import annotations

from .hub import Hub
from .importer import Source
from .sources import CLINC, LANGUAGES, MASSIVE_DA, banking77
from .sources_more import CIVIL, DEEPSET, DKHATE, EMOTION, JAILBREAK, NLBSE, PAWS, SARCASM, SMS, SST5, VITAMINC

STATIC = (
    CLINC,
    MASSIVE_DA,
    LANGUAGES,
    CIVIL,
    DKHATE,
    SMS,
    DEEPSET,
    JAILBREAK,
    PAWS,
    EMOTION,
    VITAMINC,
    SST5,
    NLBSE,
    SARCASM,
)


def catalog(hub: Hub) -> dict[str, Source]:
    sources = [banking77(hub.label_names("legacy-datasets/banking77", "default", "label")), *STATIC]
    return {s.name: s for s in sources}


def select(hub: Hub, names: str) -> list[Source]:
    """'all', 'committable' (redistributable), 'runtime' (fetch-only), or a comma-separated list."""
    every = catalog(hub)
    if names in ("all", "committable", "runtime"):
        wanted = {"all": (True, False), "committable": (True,), "runtime": (False,)}[names]
        return [s for s in every.values() if s.redistributable in wanted]
    unknown = [n for n in names.split(",") if n not in every]
    if unknown:
        raise ValueError(f"unknown source(s): {', '.join(unknown)}; known: {', '.join(every)}")
    return [every[n] for n in names.split(",")]
