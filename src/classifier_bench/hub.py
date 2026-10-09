"""Stdlib access to public datasets: the Hugging Face datasets-server rows API and pinned raw CSVs."""

from __future__ import annotations

import csv
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

ROWS_URL = "https://datasets-server.huggingface.co/rows"
INFO_URL = "https://datasets-server.huggingface.co/info"
HUB_API = "https://huggingface.co/api/datasets/"
PAGE = 100  # the rows API caps `length` at 100
_RETRYABLE = frozenset({429, 500, 502, 503, 504})

Fetch = Callable[[str], bytes]


PACE_S = 0.5  # polite spacing between requests; the rows API rate-limits bursts


def _wait(err: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = err.headers.get("Retry-After") if err.headers else None
    return float(retry_after) if retry_after and retry_after.isdigit() else min(2.0**attempt, 60.0)


def http_get(url: str, attempts: int = 8) -> bytes:
    """GET with pacing and backoff on rate limits and transient server errors (setup traffic, never measured)."""
    for attempt in range(1, attempts + 1):
        time.sleep(PACE_S)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https hosts
                body: bytes = resp.read()
                return body
        except urllib.error.HTTPError as err:
            if err.code not in _RETRYABLE or attempt == attempts:
                raise
            time.sleep(_wait(err, attempt))
    raise RuntimeError("unreachable")


class Hub:
    """Read-only dataset access through an injectable `fetch` (tests pass a fake)."""

    def __init__(self, fetch: Fetch = http_get) -> None:
        self._fetch = fetch
        self._pages: dict[str, tuple[list[dict[str, Any]], int]] = {}

    def _json(self, url: str) -> Any:
        return json.loads(self._fetch(url))

    def sha(self, dataset: str) -> str:
        return str(self._json(HUB_API + dataset).get("sha", ""))

    def page(self, dataset: str, config: str, split: str, offset: int) -> tuple[list[dict[str, Any]], int]:
        """One page of rows (each row as a dict) and the split's total row count; cached per run."""
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": PAGE}
        )
        url = f"{ROWS_URL}?{query}"
        if url not in self._pages:
            doc = self._json(url)
            self._pages[url] = ([r["row"] for r in doc["rows"]], int(doc["num_rows_total"]))
        return self._pages[url]

    def label_names(self, dataset: str, config: str, feature: str) -> list[str]:
        """Class names of a ClassLabel (or a sequence of ClassLabel) feature, from the /info endpoint."""
        info = self._json(f"{INFO_URL}?{urllib.parse.urlencode({'dataset': dataset})}")["dataset_info"][config]
        spec = info["features"][feature]
        return list(spec.get("names") or spec["feature"]["names"])

    def csv_rows(self, url: str) -> list[dict[str, str]]:
        text = self._fetch(url).decode("utf-8", errors="replace")
        return list(csv.DictReader(io.StringIO(text)))
