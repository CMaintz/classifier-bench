#!/usr/bin/env python3
"""Does a change to a ratchet baseline LOOSEN it? Exit 1 = loosening (needs the
ruleset-change label), 0 = safe.

Per-entry, not aggregate: any entry ADDED or INCREASED is a loosening; removals
are fine. This is what defeats the offset attack — fixing one violation while
suppressing another nets zero on a summed count, but shows up here as an added
entry.

Canonical copy. Onboarded repos commit this at scripts/ruleset_guard.py so both
the inline gate and foundry's reusable tier0.yml (which runs in the caller's
checkout) can call it.

Kinds:
  eslint  eslint-suppressions.json      (per file+rule count)
  snooze  habit-hooks snooze.json, and dependency-cruiser known-violations JSON
          (position-independent multiset of scalar leaves — an added violation
          increments its scalars, so it's caught; reorder/remove are safe)
  lines   ArchUnit FreezingArchRule store files (one frozen violation per line;
          run once per changed store file under archunit_store/)
  coverage  prompt-eval manifest.json — INVERSE: a REMOVED fixture is less coverage,
            so removal (not addition) is the loosening

Usage: ruleset_guard.py <eslint|snooze|lines|coverage> <base-ref> <head-ref> <path>
"""

import json
import subprocess
import sys
from collections import Counter


def eslint_counts(d):
    """eslint-suppressions.json: {file: {rule: {count: N}}} -> Counter[(file,rule)] = N."""
    c = Counter()
    for f, rules in (d or {}).items():
        for rule, v in (rules or {}).items():
            c[(f, rule)] = (v or {}).get("count", 0)
    return c


def value_counts(x):
    """snooze.json: multiset of scalar leaf values, position-independent (reorder is safe)."""
    c = Counter()

    def walk(v):
        if isinstance(v, dict):
            for w in v.values():
                walk(w)
        elif isinstance(v, list):
            for w in v:
                walk(w)
        else:
            c[v] += 1

    walk(x)
    return c


def line_counts(text):
    """ArchUnit freeze store: one violation per line -> multiset of non-empty lines."""
    c = Counter()
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln:
            c[ln] += 1
    return c


def coverage_counts(text):
    """prompt-eval manifest.json: multiset of the `fixtures` list (the coverage surface)."""
    d = json.loads(text) if text.strip() else {}
    c = Counter()
    for item in d.get("fixtures") or []:
        c[item] += 1
    return c


def loosened(kind, old_text, new_text):
    if kind == "lines":
        o, n = line_counts(old_text), line_counts(new_text)
        return [k for k in n if n[k] > o.get(k, 0)]
    if kind == "coverage":
        # INVERSE of a debt baseline: for a test-coverage surface, a REMOVED entry is
        # *less* coverage = a loosening. Adding entries is fine (tightening).
        o, n = coverage_counts(old_text), coverage_counts(new_text)
        return [k for k in o if n.get(k, 0) < o[k]]
    old = json.loads(old_text) if old_text.strip() else None
    new = json.loads(new_text) if new_text.strip() else None
    counts = eslint_counts if kind == "eslint" else value_counts
    o, n = counts(old), counts(new)
    return [k for k in n if n[k] > o.get(k, 0)]


def _git_show(ref, path):
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


if __name__ == "__main__":
    kind, base, head, path = sys.argv[1:5]
    bad = loosened(kind, _git_show(base, path), _git_show(head, path))
    for k in bad[:20]:
        print(f"  added/increased: {k}", file=sys.stderr)
    sys.exit(1 if bad else 0)
