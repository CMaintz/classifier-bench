#!/usr/bin/env python3
"""Does a change to a ratchet baseline LOOSEN it? Exit 1 = loosening (needs the
ruleset-change label), 0 = safe.

Per-entry, not aggregate: any entry ADDED or INCREASED is a loosening; removals
are fine. This is what defeats the offset attack - fixing one violation while
suppressing another nets zero on a summed count, but shows up here as an added
entry.

Canonical copy. Onboarded repos commit this at scripts/ruleset_guard.py so both
the inline gate and foundry's reusable tier0.yml (which runs in the caller's
checkout) can call it.

Kinds:
  eslint  eslint-suppressions.json      (per file+rule count)
  snooze  habit-hooks snooze.json, and dependency-cruiser known-violations JSON
          (position-independent multiset of scalar leaves - an added violation
          increments its scalars, so it's caught; reorder/remove are safe)
  lines   ArchUnit FreezingArchRule store files (one frozen violation per line;
          run once per changed store file under archunit_store/)
  coverage  prompt-eval manifest.json - INVERSE: a REMOVED fixture is less coverage,
            so removal (not addition) is the loosening
  inline  IN-SOURCE suppression directives, test-skips and config demotions, counted
          tree-wide per (file, directive) at both refs - an added eslint-disable /
          @ts-ignore / @SuppressWarnings / # noqa / #pragma warning disable /
          coverage-exclusion / it.skip / editorconfig severity=none is a loosening the
          baseline files above never see. No <path>: the base tree IS the baseline. Same
          per-entry shrink-only property as eslint, so the offset attack is still caught.
  tests   TEST DELETION: test-definition count PER FILE, INVERSE - a file with fewer
          tests at head than base lost coverage (deleted or moved out). No <path>.

Usage: ruleset_guard.py <eslint|snooze|lines|coverage> <base-ref> <head-ref> <path>
       ruleset_guard.py <inline|tests> <base-ref> <head-ref>
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
    """ArchUnit freeze store: one violation per line -> multiset of non-empty lines.

    Comment lines (starting with '#') are skipped: ArchUnit's default store index
    (`stored.rules`) is a Java Properties file, which writes a '#<timestamp>' comment
    that changes on every store update. Counting it would flag an added line on each
    legitimate prune. Real violation descriptions never start with '#'.
    """
    c = Counter()
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
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


# Never scan this script or its fixtures - they contain every directive by construction.
_EXCLUDE = [":(exclude)scripts/ruleset_guard.py", ":(exclude)tests/test_ruleset_guard.py"]
# Real source files (git pathspec globs match at any depth).
CODE_PATHS = [
    "*.ts",
    "*.tsx",
    "*.js",
    "*.jsx",
    "*.mjs",
    "*.cjs",
    "*.py",
    "*.cs",
    "*.java",
    "*.kt",
    "*.kts",
    "*.php",
    *_EXCLUDE,
]
# Gate-defining config files. A config-only PR that demotes a rule (severity = none,
# "strict": false, <NoWarn>) never touches source_paths, so the bundled ruleset-guard
# below never fires on it - this always-run scan is the only thing that catches it.
CONFIG_PATHS = ["*.editorconfig", "*.csproj", "*.props", "tsconfig*.json", *_EXCLUDE]
# Test files only. Test-SKIP directives (and the `tests` deletion count) are meaningful
# ONLY here - scanning all source for `.skip(`/`xit(` false-matches app code (a paging
# `.skip(`, `process.exit(` -> `xit(`), so those directives use this, not CODE_PATHS.
TEST_PATHS = [
    "*.test.ts",
    "*.test.tsx",
    "*.test.js",
    "*.test.jsx",
    "*.test.mjs",
    "*.test.cjs",
    "*.spec.ts",
    "*.spec.tsx",
    "*.spec.js",
    "*.spec.jsx",
    "*.spec.mjs",
    "*.spec.cjs",
    "test_*.py",
    "*_test.py",
    "*Test.java",
    "*Tests.java",
    "*IT.java",
    "*Test.kt",
    "*Tests.kt",
    "*Test.cs",
    "*Tests.cs",
    *_EXCLUDE,
]

# In-source suppressions / test-skips / config demotions, each (name, regex, pathspec).
# The regexes are BUILT FROM FRAGMENTS so this file never contains a directive literally
# (else the scan would flag its own source). Extended-regex (git grep -E); one hit per
# line, so the count is per-(file, directive) line-hits - stable at both refs, which is
# all the shrink-only comparison needs. Only LOOSENING values are listed, so a demotion
# is an added entry (flagged) and a re-promotion is a removal (safe) for free.
INLINE_DIRECTIVES = [
    ("eslint-disable", "eslint-" + "disable", CODE_PATHS),
    ("ts-ignore", "@ts-" + "ignore", CODE_PATHS),
    ("ts-expect-error", "@ts-" + "expect-error", CODE_PATHS),
    ("istanbul-ignore", "istanbul " + "ignore", CODE_PATHS),
    ("c8-ignore", "c8 " + "ignore", CODE_PATHS),
    ("noqa", "# *no" + "qa", CODE_PATHS),
    ("py-type-ignore", "# *type: *" + "ignore", CODE_PATHS),
    ("pylint-disable", "pylint: *" + "disable", CODE_PATHS),
    ("py-no-cover", "pragma: *no " + "cover", CODE_PATHS),
    ("java-suppresswarnings", "@Suppress" + "Warnings", CODE_PATHS),
    ("kotlin-suppress", "@Suppress" + r"\(", CODE_PATHS),
    ("csharp-pragma-warning", "pragma warning " + "disable", CODE_PATHS),
    ("csharp-suppressmessage", r"\[" + "SuppressMessage", CODE_PATHS),
    ("csharp-exclude-coverage", "ExcludeFrom" + "CodeCoverage", CODE_PATHS),
    ("php-phpstan-ignore", "@phpstan-" + "ignore", CODE_PATHS),
    ("php-psalm-suppress", "@psalm-" + "suppress", CODE_PATHS),
    ("phpcs-ignore", "phpcs: *" + "ignore", CODE_PATHS),
    # Test-skips: a skipped/narrowed test is a quietly disabled check. Scoped to TEST_PATHS
    # (a skip is only a skip in a test file), and js-xit is anchored with `(^|[^A-Za-z])` so
    # it does not match `exit(` / `fixit(`. Deleting a test outright is NOT caught here - the
    # `tests` kind and a coverage floor are the backstop.
    ("js-skip", r"\.skip" + r"\(", TEST_PATHS),
    ("js-only", r"\.only" + r"\(", TEST_PATHS),
    ("js-xit", "(^|[^A-Za-z])x(it|describe) *" + r"\(", TEST_PATHS),
    ("py-skip", "@(pytest.mark|unittest).*" + "skip", TEST_PATHS),
    ("junit-disabled", "@(Disabled|" + "Ignore)", TEST_PATHS),
    ("xunit-skip", r"\(Skip *=", TEST_PATHS),
    # Config demotions: flipping a rule off in a config file is the same gaming move as an
    # inline disable, and a config-only PR dodges the bundled guard.
    ("editorconfig-demote", "severity *= *(none|silent|" + "suggestion)", CONFIG_PATHS),
    ("tsconfig-unstrict", '"strict[A-Za-z]*" *: *' + "false", CONFIG_PATHS),
    ("csproj-nowarn", "<No" + "Warn>", CONFIG_PATHS),
]


def inline_counts(ref):
    """Counter[(file, directive)] of in-source suppression/skip/demotion hits at a ref."""
    c = Counter()
    for name, pat, paths in INLINE_DIRECTIVES:
        r = subprocess.run(
            ["git", "grep", "-I", "-E", "-e", pat, ref, "--", *paths],
            capture_output=True,
            text=True,
        )
        # git grep: 0 = matches, 1 = no matches (both fine), >1 = real error.
        if r.returncode > 1:
            raise SystemExit(f"ruleset_guard inline: git grep failed for {name}: {r.stderr.strip()}")
        for line in r.stdout.splitlines():
            # "<ref>:<path>:<text>" - a ref holds no colon, so field 1 is the path.
            parts = line.split(":", 2)
            if len(parts) >= 2:
                c[(parts[1], name)] += 1
    return c


def inline_loosened(base, head):
    o, n = inline_counts(base), inline_counts(head)
    return [k for k in n if n[k] > o.get(k, 0)]


# Test-definition markers, counted PER FILE over TEST_PATHS (defined above). Deleting or
# moving tests out of a file drops its count - the INVERSE of the suppression check (a
# removal is the loosening), like the coverage kind. Built from fragments for the same
# self-match reason as INLINE_DIRECTIVES. `(^|[^A-Za-z])` is a poor-man's word boundary
# (git grep -E has no portable \b) so `it`/`test` do not match inside `unit(`/`fixit(`;
# `@Test([^A-Za-z]|$)` excludes `@TestConfiguration`.
TEST_DEFS = [
    "(^|[^A-Za-z])(it|test)[.(]",
    "def " + "test",
    "@" + "Test([^A-Za-z]|$)",
    r"\[" + "Fact",
    r"\[" + "Theory",
]


def test_counts(ref):
    """Counter[file] = number of test-definition lines in that test file, at a git ref."""
    c = Counter()
    args = []
    for p in TEST_DEFS:
        args += ["-e", p]
    r = subprocess.run(
        ["git", "grep", "-I", "-c", "-E", *args, ref, "--", *TEST_PATHS],
        capture_output=True,
        text=True,
    )
    if r.returncode > 1:
        raise SystemExit(f"ruleset_guard tests: git grep failed: {r.stderr.strip()}")
    for line in r.stdout.splitlines():
        # git grep -c prints "<ref>:<path>:<count>".
        head, _, cnt = line.rpartition(":")
        path = head.split(":", 1)[1] if ":" in head else head
        c[path] = int(cnt)
    return c


def tests_loosened(base, head):
    """Files that LOST test definitions (fewer at head than base) - deletion/move-out."""
    o, n = test_counts(base), test_counts(head)
    return [f for f in o if n.get(f, 0) < o[f]]


_KINDS = {"eslint", "snooze", "lines", "coverage", "inline", "inline-count", "tests", "tests-count"}

if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else ""
    if kind not in _KINDS:
        sys.exit(
            f"ruleset_guard: unknown kind {kind!r}. Overwrite your committed "
            f"scripts/ruleset_guard.py with foundry's current copy - this workflow "
            f"needs a newer guard (foundry-init skips files that already exist)."
        )
    if kind in ("inline-count", "tests-count"):
        # Tree-wide total at one ref, for the ratchet report.
        counts = inline_counts if kind == "inline-count" else test_counts
        print(sum(counts(sys.argv[2]).values()))
        sys.exit(0)
    if kind in ("inline", "tests"):
        base, head = sys.argv[2:4]
        bad = inline_loosened(base, head) if kind == "inline" else tests_loosened(base, head)
    else:
        base, head, path = sys.argv[2:5]
        bad = loosened(kind, _git_show(base, path), _git_show(head, path))
    verb = "removed/decreased" if kind in ("coverage", "tests") else "added/increased"
    for k in bad[:20]:
        print(f"  {verb}: {k}", file=sys.stderr)
    sys.exit(1 if bad else 0)
