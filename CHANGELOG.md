# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `department_routing`: 48 authored switchboard-routing cases (38 English, 10 Danish) over eight departments, with a priority rule for callers who ask for more than one thing, keyword traps and boundary cases.

## [0.1.0] - 2026-10-09

### Added

- First release, moved out of jev-eval (where it was the unreleased `jev-eval bench`). Jev and Claude classifiers; 15 authored tasks (386 cases) and 10 committed public tasks (1,261 cases), plus 5 download-only public sets via `import`. `tasks`, `estimate`, `run`, `import` and `analyze`; latency quantiles, reliability counts, registry-priced cost, match rate, calibration (ECE, MCE, Brier), self-consistency, clustered-bootstrap intervals, kappa, McNemar, a cascade table, a padding sweep up to 32k tokens and a concurrency sweep. Cost control with `--domain`, `--sample` and `--max-cases`. `--dry-run` runs the whole pipeline offline.
