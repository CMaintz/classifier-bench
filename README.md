# classifier-bench

A head-to-head benchmark for fast typed classifiers against LLMs. On the same requests, how fast, how cheap, how well calibrated and how close to the expected labels is a purpose-built classification API compared with an LLM such as Claude Haiku or Sonnet asked for schema-constrained output?

It ships its own frozen, labeled corpus (2,321 cases across 31 tasks, counting the optional downloads), measures latency tails, reliability, cost and calibration, and writes a report with confidence intervals. Zero runtime dependencies: the Python standard library only.

**Classifiers today:** TypeSafe [Jev](https://docs.typesafe.ai) and any Claude model. **Planned:** OpenAI Decisions and Cloudflare Clef, so all three typed-classification APIs can be compared with each other and with Claude in one harness.

## Quick start

```console
$ pip install git+https://github.com/CMaintz/classifier-bench
$ classifier-bench tasks                                        # the corpus: tasks, domains, cases
$ classifier-bench estimate -c jev -c haiku                     # registry-priced cost, no calls
$ classifier-bench run -c jev -c haiku --dry-run --out /tmp/x   # synthetic providers, no keys, no cost
$ classifier-bench run -c jev -c haiku --out runs/haiku-1       # needs JEV_API_KEY + ANTHROPIC_API_KEY
$ classifier-bench analyze runs/haiku-1                         # metrics.json + report.md (offline)
```

Classifiers: `jev` (or `jev:<model>` to pin a version), `haiku` (`claude-haiku-4-5`), `sonnet` (`claude-sonnet-5-5`), `opus`, or `claude:<model id>`. The first one listed is the baseline every other one is compared with. Claude also accepts `ANTHROPIC_AUTH_TOKEN` in place of `ANTHROPIC_API_KEY`.

## The corpus

Two corpora, never pooled: `--suite authored` (default), `public`, or `all`. The report gives each its own headline.

| Suite | Tasks | Cases | Decisions |
|---|---|---|---|
| Authored | 16 | 434 | 539 |
| Public, committed | 10 | 1,261 | 1,261 |
| Public, download-only | 5 | 626 | 626 |

- **Authored:** tier routing, department routing (a phone switchboard with a fixed department list and a priority rule), support triage (team, urgency and sentiment in one call), banking intent, toxicity, PII, tool-call risk, issue triage, duplicate issues, citation support, review sentiment (5 levels), language ID (Danish vs Norwegian vs Swedish), Danish support, code-review severity, prompt injection and spam. Short, long, follow-up, tool-context, boundary and non-English cases. The criteria are written as decision rules so every label follows from the rules, and a blind second annotator agreed on 383 of 386 cases; the contested ones are excluded from the "agreed ground truth only" match rate. See [`tasks/ANNOTATION.md`](src/classifier_bench/tasks/ANNOTATION.md).
- **Public, committed:** seeded, label-stratified samples of BANKING77, CLINC150 (bank scope), MASSIVE (Danish scenarios, and language ID where the gold is the locale), Civil Comments (clear rater consensus only), DKHate, SMS Spam, jailbreak-classification, PAWS and GoEmotions (Ekman groups). Each is asked with its own original annotation definition, so the published gold answers the question being asked. Every case records dataset, revision, split and row.
- **Public, download-only:** VitaminC, SST-5, NLBSE'24 issues, iSarcasmEval and deepset prompt-injections. Their licenses or terms don't allow redistribution here, so `classifier-bench import` fetches them through the Hugging Face rows API into `~/.cache/classifier-bench`; add them to a run with `--corpus-dir`.

Bring your own tasks the same way: any directory of task files in the same JSON shape works with `--corpus-dir`.

## Cost control

Narrow a run before it costs anything:

- `--suite`, `--tasks a,b`
- `--domain routing,safety` (routing, support, safety, grounding, sentiment, language, code)
- `--sample 0.25`: that fraction of every task, stratified by label and seeded, so the same subset comes back each time
- `--max-cases 20`: a per-task cap
- `--repeats` (default 3)

`run` prints the registry-priced estimate first and refuses when it exceeds `--max-cost` (default $5). The selection is recorded in `protocol.json`. Rough figures for one pass over all 2,321 cases: Jev $0.02, Haiku $2.30, Sonnet $4.65. Small samples give wide intervals: use them for quick checks and the full suite for numbers you quote.

## The protocol

- Every classifier gets the same questions (instructions and criteria). Claude gets them as JSON with a structured-output schema and states a probability for every option; Jev gets its native questions. No prompt is tuned per classifier.
- One stdlib HTTP stack for all, zero retries, an explicit deadline (`--timeout-ms`, default 10000), no prompt caching, wall-clock timing around the single HTTP call.
- Each repeat is a seeded shuffle, and the classifier that goes first rotates from case to case. Warmup pairs run first and are excluded.
- Haiku runs without thinking; Sonnet 5.5 with `thinking: between_tools` and effort `low` (override with `--effort`).
- The run directory freezes `protocol.json`, `cases.json` and `registry.json` before the first call, appends `attempts.jsonl` (failed attempts included), optionally `wire.jsonl` (`--wire`, bodies without auth headers), and seals everything with `SHA256SUMS`. A directory that already holds attempts is refused.

## What the report measures

Per classifier and head to head:

- **Match** with the expected labels (fallback answers included, and answered calls only), every-answer-right per call, macro F1, recall and latency per expected label, top mismatches, score within-1 and mean level distance.
- **Latency:** mean, stdev, min, p25, p50, p75, p90, p95, p99, max, all attempts and successful only, per task, subset, language and pair position, and throughput.
- **Reliability:** HTTP 200s, provider errors, 429s, 529s, timeouts, network errors, unparseable 200s, fallback decisions (a failed call takes the task's fallback answer, as a router would), and which model actually answered.
- **Spend:** input, output and cache tokens, registry-priced cost in total, per call, per 1,000 calls and per correct answer. Warmup cost is reported separately.
- **Calibration:** ECE, MCE, Brier score, mean confidence and the reliability table, overall and per question. LLM probabilities are verbalized and labeled as such.
- **Self-consistency:** how often the same case gets the same answer on every repeat.
- **Head to head:** p50, p95, p99 and mean speed ratios, cost savings, paired match-rate difference and agreement, each with a 95% interval from a clustered bootstrap (whole cases resampled), plus Cohen's kappa, an exact McNemar test and agreement per task.
- **Cascade:** run the baseline on everything and escalate to the other classifier when the baseline's least confident answer is below t; escalation rate, match, cost and latency for t from 0.5 to 0.99.
- **Scale:** `--pad 0,2000,8000,16000,28000` wraps cases in unrelated background text and reports match, latency and measured input tokens per bucket, up to Jev's 32k state limit.
- **Load:** `--concurrency 1,4,16` runs one sub-run per level; `analyze` on the parent writes `load.md` with p50, p95, p99, throughput, 429s and timeouts per level.

## What it does not show

The authored labels were checked by LLM-assisted annotators, not independent human experts, and the public test sets are old enough that LLMs have probably seen them in training. Match rate is agreement with these labels, not general accuracy. Timings depend on your region and network. The report repeats these caveats at the top.

**Disclosure:** the author also builds open-source tools on top of Jev. The corpus, prompts, schemas and scoring rules are all in this repo so anyone can check them, rerun them or add a classifier.

## Development

```console
$ mise install      # Python 3.12, ruff, and the pinned mypy/pytest/pip-audit in .venv
$ mise run gate     # ruff, mypy --strict, pytest (90% coverage floor), pip-audit
```

## License

Code and the authored corpus: [MIT](LICENSE). The committed public samples stay under their own licenses (CC0, CC-BY, Apache and similar), listed in [`tasks/public/LICENSES.md`](src/classifier_bench/tasks/public/LICENSES.md).
