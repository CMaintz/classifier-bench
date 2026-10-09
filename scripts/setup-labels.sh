#!/usr/bin/env bash
# foundry — create the GitHub labels Foundry's workflows and skills rely on.
# Idempotent (gh label create --force). Run once per consumer repo; needs `gh`
# authenticated against a GitHub remote. foundry-init runs this automatically.
#
# GitHub silently drops a label an issue template / workflow references but the
# repo doesn't have — so the ticket state machine and ruleset-guard label-gate
# no-op with no error until these exist.
set -euo pipefail

label() { gh label create "$1" --color "$2" --description "$3" --force >/dev/null && echo "  label: $1"; }

# Ticket state machine — the /feature driver and repo-align (see presets/ticket-schema.md)
label "agent:ready"    0e8a16 "Groomed, complete, free to claim"
label "agent:working"  fbca04 "Claimed and assigned (WIP=1)"
label "agent:blocked"  d93f0b "Escalated to a human; reason in the issue thread"
label "align"          1d76db "repo-align target ticket"

# Gate / CI controls
label "ruleset-change" 5319e7 "Deliberate ruleset/threshold/baseline loosening (ruleset-guard gate)"
label "autofix"        c2e0c6 "Trigger autofix.yml — runs mise run fix and commits the result"

echo "Done — labels are idempotent, safe to re-run."
