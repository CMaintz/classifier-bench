# How the authored labels were checked

Every authored case is labeled against its task's written criteria, which are phrased as decision rules
(level definitions for Score questions, tie-break rules for overlapping Choice options).

1. One author wrote the cases and labels.
2. A first reviewer re-labeled every case and flagged vague criteria, keyword giveaways and imbalance; the
   criteria were rewritten as decision rules and 30 cases were relabeled, reworded or dropped.
3. After harder cases were added (386 in total), a second annotator labeled every case blind, seeing only
   the questions and the states (no labels, rationales or metadata).

Result of step 3: 383 of 386 cases agree (99.2%), 487 of 491 label decisions agree. One disagreement was
resolved in the second annotator's favor (ST-11 sentiment, the rule supports it). The other two cases stay
marked `meta.contested: true` with the second annotator's label in `meta.second_annotator`; the report
excludes them from the "agreed ground truth only" match rate.

All annotators were LLM-assisted, not independent human experts. Agreement shows the labels follow from
the written rules; it does not show the rules are the only reasonable ones.

`department_routing` (48 cases, added after the move to classifier-bench) went through the same blind
check: the second annotator saw only the instructions, criteria and states, and agreed on all 48. The
cases it found closest to the rules' edges were DR-30 (a refund after leaving: billing) and DR-36
(ending a contract because of a move: moving); both labels follow from the written criteria.
