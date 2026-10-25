# mg_goal_directed

*Last updated: October 18th, 2026.*

## What this demonstrates

Optimization toward a chemotype-selectivity objective for GSPT1 degradation. **This is a weaker claim than an
activity surrogate and the text must say so.** pDC50 is unavailable: the glue activity data has 100 numeric DC50
rows in total across all targets and RBM39 has zero, so a potency regression is not trainable. The objective here
is a one-versus-rest classifier for "degrades GSPT1 rather than another neosubstrate", which is closer to
chemotype selectivity than to potency.

**Whether to run this at all is question 1 in the report.** The alternative is to drop it and run two job classes
for glues. This directory drafts the default so it is submittable either way.

## GATE: do not submit until L1, L2 and L3 are closed

See `protac_goal_directed/instructions.md` for the checks.

## Prerequisites

`mg_transfer/jobs/transfer_arm`, and `experiments/surrogate_protac_optimization/mg_gspt1_classifier`.

## Submit

    python submit.py experiments/mg_goal_directed/jobs/rl_gspt1 --slurm

## What to check

`results/metrics.json`: `quality_filtered_top10_mean`, gated on SA score because glues are drug-like.
`top100_scaffold_count` works here, unlike for PROTACs. `vs_delta_top10_at_10k` against the screen.

## Core metrics come from

`generate_best_agent`, a fresh sample from the best agent.

## Failure mode

The classifier probability saturating at 1.0 for a single chemotype is reward hacking against a classifier, which
is easier to hack than a regressor; `top100_scaffold_count` collapsing is the signal.
