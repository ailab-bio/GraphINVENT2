# mg_conditional

*Last updated: October 18th, 2026.*

## What this demonstrates

Conditioning a glue generator on which neosubstrate is degraded, with a hard pair and a positive control built
in on purpose.

## The panel, which measures out as specified

Unique flat canonical molecules per class on the curated set: **IKZF1/3 merged 1,493; GSPT1 2,358; RBM39 526.**
All three clear the 300-molecule minimum, so the panel is the one the brief specified. IKZF1/3 and GSPT1 are both
cereblon neosubstrates reached by glutarimide-containing molecules and are the hard pair that tests whether
conditioning does anything. RBM39 is recruited through DCAF15 by aryl sulfonamides, a chemically distinct series,
and is the positive control: a model that cannot separate RBM39 from the other two has a broken mechanism rather
than a hard task. Report the two contrasts separately.

## Prerequisites

`mg_unconditional/jobs/prior_transfer`, and the curated set with the three one-hot columns.

## Submit

    python submit.py experiments/mg_conditional/jobs/preprocess_conditional --slurm

## What to check

Each `jobs/generate_*/results/metrics.json`. The RBM39 contrast should be clean; the IKZF1/3 against GSPT1
contrast is the real test. The seven categorical metrics are implemented and have **no caller** (L25), so they do
not appear yet, and the retrieval baseline does not exist. Both block the headline result.

## Core metrics come from

`generate_ikzf`, `generate_gspt1`, `generate_rbm39`, with `generate_unconditional` as the baseline.

## Failure mode

RBM39 failing to separate means the conditioning mechanism is not working at all. All three separating equally
well suggests the model keyed on something other than the label; check `condition_vector` is non-constant.
