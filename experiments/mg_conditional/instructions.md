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

`mg_transfer/jobs/prior_transfer`, and the curated set with the three one-hot columns.

## Submit

    python submit.py experiments/mg_conditional/jobs/preprocess_conditional --slurm

## What to check

Each `jobs/generate_*/results/metrics.json`. The RBM39 contrast should be clean; the IKZF1/3 against GSPT1
contrast is the real test.

Read `class_selectivity_margin` first, then `novelty` beside `class_match_*`, then the two baselines. The
headline is not the conditional number: it is the gap between `class_selectivity_margin` and
`baseline_posthoc_class_selectivity_margin`. If an unconditional model plus a one-line reranking reaches the
same margin, the conditioning bought nothing and that is the result. `baseline_lookup_*` scores
`class_match_median` 1.0 at `novelty` 0.0 by construction; a conditional model matching it on both is
retrieving, not conditioning. The per-class table is in `results/condition_per_class.json` and the bars are in
`results/conditioning.png`.

`delta_nll` is the one number that does not depend on the generated sample, so it is present even for a run that
generated nothing valid. Positive means a held-out molecule is more likely under its own condition.

## Core metrics come from

`generate_ikzf`, `generate_gspt1`, `generate_rbm39`, with `generate_unconditional` as the baseline.

## Failure mode

RBM39 failing to separate means the conditioning mechanism is not working at all. All three separating equally
well suggests the model keyed on something other than the label; check `condition_vector` is non-constant.
