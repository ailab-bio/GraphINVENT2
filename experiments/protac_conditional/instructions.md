# protac_conditional

*Last updated: October 18th, 2026.*

## What this demonstrates

Conditioning a PROTAC generator on which protein is degraded produces molecules whose warhead chemistry matches
the requested target, beyond what a nearest-neighbor retrieval lookup achieves.

## The panel, and why it is not the one the brief specified

The brief specified AR, BRD4 and SMARCA2 with at least 300 molecules per class after curation, substituting BTK,
then EGFR, then ER if SMARCA2 fell short. Measured on the curated set, unique flat canonical molecules per target:

| target | molecules | |
|---|---|---|
| AR | 3,579 | ok |
| BRD4 | 140 | **below 300** |
| SMARCA2 | 267 | **below 300** |
| BTK | 988 | ok |
| EGFR | 1,295 | ok |
| ER | 5 | below 300 |

Both BRD4 and SMARCA2 fall short, and the substitution rule covers only SMARCA2. Applying it in the stated order
gives **AR, BTK, EGFR**. The design intent survives: AR is a nuclear receptor with enzalutamide-like warheads and
is the easy class, while BTK and EGFR are both kinases with distinct chemotypes and are the hard pair a model
cannot pass by memorizing warhead families. Report the two contrasts separately. **This is a scope change and it
is question 3 in the report.**

## Prerequisites

- `preprocess_conditional`, which needs the curated set with the three one-hot columns written by the dataset
  build script.
- `experiments/protac_transfer/jobs/transfer_arm` has finished: `train` resumes from its checkpoint.

## Jobs

`preprocess_conditional` then `train`, then the generate jobs, which may run in any order.

`generate_panel` (plus `_seed2`, `_seed3`) is the job that produces this experiment's reported numbers.
It samples with `sample_conditions: all_classes`, which cycles the one-hot across batches, and so is the
only job that can produce the macro-averaged categorical metrics CLAUDE.md defines. The per-class
`generate_<class>` jobs remain for their `_<class>` breakdown; `categorical_block` refuses to emit a bare
macro-average name from a job that sampled one class, because a one-class value under the panel's name is
not the panel's number.

`generate_unconditional` samples the same model with every class set to zero. **It cannot emit a
categorical metric itself**, and it is not the baseline those metrics are reported against: with an
all-zeros condition there is no requested label to group molecules by, so every categorical key in that
job is a categorized null. Its role is to supply the sample that the per-class jobs' `baseline_posthoc_`
baseline reranks, which is why `generate_panel` points `baseline_results_dir` at its `molecules.smi`. The
two baselines that make the claim falsifiable are `baseline_lookup_` (the retrieval ceiling, drawn from
the training split and needing no configuration) and `baseline_posthoc_`.

## Submit

    python submit.py experiments/protac_conditional/jobs/preprocess_conditional --slurm

## What to check

- `jobs/train/logs/convergence.log`: `avg_valid_loss` decreasing.
- Each `jobs/generate_*/results/metrics.json`: `class_selectivity_margin` first, then `novelty` beside
  `class_match_*`. The continuous `condition_success_rate_*` keys are **not** emitted here and should not be:
  this panel is categorical, declared with `condition_classes`, and a success rate within a tolerance of 0.5 of
  a requested 1.0 measures whether the model echoed its own input.
- **The headline is the gap, not the conditional number.** Compare `class_selectivity_margin` against
  `baseline_posthoc_class_selectivity_margin`: if the unconditional model reranked per class reaches the same
  margin, the conditioning bought nothing and that is the finding. `baseline_lookup_*` is the retrieval ceiling
  and scores `class_match_median` 1.0 at `novelty` 0.0 by construction.
- Every metric here is the macro-average over the three targets, with the per-class value under the same name
  plus a `_is_ar` / `_is_btk` / `_is_egfr` suffix. The full table is `results/condition_per_class.json`.
- This dataset is bifunctional, so the per-class table is computed over warheads and adds
  `warhead_novelty_in_class`. Read it beside the global `warhead_novelty`: the two are expected to disagree, and
  the in-class one is what separates conditioning from retrieval.
- `delta_nll` does not depend on the generated sample and is present even for a run that generated nothing
  valid. Positive means a held-out molecule is more likely under its own condition.

## Core metrics come from

`generate_panel` and its two seed siblings, for every macro-averaged key. `generate_ar`, `generate_btk` and `generate_egfr` for the per-class breakdown only. The baselines are `baseline_lookup_` and `baseline_posthoc_`, the second built from `generate_unconditional`'s sample.

## Failure mode

If all three classes score near-identically, check that `condition_vector` is non-constant across rows in the
HDF5. A dataset built without the one-hot columns yields `condition_dim: 0` and the job is rejected, which is the
loud failure; a dataset built with constant columns trains silently and conditions on nothing.
