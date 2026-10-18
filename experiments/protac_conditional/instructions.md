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
- `experiments/protac_unconditional/jobs/prior_transfer` has finished: `train` resumes from its checkpoint.

## Jobs

`preprocess_conditional` then `train`, then the four generate jobs, which may run in any order.
`generate_unconditional` samples the same model with no condition set and is the baseline every categorical
metric is reported against.

## Submit

    python submit.py experiments/protac_conditional/jobs/preprocess_conditional --slurm

## What to check

- `jobs/train/logs/convergence.log`: `avg_valid_loss` decreasing.
- Each `jobs/generate_*/results/metrics.json`: `condition_success_rate_is_ar` and the sibling keys.
- **The categorical metrics are NOT yet emitted.** `delta_nll`, `knn_purity_5/10/20`, `class_match_median`,
  `class_match_rate` and `class_selectivity_margin` are implemented in `src/metrics/_conditions.py` and have no
  caller, so they never appear. That is ledger item L25 and it blocks the headline result of this experiment.
- The retrieval baseline does not exist yet either. Without it this result is not falsifiable.

## Core metrics come from

`generate_ar`, `generate_btk`, `generate_egfr`, with `generate_unconditional` as the baseline.

## Failure mode

If all three classes score near-identically, check that `condition_vector` is non-constant across rows in the
HDF5. A dataset built without the one-hot columns yields `condition_dim: 0` and the job is rejected, which is the
loud failure; a dataset built with constant columns trains silently and conditions on nothing.
