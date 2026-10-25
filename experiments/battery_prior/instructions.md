# Battery solvent prior

*Last updated: October 25th, 2026.*

## What this experiment claims

That a generative prior trained on Batt-SLM's 115,756 molecules learns
small-molecule solvent chemistry well enough to be a useful starting point. It
claims nothing about battery solvent design on its own: it is the starting
checkpoint for `battery_conditional` and `battery_goal_directed`, and the claims
that matter are made there.

This corpus is what the source work trains its own prior on. There is no
separate PubChem base corpus and no transfer-learning stage in that work: the
1.2 million PubChem molecules it mentions are the raw pull, and Batt-SLM's
115,756 are what survives its filtering and resampling. "PubChem-120K" appears
in the paper under no name, so the `battery_transfer` experiment that used to
sit beside this one was reconstructing a stage that never existed, and it has
been removed.

## Order

1. `jobs/preprocess_battslm`: build the HDF5 and the splits. Nothing else can run
   until this finishes.
2. `jobs/prior_battslm`: train the prior. The long pole of this experiment.
3. `jobs/generate_prior`, `jobs/generate_prior_seed2`, `jobs/generate_prior_seed3`.
   10,000 molecules each from the final checkpoint.

Steps 1 and 2 are strictly sequential. The three sampling jobs are independent
of each other and can run concurrently once step 2 has written its checkpoint.

## Core metrics come from

`generate_prior` and its two seed siblings, aggregated with
`scripts/aggregate-seeds.py`. Nothing is reported from `prior_battslm` itself: a
training job has no sample of its own, and the handful of keys it writes are
model efficiency and the likelihood numbers, not the core table.

Read `fcd` first. V.U.N. saturates on a prior trained on a large corpus and no
longer separates models, while distribution match does.

## What it looks like when it has gone wrong

- `jobs/prior_battslm/logs/convergence.log` with `avg_valid_loss` flat from the first
  epoch: the learning rate or the vocabulary is wrong, not the data.
- `validity` below about 0.5 after a full run on a large corpus. The prior is
  the easy case; if it cannot produce valid molecules, nothing downstream will.
- `novelty` near 1.0 with `near_duplicate_rate` also high: the model is
  memorizing and the novelty number is an artifact of canonicalization rather
  than a result. Check that `use_chirality` matches between the dataset and the
  reference.
- `fcd` unavailable with `insufficient_sample`: the test split is too small for
  the 1,000-molecule floor. That is a property of the dataset, not a bug.
