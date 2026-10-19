# mg_transfer

*Last updated: October 18th, 2026.*

## What this demonstrates

Transfer learning from ZINC250k onto the curated molecular glue set beats training on the glue set alone, with a
from-scratch control. This is the cheapest of the three programs and the right place to test the pipeline before
PROTAC jobs consume allocation.

## Prerequisites

`python scripts/download_data.py --source tpddb --source zinc`, then `preprocess_zinc` and `preprocess_glue`.

## Jobs

`preprocess_zinc` and `preprocess_glue` are independent. Then `prior_zinc`, then `prior_transfer` and
`prior_scratch`, then the paired generate jobs.

## Submit

    python submit.py experiments/mg_transfer/jobs/preprocess_glue --slurm

## What to check

`jobs/prior_transfer/logs/convergence.log` against `jobs/prior_scratch/logs/convergence.log`, column
`avg_valid_loss`, at epoch 1 and at the end. Then `jobs/generate_transfer/results/metrics.json` for the funnel.
`fcd` needs 1,000 valid unique molecules on both sides; the glue test split is roughly 600 molecules, so expect
`fcd` in `unavailable` with category `insufficient_sample`.

## Core metrics come from

`generate_transfer`, with `generate_scratch` as the control.

## Failure mode

Equal epoch-1 validation loss between transfer and scratch means the checkpoint is not loading. Validity above
1.0 means L3 is unfixed; stop.
