# mg_transfer

*Last updated: October 26th, 2026.*

## What this demonstrates

Transfer learning from ZINC250k onto the curated molecular glue set beats training on the glue set alone, with a
from-scratch control. This is the cheapest of the three programs and the right place to test the pipeline before
PROTAC jobs consume allocation.

## Prerequisites

`python scripts/download_data.py --source tpddb --source zinc`, then `preprocess_zinc` and `preprocess_glue`.

## Jobs

`preprocess_zinc` and `preprocess_glue` are independent. Then `prior_zinc`, then the six training arms, then
the paired generate jobs.

**Three seeds, and here a seed is a training run.** Both arms exist three times:
`transfer_arm{,_seed2,_seed3}` and `scratch_control{,_seed2,_seed3}`. Table 2 reports mean (standard
deviation) over seeds 1, 2 and 3, and the quantity that varies is the training run, not the sampling from a
finished checkpoint. Until 2026-10-26 each arm existed once and the three `generate_*_seedN` jobs all read
one checkpoint, so the reported interval was the spread of resampling a single run.

**The pairing is seed-matched and must stay that way.** `transfer_arm_seed2` declares
`control_job_dir: .../scratch_control_seed2`, and `generate_transfer_seed2` reads
`transfer_arm_seed2/checkpoints/`. `epochs_to_match` and the overlaid control curve are per-seed
quantities; crossing the seeds puts two sources of variance into one number.

**The prior is not re-seeded.** One `prior_zinc`, three fine-tunings. So the interval is the variance of the
transfer step, which is what the transfer claim is about, and not the variance of the whole pipeline. Say so
wherever the interval is quoted.

**Downstream experiments resume from seed 1.** `mg_conditional/jobs/train` and `mg_goal_directed/jobs/rl_gspt1*`
all read `transfer_arm/checkpoints/model_restart_20.pth` — the seed-1 arm, which is the designated
checkpoint for everything downstream. That is a deliberate choice, not an oversight: fanning the conditional
and goal-directed experiments across three transfer seeds would triple them.

Cost: six training jobs at 2 h each, so about 12 GPU-hours for this experiment.

## Submit

    python submit.py experiments/mg_transfer/jobs/preprocess_glue --slurm

## What to check

`jobs/transfer_arm/logs/convergence.log` against `jobs/scratch_control/logs/convergence.log`, column
`avg_valid_loss`, at epoch 1 and at the end — and the same for each seed against *its own* control. Then
`jobs/generate_transfer/results/metrics.json` for the funnel.

`epochs_to_match` is the transfer claim's own quantity: the epoch at which the transfer arm first reaches the
control's final validation loss. A null there with reason `undefined_for_this_run` means **transfer did not
beat from-scratch training on that seed**. That is a result and goes in the table as one; it is not a missing
measurement. `python scripts/aggregate-seeds.py experiments/mg_transfer` writes the across-seed aggregate,
and a group with one run reports no interval rather than a zero-width one.
`fcd` needs 1,000 valid unique molecules on both sides; the glue test split is roughly 600 molecules, so expect
`fcd` in `unavailable` with category `insufficient_sample`.

## Core metrics come from

`generate_transfer`, with `generate_scratch` as the control.

## Failure mode

Equal epoch-1 validation loss between transfer and scratch means the checkpoint is not loading. Validity above
1.0 means L3 is unfixed; stop.
