# protac_transfer

*Last updated: October 26th, 2026.*

## What this demonstrates

Transfer learning from a large enumerated recombinant PROTAC library onto the small curated real set beats
training on the real set alone, with a from-scratch model as the control that makes it a result.

## Prerequisites

- `python scripts/download_data.py --source tpddb` has run and `data/raw/tpddb/MANIFEST.json` exists.
- The enumerated library has been downloaded. **NOT YET IMPLEMENTED in `download_data.py`**: this is a blocker
  for `preprocess_enumerated` and `prior_enumerated` only. The TPD-DB half of this experiment runs without it.
- `bash tools/install.sh` has run, so RetroTAC and PROTAC-Splitter are present.

## Jobs, in dependency order

| job | depends on | what it does |
|---|---|---|
| `preprocess_enumerated` | the enumerated download | HDF5 for the base corpus |
| `prior_enumerated` | `preprocess_enumerated` | the base prior |
| `preprocess_tpddb` | the TPD-DB download | HDF5 for the task corpus, scaffold split recorded |
| `transfer_arm` | `prior_enumerated`, `preprocess_tpddb` | specializes the base prior onto the task corpus |
| `scratch_control` | `preprocess_tpddb` | the control: same data, no `resume_from` |
| `generate_transfer` | `transfer_arm` | **produces this experiment's core metrics** |
| `generate_scratch` | `scratch_control` | the control's core metrics |

Each of those four rows exists three times, once per seed: `transfer_arm{,_seed2,_seed3}`,
`scratch_control{,_seed2,_seed3}` and the matching `generate_*{,_seed2,_seed3}`. Both
`preprocess` jobs and `prior_enumerated` are shared by all three seeds.

### Three seeds, and here a seed is a training run

Table 2 reports mean (standard deviation) over seeds 1, 2 and 3, and the quantity that varies is the training
run itself. Until 2026-10-26 each arm existed once and the three `generate_*_seedN` jobs all read one
checkpoint, so the reported interval was the spread of resampling a single run — the same defect fixed earlier
in the goal-directed experiments by giving each seed its own `rl_*` job.

**The pairing is seed-matched and must stay that way.** `transfer_arm_seed2` declares
`control_job_dir: .../scratch_control_seed2`, and `generate_transfer_seed2` reads
`transfer_arm_seed2/checkpoints/`. `epochs_to_match` and the overlaid control curve are per-seed quantities;
crossing the seeds puts two sources of variance into one number.

**The prior is not re-seeded.** One `prior_enumerated`, three fine-tunings. The interval is therefore the
variance of the transfer step, not of the whole pipeline, and that is what the transfer claim is about. Quote
it with that scope.

**Downstream experiments resume from seed 1.** `protac_conditional/jobs/train`,
`protac_goal_directed/jobs/rl_smarca2*` and `protac_selectivity/jobs/rl_selectivity*` all read
`transfer_arm/checkpoints/model_restart_20.pth` — the seed-1 arm is the designated downstream checkpoint.
Deliberate: fanning the conditional and goal-directed experiments across three transfer seeds would roughly
triple both.

**Cost.** Six training jobs at 47 h each is about **282 GPU-hours** for this experiment alone, against 94 h
before. Size the allocation request before submitting. The MG transfer experiment is the same shape at
about 12 GPU-hours and is the right place to shake the three-seed wiring out first.

## Submit

    python submit.py experiments/protac_transfer/jobs/preprocess_tpddb --slurm

Then chain with the dependency flag, substituting the returned job id:

    sbatch --dependency=afterok:<id> experiments/protac_transfer/jobs/scratch_control/run.sbatch

## What to check when it finishes

- `jobs/transfer_arm/logs/convergence.log` and `jobs/scratch_control/logs/convergence.log`: compare column
  `avg_valid_loss` at epoch 1 and at the final epoch, and the same for each seed against *its own* control. The
  transfer model should start lower.
- `epochs_to_match` in each transfer arm's `metrics.json` is that comparison as a number: the epoch at which the
  transfer arm first reaches its control's final validation loss. A null with reason `undefined_for_this_run`
  means **transfer did not beat from-scratch training on that seed**, which is a result and belongs in the table
  as one, not in a list of unavailable metrics.
- `python scripts/aggregate-seeds.py experiments/protac_transfer` writes `results/aggregate.json` across the
  seeds. A group with one run reports no interval rather than a zero-width one, so a missing seed shows up as
  an absent interval instead of a spuriously perfect one.
- `jobs/generate_transfer/results/metrics.json`, key `validity`, `uniqueness`, `novelty`, `vun`,
  `near_duplicate_rate`.
- `fcd` is reported **only** if `fcd_n_generated` and `fcd_n_reference` are both at or above 1,000. On this task
  corpus the test split is small, so expect `fcd` in the `unavailable` block with category `insufficient_sample`.
  That is the correct outcome, not a failure.

## Core metrics come from

`generate_transfer`, and `generate_scratch` for the control. Training jobs write only the efficiency keys and the
nulled likelihood keys; they do not compute the core metric table.

## Failure mode

If `transfer_arm` shows the same epoch-1 validation loss as `scratch_control`, the checkpoint is not being loaded.
Check that `resume_from` names a file that exists and that the log line `Loading model from:` appears. A silent
no-op here previously came from using `pretrained_model_path` instead of `resume_from`, which is now rejected.

If validity on the generate jobs exceeds 1.0, L3 is not fixed and every funnel number in this experiment is
wrong. Stop.
