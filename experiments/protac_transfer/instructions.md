# protac_transfer

*Last updated: October 18th, 2026.*

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

## Submit

    python submit.py experiments/protac_transfer/jobs/preprocess_tpddb --slurm

Then chain with the dependency flag, substituting the returned job id:

    sbatch --dependency=afterok:<id> experiments/protac_transfer/jobs/scratch_control/run.sbatch

## What to check when it finishes

- `jobs/transfer_arm/logs/convergence.log` and `jobs/scratch_control/logs/convergence.log`: compare column
  `avg_valid_loss` at epoch 1 and at the final epoch. The transfer model should start lower. Record the epoch at
  which it first reaches the scratch model's final loss.
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
