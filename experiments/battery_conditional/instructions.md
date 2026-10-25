# battery_conditional

*Last updated: October 18th, 2026.*

## What this demonstrates

Conditioning on molecular dipole moment, and then measuring the achieved dipole on generated molecules by nearly
the method that produced the training labels. That closure is why this experiment is worth its compute:
`condition_mae` here means something it cannot mean for an annotation-derived categorical label.

## Blockers

- `Batt-P30K.h5` is **present**: `data/raw/battslm/Batt-P30K.h5`, 183,771,012 bytes, 29,519 molecule groups
  each carrying `smiles` and a `dipole` 3-vector, fetched 2026-10-19 and recorded in that directory's
  `MANIFEST.json` with its url and sha256. `scripts/download_data.py` fetches it under `--source battslm`.
  What is missing is the extraction: no script takes the norm of the dipole vector per molecule and writes
  `data/datasets/battp30k/battp30k_dipole.smi` (about 1 MB), so `battp30k` does not exist on disk and no job
  in this experiment can run. Write that script; the labels are sufficient.
- Measuring the achieved dipole needs the PiNet2 predictor, which is the battery oracle's electronic-property
  stage. See `battery_goal_directed/instructions.md` for whether it installs.
- **L13 is a prerequisite.** This input carries property columns, which preprocess currently drops silently
  unless `conditioning` is set. It is set here, so this specific job is safe, but the silent-drop defect is what
  would bite anyone building a variant.

## Submit

    python submit.py experiments/battery_conditional/jobs/preprocess_conditional --slurm

## What to check

`jobs/generate_low/results/metrics.json` and `generate_high`: `condition_success_rate_dipole`, and the achieved
distributions should separate and be correctly ordered. `generate_from_train` exercises the `from_train` sentinel,
which spreads requests across the training deciles; expect `condition_coverage_dipole` to populate most of its ten
deciles rather than one.

## Core metrics come from

`generate_low`, `generate_high`, `generate_from_train`, with `generate_unconditional` as the baseline.

## Failure mode

The low and high requests producing the same achieved distribution means conditioning is not working. A
`condition_coverage_dipole` with one populated decile means the request was constant, which is a config error
rather than a model failure.
