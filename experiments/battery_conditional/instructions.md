# battery_conditional

*Last updated: October 18th, 2026.*

## What this demonstrates

Conditioning on molecular dipole moment, and then measuring the achieved dipole on generated molecules by nearly
the method that produced the training labels. That closure is why this experiment is worth its compute:
`condition_mae` here means something it cannot mean for an annotation-derived categorical label.

## Blockers

- `Batt-P30K.h5` in the source repository is a **134 byte Git LFS pointer**, not the data. Either fetch it with
  `git lfs` or take it from Zenodo 10.5281/zenodo.21061162. Not yet wired into `scripts/download_data.py`.
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
