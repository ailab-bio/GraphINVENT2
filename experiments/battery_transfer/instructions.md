# battery_transfer

*Last updated: October 18th, 2026.*

## What this demonstrates

Transfer learning from a general PubChem corpus onto the battery-solvent-like corpus, with a from-scratch
control, and the only distribution-match comparison in the paper whose reference set is large enough for FCD to
mean anything.

## Blockers, both data

- **PubChem-120K is not available.** It is not distributed in the Teoroo-CMC/Batt-SLM repository (which carries
  `Batt-SLM/Batt-SLM.smi`, `Batt-SLM/filter_smiles.py`, `CPI/`, `Redox-Pot/` and a `Batt-P30K/` whose `.h5` is a
  134 byte Git LFS pointer), and the reconstruction procedure is documented only in the supporting information,
  which is not in this repository. `preprocess_pubchem120k` and `prior_pubchem120k` cannot run. **Question 4.**
- The two source PDFs are absent from `manuscript/refs/`, so no number attributed to the source work in this
  directory has been verified against the paper. All are marked unverified.

The Batt-SLM half of this experiment (`preprocess_battslm`, `prior_scratch`, `generate_scratch`) runs without
either.

## Prerequisites

`Batt-SLM.smi` downloaded (not yet wired into `scripts/download_data.py`) and
`scripts/battslm-create-dataset.py` written (not yet).

## The split

The published split, per the source work: 5,000 molecules held out as test, the remainder 9:1 train and
validation. That is not the 80/10/10 default, so `preprocess_battslm` sets the fractions explicitly.

## Submit

    python submit.py experiments/battery_transfer/jobs/preprocess_battslm --slurm

## What to check

- `jobs/generate_scratch/results/metrics.json` against the published step-0 values (unverified, from Figure S17):
  validity about 0.93, uniqueness about 0.89, novelty about 0.57. **A disagreement is a bug in us, not a
  finding.**
- `fcd`, with `fcd_n_generated` and `fcd_n_reference` beside it. This is the one experiment where both clear the
  1,000-molecule floor, so this is where FCD is reported properly.

## Core metrics come from

`generate_transfer`, with `generate_scratch` as the control.

## Failure mode

**At 93% validity, L3 fires here first**: the over-production bug puts validity above 1.0 on a good model. If
`validity > 1.0`, stop; L3 is not fixed. Also check `n_dropped_oversized` is zero in `preprocessing.yaml`.

The epoch count is 60 and not the published 1,000. That is deliberate: the published schedule is in the runbook
as a separate later step, and 60 is a first attempt whose tolerance against the published NLL has not been
established. Treat the first run as a calibration of that number.
