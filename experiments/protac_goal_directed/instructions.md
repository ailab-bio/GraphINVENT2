# protac_goal_directed

*Last updated: October 18th, 2026.*

## What this demonstrates

Policy-gradient optimization of a PROTAC generator toward predicted SMARCA2 degradation recovers more
high-scoring, synthesizable, structurally distinct candidates per surrogate call than screening the corpus it was
trained on.

## GATE: do not submit until L1, L2 and L3 are closed

The budget curve currently integrates whichever oracle component was called most rather than the composite
reward, and validity can exceed 1.0 on a good model. Both make every number this experiment reports wrong.
Confirm the fixes landed:

    python -c "import sys; sys.path.insert(0,'src'); import metrics._budget as b; print(b.__doc__)"
    grep -n "composite" src/graphinvent/ScoringFunction.py

## KILL CRITERION, checked before this runs

`experiments/surrogates/protac_pdc50_smarca2` must report a held-out R2 that beats a mean-prediction baseline.
If it does not, **this experiment does not run** and that is the finding. The target is SMARCA2 and not AR: AR has
344 numeric DC50 rows of which 159 are an engineered L702H resistance line, leaving 185 wild-type rows with 86%
from a single patent. AR appears in the paper as the documented data-poor case.

## Prerequisites

`protac_unconditional/jobs/prior_transfer`, the surrogate, and `bash tools/install.sh` for RetroTAC.

## Jobs

`rl_smarca2`, then `generate_best_agent`, then the two screening baselines and the REINVENT4 comparison, which
are independent of each other.

## Submit

    python submit.py experiments/protac_goal_directed/jobs/rl_smarca2 --slurm

## What to check

- `results/metrics.json`: `quality_filtered_top10_mean` is the headline, gated on RetroTAC because this is a
  bifunctional dataset. `top100_warhead_scaffold_count` is the primary reward-hacking detector; the
  whole-molecule `top100_scaffold_count` is reported beside it and is expected to be uninformative.
- `vs_delta_top10_at_10k` against both screens. If it is not positive, generation earned nothing.
- `top100_outside_domain_rate`: if most of the top 100 sit outside the surrogate's domain, the result is about
  the surrogate's error surface.

## Core metrics come from

`generate_best_agent`, a fresh sample from the best agent. Budget curves come from the `rl_smarca2` trajectory.
metrics.json labels which is which.

## Failure mode

The score rises smoothly while `top100_warhead_scaffold_count` collapses toward 1: that is reward hacking, not
success. A `ValueError` from `OneCycleLR` part-way through means L18 is not fixed.
