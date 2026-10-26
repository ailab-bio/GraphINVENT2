#!/bin/bash
# surrogate_protac_optimization: the reward for experiments/protac_goal_directed
# and experiments/protac_selectivity.
#
#   bash experiments/surrogate_protac_optimization/fit.sh
#
# Run from the repository root.
#
# WHAT THIS FIT SEES. It used to claim here that it fits "on the TRAINING AND
# VALIDATION SPLITS ONLY" with "no test molecule anywhere in it". That was
# false: scripts/train-surrogate.py reads the whole label table and makes its
# own 80/20 warhead-scaffold split for its own held-out estimate, and it has no
# knowledge of data/datasets/tpddb_protac/split_warhead_scaffold/ at all. So
# the reward surrogate may have been fitted on molecules that sit in the
# generative model's test split, and on the known actives the active-holdout
# rule removes from the generator's corpus.
#
# That is not leakage into the generator -- the two models share no weights --
# but it has two consequences that must be stated wherever these numbers are
# used: rediscovery is scored by structural match and never by oracle score,
# because the oracle has seen the holdout; and the held-out number below is
# the surrogate's own warhead-scaffold estimate, not an estimate on the
# generator's test molecules.
#
# Two endpoints, because potency and extent are different questions. A degrader
# can be potent and still only ever remove half the protein, and an
# optimization that reads one as the other chases the wrong molecule.
#
# THE PANEL. SMARCA2, AR and KRAS, which is the settled trio in CLAUDE.md,
# chosen to span data volumes rather than to maximize them. This script looped
# over SMARCA2, AR and BTK until 2026-10-26; BTK and EGFR are the tempting
# alternatives on molecule count and fail on pairing, and SMARCA4 is reserved
# for protac_selectivity rather than being a fourth target.
#
# CROSS-VALIDATION. Each fit runs a grouped k-fold as well as its single
# held-out split, and keeps the fold models in the pickle. Two reasons, and
# they are different. First, one 80/20 split at one seed gives a point estimate
# with no spread: on SMARCA2 pDC50 the five warhead-scaffold folds range from
# R^2 0.137 to 0.784 around a mean of 0.540, so the single split's 0.416 was
# never a number to quote alone. Second, the fold models ARE the
# reward-hacking detector CLAUDE.md specifies: their per-molecule disagreement
# is evidence that RL has pushed a molecule outside the data, which the spread
# across one forest's trees cannot show because every tree saw the same rows.
#
# --group-by warhead_scaffold estimates the error on novel molecules for a
# target the model has seen, which is what the reward does. The pooled table
# and --group-by poi estimate the error for a target it has never seen; see
# the pooled fit at the bottom of this script.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# WHERE THE ARTIFACTS GO, and which one is the reward. Four paths used to claim
# to be the reward: this script wrote one name, protac_goal_directed's
# experiment.yaml named a second, its rl_smarca2 job overrode to a third that
# does not exist, and the committed pickles' own logs name a fourth. One name
# per (target, endpoint) now, written here, and nothing else renames it.
#
# The committed smarca2/optimization.pickle is the pDC50 fit under the
# superseded name and is what protac_goal_directed loads today; re-running this
# script writes smarca2/pdc50.pickle beside it and experiment.yaml must be
# repointed in the same change.
#
# OPEN: CLAUDE.md says the two endpoints are "combined, equally weighted" into
# one composite reward, manuscript/experiment-plan.md says Dmax is descriptive
# only, and ScoringFunction combines declared components as a product. The
# configs declare the pDC50 component alone. The combination rule is a
# pharmacological decision and is not settled here.
HERE=experiments/surrogate_protac_optimization
python scripts/download_data.py --source tpddb

for TARGET in SMARCA2 AR KRAS; do
  LOWER=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')
  for ENDPOINT in pdc50 dmax; do
    python scripts/tpddb-activity-labels.py \
        --target "$TARGET" --endpoint "$ENDPOINT" \
        --output "data/surrogates/${LOWER}_${ENDPOINT}.tsv"

    # Held out by WARHEAD scaffold. A whole-molecule Bemis-Murcko scaffold of a
    # PROTAC is nearly unique per compound, so a scaffold split degenerates
    # into a random one and the held-out number would be an analogue lookup.
    python scripts/train-surrogate.py \
        --input "data/surrogates/${LOWER}_${ENDPOINT}.tsv" \
        --smiles-column smiles --label-column "$ENDPOINT" \
        --split warhead_scaffold --test-fraction 0.2 --seed 1 \
        --cv grouped --group-by warhead_scaffold --folds 5 \
        --count-fingerprint --categorical cell_line --categorical ligase \
        --output "$HERE/${LOWER}/${ENDPOINT}.pickle" \
        | tee "$HERE/${LOWER}/${ENDPOINT}.log"
  done
done

# THE POOLED FIT, and the POI-grouped estimate only it can give.
#
# With one target per file there is no unseen target to hold out, so the
# POI-grouped error -- what CLAUDE.md calls the estimate for "a target it has
# never seen" -- is not computable from the per-target fits above at all. This
# builds the pooled multi-target table and fits it once, with folds grouped by
# protein.
#
# READ THE RESULT BEFORE USING IT. Measured 2026-10-26 on 1,584 rows over 18
# canonical POIs: mean R^2 across the five POI-grouped folds is -0.150 (SD
# 0.159), and FOUR OF FIVE FOLDS ARE WORSE THAN PREDICTING THE TRAINING MEAN.
# The pooled surrogate has no demonstrated ability to score a target it has not
# seen. By this repository's own rule -- a fit that fails to beat its baseline
# is reported as a failed fit rather than used as a reward -- the POI-grouped
# arm is a failed fit, and anything that depends on scoring an unseen target
# cannot lean on it. The same table under a random split scores R^2 0.613,
# which is what that leakage looks like from the inside.
#
# This fit is therefore run and REPORTED, not wired into a reward. No
# goal_directed config names its pickle.
python scripts/tpddb-activity-labels.py \
    --target all --endpoint pdc50 \
    --output data/surrogates/pooled_pdc50.tsv

python scripts/train-surrogate.py \
    --input data/surrogates/pooled_pdc50.tsv \
    --smiles-column smiles --label-column pdc50 \
    --split random --test-fraction 0.2 --seed 1 \
    --count-fingerprint \
    --categorical cell_line --categorical ligase --categorical poi \
    --cv grouped --group-by poi --folds 5 --no-store-folds \
    --output "$HERE/pooled/pdc50.pickle" \
    | tee "$HERE/pooled/pdc50.log"
