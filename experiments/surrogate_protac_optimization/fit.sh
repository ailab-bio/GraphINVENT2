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

for TARGET in SMARCA2 AR BTK; do
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
        --count-fingerprint --categorical cell_line --categorical ligase \
        --output "$HERE/${LOWER}/${ENDPOINT}.pickle" \
        | tee "$HERE/${LOWER}/${ENDPOINT}.log"
  done
done
