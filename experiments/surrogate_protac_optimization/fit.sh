#!/bin/bash
# surrogate_protac_optimization: the reward for experiments/protac_goal_directed
# and experiments/protac_selectivity.
#
#   bash experiments/surrogate_protac_optimization/fit.sh
#
# Run from the repository root. This fits on the TRAINING AND VALIDATION SPLITS
# ONLY. No test molecule is anywhere in it, which is the whole reason this is a
# separate experiment from surrogate_protac_evaluation: the two differ in what
# data they are allowed to see, and that difference is the point.
#
# Two endpoints, because potency and extent are different questions. A degrader
# can be potent and still only ever remove half the protein, and an
# optimization that reads one as the other chases the wrong molecule.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

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
        --output "$HERE/${LOWER}_${ENDPOINT}.pickle" \
        | tee "$HERE/${LOWER}_${ENDPOINT}.log"
  done
done
