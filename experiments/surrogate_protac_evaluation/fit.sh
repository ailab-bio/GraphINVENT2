#!/bin/bash
# surrogate_protac_evaluation: the oracle that RESCORES final molecules and is
# never used for optimization.
#
#   bash experiments/surrogate_protac_evaluation/fit.sh
#
# The same featurization and model class as surrogate_protac_optimization,
# refitted on ALL the data including the test split. A large gap between the
# two on optimized molecules, when they agree on held-out real molecules, is
# evidence of exploitation rather than success. Feeding this model back into an
# optimization run would destroy that evidence, so it exists as its own
# experiment and its pickle is never named in a goal_directed `oracles:` block.
# PANEL updated 2026-10-26 to the settled SMARCA2/AR/KRAS trio; it looped over
# SMARCA2/AR/BTK before. No --cv here on purpose: this arm refits on all the
# data with --test-fraction 0.0001, so there is no held-out set and a grouped
# k-fold over it would report an in-sample number under a held-out name.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

HERE=experiments/surrogate_protac_evaluation
python scripts/download_data.py --source tpddb

for TARGET in SMARCA2 AR KRAS; do
  LOWER=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')
  for ENDPOINT in pdc50 dmax; do
    python scripts/tpddb-activity-labels.py \
        --target "$TARGET" --endpoint "$ENDPOINT" \
        --output "data/surrogates/${LOWER}_${ENDPOINT}.tsv"

    python scripts/train-surrogate.py \
        --input "data/surrogates/${LOWER}_${ENDPOINT}.tsv" \
        --smiles-column smiles --label-column "$ENDPOINT" \
        --split random --test-fraction 0.0001 --seed 1 \
        --count-fingerprint --categorical cell_line --categorical ligase \
        --output "$HERE/${LOWER}_${ENDPOINT}.pickle" \
        | tee "$HERE/${LOWER}_${ENDPOINT}.log"
  done
done
