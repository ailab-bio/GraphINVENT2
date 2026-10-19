#!/bin/bash
# surrogate_mg_evaluation: the oracle that rescores final glue molecules, refitted on all the data.
#
#   bash experiments/surrogate_mg_evaluation/fit.sh
#
# THIS CANNOT BE FITTED FROM TPD-DB AS IT STANDS. Measured 2026-10-19 by
# running scripts/tpddb-activity-labels.py --modality glue over every
# neosubstrate the conditional panel names:
#
#     target   pDC50 compounds   Dmax compounds
#     GSPT1    30                30
#     IKZF1     5                 5
#     IKZF3     1                 0
#     RBM39     0                 0
#
# The experiment plan's floor is about a hundred compounds. The largest glue
# target has thirty. A surrogate fitted on thirty compounds and held out by
# scaffold has a test set in the single digits, so its held-out number is not
# an estimate of anything, and by the kill criterion in this experiment's
# instructions it cannot drive mg_goal_directed.
#
# The script is written and runnable so the counts reproduce in one command,
# and so it works unchanged the day a larger glue activity set exists. It is
# deliberately not named in any goal-directed config.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

HERE=experiments/surrogate_mg_evaluation
python scripts/download_data.py --source tpddb

for TARGET in GSPT1 IKZF1 IKZF3 RBM39; do
  LOWER=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')
  for ENDPOINT in pdc50 dmax; do
    python scripts/tpddb-activity-labels.py --modality glue \
        --target "$TARGET" --endpoint "$ENDPOINT" \
        --output "data/surrogates/glue_${LOWER}_${ENDPOINT}.tsv" || continue

    # A glue has no warhead, so there is nothing to hold out by: the
    # warhead_scaffold split the PROTAC surrogates use has no meaning here,
    # and that is one of the four reasons the two are never pooled.
    python scripts/train-surrogate.py \
        --input "data/surrogates/glue_${LOWER}_${ENDPOINT}.tsv" \
        --smiles-column smiles --label-column "$ENDPOINT" \
        --split random --test-fraction 0.0001 --seed 1 \
        --count-fingerprint --categorical cell_line --categorical ligase \
        --output "$HERE/${LOWER}_${ENDPOINT}.pickle" \
        | tee "$HERE/${LOWER}_${ENDPOINT}.log" || true
  done
done
