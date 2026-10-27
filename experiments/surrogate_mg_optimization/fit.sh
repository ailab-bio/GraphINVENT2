#!/bin/bash
# surrogate_mg_optimization: the reward for experiments/mg_goal_directed, fitted on the training and validation splits only.
#
#   bash experiments/surrogate_mg_optimization/fit.sh
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
# PANEL updated 2026-10-26 to the settled GSPT1/VAV1/CK1a trio from CLAUDE.md;
# it looped over GSPT1/IKZF1/IKZF3/RBM39, which were chosen on molecule count.
# All three settled targets sit below the ~100-compound floor, which is why
# these fits are reported with their denominators and named in no reward.
#
# No --cv grouped: a glue has no warhead to group by, and with 30 compounds for
# the largest target a five-fold grouped k-fold would report five numbers from
# six molecules each. The single split's denominator is already the finding.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

HERE=experiments/surrogate_mg_optimization
python scripts/download_data.py --source tpddb

# The target string must match TPDdb's `Target Symbols` value exactly, and
# TPDdb writes casein kinase 1 alpha with a Greek alpha. The path slug is
# separate and ASCII, so no artifact filename carries a non-ASCII character.
# Selection resolves through data/surrogates/target_symbol_map.tsv, so --target
# takes the canonical HGNC symbol rather than the raw TPDdb spelling: CK1alpha
# is CSNK1A1, and asking for the raw string now fails loudly rather than
# matching one of its several spellings. The lowercase half of each pair is the
# artifact name and stays as it was, so no committed path moves.
for PAIR in "GSPT1:gspt1" "VAV1:vav1" "CSNK1A1:ck1a"; do
  TARGET="${PAIR%%:*}"
  LOWER="${PAIR##*:}"
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
        --split scaffold --test-fraction 0.2 --seed 1 \
        --count-fingerprint --categorical cell_line --categorical ligase \
        --output "$HERE/${LOWER}_${ENDPOINT}.pickle" \
        | tee "$HERE/${LOWER}_${ENDPOINT}.log" || true
  done
done
