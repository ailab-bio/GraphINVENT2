#!/bin/bash
# The SMARCA2 pDC50 surrogate: the reward for experiments/protac_goal_directed.
#
#   bash experiments/surrogates/protac_pdc50_smarca2/fit.sh
#
# Run from the repository root. Labels come from the single pDC50 parser, which
# writes its own provenance file; the raw table is fetched and checksummed by
# scripts/download_data.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

python scripts/download_data.py --source tpddb
python scripts/tpddb-activity-labels.py --target SMARCA2 \
    --output data/surrogates/smarca2_pdc50.tsv

HERE=experiments/surrogates/protac_pdc50_smarca2

# The optimization surrogate: train and validation only, and it is the reward.
python scripts/train-surrogate.py \
    --input data/surrogates/smarca2_pdc50.tsv \
    --smiles-column smiles --label-column pdc50 \
    --split warhead_scaffold --test-fraction 0.2 --seed 1 \
    --count-fingerprint --categorical cell_line --categorical ligase \
    --output "$HERE/optimization.pickle" | tee "$HERE/optimization.log"

# The evaluation surrogate: all the data, and it only scores final molecules.
python scripts/train-surrogate.py \
    --input data/surrogates/smarca2_pdc50.tsv \
    --smiles-column smiles --label-column pdc50 \
    --split random --test-fraction 0.0001 --seed 1 \
    --count-fingerprint --categorical cell_line --categorical ligase \
    --output "$HERE/evaluation.pickle" | tee "$HERE/evaluation.log"
