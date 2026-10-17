#!/bin/bash
# The first thing to submit on a new cluster. Runs in well under ten minutes
# and stops at the first failure, naming it.
#
#   sbatch -A <your-project>-gpu scripts/cluster_selftest.sbatch
#
# The account is passed on the command line rather than written here, because
# it is the one value that differs per person. Everything else matches the
# slurm: block in config/defaults.yaml; edit both together if you change one.
#SBATCH -p gpu
#SBATCH -J gi2-selftest
#SBATCH -t 00:20:00
#SBATCH -n 1
#SBATCH -c 72
#SBATCH --gpus=1
#SBATCH -o selftest-%j.out
#SBATCH -e selftest-%j.err

set -euo pipefail

module load GPU/buildenv-nvhpc/25.9-cu13.0

export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

echo "node:    $(hostname)"
echo "job:     ${SLURM_JOB_ID:-none}"
nvidia-smi || echo "nvidia-smi unavailable"
echo

srun python scripts/cluster_selftest.py --require-cuda
