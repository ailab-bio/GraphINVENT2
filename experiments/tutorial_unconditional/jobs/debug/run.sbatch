#!/bin/bash
#SBATCH -A naiss2024-5-630-gpu
#SBATCH -p gpu
#SBATCH -J debug
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 72
#SBATCH --gpus=1
#SBATCH -o /Users/rocom/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/GraphINVENT2/experiments/tutorial_unconditional/jobs/debug/logs/slurm-%j.out
#SBATCH -e /Users/rocom/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/GraphINVENT2/experiments/tutorial_unconditional/jobs/debug/logs/slurm-%j.err

set -euo pipefail

module load GPU/buildenv-nvhpc/25.9-cu13.0
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

cd '/Users/rocom/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/GraphINVENT2'
srun '/Users/rocom/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/GraphINVENT2/.venv/bin/python' submit.py '/Users/rocom/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/GraphINVENT2/experiments/tutorial_unconditional/jobs/debug'
