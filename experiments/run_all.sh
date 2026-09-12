#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# GraphINVENT2 — Master experiment runner
#
# Runs all four experiments in sequence:
#   1. ChEMBL v34 pretraining
#   2. DRD2 transfer learning + generation
#   3. Goal-directed optimization (PMO-style, all oracles)
#   4. Conditional generation
#
# Usage (from repository root):
#   bash experiments/run_all.sh [--pretrain-checkpoint PATH] [--dry-run]
#
# Options:
#   --pretrain-checkpoint PATH  Use an existing ChEMBL checkpoint instead of
#                               running Experiment 1 (saves time if already done).
#   --dry-run                   Print commands without executing them.
#   --skip-exp1                 Skip Experiment 1 (pretraining).
#   --skip-exp2                 Skip Experiment 2 (transfer learning).
#   --skip-exp3                 Skip Experiment 3 (goal-directed RL).
#   --skip-exp4                 Skip Experiment 4 (conditional generation).
#
# Prerequisites:
#   pip install -e ".[docking]" (only if a goal-directed run uses a Vina oracle)
#   See experiments/chembl_pretrain/README.md for data download instructions.
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PRETRAIN_CHECKPOINT=""
DRY_RUN=0
SKIP_EXP1=0
SKIP_EXP2=0
SKIP_EXP3=0
SKIP_EXP4=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pretrain-checkpoint)
      PRETRAIN_CHECKPOINT="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --skip-exp1)
      SKIP_EXP1=1; shift ;;
    --skip-exp2)
      SKIP_EXP2=1; shift ;;
    --skip-exp3)
      SKIP_EXP3=1; shift ;;
    --skip-exp4)
      SKIP_EXP4=1; shift ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

RUN() {
  echo ""
  echo ">>> $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# Resolve pretrain checkpoint
# ---------------------------------------------------------------------------
if [[ -z "$PRETRAIN_CHECKPOINT" ]]; then
  # Default: last checkpoint written by Experiment 1
  PRETRAIN_CHECKPOINT="./output/chembl_v34/unconditional/run/model_restart_200.pth"
fi

echo "============================================================"
echo "GraphINVENT2 — Full Experiment Suite"
echo "------------------------------------------------------------"
echo "Pretrain checkpoint : $PRETRAIN_CHECKPOINT"
echo "Dry-run             : $DRY_RUN"
echo "============================================================"

# ---------------------------------------------------------------------------
# Experiment 1: ChEMBL v34 Pretraining
# ---------------------------------------------------------------------------
if [[ $SKIP_EXP1 -eq 0 ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "EXPERIMENT 1: ChEMBL v34 Pretraining"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  echo "Step 1a: Check that ChEMBL SMILES file exists"
  if [[ ! -f "./data/raw/chembl_v34_filtered.smi" ]]; then
    echo "ERROR: ./data/raw/chembl_v34_filtered.smi not found."
    echo "       Follow the instructions in experiments/chembl_pretrain/README.md"
    echo "       to download and filter ChEMBL v34."
    exit 1
  fi

  echo "Step 1b: Preprocess ChEMBL v34"
  RUN python submit.py --config experiments/chembl_pretrain/preprocess_params.json

  echo "Step 1c: Pretrain model"
  RUN python submit.py --config experiments/chembl_pretrain/pretrain_params.json

  echo "Step 1 complete. Check convergence.log and update PRETRAIN_CHECKPOINT"
  echo "to the best epoch before running downstream experiments."
fi

# ---------------------------------------------------------------------------
# Experiment 2: DRD2 Transfer Learning
# ---------------------------------------------------------------------------
if [[ $SKIP_EXP2 -eq 0 ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "EXPERIMENT 2: DRD2 Transfer Learning"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  echo "Step 2a: Check that DRD2 SMILES file exists"
  if [[ ! -f "./data/raw/drd2_actives.smi" ]]; then
    echo "ERROR: ./data/raw/drd2_actives.smi not found."
    echo "       Follow the instructions in experiments/drd2_transfer/README.md."
    exit 1
  fi

  echo "Step 2b: Preprocess DRD2 actives (union vocabulary with ChEMBL)"
  RUN python submit.py --config experiments/drd2_transfer/preprocess_params.json

  echo "Step 2c: Transfer learning (fine-tune from ChEMBL checkpoint)"
  # Patch the checkpoint path into the config if --pretrain-checkpoint was given
  # Guarded by DRY_RUN: this rewrites a tracked config file in place, which a
  # dry run must not do.
  if [[ -n "$PRETRAIN_CHECKPOINT" && $DRY_RUN -eq 0 ]]; then
    python - <<EOF
import json, pathlib
cfg_path = pathlib.Path("experiments/drd2_transfer/transfer_params.json")
cfg = json.loads(cfg_path.read_text())
cfg["job"]["resume_from"] = "${PRETRAIN_CHECKPOINT}"
cfg_path.write_text(json.dumps(cfg, indent=2))
print(f"Patched resume_from -> ${PRETRAIN_CHECKPOINT}")
EOF
  elif [[ -n "$PRETRAIN_CHECKPOINT" ]]; then
    echo "[dry-run] would patch resume_from -> $PRETRAIN_CHECKPOINT in experiments/drd2_transfer/transfer_params.json"
  fi
  RUN python submit.py --config experiments/drd2_transfer/transfer_params.json

  echo "Step 2d: Generate 10,000 molecules from fine-tuned model"
  RUN python submit.py --config experiments/drd2_transfer/generate_params.json
fi

# ---------------------------------------------------------------------------
# Experiment 3: Goal-Directed Optimization (PMO-style)
# ---------------------------------------------------------------------------
if [[ $SKIP_EXP3 -eq 0 ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "EXPERIMENT 3: Goal-Directed Optimization (PMO Benchmark)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  RL_ARGS="--pretrained-model ${PRETRAIN_CHECKPOINT}"
  if [[ $DRY_RUN -eq 1 ]]; then
    RL_ARGS="$RL_ARGS --dry-run"
  fi

  echo "Step 3a: Run all oracles (6 oracles × 3 seeds = 18 runs)"
  RUN python experiments/goal_directed/run_all_oracles.py $RL_ARGS

  echo "Step 3b: Evaluate PMO results"
  RUN python experiments/goal_directed/evaluate_results.py
fi

# ---------------------------------------------------------------------------
# Experiment 4: Conditional Generation
# ---------------------------------------------------------------------------
if [[ $SKIP_EXP4 -eq 0 ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "EXPERIMENT 4: Conditional Generation"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  echo "Step 4a: Compute properties and create TSV for conditioning"
  if [[ ! -f "./data/raw/chembl_v34_cond.tsv" ]]; then
    RUN python experiments/conditional/compute_properties.py \
      --smiles data/raw/chembl_v34_filtered.smi \
      --out data/raw/chembl_v34_cond.tsv \
      --gsk3b
  else
    echo "  TSV already exists: ./data/raw/chembl_v34_cond.tsv"
  fi

  echo "Step 4b: Preprocess conditional dataset"
  RUN python submit.py --config experiments/conditional/preprocess_params.json

  echo "Step 4c: Train conditional model (fine-tune from ChEMBL checkpoint)"
  # Guarded by DRY_RUN: this rewrites a tracked config file in place, which a
  # dry run must not do.
  if [[ -n "$PRETRAIN_CHECKPOINT" && $DRY_RUN -eq 0 ]]; then
    python - <<EOF
import json, pathlib
cfg_path = pathlib.Path("experiments/conditional/train_params.json")
cfg = json.loads(cfg_path.read_text())
cfg["job"]["resume_from"] = "${PRETRAIN_CHECKPOINT}"
cfg_path.write_text(json.dumps(cfg, indent=2))
print(f"Patched resume_from -> ${PRETRAIN_CHECKPOINT}")
EOF
  elif [[ -n "$PRETRAIN_CHECKPOINT" ]]; then
    echo "[dry-run] would patch resume_from -> $PRETRAIN_CHECKPOINT in experiments/conditional/train_params.json"
  fi
  RUN python submit.py --config experiments/conditional/train_params.json

  echo "Step 4d: Evaluate conditional generation"
  COND_CHECKPOINT="./output/chembl_v34_cond/conditional/run/model_restart_100.pth"
  RUN python experiments/conditional/evaluate_conditional.py \
    --checkpoint "$COND_CHECKPOINT"
fi

# ---------------------------------------------------------------------------
# Aggregate all results
# ---------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "AGGREGATING RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RUN python experiments/aggregate_results.py

echo ""
echo "All experiments complete. Summary written to experiments/summary/."
