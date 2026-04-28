# GraphINVENT2 Experiments

End-to-end experimental evaluation suite for the GraphINVENT2 paper.  Four experiments cover all three training modes: unconditional (pretraining + transfer learning), goal-directed (RL), and conditional generation.

---

## Directory structure

```
experiments/
├── chembl_pretrain/
│   ├── preprocess_params.json    ChEMBL v34 scaffold-split preprocessing
│   ├── pretrain_params.json      Unconditional pretraining (200 epochs, large arch)
│   └── README.md                 Data download + training instructions
├── drd2_transfer/
│   ├── preprocess_params.json    Union-vocabulary preprocessing (ChEMBL + DRD2)
│   ├── transfer_params.json      Supervised fine-tuning on DRD2 actives
│   ├── generate_params.json      Generate 10,000 molecules from fine-tuned model
│   └── README.md
├── goal_directed/
│   ├── rl_params_template.json   Template config for one (oracle, seed) RL run
│   ├── oracles_config.yaml       Oracle list, TDC names, thresholds, PMO settings
│   ├── run_all_oracles.py        Launch all 18 (oracle × seed) runs
│   ├── evaluate_results.py       Compute PMO metrics and write CSV + LaTeX table
│   └── README.md
├── conditional/
│   ├── preprocess_params.json    Conditional preprocessing (4-property TSV)
│   ├── train_params.json         Fine-tune conditional model from ChEMBL prior
│   ├── generate_params.json      Generate with a specific condition vector (template)
│   ├── compute_properties.py     Create labelled TSV from a plain SMILES file
│   ├── evaluate_conditional.py   Generate + evaluate across all property ranges
│   └── README.md
├── aggregate_results.py          Combine all CSV results into summary tables + LaTeX
├── run_all.sh                    Master script that runs all experiments in order
└── README.md                     This file
```

---

## Experiment overview

| # | Name | Mode | Dependency | Est. GPU hours |
|---|------|------|------------|----------------|
| 1 | ChEMBL v34 Pretraining | Unconditional | None | 48–72 h (A100) |
| 2 | DRD2 Transfer Learning | Unconditional (resume_from) | Exp 1 | 2–6 h |
| 3 | Goal-Directed (PMO) | RL — 6 oracles × 3 seeds | Exp 1 | 12–24 h/run |
| 4 | Conditional Generation | Conditional | Exp 1 | 14–28 h |

---

## Prerequisites

### 1. Install dependencies

```bash
pip install -e ".[tdc]"   # adds PyTDC for oracle access
pip install pyyaml        # for oracles_config.yaml parsing
```

### 2. Verify the install

```bash
python -c "from tdc import Oracle; o = Oracle('DRD2'); print(o('CCO'))"
```

### 3. Download data

See [chembl_pretrain/README.md](chembl_pretrain/README.md) for ChEMBL v34 download and filtering instructions.
See [drd2_transfer/README.md](drd2_transfer/README.md) for DRD2 actives download.

---

## Running all experiments

### Option A: master script (sequential)

```bash
bash experiments/run_all.sh
```

Use `--pretrain-checkpoint` to skip retraining if a ChEMBL checkpoint already exists:

```bash
bash experiments/run_all.sh \
    --pretrain-checkpoint ./output/chembl_v34/unconditional/run/model_restart_180.pth
```

Use `--dry-run` to preview all commands without executing:

```bash
bash experiments/run_all.sh --dry-run
```

### Option B: individual experiments

Each experiment can be run independently (as long as its dependency checkpoint exists).

```bash
# Experiment 1
python submit.py --config experiments/chembl_pretrain/preprocess_params.json
python submit.py --config experiments/chembl_pretrain/pretrain_params.json

# Experiment 2
python submit.py --config experiments/drd2_transfer/preprocess_params.json
python submit.py --config experiments/drd2_transfer/transfer_params.json
python submit.py --config experiments/drd2_transfer/generate_params.json

# Experiment 3
python experiments/goal_directed/run_all_oracles.py \
    --pretrained-model ./output/chembl_v34/unconditional/run/model_restart_180.pth
python experiments/goal_directed/evaluate_results.py

# Experiment 4
python experiments/conditional/compute_properties.py \
    --smiles data/raw/chembl_v34_filtered.smi \
    --out data/raw/chembl_v34_cond.tsv --gsk3b
python submit.py --config experiments/conditional/preprocess_params.json
python submit.py --config experiments/conditional/train_params.json
python experiments/conditional/evaluate_conditional.py \
    --checkpoint output/chembl_v34_cond/conditional/run/model_restart_100.pth
```

### Option C: parallel RL runs (recommended for Experiment 3)

Each (oracle, seed) RL run is independent and can be parallelised across GPUs:

```bash
# Terminal 1
python experiments/goal_directed/run_all_oracles.py --oracle DRD2 --oracle GSK3B

# Terminal 2
python experiments/goal_directed/run_all_oracles.py --oracle JNK3 --oracle SA

# Terminal 3
python experiments/goal_directed/run_all_oracles.py --oracle QED --oracle "tdc:LogP"
```

---

## Aggregating results

After all experiments are complete:

```bash
python experiments/aggregate_results.py
```

Output:
- `experiments/summary/pmo_summary.csv` — PMO AUC Top-10 means and std across seeds
- `experiments/summary/cond_summary.csv` — Conditional accuracy per property range
- `experiments/summary/tables.tex` — Ready-to-include LaTeX tables for the paper

---

## SLURM submission

All `params.json` files include a `slurm` block.  To submit to a cluster, set `"use_slurm": true` and fill in your account name and partition settings.  Example:

```json
"submission": {
  "use_slurm": true,
  "slurm": {
    "account": "myproject",
    "run_time": "2-00:00:00",
    "gpus_per_node": "A100:1"
  }
}
```

For Experiment 3, running via SLURM is strongly recommended: each of the 18 runs requires 12–24 hours and a GPU.

---

## Checkpoint path convention

The pretrained ChEMBL checkpoint path appears in three places.  Always update all three if the best epoch changes:

1. `experiments/drd2_transfer/transfer_params.json` → `resume_from`
2. `experiments/goal_directed/rl_params_template.json` → `pretrained_model_path`
3. `experiments/conditional/train_params.json` → `resume_from`

The master script (`run_all.sh`) patches these automatically when `--pretrain-checkpoint` is passed.
