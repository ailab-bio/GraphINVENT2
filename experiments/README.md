# GraphINVENT2 Experiments

Four experiments covering the three training modes: unconditional (pretraining, then transfer learning), goal-directed (RL against oracles you define), and conditional generation. Each experiment builds on the ChEMBL prior produced by the first, so the dependency order matters more than the numbering suggests.

---

## Directory structure

```
experiments/
├── chembl_pretrain/
│   ├── preprocess_params.json    ChEMBL v34, Butina split
│   ├── pretrain_params.json      Unconditional pretraining, 200 epochs
│   └── README.md
├── drd2_transfer/
│   ├── preprocess_params.json    Union-vocabulary preprocessing (ChEMBL + DRD2)
│   ├── transfer_params.json      Supervised fine-tuning on DRD2 actives
│   ├── generate_params.json      10,000 molecules from the fine-tuned model
│   └── README.md
├── goal_directed/
│   ├── rl_params_template.json   One (target, seed) run; placeholders filled at launch
│   ├── oracles_config.yaml       Target names, oracle specs, thresholds, seeds
│   ├── run_all_oracles.py        Launches the (oracle × seed) grid
│   ├── evaluate_results.py       Reads run outputs, writes CSV + LaTeX
│   └── README.md
├── conditional/
│   ├── preprocess_params.json    Conditional preprocessing from a property TSV
│   ├── train_params.json         Conditional fine-tuning from the ChEMBL prior
│   ├── generate_params.json      Single-condition generation template
│   ├── compute_properties.py     Builds the labelled TSV from a plain SMILES file
│   ├── evaluate_conditional.py   Generates and evaluates across property ranges
│   └── README.md
├── aggregate_results.py          Combines experiment CSVs into summary tables + LaTeX
├── run_all.sh                    Runs all four experiments in order
└── README.md                     This file
```

Two directories appear only after a run: `goal_directed/configs/` holds the resolved config written for each RL run, and `goal_directed/results/` and `conditional/results/` hold the evaluation CSVs.

---

## Experiment overview

| # | Name | Mode | Depends on | Est. GPU hours |
|---|------|------|------------|----------------|
| 1 | ChEMBL v34 pretraining | Unconditional | — | 48–72 h (A100) |
| 2 | DRD2 transfer learning | Unconditional + `resume_from` | Exp 1 | 2–6 h |
| 3 | Goal-directed (PMO) | RL — one run per (target, seed) | Exp 1 | 12–24 h per run |
| 4 | Conditional generation | Conditional | Exp 1 | 14–28 h |

The GPU-hour figures are rough estimates from a single A100 and scale with dataset size, so treat them as planning numbers rather than measurements.

---

## Prerequisites

```bash
pip install pyyaml            # run_all_oracles.py and evaluate_results.py read oracles_config.yaml
pip install -e ".[docking]"   # only if an objective in Exp 3 is scored by AutoDock Vina
```

`pyyaml` is not declared in `pyproject.toml`, so it has to be installed separately even though two of the experiment scripts import it.

Experiments 3 and 4 need scoring models rather than a package. Nothing is downloaded: train a surrogate from your own labelled data with `src/graphinvent/tools/train-surrogate.py`, or prepare a receptor for docking, and point the configs at the result. The held-out metrics the training script prints are worth reading before a long run depends on the model, since a surrogate that cannot predict its own test set will still drive an RL loop and produce molecules that score well and mean nothing.

Data download and filtering instructions live with the experiments that need them: [chembl_pretrain/README.md](chembl_pretrain/README.md) for ChEMBL v34, [drd2_transfer/README.md](drd2_transfer/README.md) for the DRD2 actives.

### Importing GraphINVENT2 modules from your own scripts

The package is installed in editable mode, but the `.pth` file setuptools writes does not take effect in this checkout because the repository path contains spaces. `import metrics` and `import oracles` therefore fail unless `src/` is added to the path explicitly, and the console script `graphinvent-submit` fails for the same reason. Run everything as `python submit.py` from the repository root, and start any analysis script with:

```python
import sys
sys.path.insert(0, "src")
```

The experiment scripts under `experiments/` already do this; only ad-hoc snippets need the line added.

---

## Running the experiments

### The master script

```bash
bash experiments/run_all.sh
```

It accepts `--pretrain-checkpoint PATH` to reuse an existing ChEMBL checkpoint, `--dry-run` to print the commands without executing them, and `--skip-exp1` through `--skip-exp4` to leave individual experiments out.

```bash
bash experiments/run_all.sh \
    --pretrain-checkpoint ./output/chembl_v34/unconditional/run/model_restart_180.pth
```

The script **rewrites two tracked config files in place** — it sets `job.resume_from` in `drd2_transfer/transfer_params.json` and `conditional/train_params.json` to the checkpoint path. Because the checkpoint path defaults to `model_restart_200.pth` when the flag is omitted, this rewrite happens on every invocation, including with `--dry-run`. If you keep those configs under version control, expect a diff after each run.

The RL template is handled differently: `run_all.sh` passes the checkpoint to `run_all_oracles.py` as `--pretrained-model`, which overrides the template value at launch without editing the file.

### Individual experiments

Each experiment runs on its own once its dependency checkpoint exists.

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
    --out data/raw/chembl_v34_cond.tsv \
    --surrogate GSK3B=data/surrogates/gsk3b_rf.pkl
python submit.py --config experiments/conditional/preprocess_params.json
python submit.py --config experiments/conditional/train_params.json
python experiments/conditional/evaluate_conditional.py \
    --checkpoint output/chembl_v34_cond/conditional/run/model_restart_100.pth
```

### Parallel RL runs

The RL runs are independent, so `run_all_oracles.py` can be invoked several times with disjoint `--oracle` filters to spread them across GPUs:

```bash
python experiments/goal_directed/run_all_oracles.py --oracle target_a
python experiments/goal_directed/run_all_oracles.py --oracle target_a_selective
```

Each invocation runs its jobs sequentially, so the parallelism comes from running several invocations at once, each pinned to a different device.

---

## Aggregating results

```bash
python experiments/aggregate_results.py
```

It reads `goal_directed/results/pmo_results.csv` and `conditional/results/conditional_results.csv` and writes:

- `experiments/summary/pmo_summary.csv` — per-oracle AUC Top-10 and metric means across seeds
- `experiments/summary/cond_summary.csv` — conditional accuracy per property range
- `experiments/summary/tables.tex` — LaTeX tables

Experiment 2 is not aggregated. The script's docstring lists the DRD2 `convergence.log` among its inputs, but nothing in the code reads it, so transfer-learning numbers have to be pulled from that log by hand.

---

## SLURM submission

Every `params.json` carries a `slurm` block that is used only when `"use_slurm": true`; otherwise `submit.py` runs the job as a direct subprocess.

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

Experiment 3 is the one that really needs a scheduler, since a sweep of several targets and seeds at 12–24 hours per run will not fit in an interactive session.

---

## Checkpoint path convention

The ChEMBL checkpoint from Experiment 1 is referenced in three configs, and all three have to agree once you settle on a best epoch:

1. `experiments/drd2_transfer/transfer_params.json` → `resume_from`
2. `experiments/goal_directed/rl_params_template.json` → `pretrained_model_path`
3. `experiments/conditional/train_params.json` → `resume_from`

All three currently point at `model_restart_200.pth`, the last epoch of the pretraining schedule, which is a placeholder rather than a recommendation: the epoch with the lowest validation loss in `convergence.log` is usually earlier.

A checkpoint path is more than a path here, because `src/graphinvent/parameters/config.py` reads the pretrained run's `params_all.json` from the same directory to recover the GGNN architecture. That inheritance only applies to architecture keys **absent** from the job's own config block. Every key listed explicitly in a job config wins over the checkpoint's value, so a config that spells out `hidden_node_features`, `message_passes` and the MLP dimensions will build a model from those numbers and fail to load the checkpoint if they disagree with it. When fine-tuning, delete the architecture block rather than trusting it to be ignored.
