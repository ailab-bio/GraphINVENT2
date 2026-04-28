# Experiment 3: Goal-Directed Optimization (PMO Benchmark)

Evaluate GraphINVENT2's reinforcement learning optimizer on the Practical Molecular Optimization (PMO) benchmark (Gao et al., NeurIPS 2022).

---

## Purpose

The PMO benchmark measures **sample efficiency**: how quickly an optimizer finds high-scoring molecules under a fixed oracle-call budget (10,000 unique evaluations).  The primary metric is **AUC Top-10**: the area under the curve of the top-10 average oracle score versus oracle calls, normalised to [0, 1].

This experiment runs GraphINVENT2's goal-directed mode (augmented log-likelihood RL with best-agent-so-far replay) on six oracle tasks and three random seeds, matching the PMO evaluation protocol.

---

## Dependencies

```bash
pip install -e ".[tdc]"   # required for TDC oracle access
```

Surrogate models (DRD2, GSK3B, JNK3, SA) are downloaded automatically by TDC to `data/surrogates/` on first use.

---

## Oracle tasks

| Oracle | Task type | Success threshold | Description |
|--------|-----------|-------------------|-------------|
| `GSK3B` | Bioactivity | ≥ 0.5 | GSK3β inhibition (RF on ECFP6) |
| `JNK3` | Bioactivity | ≥ 0.5 | JNK3 inhibition (RF on ECFP6) |
| `DRD2` | Bioactivity | ≥ 0.5 | DRD2 activity (SVM on ECFP6) |
| `SA` | Synthesizability | ≥ 0.67 | SA score (RDKit, normalised; ≥ 0.67 ≈ raw SA ≤ 3.0) |
| `QED` | Drug-likeness | ≥ 0.9 | QED composite score (RDKit) |
| `tdc:LogP` | Physicochemical | none | LogP lipophilicity (TDC normalised) |

See `oracles_config.yaml` for full descriptions and TDC oracle names.

---

## Step 1 — Ensure the pretrained checkpoint exists

Update the `pretrained_model_path` in `rl_params_template.json` to point at the best ChEMBL checkpoint from Experiment 1:

```bash
# Edit rl_params_template.json and replace:
#   "pretrained_model_path": "./output/chembl_v34/unconditional/run/model_restart_200.pth"
# with the actual best epoch path.
```

Or pass it at runtime via `--pretrained-model`:

```bash
python experiments/goal_directed/run_all_oracles.py \
    --pretrained-model ./output/chembl_v34/unconditional/run/model_restart_180.pth
```

---

## Step 2 — Run all oracle experiments

```bash
python experiments/goal_directed/run_all_oracles.py
```

This generates a resolved `params.json` for each (oracle, seed) combination in `experiments/goal_directed/configs/` and launches `submit.py` for each.  A total of 6 oracles × 3 seeds = **18 runs**.

To run a subset (e.g. only DRD2 and GSK3B, one seed):

```bash
python experiments/goal_directed/run_all_oracles.py \
    --oracle DRD2 --oracle GSK3B \
    --seed 42
```

To preview what would run without executing:

```bash
python experiments/goal_directed/run_all_oracles.py --dry-run
```

**Expected output directories:**
```
output/chembl_v34/goal_directed/DRD2_seed42/
output/chembl_v34/goal_directed/DRD2_seed123/
output/chembl_v34/goal_directed/DRD2_seed456/
output/chembl_v34/goal_directed/GSK3B_seed42/
...
```

**Expected runtime:** 12–24 hours per run on a single GPU (A100 recommended).  All 18 runs can be parallelised across GPUs by running `run_all_oracles.py` multiple times with different `--oracle` and `--seed` filters.

---

## RL training details

The goal-directed training loop:

1. **Initialise:** copy the ChEMBL pretrained model as both the agent and the frozen prior.
2. **Generate:** sample a batch of molecules from the agent.
3. **Score:** call the TDC oracle (cached; only new unique SMILES count toward the budget).
4. **AugLL loss:** compute `(log p_agent - log p_prior - σ·score)²` and back-propagate.
5. **BASF update:** if the agent's mean reward improves, update the best-agent-so-far (BASF) model.
6. **Replay:** mix BASF-generated molecules into the next training batch.
7. **Checkpoint:** save model and evaluation SMILES at 1K, 3K, 5K, 10K oracle calls.
8. **Stop:** terminate when the oracle call count reaches `oracle_budget=10000`.

Key RL parameters (see `rl_params_template.json`):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `sigma` | 20 | Scales the reward term in AugLL; 20 is the REINVENT default |
| `alpha` | 0.5 | Weight of BASF replay vs fresh agent samples |
| `batch_size` | 64 | Small batch keeps oracle calls per step manageable |
| `score_type` | `continuous` | Raw oracle score used as reward (not binarised) |

---

## Step 3 — Evaluate results

```bash
python experiments/goal_directed/evaluate_results.py
```

This script reads the optimization log and checkpoint `.smi` files for each run, computes all metrics, and writes:
- `experiments/goal_directed/results/pmo_results.csv` — per-checkpoint metrics for every run
- `experiments/goal_directed/results/pmo_table.tex` — LaTeX table of AUC Top-10 mean ± std

---

## Metrics

At each oracle-call checkpoint (1K, 3K, 5K, 10K):

| Metric | Description |
|--------|-------------|
| **AUC Top-10** | Primary PMO metric: area under the top-10-average-score curve, normalised to [0, 1] |
| Top-1 / Top-10 / Top-100 | Mean of the best 1 / 10 / 100 unique oracle scores seen so far |
| Validity | Fraction of generated SMILES that parse correctly |
| Uniqueness | Fraction of valid SMILES that are unique |
| Novelty | Fraction of unique SMILES not in the ChEMBL training set |
| Internal diversity | Mean pairwise Tanimoto distance (ECFP4) of the top-100 molecules |
| SA score | Mean raw SA score of generated molecules (RDKit) |
| Success rate | Fraction of molecules above the oracle-specific threshold |

---

## How to interpret the outputs

- **AUC Top-10** is the single number to compare against baselines in the PMO paper.  Higher is better.  Published REINVENT results are approximately 0.4–0.6 depending on the oracle.
- **Budget curves** (Top-10 score vs oracle calls) reveal whether the optimizer finds good molecules early (high efficiency) or only near the budget limit.
- **Diversity of top-100** distinguishes a model that finds many good molecules vs one that exploits a single high-scoring scaffold.

---

## Adding new oracles

To add an oracle not listed in `oracles_config.yaml`, add an entry with the TDC oracle name and a threshold.  Any TDC oracle can be referenced as `tdc:<tdc_oracle_name>` in `score_components`.

To add a custom (non-TDC) oracle, implement it as a `BaseOracle` subclass in `src/oracles/` and register it in the `ScoringFunction` (see `src/graphinvent/ScoringFunction.py` for examples).
