# Tutorial: Evaluation in GraphINVENT2

This document is the single reference for all metrics computed by GraphINVENT2.

---

## 1. Overview

GraphINVENT2 uses a **hierarchical evaluation framework**: each successive training setting computes all metrics from the simpler settings plus its own additions.

```
Unconditional generation
  └── + Success rate, conditional V.U.N., conditional diversity, conditional similarity
        (Conditional generation)
          └── + AUC Top-K, oracle-call tracking, optimization curve
                (Goal-directed / RL generation)
```

Metrics are written to `generation.log` (CSV) and, when `use_tensorboard: true`, to TensorBoard at every `sample_every` evaluation step.

---

## 2. Metrics by evaluation setting

### 2.1 Unconditional generation

Applies to `job_type: unconditional` (pre-training and transfer learning).

---

#### Validity

**What it measures:** The fraction of generated SMILES strings that RDKit can parse into a chemically valid molecule.

**Formula:**

    validity = n_valid / n_generated

**Interpretation:** 1.0 is perfect; values < 0.7 typically indicate the model has not converged or the vocabulary is misspecified. A newly initialised model starts near 0.

---

#### Uniqueness

**What it measures:** The fraction of valid molecules that are non-duplicate, measured by canonical SMILES.

**Formula:**

    uniqueness = |unique canonical SMILES among valid| / n_valid

**Interpretation:** 1.0 means every valid molecule is structurally distinct. Low uniqueness (< 0.5) indicates mode collapse — the model is repeatedly generating the same few structures.

---

#### Novelty

**What it measures:** The fraction of unique valid molecules not found verbatim in the training set.

**Formula:**

    novelty = |unique_valid ∩ complement(training_set)| / |unique_valid|

**Interpretation:** 1.0 means the model generates only molecules unseen during training. Very low novelty (< 0.3) suggests memorisation. Novelty is `None` when the training set is unavailable.

---

#### V.U.N.

**What it measures:** The joint score across validity, uniqueness, and novelty — the fraction of generated molecules that are simultaneously valid, unique, and novel.

**Formula:**

    vun = validity × uniqueness × novelty

**Interpretation:** The primary single-number summary for unconditional generation quality. A model that generates valid and diverse molecules while exploring beyond the training set will have a high V.U.N. score.

---

#### Internal diversity

**What it measures:** How chemically distinct the generated molecules are from one another. Follows the MOSES benchmark definition.

**Formula:**

    internal_diversity = 1 − mean(pairwise Tanimoto similarities)

where pairwise similarities are computed over the upper triangle (excluding diagonal) of the generated set using Morgan fingerprints (ECFP4, radius 2, 2048 bits). SMILES are canonicalised and deduplicated before fingerprinting.

**Interpretation:**
- 1.0 — all molecules are maximally different from each other
- 0.0 — all molecules are identical
- Values > 0.7 are typical for well-trained unconditional models on drug-like datasets.
- `n_duplicates_removed` is reported separately and is itself a diversity signal.

**Configurable parameters:**

| Parameter | Default | Description |
|---|---|---|
| `compute_internal_diversity` | `true` | Set to `false` to skip (saves O(n²) time). |
| `diversity_max_molecules` | `10000` | Cap on molecules used for pairwise computation. When exceeded, a random subsample of this size is drawn and a warning is printed. Set to `null` to disable. |

**Additional reported statistics:** `mean_internal_similarity`, `median_internal_similarity`, `max_internal_similarity`, and the fraction of pairs with similarity > 0.4 / 0.6 / 0.8 / 0.9. For RL jobs, `internal_diversity_successful` is also computed on the subset of molecules that exceed the success threshold.

---

#### Test-set similarity

**What it measures:** How close the generated molecules are to the held-out test set — a continuous version of the binary novelty check. For each valid generated molecule, the maximum Tanimoto similarity to any molecule in the test set is recorded (nearest-neighbour similarity).

**Formula:**

    nn_sim(g) = max_{t ∈ test_set} Tanimoto(fp(g), fp(t))

Aggregate statistics: mean, median, top-K mean, threshold fractions (> 0.4 / 0.6 / 0.8 / 0.9), and exact rediscovery count (nn_sim = 1.0).

**Interpretation:**
- High mean similarity (> 0.7) suggests the model is interpolating within the training distribution.
- Low mean similarity (< 0.3) suggests the model is exploring novel chemical space but may be generating unrealistic molecules.
- `exact_rediscovery_count` is the number of generated molecules that are identical to a test molecule (Tanimoto = 1.0).

**Configurable parameters:**

| Parameter | Default | Description |
|---|---|---|
| `compute_test_similarity` | `true` | Set to `false` to skip (useful for large test sets). |
| `test_similarity_max_refs` | `null` | Cap on test-set reference molecules. Deterministic stride-based subsampling. |
| `test_similarity_top_k` | `10` | K for the top-K mean similarity statistic. |

---

#### FCD (Fréchet ChemNet Distance)

**What it measures:** The distributional distance between the generated set and a reference set (training or test), computed in the latent space of ChemNet — a neural network trained on molecular property prediction. Analogous to the Fréchet Inception Distance used in image generation.

**Formula:** Modelled as multivariate Gaussians in ChemNet's penultimate-layer activations:

    FCD = ||μ_gen − μ_ref||² + Tr(Σ_gen + Σ_ref − 2(Σ_gen Σ_ref)^{1/2})

**Interpretation:** Lower FCD means the generated distribution is closer to the reference. FCD < 1 is excellent; FCD > 10 indicates a substantial distributional shift. Returns `None` if `fcd_torch` is not installed.

---

#### SA score (Synthetic Accessibility)

**What it measures:** Estimates how easy a molecule is to synthesise, on a scale from 1 (easy) to 10 (hard), based on fragment frequencies in a large database of known compounds (Ertl & Schuffenhauer, 2009).

**Reported statistics:** mean, median, and standard deviation across all valid generated molecules.

**Interpretation:** Drug-like molecules typically score between 2 and 5. Scores above 6 indicate complex or possibly unrealisable chemistry.

---

#### UC-JSD (Jensen–Shannon Divergence of NLL distributions)

**What it measures:** A convergence diagnostic that compares the model's NLL distribution on generated molecules to the NLL distribution on training molecules. A converged model assigns similar likelihoods to both.

**Formula:**

    UC-JSD = JSD(P_generated || P_training)

where distributions are over per-token negative log-likelihoods and JSD is the symmetric Jensen–Shannon divergence.

**Interpretation:** UC-JSD → 0 as training converges. Written to `validation.log`.

---

### 2.2 Conditional generation

Applies to `job_type: conditional`. Computes all unconditional metrics, plus:

---

#### Success rate

**What it measures:** The fraction of valid generated molecules that satisfy all target property criteria defined in `success_criteria`.

**Formula:**

    success_rate = |{mol ∈ valid_generated : passes_all_criteria(mol)}| / n_valid

**Criteria types** (configured via `SuccessCriterion`):

| Type | Condition |
|---|---|
| `threshold` | `property > value` (direction: `"greater"`) or `property < value` (direction: `"less"`) |
| `range` | `min ≤ property ≤ max` |
| `target` | `|property − value| ≤ tolerance` |

---

#### Conditional V.U.N.

**What it measures:** V.U.N. restricted to molecules that pass all target criteria.

**Formula:**

    conditional_vun = success_rate × uniqueness × novelty

---

#### Conditional internal diversity

**What it measures:** Internal diversity (1 − mean pairwise Tanimoto) computed only on the subset of generated molecules that pass all target criteria. This answers "are the successful molecules themselves diverse, or does the model find one solution and repeat it?"

Computed and logged as `internal_diversity_successful` alongside the full-set `internal_diversity`.

---

#### Conditional test-set similarity

**What it measures:** Nearest-neighbour Tanimoto similarity against the subset of the test set whose property values match the target condition within the specified tolerance (default ±0.3).

**How it works:** When `sample_conditions` is set in the job config (e.g. `{"pLogS": -1.5}`), the test set is automatically filtered to molecules whose condition values fall within `tolerance` of the target before comparison. Property names are read from the tab-separated header of `test.smi` produced during preprocessing.

---

### 2.3 Goal-directed generation (RL)

Applies to `job_type: goal_directed`. Computes all conditional and unconditional metrics, plus:

---

#### Sample efficiency / AUC Top-K

**What it measures:** The area under the top-K average score curve, normalised by oracle budget. At each oracle call `t`, the running top-K scores are maintained and their mean `f(t)` is recorded. The AUC is the integral of this curve from 0 to the budget, divided by the budget. Follows the PMO benchmark (Gao et al., 2022).

**Formula:**

    AUC_top_K = (1 / budget) ∫₀^budget f(t) dt  ≈  (1 / T) Σ_{t=1}^{T} mean(top-K scores up to t)

For constrained RL, only molecules that pass all constraints count toward the top-K scores.

**Interpretation:** Higher AUC Top-K means the model found high-scoring molecules earlier in the optimisation. Gao et al. recommend K = 10, budget = 10 000, 5 independent runs, reported as mean ± std.

---

#### Oracle call tracking

GraphINVENT2 deduplicates oracle calls: if the same SMILES is queried again, the cached score is returned without incrementing the oracle counter. The total oracle call count is logged at each RL step.

---

#### Optimization curve

The top-K average score vs. oracle calls is logged to TensorBoard at each step (`Evaluation/score`). This curve shows how quickly the policy learns to propose high-scoring molecules.

---

## 3. How to run evaluation

### Using the `generate` job type

Set `sample_mode: evaluate` to evaluate a trained model on the test set (computes NLL / UC-JSD):

```json
{
  "job": {
    "job_type": "generate",
    "sample_mode": "evaluate",
    "generation_epoch": 100
  }
}
```

Set `sample_mode: generate` to sample molecules and compute all generation metrics:

```json
{
  "job": {
    "job_type": "generate",
    "sample_mode": "generate",
    "n_samples": 10000,
    "generation_epoch": 100
  }
}
```

Run:

```bash
python submit.py --config jobs/generate/params.json
```

Results are written to `output/<dataset>/generate/<job_name>/generation/` and `generation.log`.

---

### Standalone usage of `src/metrics/`

The evaluation functions in `src/metrics/` are fully independent of GraphINVENT2's training pipeline and can be used directly:

```python
from metrics import evaluate_unconditional, compute_internal_diversity, compute_test_set_similarity

# Load your SMILES
generated = [...]   # list of generated SMILES strings
test_set   = [...]  # list of held-out test set SMILES strings
train_set  = set([...])  # set of training SMILES strings

# Unconditional metrics
results = evaluate_unconditional(
    generated,
    reference_mols=test_set,
    training_smiles=train_set,
    include_fcd=True,
)
print(results)

# Internal diversity
div = compute_internal_diversity(generated, max_mols=5000)
print(f"Internal diversity: {div['internal_diversity']:.3f}")

# Test-set nearest-neighbour similarity
sim = compute_test_set_similarity(generated, test_set, top_k=10)
print(f"Mean NN similarity: {sim['mean_similarity']:.3f}")
```

For conditional evaluation:

```python
from metrics import evaluate_conditional, SuccessCriterion

criteria = [
    SuccessCriterion(property="qed", type="threshold", value=0.6, direction="greater"),
    SuccessCriterion(property="sa_score", type="threshold", value=4.0, direction="less"),
]
results = evaluate_conditional(generated, test_set, criteria, training_smiles=train_set)
print(f"Success rate: {results['success_rate']:.3f}")
```

---

### Reading results

| File | Contents |
|---|---|
| `generation.log` | One row per evaluation epoch/step; all scalar metrics as CSV columns. |
| `oracle_eval.csv` | One row per oracle-call milestone (constrained RL only). |
| `validation.log` | NLL and UC-JSD per epoch (evaluate mode). |
| `convergence.log` | Training and validation loss per epoch. |
| `progress.png` | Auto-generated plot of key metrics over training. |
| `tensorboard/` | Full metric time-series; view with `tensorboard --logdir output/.../tensorboard/`. |

---

## 4. Configurable evaluation parameters

| Parameter | Default | Description |
|---|---|---|
| `compute_internal_diversity` | `true` | Compute pairwise fingerprint diversity within the generated set. |
| `diversity_max_molecules` | `10000` | Subsample cap for pairwise diversity; `null` = no limit. |
| `compute_test_similarity` | `true` | Compute nearest-neighbour similarity to the test set. |
| `test_similarity_max_refs` | `null` | Subsample cap for test-set references. |
| `test_similarity_top_k` | `10` | K for top-K mean similarity statistic. |
| `sample_every` | `10` | Frequency (in epochs/steps) at which generation and evaluation run. |
| `n_samples` | `100` | Number of molecules generated per evaluation step. |
| `success_threshold` | `0.5` | Score threshold for success rate in RL evaluation. |

---

## 5. Worked examples

### Unconditional: evaluating a pretrained model on ZINC

1. Preprocess ZINC into HDF5 format (see Tutorial 01).
2. Train with `job_type: unconditional`. Molecules are evaluated every `sample_every` epochs; results appear in `generation.log`.
3. After training, inspect:

```bash
# Open the progress plot
open output/ZINC/unconditional/run1/progress.png

# Check the latest epoch's metrics
tail -1 output/ZINC/unconditional/run1/generation.log
```

Key columns to check: `fraction_valid`, `fraction_unique`, `novelty`, `internal_diversity`, `sim_mean` (should be < 0.5 for a generative model exploring novel space).

---

### Conditional: evaluating molecules sampled at pLogS = −1.5

1. Preprocess with a tab-separated TSV containing a `pLogS` column (see Tutorial 05).
2. Train with `job_type: conditional`, `condition_dim: 1`.
3. Sample with:

```json
{
  "job": {
    "job_type": "generate",
    "sample_mode": "generate",
    "condition_dim": 1,
    "sample_conditions": {"pLogS": -1.5},
    "n_samples": 1000,
    "generation_epoch": 100
  }
}
```

Evaluation automatically:
- Computes `success_rate` using any `success_criteria` you define.
- Filters the test set to molecules with pLogS within ±0.3 of −1.5 before computing `sim_mean`.
- Logs `internal_diversity_successful` alongside the overall `internal_diversity`.

---

### Goal-directed: assessing DRD2 + SA optimisation

After an RL run targeting DRD2 activity and SA score:

1. Open `oracle_eval.csv` to see how validity, success rate, diversity, and FCD evolved at each oracle-call milestone (1K, 3K, 10K calls).
2. For AUC Top-10, the running top-10 scores are tracked throughout training; the area under that curve is written to `oracle_eval.csv` as `auc_top_10`.
3. For diversity of the top hits:

```python
import csv
from metrics import compute_internal_diversity

# Load the top-scoring generated SMILES from the final checkpoint
smiles = [row["smiles"] for row in csv.DictReader(open("top_hits.csv"))]
div = compute_internal_diversity(smiles)
print(f"Top-hit internal diversity: {div['internal_diversity']:.3f}")
print(f"Most similar pair: {div['max_internal_similarity']:.3f}")
```

Following PMO benchmark best practice (Gao et al., 2022): report results at 10K oracle calls over 5 independent seeds as mean ± std.
