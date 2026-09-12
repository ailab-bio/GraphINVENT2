# Evaluation in GraphINVENT2

This is the reference for the metrics GraphINVENT2 computes, what each one does and does not
tell you, and which of them a given job type actually writes to disk.

That last distinction matters more than it might seem. GraphINVENT2 contains two evaluation
paths that are easy to confuse: the **in-pipeline** metrics, computed by `Analyzer` during
training, generation and RL and appended to `generation.log`; and the **standalone** functions
in `src/metrics/`, which implement a wider set including V.U.N., FCD, and success criteria but
which the training pipeline does not call. If you want the second set you have to run it
yourself against a generated SMILES file. Each section below says which path a metric belongs
to.

---

## 1. What each job type writes

| Job type | In-pipeline metrics written to `generation.log` |
|----------|--------------------------------------------------|
| `unconditional` | Validity, proper termination, uniqueness, novelty, SA scores, internal diversity, test-set similarity, property histograms. UC-JSD goes to `validation.log`. |
| `conditional` | The same set. No conditional-specific metric is computed automatically; the test-set similarity is however filtered to molecules matching `sample_conditions`. |
| `generate` | The same set, one row. Computed from the first generation batch only, not from the full `n_samples` sample. |
| `goal_directed` | The same set plus `success_rate`, `internal_diversity_successful`, and one `score_<component>_mean` per scoring component. Mean score also goes to `score.log`. |
| `goal_directed` with `oracle_budget` | Additionally, `oracle_eval.csv` with a full metrics row per milestone checkpoint, including diversity, FCD, and rediscovery rate. |

FCD and V.U.N. are **not** computed on the ordinary path. `Analyzer._compute_extended_metrics`
takes an `include_expensive` flag that is left false everywhere except the oracle-milestone
evaluation, so FCD appears only in `oracle_eval.csv`, and V.U.N. only if you call
`evaluate_unconditional` yourself.

---

## 2. Distribution-level metrics

### Validity

The fraction of generated graphs that RDKit can sanitise.

    validity = n_valid / n_generated

A freshly initialised model is near 0 and a converged one on drug-like data is usually above
0.7. Invalid graphs are written into the `.smi` output as the placeholder `[Xe]`, which RDKit
parses successfully as a xenon atom, so validity recomputed by counting successful
`MolFromSmiles` calls on a raw output file will be far too high; use the `.valid` file. Persistently low validity late in training points at the vocabulary or the bond encoding
rather than at undertraining — in particular, `use_aromatic_bonds: true` lets the model emit
aromatic systems that fail sanitisation, and Kekulé encoding lets it emit rings with the wrong
single/double parity.

Validity is a floor, not a result. It measures whether RDKit accepts the output, which is a
much weaker property than the molecule being reasonable.

### Proper termination

`fraction_pt` is the share of graphs that ended because the model drew the terminate action;
the rest were cut off at `max_n_nodes` or by an invalid action. `fraction_valid_pt` combines
both conditions.

This pair deserves attention because a model can post high validity while almost never choosing
to stop, in which case the size distribution of its output is set by `max_n_nodes` rather than
by anything the model learned, and the molecules are truncated fragments that happen to
sanitise.

### Uniqueness

    uniqueness = |distinct canonical SMILES among valid| / n_valid

Uniqueness below about 0.5 is mode collapse. Note that it is computed within a single generated
batch, so it depends on `n_samples`: a small sample will show high uniqueness even from a model
with a narrow distribution, because it has not drawn enough molecules to repeat itself.

### Novelty

    novelty = |unique valid molecules not in the training set| / |unique valid|

Novelty compares canonical SMILES exactly, which makes it a blunt instrument: a generated
molecule differing from a training molecule by one methyl counts as fully novel. It is returned
as `None` when the training set is unavailable. Very low novelty indicates memorisation; high
novelty on its own indicates nothing, since a model producing implausible structures is
maximally novel.

The continuous version of this question is test-set similarity, below.

### V.U.N. (standalone only)

    vun = validity × uniqueness × novelty

A single-number summary of the three above, returned by `evaluate_unconditional` in
`src/metrics/`. It is not written to `generation.log`. Treating it as the score for a
generative model is a mistake this codebase deliberately avoids making automatically: the
product is maximised by a model producing valid, distinct structures unrelated to anything in
the training data, which is exactly what a poorly-trained model does.

### Internal diversity

How different the generated molecules are from each other, following the MOSES definition:

    internal_diversity = 1 − mean(pairwise Tanimoto similarity)

computed over the upper triangle of the pairwise matrix using Morgan fingerprints (ECFP4,
radius 2, 2048 bits). SMILES are canonicalised and deduplicated first, and the number of
duplicates removed is reported separately as a diversity signal in its own right.

Values above roughly 0.7 are typical for a well-trained unconditional model on drug-like data.
A set with fewer than two unique valid molecules returns 0.0 with a warning.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `compute_internal_diversity` | `true` | Set false to skip the O(n²) comparison |
| `diversity_max_molecules` | `10000` | Random subsample cap when the generated set is larger; a warning is printed. `null` disables the cap |

`generation.log` carries `internal_diversity`, `mean_internal_similarity`, and
`max_internal_similarity`. The function also returns the median and the fraction of pairs above
0.4/0.6/0.8/0.9 if you call it directly. For goal-directed jobs,
`internal_diversity_successful` repeats the calculation over only the molecules scoring above
`success_threshold`, which is the number that distinguishes an optimiser that found many good
molecules from one that found one and copied it.

### Test-set similarity

For each valid generated molecule, the maximum Tanimoto similarity to any molecule in the test
set:

    nn_sim(g) = max over test molecules t of Tanimoto(fp(g), fp(t))

Reported as `sim_mean`, `sim_median`, `sim_top<K>`, the fractions above 0.4/0.6/0.8/0.9, and
`exact_rediscovery_count` — the number of *distinct test molecules* reproduced exactly,
determined by canonical-SMILES identity rather than by a fingerprint similarity of 1.0, since
distinct molecules can share a fingerprint.

Interpretation is two-sided and neither end is good. A mean above about 0.7 says the model is
interpolating within the region it was trained on; a mean below about 0.3 says the samples are
unrelated to the reference set, which is as consistent with generating nonsense as with
exploring usefully. The distribution of `nn_sim` across the generated set is more informative
than its mean, because a bimodal distribution — some near-copies, some far-out structures — has
the same mean as a uniformly mediocre one.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `compute_test_similarity` | `true` | Set false to skip |
| `test_similarity_max_refs` | `null` | Deterministic stride-based subsample of the test set; `null` uses all |
| `test_similarity_top_k` | `10` | K for `sim_top<K>` |

When `sample_conditions` is set, the test set is filtered to molecules whose condition values
lie within ±0.3 of the target before the comparison, so a conditional run is compared against
the relevant slice of the test set rather than all of it. The tolerance is hard-coded.

### SA score

Synthetic accessibility on the Ertl & Schuffenhauer scale, 1 easy to 10 hard, estimated from
fragment frequencies in a database of known compounds. `sa_score_mean`, `sa_score_median`, and
`sa_score_std` are reported over the valid molecules.

Drug-like molecules typically fall between 2 and 5. The score is a fragment-frequency heuristic
and not a retrosynthetic assessment: it penalises unusual substructures whether or not they are
actually hard to make, and it will happily give a low score to a molecule no route reaches. It
is useful as a filter against obviously unrealisable output and misleading as a synthesizability
claim.

### FCD (oracle-milestone evaluation only)

Fréchet ChemNet Distance between the generated and reference distributions, computed as
multivariate Gaussians in ChemNet's penultimate-layer activations:

    FCD = ||μ_gen − μ_ref||² + Tr(Σ_gen + Σ_ref − 2(Σ_gen Σ_ref)^{1/2})

Lower is closer. Below 1 is a close match; above 10 is a substantial distributional shift.
Requires `fcd_torch`; returns `None` if it is not installed. It appears in `oracle_eval.csv` and
nowhere else on the automatic path, and is available through the `include_fcd` argument of the
standalone `evaluate_*` functions.

### UC-JSD

The Jensen–Shannon divergence between the per-molecule NLL distribution of the generated set
and that of the training set, written to `validation.log` as `uc_jsd` and echoed into
`convergence.log` as `model_score`.

It approaches 0 as training converges, which makes it a convergence diagnostic rather than a
quality measure: a model that has memorised the training set scores well on it, and so does a
model whose samples happen to have training-like likelihoods for the wrong reasons.

---

## 3. Success criteria and conditional metrics

Two different notions of "success" exist in this codebase and they are not connected.

**In the pipeline**, `success_rate` is the fraction of generated molecules whose RL score
exceeds `success_threshold` (default 0.5). It is computed only when a score tensor exists,
which means only for `goal_directed` jobs. A conditional job produces no `success_rate` column.

**In `src/metrics/`**, `SuccessCriterion` describes a property-based criterion and
`evaluate_conditional` computes a `success_rate` and a `conditional_vun` from a list of them.

| Type | Condition |
|------|-----------|
| `threshold` | `property > value` with `direction="greater"`, or `< value` with `"less"` |
| `range` | `min ≤ property ≤ max` |
| `target` | `|property − value| ≤ tolerance` |

`property` must name a key in `PROPERTY_REGISTRY` — currently `qed`, `sa_score`, `mol_weight`,
and `logp` — or you must pass the function through the `property_fns` argument.

`src/metrics/_criteria.py` also provides `load_criteria_from_config`, which reads
`config["job"]["success_criteria"]` from a params dict. Nothing in `src/graphinvent/` calls it,
`success_criteria` is not a key in `defaults.py`, and putting it in a job config has no effect
on a training or generation run. Use the criteria system by calling `evaluate_conditional`
directly on a generated SMILES file, as shown below.

---

## 4. Sample efficiency for goal-directed runs

### AUC Top-K

The PMO benchmark's primary metric (Gao et al., 2022). At each oracle call *t* the mean of the
top *k* scores seen so far is recorded, dividing by *k* even when fewer than *k* molecules
exist so that early calls are penalised; AUC Top-K is the area under that curve divided by the
budget, so it lies in [0, 1]. Under a constrained variant, only molecules satisfying all
constraints count toward the top *k*.

It is implemented as `compute_auc_top_k` in `src/oracles/_auc.py` and is **not** computed
automatically. Nothing in `src/graphinvent/` calls it, `oracle_eval.csv` has no `auc_top_10`
column, and the `CachedOracle.optimization_log` it needs is not persisted by the training loop.
Computing it currently means driving a `CachedOracle` yourself, or reconstructing the curve
from the per-milestone `.smi` files.

AUC Top-K measures how quickly high scores are reached, which is the right question when oracle
calls are the binding constraint. It says nothing about whether the top *k* molecules are
distinct from one another, synthesisable, or plausible, and a run can score well by exploiting
a region the surrogate model overrates. Report it beside the diversity and validity columns.

### Oracle call counting

`CachedOracle` returns a cached score for a repeated SMILES without incrementing its own
counter, so its `call_count` reflects unique evaluations. The budget loop in
`Workflow.constrained_rl_training_phase`, however, increments its own counter by `batch_size`
at every step regardless of validity or duplication. A nominal 10 000-call budget therefore
corresponds to fewer than 10 000 distinct molecules evaluated, and is not directly comparable
to a PMO figure without accounting for that.

---

## 5. Running the evaluation

### Through the `generate` job

Sampling and computing the generation metrics:

```json
{
  "job": {
    "job_type": "generate",
    "sample_mode": "generate",
    "n_samples": 10000,
    "pretrained_model_path": "./output/debug/unconditional/run/model_restart_100.pth"
  }
}
```

Computing NLL and UC-JSD on the test set instead:

```json
{
  "job": {
    "job_type": "generate",
    "sample_mode": "evaluate",
    "pretrained_model_path": "./output/debug/unconditional/run/model_restart_100.pth"
  }
}
```

```bash
python submit.py --config jobs/generate/params.json
```

`sample_mode: evaluate` loads `train.h5`, `valid.h5`, and `test.h5`, so the dataset must be
preprocessed and present. Results land in `output/<dataset>/generate/<job_name>/`.

### Directly, through `src/metrics/`

The functions in `src/metrics/` do not depend on the training pipeline. The editable install
does not reliably put `src/` on the import path in this repository, so add it explicitly and
run from the repository root:

```python
import sys
sys.path.insert(0, "src")

from metrics import (
    evaluate_unconditional,
    compute_internal_diversity,
    compute_test_set_similarity,
)

generated = [...]        # list of generated SMILES
test_set  = [...]        # list of held-out test SMILES
train_set = {...}        # set of training SMILES

results = evaluate_unconditional(
    generated,
    test_set,
    training_smiles=train_set,
    include_fcd=True,
)
# keys: validity, uniqueness, novelty, vun, diversity, sa_mean, sa_median, sa_std, fcd
print(results)

div = compute_internal_diversity(generated, max_mols=5000)
print(f"Internal diversity: {div['internal_diversity']:.3f}")

sim = compute_test_set_similarity(generated, test_set, top_k=10)
print(f"Mean NN similarity: {sim['mean_similarity']:.3f}")
```

Conditional evaluation with explicit criteria:

```python
import sys
sys.path.insert(0, "src")

from metrics import evaluate_conditional, SuccessCriterion

criteria = [
    SuccessCriterion(property="qed", type="threshold", value=0.6, direction="greater"),
    SuccessCriterion(property="sa_score", type="threshold", value=4.0, direction="less"),
]
results = evaluate_conditional(generated, test_set, criteria, training_smiles=train_set)
print(f"Success rate: {results['success_rate']:.3f}")
print(f"Conditional V.U.N.: {results['conditional_vun']}")
```

---

## 6. Where results are written

| File | Contents |
|------|----------|
| `generation.log` | One row per evaluation epoch or step; all scalar metrics as CSV columns, plus normalised property histograms |
| `validation.log` | Per-molecule NLL on the validation, training, and generated sets, and UC-JSD |
| `convergence.log` | Learning rate and losses per epoch (or per step, for RL) |
| `score.log` | `Step, Score` for goal-directed jobs |
| `oracle_eval.csv` | One row per oracle milestone; goal-directed jobs with `oracle_budget` only |
| `progress.png` | Nine-panel plot regenerated from `generation.log` and `convergence.log` at every evaluation |
| `tensorboard/` | Scalar time series when `use_tensorboard` is true |

`oracle_eval.csv` columns: `oracle_count`, `fraction_valid`, `fraction_unique`, `novelty`,
`sa_score_mean`, `sa_score_median`, `sa_score_std`, `success_rate`, `diversity`, `fcd`,
`rediscovery_rate`, and `score_<component>_mean` for each scoring component.

---

## 7. Configurable evaluation parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compute_internal_diversity` | `true` | Pairwise fingerprint diversity within the generated set |
| `diversity_max_molecules` | `10000` | Random subsample cap for that comparison; `null` disables |
| `compute_test_similarity` | `true` | Nearest-neighbour similarity against the test set |
| `test_similarity_max_refs` | `null` | Stride-based subsample cap on test references |
| `test_similarity_top_k` | `10` | K for the top-K mean similarity statistic |
| `sample_every` | `10` | Epochs or steps between evaluations |
| `n_samples` | `2000` | Molecules generated per evaluation |
| `success_threshold` | `0.5` | RL score above which a molecule counts as successful |
| `eval_sample_size` | `30000` | Molecules generated per oracle-milestone checkpoint |
| `checkpoint_oracle_counts` | `[1000, 3000, 10000]` | Oracle-call milestones at which to checkpoint and evaluate |

Note that `n_samples` defaults to 2000 but the shipped job templates set it to 100, which is
enough to confirm the pipeline runs and far too few for uniqueness, diversity, or similarity
statistics to be stable.

---

## 8. Worked examples

### An unconditional model

Train with `job_type: unconditional`; molecules are sampled and evaluated every `sample_every`
epochs.

```bash
open output/ZINC/unconditional/run/progress.png
tail -1 output/ZINC/unconditional/run/generation.log
```

The columns worth reading together are `fraction_valid_pt` (is the model deciding to stop?),
`fraction_unique` and `internal_diversity` (is it exploring?), `novelty` and `sim_mean` (is it
copying?), and the property histograms (is it in the right region at all?). Any one of them
read alone can be satisfied by a model that is failing in a way the others would expose.

### A conditional model

Sample at a target and compute the property yourself:

```json
{
  "job": {
    "job_type": "generate",
    "sample_mode": "generate",
    "n_samples": 1000,
    "pretrained_model_path": "./output/chembl_cond/conditional/run/model_restart_100.pth",
    "conditioning": {"properties": ["pLogS"], "source": "smiles_file"},
    "sample_conditions": {"pLogS": -1.5}
  }
}
```

The pipeline will filter the test-set similarity comparison to molecules with pLogS within ±0.3
of −1.5 and will log `internal_diversity`, but it will not tell you whether the generated
molecules have the requested pLogS. Compute the property on the output and compare its
distribution against the target, and against a second run at a well-separated target — a model
ignoring its condition produces the same distribution for both.

### A goal-directed run

After an RL run, `oracle_eval.csv` gives validity, success rate, diversity, and FCD at each
oracle milestone, and `score.log` gives the optimisation trace. For the diversity of the top
hits:

```python
import sys
sys.path.insert(0, "src")

import csv
from metrics import compute_internal_diversity

smiles = [row["smiles"] for row in csv.DictReader(open("top_hits.csv"))]
div = compute_internal_diversity(smiles)
print(f"Top-hit internal diversity: {div['internal_diversity']:.3f}")
print(f"Most similar pair: {div['max_internal_similarity']:.3f}")
```

Following PMO practice, report at a fixed oracle budget over at least five seeds as mean ± std,
set with `"seed": N` for N > 0. Report the diversity of the successful subset alongside the
score: a mean score that rises while `internal_diversity_successful` falls describes an
optimiser that has found one exploit, and reporting only the former hides that.
