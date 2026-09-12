# Tutorial 4: Reinforcement learning

Goal-directed generation takes a trained model and moves its distribution toward molecules that
score well on a user-supplied objective, while penalising how far it drifts from the original
model. The job type is `goal_directed`; the same job type covers both the unconstrained variant
and the oracle-budget-capped variant, selected by whether `oracle_budget` is set.

The objective is the augmented log-likelihood used in REINVENT (Olivecrona et al. 2017;
Blaschke et al. 2020; Atance et al. 2021):

```
loss = ( log p_agent(x) - log p_prior(x) - σ · score(x) )²
```

`p_agent` is the model being updated, `p_prior` is a frozen copy of the starting model, and
`score(x)` lies in [0, 1]. The squared term is minimised when the agent assigns molecule *x* a
log-likelihood exactly `σ · score(x)` above the prior's, so the agent is pushed to upweight
high-scoring molecules by an amount proportional to their score and to leave everything else
where the prior had it. The prior is what stops the run from collapsing onto whatever
degenerate structure maximises the score — without it the optimiser has no reason to keep
producing molecules at all.

A best-agent-so-far (BASF) copy is also kept. Its loss is mixed into the total with weight
`alpha`, so the update is `(1 - alpha) · loss_agent + alpha · loss_BASF`. The BASF model is
replaced whenever an evaluation step produces a mean score above the best seen so far.

---

## Prerequisites

1. A checkpoint from [Tutorial 2](./02_pretraining.md) or [Tutorial 3](./03_transfer_learning.md),
   with its `params_all.json` in the same directory.
2. A preprocessed dataset with the matching vocabulary. It is needed for the reference property
   distributions and for `preprocessing_params.json`, even though RL itself trains on the
   agent's own samples rather than on the dataset.

---

## Scoring

Each generated molecule gets a score in [0, 1] built from one or more components listed in
`score_components`.

| Component | What it computes |
|-----------|------------------|
| `"QED"` | RDKit's quantitative estimate of drug-likeness |
| `"target_size=N"` | `1 - |n_nodes - N| / (max_n_nodes - N)`; N must satisfy `0 < N < max_n_nodes`, since N equal to `max_n_nodes` divides by zero |
| `"logp_target=X"` | `1 - |logP - X| / 5`, clamped at 0; drives Crippen logP toward a value rather than maximising it |
| `"<name>_activity"` | `predict_proba` of a scikit-learn classifier named by `<name>` in `qsar_models` |
| any name declared in the `oracles` block | Whatever that oracle computes: a surrogate you trained, a Python callable, or an AutoDock Vina docking run |

Those first four are built in and computed directly from the molecule. Everything else has to
be declared as an oracle, described [below](#oracles) and in depth in
[Tutorial 6](./06_custom_oracles.md). A component that is neither built in nor declared raises
a `ValueError` naming it when `ScoringFunction` is constructed, before any molecules are
generated, rather than part-way into the first training step.

The final score is masked to zero for molecules that are invalid, duplicated within the batch,
or force-terminated. This means the reward already contains an implicit validity and
uniqueness objective, and a rising mean score can reflect improving validity rather than
improving chemistry — a reason to read `score.log` alongside the component means in
`generation.log` rather than on its own.

### Combining components

| `score_type` | Behaviour |
|--------------|-----------|
| `"binary"` | 1 if every component is strictly above its `score_thresholds` entry, else 0 |
| `"continuous"` | Product of the component scores; thresholds are ignored |

With exactly one component listed, `"continuous"` passes that component's score through
unchanged and its threshold is ignored, while `"binary"` still applies the threshold and turns
the score into 0 or 1.

The two differ in what gradient signal they provide. Binary scoring gives no credit for
partial progress, which forces the agent to satisfy every criterion at once but leaves it with
a flat reward until it does. The product is smooth but lets a high score on one objective
compensate for a poor one on another, which is usually not what a multi-objective design
problem means. Neither is obviously right; the choice depends on whether the criteria are
genuinely conjunctive.

`submit.py` requires `score_components` and `score_thresholds` in the job block for
`goal_directed` jobs, and rejects the config if their lengths differ.

### QSAR models

An `"*_activity"` component needs a pickled dict with the key `classifier_sv` holding a
scikit-learn classifier that implements `predict_proba` and accepts a 2048-bit ECFP4 (Morgan
radius 2) fingerprint:

```python
import pickle
from sklearn.svm import SVC

clf = SVC(probability=True)
# fit clf on labelled ECFP4 fingerprints

with open("data/surrogates/my_qsar_model.pickle", "wb") as f:
    pickle.dump({"classifier_sv": clf}, f)
```

```json
"score_components": ["QED", "drd2_activity"],
"score_thresholds": [0.5, 0.5],
"qsar_models": {"drd2_activity": "data/surrogates/my_qsar_model.pickle"}
```

Paths are resolved relative to the directory `submit.py` runs from.

`data/surrogates/QSAR_model_example.pickle` in this repository is a zero-byte placeholder, not
a model. It is the default value of `qsar_models` in `defaults.py`, but entries whose key does
not appear in `score_components` are skipped, so it only causes a problem if you actually list
`drd2_activity`. Doing so fails at startup with `QSAR model file is empty or corrupt`. Train a
model of your own with `src/graphinvent/tools/train-surrogate.py`; the `qsar_models` route is
kept for backwards compatibility, and a new objective is better declared as an `sklearn` oracle,
which accepts a plain pickled estimator and can also report uncertainty.

---

## Parameters

| Parameter | Default in `defaults.py` | Value in `jobs/goal_directed/params.json` | Meaning |
|-----------|--------------------------|--------------------------------------------|---------|
| `pretrained_model_path` | `""` | `""` | Checkpoint to load as both the initial agent and the frozen prior; must be set |
| `score_components` | `["QED", "drd2_activity", "target_size=12"]` | `["QED"]` | Components to combine |
| `score_thresholds` | `[0.5, 0.5, 0.0]` | `[0.5]` | Per-component thresholds, used only by binary scoring |
| `score_type` | `"binary"` | `"binary"` | How components combine |
| `qsar_models` | `{"drd2_activity": "data/surrogates/QSAR_model_example.pickle"}` | `{}` | Component name to pickle path |
| `oracles` | `{}` | `{}` | Oracle definitions keyed by the name used in `score_components` |
| `uncertainty_modulation` | `{}` | `{}` | Optional damping of reward or gradient by oracle uncertainty |
| `sigma` | `20` | `20` | Scale of the score term in the augmented log-likelihood |
| `alpha` | `0.5` | `0.5` | BASF weight in the loss; 0 uses the agent alone |
| `epochs` | `100` | `100` | RL steps, when `oracle_budget` is null |
| `batch_size` | `1000` | `50` | Molecules generated and scored per step |
| `accumulation_steps` | `10` | `10` | Steps accumulated before an optimiser update |
| `sample_every` | `10` | `10` | Steps between evaluation, checkpointing, and BASF updates |
| `n_samples` | `2000` | `100` | Molecules generated at each evaluation |
| `oracle_budget` | `null` | `null` | Integer caps the run by oracle calls instead of steps |

The defaults in `defaults.py` are inherited by any key you omit, which is why the two columns
differ and why it is worth setting `batch_size` explicitly: RL generates and scores every
molecule in the batch at every step, so the pretraining default of 1000 is far more expensive
per step than the template's 50.

`init_lr`, `max_rel_lr`, and `min_rel_lr` work as in supervised training, except that the
one-cycle schedule is sized from the number of RL steps (or, under a budget, from
`ceil(oracle_budget / batch_size)`) divided by `accumulation_steps`.

### Choosing `sigma`

`sigma` sets how large a likelihood shift the score is worth, and so trades exploration against
exploitation.

| Situation | Rough range |
|-----------|-------------|
| Gentle push, wide exploration | 5–10 |
| Standard fine-tuning | 20 |
| Aggressive optimisation, mode collapse likely | 50–100 |

These are rules of thumb from published REINVENT settings rather than values validated on this
codebase. Start at 20. If validity falls and the generated molecules converge on one scaffold,
lower it; if the agent has barely moved from the prior after a few hundred steps, raise it.

---

## Configuration file

```bash
cp jobs/goal_directed/params.json jobs/goal_directed/my_experiment.json
```

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./src/graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "debug",
    "job_name": "run",
    "use_slurm": false,
    "slurm": {
      "account": "XXXXXXXXXX",
      "run_time": "0-12:00:00",
      "gpus_per_node": "T4:1"
    }
  },
  "job": {
    "job_type": "goal_directed",
    "oracle_budget": null,
    "device": "cuda",
    "batch_size": 50,
    "block_size": 100000,
    "accumulation_steps": 10,
    "epochs": 100,
    "init_lr": 1e-4,
    "max_rel_lr": 10,
    "min_rel_lr": 0.0001,
    "sample_every": 10,
    "n_samples": 100,
    "n_workers": 0,
    "restart": false,
    "use_tensorboard": false,
    "pretrained_model_path": "./output/debug/unconditional/run/model_restart_100.pth",
    "score_components": ["QED", "target_size=12"],
    "score_thresholds": [0.5, 0.0],
    "score_type": "binary",
    "qsar_models": {},
    "sigma": 20,
    "alpha": 0.5
  }
}
```

`dataset` and `data_path` may be omitted when `pretrained_model_path` is set: `submit.py` reads
`dataset_dir` from the checkpoint's `params_all.json` and derives both. Set them explicitly only
to evaluate against a different dataset than the one the model was trained on.

The GGNN architecture is inherited from the checkpoint's `params_all.json`, so it does not
belong in this config. That inheritance applies only to keys you have not written yourself —
any architecture key present in the job block overrides the checkpoint's value, which is a
reliable way to produce a shape mismatch.

A `target_size` threshold of `0.0` is close to no constraint, but not quite none: the size
score is `1 - |n_nodes - N| / (max_n_nodes - N)` clamped at 0, and the binary test is strictly
greater than the threshold, so a molecule further than `max_n_nodes - N` atoms from the target
scores exactly 0 and still fails. Raise the threshold to demand molecules genuinely close to
the target.

---

## Running the job

```bash
python submit.py --config jobs/goal_directed/my_experiment.json
```

---

## Output

Written to `output/<dataset>/goal_directed/<job_name>/`.

| File | Contents |
|------|----------|
| `params_all.json` | Resolved parameters, library versions, git hash, seed |
| `score.log` | `Step, Score` — the mean agent score at each evaluation step |
| `convergence.log` | `step, lr, avg_train_loss, model_score`, one row per step |
| `generation.log` | Per-evaluation-step molecule statistics, including `success_rate`, `internal_diversity_successful`, and a `score_<component>_mean` column per component |
| `model_restart_<step>.pth` | Agent weights, optimiser and scheduler state, at each evaluation step |
| `generation/` | Sampled SMILES and feature plots per step, labelled `agent`, `BASF`, `eval`, and `pre-fine-tuning` |
| `checkpoint_oracle_<N>.pth` | Only when `oracle_budget` is set: the agent at each milestone in `checkpoint_oracle_counts` |
| `oracle_eval.csv` | Only when `oracle_budget` is set: full metrics for each milestone checkpoint |

### Reading `score.log` and `generation.log`

The mean score should rise. That alone does not distinguish a model that has learned the
objective from one that has found a single high-scoring molecule and stopped exploring, which
is why `generation.log` also carries `fraction_unique`, `internal_diversity`, and
`internal_diversity_successful` — the last being diversity computed only over molecules above
`success_threshold`. A run where the score climbs while `internal_diversity_successful` falls
has collapsed onto one scaffold, and the headline score is then a property of the scoring
function rather than of the model.

`fraction_valid` and `fraction_valid_pt` falling sharply is the other collapse signature, and
usually calls for a lower `sigma` or a lower learning rate.

### TensorBoard

```bash
tensorboard --logdir output/<dataset>/goal_directed/<job_name>/tensorboard/
```

---

## Oracles

Most objectives worth optimising cannot be read off the graph. They come from a model fitted to
assay data, from a docking calculation, or from an external service, and GraphINVENT calls all
of these oracles. An oracle is declared in the `oracles` block of the job config under the
name it will be referenced by, and then listed in `score_components`:

```json
"oracles": {
  "EGFR": {
    "type": "sklearn",
    "path": "data/surrogates/egfr_rf.pkl",
    "output": "proba",
    "radius": 2,
    "n_bits": 2048
  }
},
"score_components": ["EGFR", "QED"],
"score_thresholds": [0.5, 0.5]
```

Nothing is downloaded and no catalogue of ready-made objectives is bundled. The model is one you
train or write, which means its provenance, its applicability domain, and its failure modes are
known to you and can be reported alongside the generated molecules. Three oracle types cover
most cases:

| `type` | Backed by | Principal keys |
|--------|-----------|----------------|
| `"sklearn"` | A pickled scikit-learn estimator over Morgan fingerprints | `path`, `output` (`"proba"`, `"predict"`, or `"decision"`), `radius`, `n_bits`, `model_key` |
| `"python"` | Any importable `f(list[str]) -> list[float]` | `target`, written as `"package.module:function"`, plus optional `kwargs` |
| `"vina"` | AutoDock Vina docking against a prepared receptor | `receptor` (a PDBQT file), `center`, `box_size`, `exhaustiveness`, `n_workers` |

`src/graphinvent/tools/train-surrogate.py` produces a model in the form the `sklearn` type
expects from a table of SMILES and labels, and prints the config block to paste in. The
`python` type is the escape hatch for anything that does not fit a pickled estimator: a PyTorch
model, an in-house web service, a physics calculation.

### Transforms and directions

An oracle reports its native quantity, whether that is a docking energy in kcal/mol, a predicted
pIC50, or a classifier probability, and a `transform` maps it onto the [0, 1] desirability the
RL objective needs. The bound is not cosmetic: the augmented log-likelihood adds `σ · score`, so an
unbounded component could dominate the loss without limit.

```json
"docking": {
  "type": "vina",
  "receptor": "data/receptors/egfr.pdbqt",
  "center": [12.4, -3.1, 22.8],
  "box_size": [20.0, 20.0, 20.0],
  "transform": {"type": "clipped_linear", "low": -4.0, "high": -11.0}
}
```

Writing `high` below `low` is how a quantity where smaller is better is expressed. The available
types are `clipped_linear`, `sigmoid`, `step`, and `identity`; omitting `transform` clamps the
raw value to [0, 1], which is only correct for an oracle that already returns a probability.
Choosing the endpoints is a modelling decision. The window above says that below -11 kcal/mol
nothing is gained, which is defensible for docking because the scoring function cannot rank very
strong binders, but it is an assumption and belongs in the write-up.

`direction` is `"maximize"` by default; `"minimize"` inverts the transformed score. Selectivity
is expressed with it rather than with a second differently-shaped transform: binding one target
while avoiding another is two oracles over the same kind of quantity pointing opposite ways.

```json
"oracles": {
  "GSK3B": {"type": "sklearn", "path": "data/surrogates/gsk3b_rf.pkl"},
  "JNK3":  {"type": "sklearn", "path": "data/surrogates/jnk3_rf.pkl",
            "direction": "minimize"}
},
"score_components": ["GSK3B", "JNK3", "QED"],
"score_thresholds": [0.5, 0.5, 0.4],
"score_type": "binary"
```

With `binary` scoring a molecule counts only if it clears all three thresholds, which is the
intended reading of a conjunctive design goal. This makes the reward sparse at the start of a
run, when few molecules clear anything, and each surrogate is independently imperfect: a
molecule above threshold on all three is a statement about three models, not about a compound.

### The cost of docking

A single Vina pose search takes on the order of a second per molecule per core, so a few hundred
RL steps at batch 50 is tens of thousands of dockings. Deduplication helps, since a converging
agent re-proposes molecules constantly and every oracle is wrapped in a cache that scores a
repeated SMILES for free, but a docking-driven run is still hours to days rather than minutes.
The scoring function is the deeper problem: Vina's empirical score correlates only loosely with
measured affinity, and an RL agent is an efficient adversary against an imperfect objective. It
will find molecules that dock well and do not bind. Combine docking with property constraints,
and treat the result as a filter rather than as evidence.

### Uncertainty modulation

A surrogate that reports a predictive spread, as the `sklearn` type does whenever the pickled
estimator is an ensemble, can have that spread damp its own influence, so the agent is not
rewarded for wandering into the region where the model is extrapolating. `uncertainty_modulation`
configures this per component, either by folding reliability into the score or by reweighting the
per-molecule gradient contributions:

```json
"uncertainty_modulation": {
  "mode": "loss",
  "components": {
    "EGFR": {"method": "sigmoid", "beta": 0.4, "alpha": 10.0}
  }
}
```

It is off by default, and a component configured here whose oracle reports no uncertainty is left
alone with a warning. The two modes differ in what they claim: score modulation says an
untrustworthy molecule is worse as a molecule and changes the optimum being sought, while loss
modulation leaves the objective intact and merely declines to learn much from molecules the
surrogate cannot vouch for. The implementation follows Medina and Janet
([arXiv:2606.24990](https://arxiv.org/abs/2606.24990)).

[Tutorial 6](./06_custom_oracles.md) covers all of this in detail: every oracle type's full key
set, the transform parameterisations, the uncertainty methods, and worked configurations.

---

## Oracle budgets and sample efficiency

Setting `"oracle_budget": <int>` switches the run from a fixed number of steps to a fixed
number of oracle calls. Training stops once the count reaches the budget, checkpoints are saved
as each milestone in `checkpoint_oracle_counts` is crossed, and afterwards each milestone
checkpoint is loaded, used to generate `eval_sample_size` molecules, and scored into
`oracle_eval.csv`.

Two things about the counting are worth being precise about, because they affect whether a
number is comparable to published PMO results:

- The budget loop reads `ScoringFunction.oracle_calls`, which is the oracle cache's count of
  *unique* molecules evaluated, matching the PMO protocol. A repeated SMILES is served from the
  cache and costs nothing, and both batches scored per step — the agent's and the
  best-agent-so-far's — contribute. With no oracle declared, and the objective made only of
  built-ins such as QED, there is nothing to meter, so the count falls back to the number of
  molecules scored and a budget then behaves much like a step limit.
- `oracle_eval.csv` contains `oracle_count`, `fraction_valid`, `fraction_unique`, `novelty`,
  the SA statistics, `success_rate`, `diversity`, `fcd`, `rediscovery_rate`, and one
  `score_<component>_mean` per component. It does **not** contain AUC Top-10. The run does
  write `optimization_log.jsonl`, one `{"oracle_calls": ..., "score": ...}` record per unique
  molecule evaluated, which is the curve `compute_auc_top_k` integrates; the metric has to be
  computed from it afterwards.

### AUC Top-K

`compute_auc_top_k` implements the PMO benchmark's primary metric. At each oracle call *t* it
takes the mean of the top *k* scores seen so far, averaging over however many qualifying
molecules have been found rather than dividing by a fixed *k*, and returns the area under that
curve divided by the span integrated. The result is in [0, 1]: a method that finds *k* perfect
molecules immediately approaches 1, one that finds them only at the budget limit approaches 0.

Apply it to the log the run wrote:

```python
import json
import sys

sys.path.insert(0, "src")   # the editable install does not put src/ on the path

from oracles import compute_auc_top_k

with open("output/<dataset>/goal_directed/<job_name>/optimization_log.jsonl") as f:
    log = [(r["oracle_calls"], r["score"]) for r in map(json.loads, f)]

auc = compute_auc_top_k(
    optimization_log=log,   # [(call_count, score), ...]
    k=10,
    budget=10000,
    finish=True,   # flat-extend the curve to the budget only if the run stopped legitimately
)
```

A constrained variant takes a per-entry boolean list so that only molecules satisfying every
constraint count toward the top *k*:

```python
satisfied = [score > 0.5 for _, score in log]
auc = compute_auc_top_k(log, k=10, budget=10000, constraints=satisfied)
```

AUC Top-10 rewards reaching high scores quickly, which is the right thing to measure if oracle
calls are the binding cost. It says nothing about whether the top ten molecules are chemically
distinct, synthesisable, or plausible, and a run can score well by exploiting a single scaffold
the surrogate model happens to overrate. Report it with the diversity and validity columns, not
in place of them.

---

## Adding a scoring component

### A scoring function of your own

No code change is needed. Write a function that takes a list of SMILES and returns one float per
input, in order, and point a `python` oracle at it:

```python
# mypackage/scoring.py
def penalised_logp(smiles: list) -> list:
    """LogP minus a synthetic-accessibility penalty. Returns raw values, not scores."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    values = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        values.append(-10.0 if mol is None else Descriptors.MolLogP(mol))
    return values
```

```json
"oracles": {
  "plogp": {
    "type": "python",
    "target": "mypackage.scoring:penalised_logp",
    "transform": {"type": "clipped_linear", "low": -5.0, "high": 5.0}
  }
},
"score_components": ["plogp"],
"score_thresholds": [0.5]
```

The function returns its native quantity and the transform turns that into a score, so the same
function can be reused with a different window without editing it. It must handle `None` and
unparseable SMILES itself, since invalid molecules are routine during generation, and it must
return exactly one value per input — a length mismatch is caught and raised rather than allowed
to misalign scores with molecules.

### A new oracle type

Worth doing when the *mechanism* is new, not merely the objective: a different featurisation, a
different model framework, a different external service. Subclass `BaseOracle`, implement
`predict` returning native values, and register the class in `ORACLE_TYPES` in
`src/oracles/_factory.py`. The constructor signature then becomes the configuration schema, since
the factory passes the remaining spec keys through as keyword arguments.

```python
from oracles import BaseOracle
from oracles._factory import ORACLE_TYPES


class MyModelOracle(BaseOracle):
    def __init__(self, name, checkpoint, transform=None, direction="maximize"):
        super().__init__(name=name, transform=transform, direction=direction)
        self.model = load_my_model(checkpoint)

    def predict(self, smiles):
        return [self.model(s) if s else 0.0 for s in smiles]


ORACLE_TYPES["mymodel"] = MyModelOracle
```

The base class applies the transform and direction, so `predict` should not clamp or invert
anything itself. Override `predict_with_uncertainty` as well if the model can report a spread;
leaving it alone reports no uncertainty, which callers treat as "unavailable" rather than as an
error.

### A component computed directly from the graph

For anything that can be computed from an RDKit molecule without an external model, add a branch
to `get_contributions_to_score` in `src/graphinvent/ScoringFunction.py`:

```python
elif score_component == "my_property":
    scores = torch.tensor(
        [compute_my_property(graph.molecule) for graph in graphs],
        device=self.device,
    )
    contributions_to_score.append(scores)
```

The branch must append a tensor of length `self.n_graphs` with values in [0, 1], since the
combination logic assumes both. Then list `"my_property"` in `score_components` with a matching
`score_thresholds` entry.

Branch order in `get_contributions_to_score` matters. `target_size=`, `logp_target=`, and `QED`
are matched first, then any component name containing the substring `activity`, and only then
the declared oracles. An oracle called `kinase_activity` is therefore routed to the QSAR branch
and fails on a missing `qsar_models` entry, so avoid `activity` in an oracle name unless you
intend the QSAR lookup.

---

## Next step

Generate a larger set from the best checkpoint, chosen from `score.log` together with the
diversity columns of `generation.log`: [Tutorial 5: Sampling](./05_sampling.md), with
`pretrained_model_path` set to for example
`"./output/debug/goal_directed/run/model_restart_50.pth"`.
