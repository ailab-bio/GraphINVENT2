# Defining your own scoring oracles

Goal-directed generation optimises whatever you tell it to optimise, so the
scoring function is the part of the setup that determines what you actually
get. GraphINVENT2 does not ship a catalogue of pre-trained objectives. Instead
you declare oracles in the job configuration: a model you trained, a docking
run against a receptor you prepared, or an arbitrary Python callable. The
provenance and the quality of the objective are then yours to control and to
report, which matters because an RL agent will exploit any weakness in the
scoring function it is given.

## The configuration shape

An oracle is an entry in the `oracles` block of a `goal_directed` job config.
The `type` field selects the implementation and the remaining keys are passed
to it. `score_components` then references oracles by name, alongside the
built-in components (`QED`, `target_size=<int>`, `logp_target=<float>`).

```json
"oracles": {
  "EGFR": {
    "type": "sklearn",
    "path": "data/surrogates/egfr_rf.pkl",
    "output": "proba",
    "direction": "maximize"
  }
},
"score_components": ["EGFR", "QED"],
"score_thresholds": [0.5, 0.3],
"score_type": "continuous"
```

Two fields are common to every oracle and carry most of the modelling
decisions.

`transform` maps the oracle's native output onto a [0, 1] desirability. This
matters whenever the oracle does not already return a probability: a docking
energy of -8.5 kcal/mol or a predicted pIC50 of 7.2 means nothing to the RL
objective until you say how good it is. The available transforms are
`clipped_linear` (a ramp between `low` and `high`), `sigmoid` (the same, but
smooth, so molecules outside the window still produce gradient), `step` (a hard
constraint), and `identity`.

`direction` is `"maximize"` by default; `"minimize"` inverts the score after
the transform. This is how an anti-target is expressed, and it is the whole of
what multi-objective selectivity requires.

## Training a surrogate

`src/graphinvent/tools/train-surrogate.py` takes a table of SMILES and labels
and produces a pickled random forest in the format the `sklearn` oracle
expects:

```bash
python src/graphinvent/tools/train-surrogate.py \
    --input data/assays/egfr.csv --smiles-column smiles --label-column pIC50 \
    --threshold 6.0 --split scaffold --output data/surrogates/egfr_rf.pkl
```

The default is a random forest rather than a single model for two reasons. Its
per-tree predictions give an ensemble spread at no extra cost, which is what
the uncertainty modulation below consumes, and it is hard to make a random
forest fail badly on fingerprint data, which keeps attention on the dataset
where it belongs.

The script defaults to a scaffold split and prints held-out metrics. Read them
before using the model. A dataset assembled from congeneric series will give a
flattering score under a random split because near-duplicates land on both
sides, and a surrogate that cannot predict its own test set will still drive an
RL run perfectly happily, producing molecules that score well and mean nothing.

The fingerprint settings used at training time must be repeated in the oracle
config. Nothing can verify this at load time, and a mismatch does not raise; it
shows up as a model that predicts poorly for no visible reason.

## Multi-objective and selectivity

Multi-objective optimisation is a longer `score_components` list. Components
combine as a product when `score_type` is `"continuous"`, or as an AND over
`score_thresholds` when it is `"binary"`, so a molecule must satisfy every
criterion to score well.

Designing for selectivity — bind one target, avoid another — is two oracles
over the same kind of quantity with opposite directions:

```json
"oracles": {
  "on_target":  {"type": "sklearn", "path": "data/surrogates/target_a.pkl"},
  "off_target": {"type": "sklearn", "path": "data/surrogates/target_b.pkl",
                 "direction": "minimize"}
},
"score_components": ["on_target", "off_target", "QED"],
"score_thresholds": [0.5, 0.5, 0.3],
"score_type": "continuous"
```

One caution about the product aggregation: each additional component multiplies
the score down, so a three-component objective where every component sits near
0.5 yields an aggregate near 0.125. Since the reward enters the loss as
`sigma * score`, adding components weakens the learning signal unless `sigma`
is raised to compensate. This is visible in practice rather than theoretical;
check that `sigma * score` is comparable to the agent-prior log-likelihood gap
in your `convergence.log` before concluding that an objective does not work.

## Docking with AutoDock Vina

The `vina` oracle gives a structure-based objective without needing assay data
first, which is its main attraction. It requires the `vina` Python package or
executable, `meeko` (or `obabel`) for ligand preparation, and a receptor
already prepared as PDBQT with a search box.

```json
"EGFR_dock": {
  "type": "vina",
  "receptor": "data/receptors/egfr.pdbqt",
  "center": [12.0, 3.4, -8.1],
  "box_size": [20.0, 20.0, 20.0],
  "exhaustiveness": 8,
  "n_workers": 8,
  "transform": {"type": "clipped_linear", "low": -4.0, "high": -11.0}
}
```

Note that `high` is numerically below `low`, because a docking energy improves
as it becomes more negative. The default transform maps -4 kcal/mol to 0 and
-11 to 1, which spans roughly "no meaningful binding" to "as good as docking
usefully resolves". Those numbers are a convention, not a calibration, and are
worth revisiting per target.

Two limitations deserve stating plainly. Docking costs on the order of a second
per molecule per core, so an RL run of a few hundred steps at batch 64 is tens
of thousands of pose searches, and the run is measured in hours or days rather
than minutes; caching helps, since a converging agent re-proposes molecules
often, but it does not change the order of magnitude. More seriously, Vina's
empirical score correlates only loosely with measured affinity, and an RL agent
is an efficient adversary against exactly that kind of imperfect objective. A
docking score is more useful combined with property constraints than alone, and
it is better read as a filter than as ground truth.

## Arbitrary scoring code

The `python` oracle imports any callable of the form
`f(list[str]) -> list[float]`, named as `"module.path:function"`:

```json
"my_score": {
  "type": "python",
  "target": "my_project.scoring:predict_affinity",
  "kwargs": {"model_dir": "checkpoints/v3"},
  "transform": {"type": "sigmoid", "low": 5.0, "high": 9.0}
}
```

The callable must return exactly one value per input, in order, and must handle
`None` and unparseable SMILES itself, since generated molecules are frequently
invalid. The length contract is checked on every call: a mismatch would
misalign scores with molecules and corrupt the reward with no visible error.

Use this for a PyTorch model, an internal web service, or anything else that
does not fit the scikit-learn interface. Adding a new oracle *type* to the
codebase means subclassing `BaseOracle` and registering it in `ORACLE_TYPES`;
implement `predict` to return your native quantity, and let the base class
apply the transform and direction.

## Uncertainty-modulated rewards

A surrogate trained on finite data is not an oracle, but the RL objective
treats it as one: the agent is rewarded for the predicted property, so it will
drift into regions where the surrogate is extrapolating and its high
predictions are unsupported. Letting predictive uncertainty damp the signal is
a partial remedy, implemented here following Medina and Janet,
[arXiv:2606.24990](https://arxiv.org/abs/2606.24990).

Two strategies are available and can be combined.

Score modulation folds reliability into the objective, so an uncertain molecule
is worth less as a molecule and the agent is steered toward the surrogate's
applicability domain. It changes what is being optimised.

Loss modulation leaves the score alone and reweights how much each sampled
molecule contributes to the gradient, following the paper's Eq. 8:

```
L = (1/N) * sum_j [ w_j / ((1/N) * sum_l w_l) ] * L_j
```

The division by the batch-mean weight is what makes this a reweighting rather
than a learning-rate change: without it a uniformly uncertain batch would
simply shrink the loss. Loss modulation makes no claim about molecular quality;
it declines to learn much from predictions it cannot trust, leaving the
objective intact.

Configuration is per component, which is not a convenience but a requirement.
Each oracle reports uncertainty in its own units — a docking spread in
kcal/mol, an ensemble standard deviation in probability, a conformal p-value —
so no single threshold can be meaningful across all of them:

```json
"uncertainty_modulation": {
  "mode": "loss",
  "components": {
    "EGFR":      {"method": "linear", "max_uncertainty": 0.35},
    "off_target": {"method": "sigmoid", "beta": 0.25, "alpha": 12.0}
  }
}
```

`mode` is one of `"none"`, `"score"`, `"loss"`, or `"both"`. Components absent
from the block are unmodulated, as are components whose oracle reports no
uncertainty; a non-ensemble scikit-learn model, for instance, cannot provide
one and will be left alone with a warning.

The available methods are `linear` (`1 - u/max_uncertainty`, the paper's Eq. 14
with an explicit scale), `sigmoid` (their Eq. 13, useful when there is a real
in-domain/out-of-domain boundary rather than a gradual decay), `inverse`, and
`exponential`. Picking `max_uncertainty` or `beta` requires knowing what
uncertainties your model actually produces, so it is worth scoring a sample of
molecules and looking at the distribution before choosing.

What this does not do is calibrate uncertainties against each other. After the
mapping every component yields a weight in [0, 1], but a 0.5 from one component
does not represent the same degree of doubt as a 0.5 from another. The mapping
is a per-component modelling choice, not a principled cross-component
normalisation, and results should be described that way.

Score modulation compounds across components even using the geometric mean the
paper specifies, so with several uncertain components the reward can become
small enough that `sigma * score` no longer moves the agent. If a
score-modulated multi-objective run appears not to learn, compare
`sigma * score` against the loss magnitude in `convergence.log` before looking
for a bug.
