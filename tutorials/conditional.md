# Conditional Generation

*Last updated: October 18th, 2026.*

A `conditional` job trains a model that takes a property value as input, so you can ask for a molecule with a
particular logP rather than sampling whatever the prior produces. It needs a dataset preprocessed with property
columns, and it is the one job type that fails loudly if you give it the wrong kind of dataset.

## Run it

```bash
python submit.py experiments/tutorial_conditional/jobs/preprocess     # 19 s
python submit.py experiments/tutorial_conditional/jobs/logp           # ~3 min
python submit.py experiments/tutorial_conditional/jobs/sample_logp    # 12 s
```

In that order. The input is the same 1,000 ZINC molecules with two computed property columns:

```
SMILES	logp	mol_weight
O=C(CCCN1C(=O)CCC1=O)NCC1(c2ccccc2)CCC1	2.1537	328.412
```

Preprocessing has to be told to use them:

```yaml
condition_dim: 2
conditioning:
  properties: ["logp", "mol_weight"]
  source: smiles_file
```

Omit those and the columns are silently dropped: you get a valid unconditional dataset and the only evidence is
`condition_dim: 0` in `preprocessing.yaml`. Check it before training.

## How the conditioning works

The condition vector is a float32 array of shape `n_subgraphs × condition_dim`, stored alongside the graphs. At
each construction step the model sees the same vector, so every action is taken in the knowledge of what was
asked for.

Mechanically, a two-layer MLP encodes the condition vector into a seed embedding, and that embedding initializes
a **virtual node** prepended to the graph. The virtual node is connected to every real node by an extra edge
type with its own message network, so each real atom receives the condition at every message-passing round. The
seed is stripped before readout, which keeps the output shape identical to the unconditional model.

Stripping the seed is not the whole story, and the part that is easy to miss is that the readout *input* changes
anyway: the condition embedding is concatenated to the graph embedding, making
`graph_emb_size = gather_width + condition_embedding_dim`. That concatenation is load-bearing. The first action
of every molecule is taken on an empty graph, where no real node exists to carry the seed's message, so without
the concatenation the model would be unconditional for exactly the decision that picks the first atom.

## Training it

```yaml
job_type: conditional
epochs: 4
sample_every: 2
sample_conditions:
  logp: 2.5
  mol_weight: 350.0
condition_tolerance:
  logp: 0.5
  mol_weight: 25.0
```

A conditional model cannot be sampled without being told what to ask for, and training samples itself at
`sample_every` to score its own progress, so `sample_conditions` is required here rather than only at generation
time. `condition_tolerance` says how close counts as a hit.

```
epoch, lr, avg_train_loss, avg_valid_loss, model_score
Epoch 1, 0.00094086, 7.06491280, 4.32926416, NA
Epoch 2, 0.00081071, 3.79346800, 3.61663866, 0.440057
Epoch 3, 0.00028188, 2.97002840, 2.81868124, NA
Epoch 4, 0.00000001, 2.24526167, 2.60681224, 0.580993
```

The losses start higher and end higher than the unconditional run on the same molecules, which is expected: the
model is solving a harder problem with the same capacity and the same four epochs.

Giving a `conditional` job an unconditional dataset is refused rather than run:

```
ValueError: job_type: conditional, but <dataset> was preprocessed with condition_dim: 0,
so its HDF5 carries no condition_vector. Preprocess a tab-separated file with property
columns instead.
```

That failure is in the test suite deliberately. A conditional job that quietly trains unconditionally looks
exactly like a model that failed to learn.

## Asking for something

```yaml
job_type: generate
sample_mode: generate
n_samples: 200
sample_conditions:
  logp: from_train
  mol_weight: from_train
condition_tolerance:
  logp: 0.5
  mol_weight: 25.0
```

`from_train` is the useful mode. Instead of one value for the whole sample, it spreads the requests across the
deciles of the training property distribution and records which value was asked of which molecule in
`results/condition_requests.json`. A single fixed value can only ever populate one decile of
`condition_coverage`, so the metric designed to expose "works only near the middle of the distribution" could
never fire.

The difference is visible: with `from_train`, nine of the ten deciles are populated, against one for a fixed
request.

## Reading the result

From the run above:

```
validity                             0.0
n_valid                              0
n_sampled                            200
validity_n_generated_raw             200
condition_success_rate_logp          null
condition_mae_logp                   null
condition_coverage_logp              null
```

`condition_success_rate_<property>` is the fraction of generated molecules within `condition_tolerance` of the
value requested of them. `condition_coverage_<property>` is that same rate per decile of the training
distribution, as ten numbers, with `null` for a decile no request landed in. A decile with no requests records
null rather than zero, so an untested decile is not read as a failed one.

`condition_mae_<property>` is the mean absolute miss, in the property's own units, over the same molecules the
success rate is denominated over. It is there because the success rate is a step function of
`condition_tolerance`: two models whose typical error differs threefold score the same rate when both errors sit
inside the tolerance, and a model that halves its error does not move the rate at all until it crosses the
threshold. Read the pair together. Here the requested tolerances were 0.5 and 25.0, and the errors are 2.19 and
188.95, so this model misses by about four tolerances on logP and by seven and a half on molecular weight. That
is a much sharper statement than a 0.2 hit rate, and it is the number that will move first when the model starts
to work.

**This model does not condition, and the tutorial says so rather than dressing it up.** Twenty valid molecules
from two hundred requests, a 20% hit rate on logP and 0% on molecular weight, with successes only in the low
deciles: that is a four-epoch model on 689 training molecules. It demonstrates the mechanism and the metrics, not
the capability. Treat a conditional result as evidence only once the model clears a stated bar, for instance 60%
validity and a median molecular weight within 30% of its training set's. Below that, the conditional metrics are
measuring the model's inability to make molecules at all.

### The baseline is not optional

```
baseline_condition_success_rate_logp    null
baseline_condition_mae_logp             null
unavailable: "no unconditional baseline sample is declared; set baseline_results_dir …"
```

Every conditional metric is meant to be read beside its unconditional baseline, which means a second sample from
**the same model** with no condition set. Without it the number cannot be interpreted, because nothing separates
the effect of conditioning from the base rate of the corpus: if 30% of the training set has logP near 2.5, then a
30% hit rate is what a model that ignores the condition entirely would score. The gap between conditional and
baseline is the result; the conditional number alone is not.

## Categorical conditions

A one-hot label is numeric columns, one per class, holding 0 or 1. Declaring the same columns a second time as
`condition_classes` is what selects the categorical metric set:

```yaml
condition_dim: 3
conditioning:
  properties: ["is_class_a", "is_class_b", "is_class_c"]
  source: smiles_file
condition_classes: ["is_class_a", "is_class_b", "is_class_c"]
```

Two declarations rather than one, because nothing else can tell a class column from a property that happens to
take the values 0 and 1, and the two metric sets are disjoint. Get it wrong and the job reports
`condition_success_rate` on a one-hot, which measures whether the model echoed its own input. The names and
their order must match `conditioning.properties` exactly; a different order attributes every per-class number to
the wrong class while still producing plausible numbers, so it is an error rather than a silent reorder.

Then ask for the whole panel in one job:

```yaml
sample_conditions: all_classes
```

which cycles the one-hot across generation batches. A single one-hot would leave every class but one with a null
row, and the headline here is the macro-average across the panel.

### What it reports

`delta_nll`, `knn_purity_<5|10|20>`, `class_match_median`, `class_match_rate`, `class_selectivity_margin` and
`novelty`, each as the macro-average over the panel, with the per-class value under the same name plus a
`_<class>` suffix. The full table for all three blocks goes to `results/condition_per_class.json`.

`class_selectivity_margin` is the one to read first, because it is self-normalizing: an unconditional model
scores about zero by construction, and the metric cannot be satisfied by generating generic chemistry.

`delta_nll` is the only one that is a property of the model rather than of a sample. It scores each held-out
molecule under every class in the panel, which is k forward passes per molecule, and needs a preprocessed test
split carrying the class columns. A multi-label molecule's other legitimate classes are excluded from the
competing set and the number excluded is reported as `delta_nll_n_alternatives_excluded`, because a panel where
every molecule belongs to three classes has very little left to compete against.

### Two baselines, and the metrics are not evidence without them

| Prefix | What it is | What it tells you |
|---|---|---|
| `baseline_lookup_` | the training molecules of the requested class, sampled to the same N per class | the retrieval ceiling. It scores `class_match_median` 1.0 at `novelty` 0.0 by construction, so a conditional model that matches it on *both* is retrieving, not conditioning |
| `baseline_posthoc_` | the unconditional model sampled the same way, reranked per class by similarity to that class's training molecules, cut to the same N | **the one that can falsify the claim.** If an unconditional model plus a one-line reranking reaches the same `class_selectivity_margin`, the conditioning bought nothing |

The post-hoc baseline needs `baseline_results_dir` pointing at the unconditional model's `molecules.smi`.
Without it those keys record an explicit null carrying that reason, and the conditional numbers can be reported
but not falsified.

For a bifunctional dataset the per-class table is computed over warheads rather than whole molecules, and adds
`warhead_novelty_in_class_<class>` beside the global `warhead_novelty`. The two ask different questions and are
expected to disagree: global novelty asks whether the warhead appears anywhere in the corpus, and a model that
has learned the conditioning will score low on it, because the right answer for a requested class is a warhead
resembling that class's known ligands. In-class novelty asks whether it is absent from *that class's* warheads,
which is the question that separates conditioning from retrieval.

## When it goes wrong

All classes or all requests scoring alike usually means the condition vector is constant across rows. Check it
directly:

```bash
python -c "import h5py; f=h5py.File('data/datasets/<ds>/train.h5'); \
print(f['condition_vector'].shape, f['condition_vector'][:5])"
```

A `condition_coverage` with one populated decile means the request was constant, which is a config choice rather
than a model failure: use `from_train`.
