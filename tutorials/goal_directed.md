# Goal-Directed Optimization

*Last updated: October 18th, 2026.*

A `goal_directed` job fine-tunes a trained prior toward a scoring function with policy-gradient reinforcement
learning. It is the job type where the metrics matter most, because optimizing against a cheap scoring function
is the easiest way in this repository to produce a number that looks good and means nothing.

## Run it

```bash
python submit.py experiments/tutorial_goal_directed/jobs/qed      # 26 s
python submit.py experiments/tutorial_goal_directed/jobs/budget   # 29 s
```

Both start from the prior built in [Unconditional training](./unconditional.md). The two jobs differ only in
what stops them: `qed` runs a fixed number of learning steps, `budget` runs until an oracle budget is spent.

```yaml
job_type: goal_directed
pretrained_model_path: experiments/tutorial_unconditional/jobs/debug/checkpoints/model_restart_4.pth
score_components: ["QED"]
score_thresholds: [0.0]
score_type: continuous
rl_steps: 4
rl_eval_every: 2
batch_size: 20
n_samples: 20
device: cpu
```

## The RL schedule has its own keys

`rl_steps`, `rl_eval_every`, `rl_best_agent_every` and `rl_batch_size`. A `goal_directed` job that sets `epochs`
or `sample_every` is rejected with an error naming the key it should have used:

```
ValueError: job_type: goal_directed does not read 'epochs'.
A learning step is not an epoch; use 'rl_steps' instead.
```

A learning step is not an epoch. The two paths want different values, and sharing the keys meant a change to the
supervised defaults silently changed every optimization run. The defaults come from Atance et al. 2022:
`rl_steps: 200`, `rl_best_agent_every: 5`, `rl_eval_every: 5`, with `rl_batch_size` set per experiment.

The other RL settings are fixed by CLAUDE.md and are not yours to tune: the best-agent loss with `alpha = 0.5`,
`sigma = 10` for property-shaping objectives and `20` for activity objectives, and a score of zero for invalid,
improperly terminated and duplicate molecules.

## Watching it optimize

`logs/score.log` is the trajectory:

```
Step, Score
Step 1, 0.15419988
Step 2, 0.15989959
Step 4, 0.17071086
```

Rising is the thing to check, and ending above the prior's starting score is the minimum bar. Four steps on a
weak prior moves QED from 0.154 to 0.171, which demonstrates the loop rather than the capability.

The per-step record of validity, uniqueness, novelty and internal diversity goes to TensorBoard, tagged by model:

```bash
tensorboard --logdir experiments/tutorial_goal_directed/jobs/qed/logs/tensorboard/
```

Tags look like `RL/agent/internal_diversity`, `RL/eval/novelty`, `RL/BASF/fraction_unique`, where `BASF` is the
best-agent-so-far. **This needs `use_tensorboard: true`**, and without it an optimization run keeps no per-step
record at all. Any run you intend to report should set it, because the collapse of diversity over steps is the
thing the metric suite exists to make visible and it is only observable as a trajectory.

## Reading the result

From the `qed` run:

```
auc_top10                      0.796      success_rate              0.118
best_at_1k                     0.870      success_threshold         0.5
top10_mean_at_1k               0.800      validity                  0.450
top100_scaffold_count          13         quality_filtered_top10_mean 0.354
top100_internal_diversity      0.941      n_unique_valid            34
```

`quality_filtered_top10_mean` is the headline, not `best_at_1k`. It is the top-10 mean among molecules that pass
the structural alerts and a synthesizability gate, so it answers "how good are the best molecules I could
actually make" rather than "how high did the score get". Here it is 0.354 against an unfiltered
`top10_mean_at_1k` of 0.800, and that gap is the cost of insisting the winners be real chemistry. On this weak a
prior the gap is most of the score.

`success_rate` is the fraction of unique valid molecules scoring at or above `success_threshold`, compared with
`>=` because the config key reads as "at least". It is 0.118 at a threshold of 0.5.

### The budget metrics measure the composite, not one component

`auc_top10` and every `*_at_<budget>` integrate against **unique molecules submitted to the oracle**. An oracle
call is one unique canonical SMILES; cache hits are free and are not counted, which is what makes the accounting
meaningful rather than a measure of how often the code happened to call a function.

They integrate the **composite** score the policy was trained on. With a single component that is the same thing,
but a multi-component objective combines its parts, and a curve over one part describes something the model was
not optimizing. A six-component geometric mean is zeroed by a single failed component, so a curve over the
busiest component would be unrelated to the reward.

### success_threshold is per experiment, never inherited

0.5 is meaningful for QED, which is a smooth score on the unit interval. It is meaningless for a geometric mean
containing binary filters, where a single failed filter zeroes the product and 0.5 is unreachable rather than
good. Set `success_threshold` in the experiment's own config, and record a `success_definition` string saying in
words what was counted. A threshold inherited from a global default is a number nobody chose.

### The detectors, which are the point

Optimizing against a cheap scoring function produces reward hacking by default, and the way it shows is specific:
overall diversity stays high while the top-scoring set collapses onto one scaffold. So `internal_diversity` is
not the detector and `top100_scaffold_count` is. Watch it fall toward 1 while the score rises. Here it is 13 for
`qed` and 10 for `budget`, over the 34 and 23 unique valid molecules those runs produced. When fewer than 100
molecules survive, the "top 100" is all of them, and the count has to be read against `n_unique_valid`: 13 of 29
is not the same finding as 13 of 100.

For bifunctional molecules the whole-molecule count does not work at all, because a Bemis-Murcko scaffold of a
PROTAC contains every ring system and is nearly the whole molecule. There `top100_warhead_scaffold_count` is the
detector: the top 100 are decomposed, and the scaffolds are counted over the target-binding components. It is
emitted only when the job's dataset declares `bifunctional: true`, alongside `top100_split_success_rate` over the
same hundred molecules, because a count over the molecules that decomposed cannot be read without knowing how
many did. The whole-molecule count stays reported beside it and is expected to be uninformative. Measured on 12
real degraders: 12 distinct whole-molecule scaffolds, against 10 distinct warhead scaffolds over the 11 that
decomposed.

Decomposition runs in PROTAC-Splitter's own interpreter at roughly 2 s per molecule, so this costs a few minutes
per `goal_directed` job and nothing on a drug-like dataset, where the key is absent rather than null.

## The budget-capped variant

```yaml
oracle_budget: 200
checkpoint_oracle_counts: [100]
eval_sample_size: 20
```

The run stops when the budget is spent rather than after a fixed number of steps, and it checkpoints at the
oracle counts you name so you can evaluate the model as it was at a given spend. It writes two extra files:

```
results/optimization_log.jsonl        (cumulative_unique_molecules, composite_score) per molecule
results/oracle_eval.csv               one row per evaluated checkpoint
results/checkpoint_100_samples.smi    molecules from the checkpoint at 100 calls
```

`optimization_log.jsonl` is the authoritative record of the budget spent and the input the budget curves are
computed from.

Two cautions on this path. The post-training checkpoint evaluation scores `eval_sample_size` molecules through
the same scoring function *after* the budget is exhausted, so the live call counter overshoots the budget the run
was supposed to respect; `optimization_log.jsonl` is written before that evaluation and is unaffected. And the
scheduler is sized from an estimate of how many steps the budget allows, while the loop counts unique molecules,
so the two can disagree on a long run.

## Your own objective

`QED` is a built-in, and built-ins are for tutorials. A real objective is a model you own, declared in the
`oracles:` block and referenced by name from `score_components`:

```yaml
score_components: ["gsk3b"]
score_type: continuous
oracles:
  gsk3b:
    type: sklearn
    path: data/surrogates/gsk3b.pkl
```

Three types are accepted. `sklearn` loads a pickled scikit-learn model over Morgan fingerprints, which is what
`scripts/train-surrogate.py` produces. `python` calls an arbitrary callable, so an objective needs to know
nothing about this repository's internals: it takes SMILES and returns scores. `vina` runs a docking calculation.
Each spec takes its type's parameters plus optional `transform` and `direction`, the latter for objectives where
lower is better.

Every oracle is wrapped in a cache keyed on canonical SMILES, which is what makes the budget accounting correct:
an oracle call is one unique molecule, and a cache hit costs nothing and is not counted. That is not an
optimization detail. Changing the cache changes the reported sample efficiency.

Train the surrogate before the run that uses it, and publish its held-out performance against a
mean-prediction baseline. A surrogate that cannot predict its own test set will still happily drive an
optimization run and produce molecules that score well and mean nothing, which is the failure every detector in
this tutorial exists to catch.

## When it goes wrong

A score that rises while `top100_scaffold_count` falls toward 1 is reward hacking, not success. That is the
result, and it should be reported rather than retried with a different seed.

A score that does not move usually means `sigma` is too small for the score's scale, or every molecule is scoring
zero because an invalid, duplicate or improperly terminated molecule scores zero by design and the prior is too
weak to produce anything else.

`quality_filtered_top10_mean` reported as null with a reason means nothing survived the filters, which on a weak
prior is a real outcome rather than a bug.
