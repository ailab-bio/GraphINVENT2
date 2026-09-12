# Experiment 3: Goal-directed optimisation (PMO-style)

Run GraphINVENT2's reinforcement learning mode against several targets and several seeds, following the protocol of the Practical Molecular Optimization benchmark (Gao et al., NeurIPS 2022). The targets are yours: each entry in `oracles_config.yaml` names a model you trained or a docking setup you prepared, so the sweep measures your prior against your objectives rather than against a fixed catalogue.

Read the [status section](#status-of-the-pmo-metric-pipeline) before planning a paper around these numbers: the RL runs themselves work, but `evaluate_results.py` has three defects that leave the AUC Top-10 table empty.

---

## What the benchmark measures

Optimisation against a molecular oracle is easy to do badly in a way that looks good. Given unlimited evaluations, almost any search procedure eventually finds high-scoring molecules, so a final top score says more about the budget than about the method. PMO fixes the budget at 10,000 oracle evaluations and scores methods on the *area under the top-10 average score curve*, which rewards finding good molecules early and penalises spending the budget before the score moves.

AUC Top-10 is a sample-efficiency proxy and inherits the limitations of the oracle it is computed from. Where that oracle is a fingerprint classifier trained on a specific labelled set, a high score means the generated molecules resemble that set's actives in ECFP space rather than that they are active. Optimisers are effective at exploiting exactly that gap. A single AUC number also says nothing about whether the high scorers are one series or many, which is why the diversity and synthesisability columns below belong in any report of these results rather than in an appendix.

---

## Dependencies

```bash
pip install pyyaml            # oracles_config.yaml is read by both scripts here
pip install -e ".[docking]"   # only if a target is scored by AutoDock Vina
```

The objectives themselves are not dependencies you install but models you provide. For an activity target, train one from your own assay data:

```bash
python src/graphinvent/tools/train-surrogate.py \
    --input data/assays/target_a.csv --smiles-column smiles --label-column pIC50 \
    --threshold 6.0 --split scaffold --output data/surrogates/target_a_rf.pkl
```

Read the held-out ROC-AUC it prints before committing to a 12-hour run. A surrogate that cannot predict its own scaffold-split test set will still drive the RL loop perfectly happily, and the molecules it produces will score well and mean nothing.

---

## Targets

`oracles_config.yaml` defines the sweep. Each entry has a `name` (the score component the run optimises), an `oracle` block copied verbatim into the job's `oracles` config, a success `threshold`, and a `description` recording where the model came from. The shipped file defines two entries and three seeds, so six runs; the paths in it are placeholders for models that do not ship with the repository.

```yaml
oracles:
  - name: target_a
    oracle:
      type: sklearn
      path: data/surrogates/target_a_rf.pkl
      output: proba
      direction: maximize
    threshold: 0.5
    description: >
      Activity against the primary target, from a random forest over Morgan
      fingerprints trained on in-house or ChEMBL assay data.

seeds: [0, 1, 2]
```

The oracle type decides where the score comes from: `sklearn` for a pickled estimator over fingerprints, `python` for any importable `f(list[str]) -> list[float]`, `vina` for docking against a prepared receptor. A commented-out `vina` entry in the file shows the structure-based route, which suits a target with a usable structure and too little assay data to fit a surrogate. [Tutorial 6](../../tutorials/06_custom_oracles.md) documents every key.

Two things about the entries are worth stating explicitly in any write-up built on them. A fingerprint classifier scores a molecule by its resemblance to the actives it was trained on, so a high score is a similarity claim rather than an activity claim, and an optimiser is good at exploiting that gap. A docking score is not much better placed: Vina's empirical function correlates only loosely with measured affinity, and at roughly a second per molecule per core it also sets the pace of the whole run.

A built-in component such as `QED` can be named in `score_components` too, but it makes a poor sweep entry. It is computed directly from the molecule with no oracle involved, so the run is not oracle-limited in any meaningful sense, and Gao et al. classify QED and penalised LogP as saturated benchmarks that most generative models reach the ceiling of. Use them as pipeline checks, not as evidence of optimisation ability.

---

## Step 1 — Point the template at the pretrained prior

`rl_params_template.json` ships with `"pretrained_model_path": "./output/chembl_v34/unconditional/run/model_restart_200.pth"`. Either edit that value to the checkpoint chosen in Experiment 1, or override it per launch:

```bash
python experiments/goal_directed/run_all_oracles.py \
    --pretrained-model ./output/chembl_v34/unconditional/run/model_restart_180.pth
```

The RL loop copies this checkpoint three ways: the agent that gets updated, a frozen prior that anchors the augmented log-likelihood loss, and the best-agent-so-far model used for replay. The prior is what stops the agent from drifting into high-scoring nonsense, so its quality bounds the whole experiment.

Architecture keys are inherited from the checkpoint's `params_all.json`, but only where the job config leaves them unset. The template correctly omits them; do not add them back.

---

## Step 2 — Launch the runs

```bash
python experiments/goal_directed/run_all_oracles.py
```

For each (target, seed) pair this fills in the template, writes the resolved config to `experiments/goal_directed/configs/<name>_seed<seed>/params.json`, and invokes `submit.py` on it. The runs execute sequentially. The oracle spec is copied into the resolved config rather than referenced, so that file alone reproduces the run without the sweep config beside it.

Filters, all repeatable:

```bash
python experiments/goal_directed/run_all_oracles.py --oracle target_a --seed 0
python experiments/goal_directed/run_all_oracles.py --dry-run
```

`--config` points at an alternative `oracles_config.yaml`. A dry run prints the commands and writes nothing. Output directories are named from the job name, with `:` replaced by `_`:

```
output/chembl_v34/goal_directed/target_a_seed0/
output/chembl_v34/goal_directed/target_a_seed1/
output/chembl_v34/goal_directed/target_a_selective_seed0/
...
```

Expect 12–24 hours per run on one GPU. The runs are independent, so several invocations with disjoint `--oracle` filters can proceed in parallel on different devices.

---

## How the RL loop works

1. The pretrained checkpoint is loaded as agent, frozen prior and best-agent-so-far.
2. A batch of molecules is sampled from the agent.
3. Each molecule is scored. Every declared oracle is wrapped in `CachedOracle`, which returns a cached score for a repeated SMILES rather than re-evaluating it, so a converging agent stops paying for the molecules it keeps re-proposing.
4. The loss `(log p_agent − log p_prior − σ·score)²` is computed and back-propagated.
5. When the agent's mean evaluation score improves, the best-agent-so-far copy is updated.
6. Checkpoints are written at each milestone in `checkpoint_oracle_counts`.
7. Training stops when the oracle call counter reaches `oracle_budget`.

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `sigma` | 20 | Scales the reward term against the prior; the REINVENT default |
| `alpha` | 0.5 | Weight of best-agent-so-far replay against fresh agent samples |
| `batch_size` | 64 | Molecules generated and scored per step |
| `accumulation_steps` | 1 | One optimiser update per step |
| `oracle_budget` | 10000 | Sets the number of steps and the AUC normalisation |
| `sample_every` | 1 | Evaluation of 200 molecules at every step |
| `score_type` | `continuous` | With a single component the raw oracle score is used unchanged |

Because these runs are single-component and continuous, `score_thresholds` is ignored: the threshold from `oracles_config.yaml` reaches the run only as `success_threshold`, which decides what counts as a success in the reported metrics rather than what the agent is rewarded for. Under `"binary"` the same single component would be thresholded into 0 or 1, which is the multi-objective setting discussed below.

### How the budget is counted

`constrained_rl_training_phase` reads `ScoringFunction.oracle_calls` after every step, which is the oracle cache's count of unique molecules evaluated. Duplicates and repeated structures cost nothing, matching the PMO protocol. Both batches scored per step contribute, the agent's and the best-agent-so-far's, and the molecules generated for the per-step evaluation are scored through the same cached oracle and so also consume budget. A run whose objective is made only of built-in components has no oracle to meter, and the count falls back to the number of molecules scored, at which point a budget behaves much like a step limit.

---

## Step 3 — Evaluate

```bash
python experiments/goal_directed/evaluate_results.py
```

Options: `--output-root` (default `./output`), `--dataset` (default `chembl_v34`), `--config`, `--budget` (default 10000), `--k` (default 10), `--out` (default `experiments/goal_directed/results/`), and a repeatable `--oracle` filter.

It writes `results/pmo_results.csv` with one row per (oracle, seed, checkpoint) and `results/pmo_table.tex` with AUC Top-10 as mean and standard deviation across seeds.

### Status of the PMO metric pipeline

The training side is complete. Each run writes `optimization_log.jsonl`, one `{"oracle_calls": ..., "score": ...}` record per unique molecule evaluated, and `checkpoint_<N>_samples.smi` for every milestone in `checkpoint_oracle_counts`, which are the two inputs the evaluation needs.

`evaluate_results.py` does not yet read them correctly. Three defects, all of which fail quietly:

1. It instantiates the oracle as `OracleFactory.create_cached(oracle_name)`, without the spec the factory needs as its second argument. The resulting `TypeError` is caught by a broad `except Exception`, printed as "Could not load oracle ... Skipping", and every target is skipped. The spec is available as `oracle_info["oracle"]`.
2. It reads the seed list from a nested `pmo` block, `oracle_cfg.get("pmo", {})`, while `oracles_config.yaml` now carries `seeds` at the top level. The lookup misses and the hard-coded fallback `[42, 123, 456]` is used, so no output directory matches. `run_all_oracles.py` accepts both spellings; this script does not.
3. `oracle_scores_from_log` reads the key `oracle_call`, while the training loop writes `oracle_calls`. Every line raises `KeyError`, is swallowed, and the log parses as empty, so `compute_auc_top_k` receives nothing and returns 0.0.

Until those are fixed, read `oracle_eval.csv` in each job directory for the per-milestone metrics and compute AUC Top-10 from the log yourself:

```python
import json
import sys

sys.path.insert(0, "src")
from oracles import compute_auc_top_k

job_dir = "output/chembl_v34/goal_directed/target_a_seed0"
with open(f"{job_dir}/optimization_log.jsonl") as f:
    log = [(r["oracle_calls"], r["score"]) for r in map(json.loads, f)]

auc = compute_auc_top_k(log, k=10, budget=10000)
```

`compute_auc_top_k` itself is correct and tested. The `finish` argument controls whether the curve is flat-extended from the last call out to the budget, which is appropriate only when a run stopped legitimately rather than crashing, and `constraints` accepts a per-entry boolean so that only molecules satisfying every constraint count toward the top-k, which is the multi-objective variant.

`oracle_eval.csv` is written by `Analyzer.evaluate_checkpoint_molecules` after training and has one row per milestone with the columns `oracle_count`, `fraction_valid`, `fraction_unique`, `novelty`, `sa_score_mean`, `sa_score_median`, `sa_score_std`, `success_rate`, `diversity`, `fcd`, `rediscovery_rate`, and one `score_<component>_mean` per score component. There is no `auc_top_10` column.

---

## Metrics and how to read them

| Metric | Source | Meaning |
|--------|--------|---------|
| AUC Top-10 | `optimization_log.jsonl`, via `compute_auc_top_k` | Area under the top-10 average score curve, normalised by budget |
| Top-1 / Top-10 / Top-100 | `evaluate_results.py`, once it can instantiate the oracle | Mean of the best 1 / 10 / 100 unique oracle scores |
| `fraction_valid` | `oracle_eval.csv` | Generated graphs passing RDKit sanitisation |
| `fraction_unique` | `oracle_eval.csv` | Distinct canonical SMILES among the valid molecules |
| `novelty` | `oracle_eval.csv` | Fraction not present verbatim in the training set |
| `diversity` | `oracle_eval.csv` | 1 − mean pairwise Tanimoto over ECFP4 |
| `sa_score_mean` | `oracle_eval.csv` | Mean raw SA score, 1 easy to 10 hard |
| `success_rate` | `oracle_eval.csv` | Fraction scoring above `success_threshold` |
| `fcd` | `oracle_eval.csv` | Fréchet ChemNet Distance to the test set; `None` without `fcd_torch` |

Validity, uniqueness and novelty are necessary conditions rather than results. An agent that has collapsed onto a single high-scoring scaffold can report perfect validity, respectable uniqueness from decorating that scaffold, and high novelty because none of the decorations appeared verbatim in training. The combination that identifies this is a high success rate with falling `diversity`, and it is the reason the diversity column should be reported alongside the score rather than after it.

The SA distribution is the second guardrail. A fingerprint classifier has no notion of synthetic feasibility, so an unconstrained agent will drift toward structures that score well and could not be made. A rising `sa_score_mean` across milestones is the signature. There is no built-in synthesisability score component, so constraining it means adding an oracle of your own — a `python` oracle wrapping the RDKit SA scorer is a few lines — and paying for it with a harder task.

Multi-objective runs use binary scoring so that a molecule must clear every threshold. Selectivity is the usual case, and it is two oracles over the same kind of quantity with opposite directions:

```json
"oracles": {
  "target_a": {"type": "sklearn", "path": "data/surrogates/target_a_rf.pkl"},
  "anti_target": {"type": "sklearn", "path": "data/surrogates/anti_target_rf.pkl",
                  "direction": "minimize"}
},
"score_components": ["target_a", "anti_target", "QED"],
"score_thresholds": [0.5, 0.5, 0.4],
"score_type": "binary"
```

With `"continuous"` the components are multiplied instead, which lets a very high score on one objective compensate for a poor score on another — rarely what a multi-objective design problem actually wants.

---

## Adding a target to the sweep

Add an entry to `oracles_config.yaml` with `name`, `oracle`, `threshold` and `description`. No code change is needed for any of the three oracle types: `run_all_oracles.py` copies the `oracle` block straight into the job config, and `ScoringFunction` builds it at job start, failing immediately on a missing model file or an unknown key rather than after the first batch.

A genuinely new oracle *type* — a different model framework or an external service that does not fit a pickled estimator or a Python callable — means subclassing `BaseOracle` and registering the class in `ORACLE_TYPES` in `src/oracles/_factory.py`. [Tutorial 6](../../tutorials/06_custom_oracles.md) walks through it.

Watch the name. `ScoringFunction.get_contributions_to_score` matches `target_size=`, `logp_target=` and `QED` first, then any component whose name contains the substring `activity`, and only then the declared oracles. An oracle named `target_a_activity` is routed to the QSAR branch and fails on a missing `qsar_models` entry.
