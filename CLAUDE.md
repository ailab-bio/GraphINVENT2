# CLAUDE.md

The specification for GraphINVENT2: what the system is, what it guarantees, and what every job type
and every metric means. Operations live in `RUN.md`, the science in `manuscript/experiment-plan.md`,
and open defects in `TODO.md`. Nothing is duplicated between them; where one needs another, it points.

GraphINVENT2 is a graph-based molecular generative model for 2D molecular generation. It extends the
original GraphINVENT with conditional generation, and keeps transfer learning and RL-guided
optimization while making both easier to use and more efficient. It is a research codebase with one
maintainer. It is not a library, a product, or a framework.

**This document describes the target design.** Where the code does not yet match it, the gap is in
[Status](#status) and nowhere else. Build toward this document; do not preserve current behavior that
contradicts it. A capability asserted here and absent from the code is a bug in one of the two.

---

## Working principles

These exist because agent-written code drifts toward defensive complexity. Follow them literally.

- **Prefer deleting to adding.** A change that removes code and keeps behavior is a good change.
- **No new abstraction without approval.** No base classes, registries, plugin systems, factory
  functions, or dependency injection. If you think one is needed, say why and wait.
- **No new config parameter unless a planned experiment varies it.** A parameter that always takes
  the same value is a constant. Put it in the code.
- **No new test file without a stated justification.** See [Tests](#tests).
- **No backwards compatibility.** No shims, no deprecation warnings, no aliases for renamed things.
  Breaking changes are free here.
- **Fix bugs, do not document them.** If you find broken behavior, fix it. Do not add a note
  describing the breakage and route around it.
- **Fail loudly.** Do not catch an exception to continue with a default. Crash with a message that
  names the key, the value, and the values that are implemented. A configuration value that selects
  a behavior the code does not implement must never fall through to a default: see
  [Behavior-selecting keys](#behavior-selecting-keys).
- **Inline single-use helpers** unless the name carries real meaning.
- **Do not add logging, progress bars, or output files** beyond what is specified here.
- **Do not tune hyperparameters.** See [Hyperparameters](#hyperparameters).
- **American spelling throughout**, in code, comments, docstrings, documentation, tutorials and
  output messages: optimize, normalize, analyze, behavior, color, center, modeling, labeled,
  initialize, visualization.

When a task is ambiguous, ask rather than building the general version that covers both readings.

---

## Platform guarantees

Two environments, and the code must work in both without edits: a MacBook on Apple silicon for
development, and a SLURM cluster for the real runs. Everything small runs locally, which includes the
tests, the development experiments, every tutorial and any debugging. If a change makes the laptop
path slower or harder, that is a regression even when the cluster path gets faster.

- **Three devices: CPU, CUDA and MPS.** Device checks follow one pattern and `pin_memory` stays
  CUDA-only. A job produces the same metrics on all three within floating-point tolerance, and the
  tests pass on a laptop with no GPU.
- **An accelerator that is asked for and absent falls back interactively and is fatal inside a SLURM
  allocation.** Falling back there spends the whole walltime on the wrong device and still writes
  plausible metrics stamped with a device nobody reads, so the job crashes instead.
- **`n_workers` is 0, and a non-zero value is rejected.** The HDF5 dataset holds an open file handle,
  which cannot be pickled for a spawned worker and is not fork-safe for a forked one, where it
  corrupts the loaded graphs rather than failing. Control loading cost with `block_size`.
- **No cluster-specific code paths.** Anything that differs between Arrhenius and another SLURM
  cluster is a value under `slurm:`, never a branch in Python. If you find yourself writing
  `if cluster == "arrhenius"`, stop.
- **No architecture assumptions.** Arrhenius GPU nodes are aarch64 with CUDA 13, so every dependency
  must resolve on both macOS arm64 and Linux aarch64. A dependency that only ships x86 wheels is not
  usable here.
- **Paths are configuration.** Cluster home directories have small quotas and data belongs on project
  filesystems, so the `data/` and `experiments/` roots are configurable and never hard-coded.
- **Nothing requires a GPU.** Every job type runs on CPU, slowly. A code path that only works on CUDA
  is a bug.

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; that pair is the single source
of truth, on every machine, and nothing else declares a dependency. The lock file is not yet
generated, which [Status](#status) records. RDKit, PyTorch, h5py, numpy, PyYAML and matplotlib are core; fcd,
XGBoost, PROTAC-Splitter and AutoDock Vina are optional extras used by evaluation and RL only. The
install commands are in `RUN.md`. Adding a core dependency needs a reason.

---

## Hyperparameters

Hyperparameter search is out of scope. Values come from prior GraphINVENT work, principally Atance et
al. (2022), *J. Chem. Inf. Model.* 62:4863, and Nori et al. (2022), arXiv:2211.02660, and from the
defaults in `github.com/olsson-group/RL-GraphINVENT`.

The RL settings are fixed: BAR loss with `alpha = 0.5`, `sigma = 10` for property-shaping objectives
and `20` for activity objectives, best agent updated every 5 learning steps, 1,000 molecules per
learning step, 200 learning steps, and a score of zero for invalid, improperly terminated and
duplicate molecules.

`max_nodes` is the exception, because the data sets it rather than anyone choosing it: take the 99th
percentile heavy-atom count of the dataset and record the number. Measured values are 108 for the
TPD-DB degrader corpus (median 63, maximum 141), 54 for molecular glues (median 34), and 16 for
battery solvents, where the dataset criteria bound it. Preprocessing decomposes each molecule into
one subgraph per construction action, so training-set size, HDF5 size and epoch time all grow
superlinearly with molecule size. Measure on a small subset before committing to a full run.

Any other change to a hyperparameter needs a stated reason and the maintainer's approval first.

---

## Repository layout

```
src/graphinvent/      model, training, generation
src/metrics/          evaluation (see Evaluation metrics)
src/oracles/          oracle wrappers, surrogates, and the call-counting cache
config/defaults.yaml  every parameter and its default value
config/slurm.sh.template  the one SLURM template
data/datasets/<name>/ source SMILES and provenance, plus one `split_<type>/` subdirectory
                      per split set holding that split's files (see Data splits)
experiments/          one directory per experiment (see below)
tools/                external tools and data this repository depends on
tutorials/            one walkthrough per job type
scripts/              data download, dataset construction, surrogate fitting
submit.py             entry point
visualize.py          render a .smi file to a grid image
manuscript/           the paper, its figures, and the provenance of every number in it
```

### External tools

Many metrics and oracles depend on software that is not a plain PyPI install: packages distributed
only from GitHub, data files extracted from other projects, and model weights downloaded on first
use. An invisible dependency is one that fails on a cluster at midnight, so they are collected in one
place.

```
tools/
    README.md      the inventory: one entry per tool or data file
    install.sh     clones and downloads everything at its pinned version; idempotent
    data/          small externally sourced data files, committed, provenance in the README
    splitter-cache/   decomposition results, keyed by splitter model; not committed
    .gitignore     clones and downloaded weights are not committed
```

Every entry in `README.md` records what the tool is, which metrics or oracles need it, the source
URL, the exact pinned version or commit, the license, the install command, and whether it is required
or optional. When an optional tool is absent, the README says which metrics record null, so a sparse
`metrics.json` is explicable without reading code. The inventory covers PROTAC-Splitter, RetroTAC,
REINVENT4 and the battery assets.

Downloaded model weights are pinned to a path under `tools/`, never left to a library's default
cache. A dependency that is used but not recorded in `tools/README.md` is a bug.

### Experiments

Every run lives in an experiment directory. Nothing is written outside one.

```
experiments/<experiment_name>/
    experiment.yaml              settings shared by all jobs here
    instructions.md              what this experiment claims, in what order its jobs run, which
                                 generate job produces its core metrics, and what it looks like
                                 when it has gone wrong
    results/                     cross-job aggregates only, such as aggregate.json over seeds
    jobs/
        <job_name>/
            config.yaml          this job's settings
            run.sbatch           rendered from the one SLURM template
            results/             metrics.json, molecules, oracle logs, report.html, figures
            checkpoints/         model state
            logs/                stdout, stderr, tensorboard
```

`metrics.json` and `report.html` are written by every job type that produces metrics, which is
`generate` in both modes, `goal_directed`, `unconditional` and `conditional`, since model efficiency
is reported for every trained model.

A `preprocess` job writes nothing to its own `results/` but the resolved config and its two figures.
Splits, HDF5, `train.csv` and `preprocessing.yaml` belong to the dataset, under
`data/datasets/<name>/split_<type>/`, because every job in every experiment on that dataset reads the
same split set.

`<experiment_name>` describes the question, not the method: `conditional_logp_zinc`, not `run3`. A
job directory is self-contained and can be copied or deleted on its own. The old `jobs/` and
`output/` directories at the repo root are gone; do not reintroduce them.

### What one experiment is

An experiment is **one trained model, or one optimization campaign, together with everything needed
to evaluate it**. Filters, decomposition and analysis are not experiments; they are the analysis of
one. Baselines are not experiments; they are arms of the experiment they are a baseline for.

| Experiment | What it is |
|---|---|
| `<program>_prior` | unconditional training on the base corpus |
| `<program>_transfer` | transfer onto the task corpus with `resume_from`, **together with** the from-scratch control on the same corpus. Both arms live in one experiment because neither means anything alone: the comparison is the result |
| `<program>_conditional` | conditional training on the panel. One experiment, not one per class: all classes are trained in one model and evaluated together |
| `<program>_goal_directed` | optimization toward the program's oracle, with the virtual screening arm inside it and the REINVENT4 arm mirroring it from `reinvent4_<program>_goal_directed` |
| `protac_selectivity` | the bonus campaign, optimizing for degradation of one target and against a second. Virtual screening is its only baseline |

Three programs — `protac`, `mg`, `battery` — give twelve, plus `protac_selectivity`: **thirteen**.
Four surrogate fits sit beside them as their own experiments, because each level differs in what
data it may see and that difference is the point: `surrogate_<program>_optimization` is fitted on
the training and validation splits only and is the RL reward,
`surrogate_<program>_evaluation` is the same featurization and model class refitted on all data
including the test split and only ever rescores final molecules. Five tutorial experiments and six
REINVENT4 baselines sit beside those, so `experiments/` holds twenty-eight directories of which
thirteen are the result.

**All three goal-directed experiments carry both baseline arms**, virtual screening and REINVENT4,
not the drug-discovery ones only. `protac_selectivity` is the single exception and takes virtual
screening alone.

**The virtual screening baseline is not a job type.** `scripts/screen-library.py` takes a
goal-directed job directory and a library, draws uniformly at random, scores through the same
`CachedOracle` the RL job uses, and writes `results/screening_<library>.json`; that job's
`screening_baseline_results` key points at the file and the metrics layer computes the `vs_*` keys
from it. Screening is a random draw plus surrogate calls at milliseconds each, so it needs no SLURM
allocation. It runs for all four optimization experiments and for no other job type, since there is
nothing to spend a budget against outside an optimization campaign.

**Six further experiments run the REINVENT4 baseline**, named
`reinvent4_<program>_<prior|goal_directed>`. REINVENT4 is an external tool, not part of this
codebase: it lives at its pinned commit under `tools/`, is launched by
`experiments/reinvent4_*/submit_reinvent.py` with its own config structure, and is never reachable
from `submit.py` or from a `job_type`. Each has an `instructions.md` naming the single GraphINVENT2
job it mirrors, and its `config.yaml` records the fairness conditions in machine-readable form: the
mirrored job, the prior's split directory, the oracle block, the oracle-call budget, the sample size
and the seeds. `submit_reinvent.py` asserts each against the mirrored job's
`results/config.resolved.yaml` and refuses to run on a mismatch, so the fairness conditions are a
check rather than a paragraph. Its generated molecules are scored by this repository's own metric
functions through `evaluate_external_sample`, because a baseline scored by other code is not a
comparison.

Three priors and three campaigns, not six of each. The prior comparison answers whether a
graph-based model matches a SMILES-based one at distribution matching, which is the unconditional
claim. There is no REINVENT4 transfer arm, because comparing two different fine-tuning mechanisms
measures REINVENT4's implementation rather than anything about GraphINVENT2, and the transfer claim
already has its control in the from-scratch arm. There is no REINVENT4 conditional arm; the
conditional claim is falsified by `baseline_posthoc_`, not by another model.

**The three priors are never shared.** A PROTAC prior is not a starting point for a glue model and
neither is a starting point for a solvent model: the size regimes, the atom vocabularies and the
`max_nodes` values all differ, and a shared prior would silently impose one domain's geometry on
another.

**Every program follows the same transfer-learning shape**, which is a structural feature of the
system rather than a choice an experiment makes: a prior trained on a large base corpus, fine-tuned
on the smaller task corpus with `resume_from`, against a from-scratch control trained on the task
corpus alone. The comparison between those two arms is what the transfer-learning claim rests on, so
both are always run.

**Every experiment ends with a sampling job of 10,000 molecules over three seeds**, from the final
checkpoint or the best agent, and that sample is what the reported metrics are computed on. No
experiment reports metrics from a training job, from a trajectory, or from a smaller sample. Each
`instructions.md` names the sampling job that produces its numbers.

---

## Job types

Five job types. These names are the only accepted values; there are no aliases.

A job type is a distinct top-level entry point into the code, which is why the list is five and not
three or seven: the five are the dispatch keys in `main.py` and the five evaluation functions in
`src/metrics/_run.py`. A variation that reuses an entry point is configuration, not a job type.

| job_type | Purpose | Needs | Produces |
|---|---|---|---|
| `preprocess` | Turn a SMILES file into training-ready HDF5 and create the data splits | SMILES or TSV file | HDF5 per split (nodes, edges, action probabilities; `condition_vector` if property columns are present), the three split files, `preprocessing.yaml` recording exactly how the splits were made |
| `unconditional` | Train a generative model on a dataset | preprocessed HDF5 | checkpoint, training curves |
| `conditional` | Train a model that conditions generation on property values or a categorical label | preprocessed HDF5 containing `condition_vector` | checkpoint, training curves |
| `goal_directed` | Optimize a trained model toward an oracle with policy-gradient RL | checkpoint, oracle name | checkpoint, oracle call log, score trajectory |
| `generate` | Sample molecules from a checkpoint and evaluate them | checkpoint | `molecules.smi`, `metrics.json`, `report.html` |

Notes that are easy to get wrong:

- **Transfer learning is `unconditional` with `resume_from` set.** It is not a separate job type.
- **`conditional` requires** `condition_dim > 0` and a `condition_vector` dataset in the HDF5. Both
  are checked up front with a message naming what is missing.
- **`generate` has two modes.** `sample_mode: generate` writes molecules and computes the metrics
  below. `sample_mode: evaluate` computes NLL and UC-JSD.
- **Training jobs do not compute the core metric table.** A training job has no sample of its own and
  generating one doubles the cost of training, so it reports model efficiency and its training
  curves. Each experiment's `instructions.md` names the `generate` job that produces its core
  metrics, so nobody reads a training job's handful of keys as the whole table.

---

## Data splits

The splits are the most consequential thing `preprocess` does. Every number this project reports
depends on them, so they are defined once per dataset and never redefined afterward.

| Split | Used for | Never used for |
|---|---|---|
| **train** | Fitting model parameters. The reference set for novelty. | Anything reported as a result |
| **validation** | Model selection only: choosing the best checkpoint and early stopping, in `unconditional`, `conditional` and `goal_directed` training alike | Any evaluation metric |
| **test** | Held out until final evaluation. The reference set for all distribution-matching metrics, for NLL, and for sampling condition values in conditional evaluation | Training, early stopping, or any choice made during development |

A model selected on the test set has no held-out set left, so the number it reports is not an estimate
of anything. Keep the test split closed until the model is fixed.

**Splitting happens at the molecule level, before subgraph decomposition.** Preprocessing decomposes
each molecule into many subgraphs, so a split applied afterward would scatter subgraphs of the same
molecule across train, validation and test, and every reported number would be leakage. Split the
SMILES file first, then decompose each split separately.

**Four strategies, and the choice is recorded rather than assumed.** `preprocessing.yaml` records the
strategy, the seed, the fractions and the resulting counts.

| `split_type` | What it groups by |
|---|---|
| `scaffold` | Bemis-Murcko scaffold of the whole molecule. The default, because a random split of a molecular dataset puts close analogues on both sides and overstates generalization |
| `warhead_scaffold` | Bemis-Murcko scaffold of the target-binding component, for bifunctional molecules |
| `random` | Nothing. Use only when comparing against published numbers that used it, and say so |
| `custom` | A hook with no implementation; selecting it raises |

Groups are assigned largest first, with equal-sized groups shuffled under the seed rather than sorted
by scaffold string. Alphabetical tie-breaking looks harmless and is not: for a bifunctional corpus
most scaffolds are singletons, and every CRBN degrader's Murcko scaffold shares the same
glutarimide-phthalimide prefix, so they formed one contiguous alphabetical block that landed entirely
in the tail. Measured on the 21,429-molecule TPD-DB set, that put 64.5% CRBN in train against 99.6%
and 99.4% in validation and test, which made `fcd`, `nll_test` and checkpoint selection measurements
against one E3 chemistry.

**Bifunctional molecules need `warhead_scaffold`.** A Bemis-Murcko scaffold of a PROTAC contains every
ring system in the molecule and so is nearly unique per compound: measured on 476 decomposed TPD-DB
degraders, 95.4% have a scaffold of their own, against 51.5% for their warheads. A whole-molecule
scaffold split therefore degenerates into a random one and leaks analogues. On that same sample it
left 44.7% of test molecules sharing a warhead scaffold with training, against 0% under
`warhead_scaffold`. The strategy decomposes with PROTAC-Splitter at about 2 s per molecule, cached
under `tools/splitter-cache/`, and a molecule that yields no warhead scaffold forms its own group and
is counted, so the no-shared-warhead guarantee is stated over the molecules that yielded one.

**Splits are fixed per split set and reused.** A split set is identified by `split_type`, `seed`,
`train_frac` and `valid_frac`, and lives in `data/datasets/<name>/split_<type>/`. Every job in an
experiment, and every experiment on the same dataset, reads the same one; `split_type` is declared in
`experiment.yaml` and inherited by every job, exactly as `dataset_dir` is. Two models evaluated on
different test sets are not comparable, so a preprocess job pointed at an existing split directory
whose recorded `seed`, `train_frac` or `valid_frac` differ from the current config **raises**, naming
the key, both values, and the instruction to use a new `dataset_dir`. Re-running with identical split
parameters is the only case that proceeds. There is no force-resplit flag and no `restart` key: a
mismatch is never what anyone wanted, and the old behavior — move the previous splits into
`_previous_run_<timestamp>/` and write new ones — meant a model trained on one split was evaluated
against another's test set with nothing failing.

**Conditional jobs** draw condition values at generation time from the test split's distribution, not
the training split's. **Goal-directed jobs** train the prior on train; screening libraries for the
virtual screening baseline are declared per experiment and are none of these three splits.

---

## Configuration

Three layers, merged key by key, later wins:

1. `config/defaults.yaml` — every parameter, with its default and a one-line comment
2. `experiments/<exp>/experiment.yaml` — the dataset and paths shared across the experiment's jobs
3. `experiments/<exp>/jobs/<job>/config.yaml` — this job's settings

A key left unset is inherited; omitting a key *is* inheriting it, and there is no flag to turn that
on or off. Nested blocks merge key by key as well, so setting one key under `slurm:` must not discard
the rest of it. The merged configuration is written to the job's `results/config.resolved.yaml`, so
every run records exactly what it ran with. Config files are YAML, not JSON, so parameters can carry
comments; a comment that contradicts the code is a bug.

**Architecture keys come from the checkpoint.** When `resume_from` or `pretrained_model_path` is set,
all architecture keys (GGNN sizes, `condition_dim`, `condition_embedding_dim`, `condition_type`,
`conditioning`) are read from the checkpoint and cannot be overridden. A config that sets one to a
conflicting value is an error, not a silent override, because loading a model with the wrong
architecture either crashes confusingly or produces a silently wrong model.

**Reinforcement learning has its own schedule keys:** `rl_steps`, `rl_batch_size`,
`rl_best_agent_every` and `rl_eval_every`. A learning step is not an epoch, the two paths want
different values, and sharing the keys meant a change to the supervised defaults silently changed
every optimization run. A `goal_directed` config that sets `epochs` or `sample_every` is rejected
with an error naming the key it should have used. Defaults from Atance et al.: `rl_steps: 200`,
`rl_best_agent_every: 5`, `rl_eval_every: 5`, with `rl_batch_size` set per experiment.

**Three values are set per experiment and never inherited**, because a number nobody chose is worse
than no number: `success_threshold` with a `success_definition` string saying in words what was
counted, and `property_window` giving the bounds for `property_window_rate`. A global default for
these is meaningful for QED on the unit interval and meaningless for a geometric mean containing
binary filters, where a single failed component zeroes the product.

### Behavior-selecting keys

A key whose value selects a behavior validates that value against the behaviors the code implements,
and raises naming both the key and the implemented set. This is where "fail loudly" is either true or
decorative: a value that falls through to a default costs an afternoon at best, and at worst produces
results that look right and are not. The keys are `job_type`, `split_type`, `sample_mode`, `device`,
`score_type`, `score_components`, `decoding_route`, `condition_type`, `conditioning.source`, and each
oracle's `type`.

Two of these need more than a value check, because the value is consistent and the *correspondence*
is not. `conditioning.properties` must name the property columns of the SMILES file in the file's
order: the condition vector is built positionally from those columns while generation labels each
slot from the config, so a transposed declaration conditions on one property and reports it as
another, and both orderings write a byte-identical HDF5. And property columns present in the file
while `conditioning` is unset are an error, not a silent drop to an unconditional dataset.

---

## Evaluation metrics

Every metric has exactly one implementation and it lives in `src/metrics/`. Do not compute a metric
inline anywhere else. A sample from an external model is scored by the same functions:
`evaluate_external_sample` takes a `.smi` path and computes the core table, FCD, the triage stack and
the component metrics exactly as `evaluate_generation` does. It differs in one respect and it is the
one that matters: novelty's reference corpora cannot be discovered by walking a checkpoint lineage,
because there is no checkpoint, so they are passed explicitly and **asserted equal** to those of the
arm being compared against — a novelty comparison against a different reference set is not a
comparison. `nll_*`, `uc_jsd` and `delta_nll` are properties of a model this code cannot evaluate, so
they record `not_applicable_to_job_type` nulls. `Analyzer` owns the model forward pass that produces the likelihood tensors,
because only it holds the model and the loaders, but the arithmetic that turns them into `nll_*` and
`uc_jsd` is in `src/metrics/_likelihood.py` with every other metric.

### The metrics.json envelope

```
{
  "schema_version": 1,
  "run":         { job_name, job_type, dataset, seed, device, timestamp },
  "metrics":     { flat, keys exactly as tabulated below },
  "unavailable": { "<key>": { "category": ..., "reason": ... } }
}
```

The `metrics` block is flat, so runs compare without a translation layer. Every key whose value is
null has an entry in `unavailable` carrying a human-readable reason and a category from exactly this
set: `optional_tool_missing`, `insufficient_sample`, `not_applicable_to_job_type`, `not_implemented`.
A null that carries its reason is worth more than a flat dictionary, because a reader cannot
otherwise tell "undefined for this run" from "the computation returned nothing". One reader is used
by the HTML report, the cross-seed aggregator and any plotting.

### Reporting conventions

- Sample **N = 10,000** molecules and run **3 seeds**; report mean and standard deviation. What a seed
  controls differs by job: in a `generate` job it controls sampling from a fixed checkpoint, so one
  job runs the seed list and reports the spread across them; in a `goal_directed` job it controls the
  optimization run itself, so each seed is its own job and the aggregate is written to
  `experiments/<exp>/results/aggregate.json`. Nothing reported comes from a single seed, and nothing
  is aggregated by hand in a notebook.
- **Key names are exactly as tabulated below.** No reference-set suffixes: the reference set for each
  metric is fixed by its definition here, and the definition is the documentation. Four additions are
  allowed: a `baseline_lookup_` or `baseline_posthoc_` prefix for a conditional metric's baseline; a
  `_<class>` suffix for the per-class breakdown of a categorical conditional metric; the funnel counts
  `n_sampled`, `n_valid`, `n_unique_valid` and `n_unique_valid_novel`; and a **supporting field**
  named `<metric>_<field>` recording a denominator, a threshold or a parameter its metric was
  computed with. A number whose meaning depends on how many molecules reached it, or on a threshold,
  is not interpretable without them, and a reader who has to open the code to find the denominator
  will not. The supporting fields are `fcd_n_generated` and `fcd_n_reference`,
  `validity_n_generated_raw`, `property_window_rate_bounds`, `success_threshold`,
  `novelty_reference_corpora` and `novelty_n_reference_molecules`,
  `reinvent4_commit` and `reinvent4_mirrored_job`,
  `success_definition`, `condition_dropout`, `triage_stages`, `triage_pass_rate`,
  `triage_n_entered`, `triage_sa_max`, `triage_retrotac_min`, `<stage>_pass_rate` and
  `<stage>_n_reached` for each triage stage,
  `n_scaffolds`, `n_split_successfully`, `n_warheads_usable`,
  `split_status_counts`, `top100_n_split`, `top100_split_success_rate`, `screening_library`, `screening_library_size`,
  `sa_score_n_unscored`, `e3_similarity_floor`,
  `e3_max_similarities`, `e3_n_tested`, `e3_n_cores`, `ecfp4_radius`, `warhead_representation`,
  `warhead_fingerprint_radius`, `warhead_fingerprint_bits`, `knn_reference_pool_size`,
  `class_match_threshold`, `class_panel`, `n_unique_valid_<class>`, `delta_nll_n_molecules`,
  `delta_nll_n_alternatives_excluded`, the splitter model and version beside every component metric,
  and the subsample size and seed wherever a metric is subsampled (`component_sample_size` and
  `component_sample_seed` for the component metrics). A supporting field is not a metric and
  nothing is ranked on one.
- **Metrics come from the sample that matches their meaning.** For a goal-directed job, budget and
  top-100 metrics are computed over the optimization trajectory, because that is what spending oracle
  calls means; generation-quality metrics are computed over a fresh sample from the best agent,
  because that is the model you would deploy. Each metric's docstring and the report state which.
- **Validity, uniqueness and novelty compose as a funnel**, computed sequentially on the surviving
  subset, each reported both as a conditional rate and as a fraction of N.
- **Validity means parsed and sanitized without repair.** RDKit is not permitted to fix valences or
  charges. A molecule that only parses after repair is invalid.
- **The generated list is truncated to exactly `n_samples` before any metric touches it.** Batches
  over-produce, and a denominator of `n_samples` against an untruncated list puts validity above 1.0
  on exactly the good models worth reporting. `validity_n_generated_raw` records what the generator
  produced, so the over-production is visible rather than hidden.
- **Both sides of every comparison are canonicalized identically**, at the level of chemistry the
  model can express. Comparing a flattened generated set against a stereochemistry-bearing reference
  makes a molecule the model copied exactly count as novel, which pins novelty near 1.0.
- **Every conditional metric is reported alongside its unconditional baseline**, meaning the same
  model sampled the same way with no condition set. A conditional number with no baseline cannot be
  read, because nothing separates conditioning from the base rate of the corpus.
- **Every sample metric is computed twice, before and after triage, and both are reported.** The
  second set carries a `_filtered` suffix so the two are unmistakable in one table. Both numbers
  matter and neither replaces the other: the unfiltered set says what the model actually produces,
  which is what a generative result is, and the filtered set says what survives triage, which is
  what a project would carry forward. A paper quoting only the filtered numbers is describing the
  filter; one quoting only the unfiltered numbers is describing something nobody would make. The
  denominator of a `_filtered` rate is still `n_samples`, so `validity_filtered` reads as the
  fraction of the sample that both parsed and survived. Model properties have no filtered twin,
  because removing molecules from a sample they were never computed over changes nothing: `nll_*`,
  `uc_jsd`, `delta_nll` and the efficiency keys.

### The triage stack

Named in the output as `triage_stages`, with a pass rate at each stage denominated over the
molecules that reached it, and `triage_pass_rate` end to end.

| Dataset | Stages |
|---|---|
| drug-like | `structural_alerts` (MCF + PAINS), then `synthesizability` (`sa_score < 4.5`) |
| bifunctional | `structural_alerts`, then `retrotac` (`retrotac >= 0.30` on the whole molecule), then `warhead_synthesizability` (`sa_score < 4.5` on the decomposed warhead) |

Two stacks, because synthesizability is not one measurement. A drug-like molecule is gated on its
own SA score. A bifunctional one cannot be: SA score reads a PROTAC's size as difficulty and rejects
chemistry that demonstrably exists, so the whole molecule is gated on RetroTAC and the SA gate moves
to the warhead, where the fragment is drug-sized and SA behaves. A molecule with no RetroTAC score
or no usable warhead fails that stage rather than passing it, and the stage rates make that visible:
a collapsing `warhead_synthesizability_pass_rate` with a collapsing `split_success_rate` beside it
is a decomposition problem, not a chemistry one.

### Core metrics, reported for every setting

| Key | Definition |
|---|---|
| `validity` | valid / N |
| `uniqueness` | unique canonical SMILES among valid / valid |
| `novelty` | unique valid molecules absent from **every** training split in the checkpoint's lineage / unique valid. A transfer-learned model that reproduces a base-corpus molecule has not produced a novel one, so the reference is every corpus the model has seen and `novelty_reference_corpora` records which |
| `vun` | unique, valid, novel / N (the funnel composite) |
| `near_duplicate_rate` | fraction of unique valid molecules whose maximum ECFP4 Tanimoto to any training molecule exceeds 0.9 |
| `internal_diversity` | mean pairwise Tanimoto distance (1 − similarity) on ECFP4 (Morgan, radius 2, 2048 bits) over unique valid molecules, subsampled to 10,000 |
| `scaffold_diversity` | unique Bemis-Murcko scaffolds / unique valid molecules |
| `sa_score_mean`, `sa_score_median` | synthetic accessibility over unique valid molecules. Whole-molecule for a drug-like dataset; over the decomposed warheads for a bifunctional one, because SA score reads a PROTAC's size as difficulty |
| `filter_pass_rate` | fraction of unique valid molecules passing MCF and PAINS structural alerts |
| `Ro5_pass_rate` | fraction of unique valid molecules with at most one Rule of 5 violation |
| `property_window_rate` | fraction of unique valid molecules whose every named property falls inside its window, with the bounds echoed verbatim into `property_window_rate_bounds` |

`near_duplicate_rate` exists because binary novelty counts a molecule differing by one methyl as
novel. A model with high novelty and a high near-duplicate rate is memorizing.

`property_window_rate` is one parameterized metric rather than a family of per-domain ones, because
the alternative was a degrader-only metric, then a battery-only third and a development-set fourth.
It replaced `bRo5_pass_rate`, which was ill-posed: "beyond Rule of 5" names the region compounds fall
*into* by exceeding Ro5, not an envelope they stay *within*, so a pass rate against it had no
definition. Ro5 bounds are the default, a degrader window and the objective's property set are set by
those programs, and every table says which window it used.

### Unconditional only

| Key | Definition |
|---|---|
| `fcd` | Fréchet ChemNet Distance between generated molecules and the test split |
| `nll_train`, `nll_validation`, `nll_test` | mean negative log-likelihood of a 10k sample from each split under the model (`sample_mode: evaluate`), per molecule and summed over construction actions. Comparable across splits of one dataset, not across datasets with different molecule sizes |
| `uc_jsd` | uniformity-completeness JSD (`sample_mode: evaluate`) |

FCD is the headline number for unconditional generation, because V.U.N. saturates on modern models
and no longer separates them while distribution match does. It needs **at least 1,000 unique valid
molecules on each side**; below that it is not computed and goes to `unavailable` with category
`insufficient_sample` and both counts in the reason. ChemNet activations are 512-dimensional, so a
sample covariance from fewer molecules is singular and the result is a different and much larger
quantity rather than a noisy estimate of the same one. Between 1,000 and 10,000 it is computed and
flagged as biased upward and not comparable to published numbers, which use 10,000 (GuacaMol) or
30,000 (MOSES). `fcd_n_generated` and `fcd_n_reference` accompany every value. On a 1,000-molecule
development set the reference split is around a hundred molecules, so `fcd` is unavailable there by
design and no tutorial presents a development-set FCD as a result.

### Conditional, continuous properties

| Key | Definition |
|---|---|
| `condition_success_rate_<property>` | fraction of generated molecules within `condition_tolerance` of the value requested of them |
| `condition_mae_<property>` | mean absolute difference between the requested and the achieved value, in the property's units, over the same molecules |
| `condition_coverage_<property>` | per-decile success rate across the training property distribution, reported as ten numbers, with null for a decile no request fell into |

`condition_mae_<property>` is reported beside the success rate because that rate is a step function of
`condition_tolerance`: two models whose typical error differs threefold score identically when both
errors sit inside the tolerance, and a model that halves its error moves the rate not at all until it
crosses the threshold. The rate says whether the target was hit, the error says by how much it was
missed, and no tolerance chosen after the fact can flatter the second. Requests are drawn across the
training deciles rather than held at one value, since a single value populates only one decile and
`condition_coverage` exists to detect a model that works only near the middle of the distribution.

### Conditional, categorical labels

Used when the condition is a class rather than a value, such as a target protein. The continuous
metrics above have no definition in this case.

**The condition is a one-hot vector that the model embeds.** The HDF5 `condition_vector` is a
float32 one-hot of width equal to the number of classes, built positionally from the file's columns;
`condition_dim` equals the number of classes; and `ConditionEncoder`, a two-layer MLP, learns the
embedding from it end to end. No protein language model embeddings, no sequence features, no
pretrained protein representations. With three classes and a few thousand molecules a learned
embedding of a one-hot is everything the data supports, and a thousand-dimensional sequence
embedding would add variance the training set cannot constrain.

**`conditioning.properties` IS the class-to-index mapping**, recorded in the dataset's
`preprocessing.yaml` and in the checkpoint's resolved config. It is positional: slot *i* of the
condition vector is class *i* of that list, in both places. A permuted list therefore conditions on
one class, reports it as another, and writes a byte-identical HDF5 — there is no shape, no count and
no checksum that differs — so a dataset whose mapping is not the checkpoint's raises, and a class
named in `sample_conditions` that is not in the panel raises and names the classes that exist.

**Conditional training drops the condition on `condition_dropout` of its training examples**,
default 0.1, drawn under the job seed. This is not a regularizer, though it looks like one. Every
conditional metric is reported against an unconditional baseline, meaning the same model sampled
with no condition set, and with a one-hot condition "no condition" is the all-zeros vector. A model
trained only on one-hot inputs has never seen it, so the baseline would measure what the model does
when asked a question outside its training distribution, and the conditional-minus-baseline gap —
which is the entire claim — would be partly an artifact of that. Dropping the condition makes zeros
a legitimate input meaning "unspecified", so the baseline becomes a real measurement and the model
keeps a usable unconditional mode. Validation keeps its conditions, so model selection picks the
best conditional model rather than the best average of two modes. The value is recorded in
`metrics.json`, because a conditional number from a run that trained without it cannot be compared
with one that did.

The panel is declared with `condition_classes`, a list of the one-hot column names. Nothing else can
tell a class column from a property that happens to take the values 0 and 1, and the two metric sets
are disjoint: `condition_success_rate` within a tolerance of 0.5 of a requested 1.0 measures whether
the model echoed its own input.

| Key | Definition |
|---|---|
| `delta_nll` | for a held-out molecule of class *p*, the mean over competing classes *q* of NLL(x\|q) − NLL(x\|p), excluding legitimate alternative classes of a multi-label molecule. Positive means the molecule is more likely under its own condition. Supporting: `delta_nll_n_molecules`, `delta_nll_n_alternatives_excluded` |
| `knn_purity_<5\|10\|20>` | fraction of each generated molecule's *k* nearest neighbors in the held-out reference pool, by Morgan-Tanimoto, carrying the requested label. Supporting: `knn_reference_pool_size` |
| `class_match_median`, `class_match_rate` | median, and fraction above `class_match_threshold`, of the maximum ECFP4 Tanimoto from each generated molecule to the training molecules carrying the requested label |
| `class_selectivity_margin` | the same maximum Tanimoto computed against the other classes in the panel, subtracted from the on-class value |
| `novelty` | the core metric, reported here too and per class, because it is what separates conditioning from retrieval |

**Every key above is the macro-average across the panel**, with the per-class value under the same
name plus a `_<class>` suffix and `n_unique_valid_<class>` as its denominator. Macro rather than
micro, because a panel is never balanced: the classes differ by an order of magnitude in how many
training molecules they have, and a micro-average reports the largest class's number under a name
that reads as the panel's. The full per-class table, for all three blocks below, is written to
`results/condition_per_class.json`. One `generate` job covers the whole panel with
`sample_conditions: all_classes`, which cycles the one-hot across generation batches.

`class_selectivity_margin` is the cleanest single number here because it is self-normalizing: an
unconditional model scores approximately zero by construction, and the metric cannot be satisfied by
generating generic chemistry.

**Two baselines, and the metrics are not evidence without them.** Both are computed by the same
function as the conditional sample, so all three have identical shape, and both are prefixed:

| Prefix | Baseline |
|---|---|
| `baseline_lookup_` | the training molecules carrying the requested label, sampled to the same N per class with the job's seed. The retrieval ceiling. A conditional model cannot beat it on `class_match_*` by construction, since the lookup set *is* training chemistry, so the comparison that matters is novelty: it scores `class_match_median` 1.0 at `novelty` 0.0, and a conditional model that matches it on both is retrieving rather than conditioning |
| `baseline_posthoc_` | the unconditional model sampled the same way, then reranked per class by maximum similarity to that class's training molecules and cut to the same N. **This is the one that can falsify the claim.** If an unconditional model plus a one-line reranking reaches the same `class_selectivity_margin`, the conditioning bought nothing |

The post-hoc baseline needs `baseline_results_dir` pointing at the unconditional model's
`molecules.smi`; without it those keys are an explicit null carrying that reason, and the conditional
numbers can be reported but not falsified. A molecule may be selected for more than one class in the
post-hoc baseline, which is correct: it is not a partition of the unconditional sample but what the
best post-hoc filter per class would return.

### Component-level metrics, for bifunctional molecules

Whole-molecule statistics for a PROTAC are dominated by the linker, which is the least informative
part, so the conditional and goal-directed metrics above are computed on the decomposed
target-binding component rather than the whole molecule. A dataset declares itself with
`bifunctional: true`, which gates every metric in this section.

**Synthesizability splits by level.** SA score cannot be used at the whole-PROTAC level: it reads
size as difficulty, and on 500 real TPD-DB degraders the median SA is 4.71 with 58.6% failing
`sa_score < 4.5`, against 2.87 and 6.2% for drug-like ZINC. Those degraders were made and tested, so
the measure rejects chemistry that demonstrably exists. So **RetroTAC scores the whole PROTAC and SA
score scores the warheads** after decomposition, where the fragments are drug-sized and SA behaves.
Both are reported and neither replaces the other. Molecular glues and any drug-like dataset keep SA
score at the whole-molecule level and record no RetroTAC.

| Key | Definition |
|---|---|
| `retrotac_mean`, `retrotac_median` | route-derived synthesizability of the **whole** PROTAC, over unique valid molecules, from the released RetroTAC GNN refit `gnn_20260828_182305_final` (Ribes et al., arXiv:2610.13639; held-out R² 0.615, RMSE 0.140), which is the synthesizability model this work uses. Not clipped to [0, 1]: a value outside it means the molecule is far from the model's training distribution |
| `split_success_rate` | **stage one only**: fraction of the sample PROTAC-Splitter decomposed into three fragments. Supporting: `n_split_successfully` |
| `warhead_usable_rate` | **stage two**: of those, the fraction whose target-binding fragment survived capping and canonicalization into a molecule that can be fingerprinted. Supporting: `n_warheads_usable`, `split_status_counts` |
| `warhead_*` | the core and conditional metrics recomputed over target-binding components, denominated over the usable warheads |
| `warhead_novelty_in_class` | of the warheads generated for one requested class, the fraction absent from *that class's* training warheads; macro-averaged over the panel, with `_<class>` per class |
| `e3_recognizable_rate` | fraction of usable splits whose predicted E3 fragment contains a recognized E3-binder core as an exact substructure. A validity check, not a diversity analysis. Supporting: `e3_n_tested`, `e3_n_cores`, `e3_max_similarities`, `e3_similarity_floor` |
| `top100_warhead_scaffold_count` | distinct Bemis-Murcko scaffolds among the target-binding components of the 100 highest-scoring molecules of a `goal_directed` job |

**Two denominators, not one.** `split_success_rate` and `warhead_usable_rate` are separate stages
whose product is the end-to-end rate, and the two counts make it reconstructable. One combined rate
hid which stage was failing, and the stages fail for unrelated reasons: a molecule the splitter would
not decompose is a decomposition problem, and a fragment that will not survive capping is a problem
with the cut it chose. `split_status_counts` breaks the second down by reason, drawn from a fixed
vocabulary: `ok`, `no_prediction`, `malformed_prediction`, `parse_failed`, `capping_failed`,
`canonicalization_failed`. The per-molecule record — requested class, canonical molecule, raw
fragment, processed warhead, status and splitter model — is written to
`results/component_records.json`, so both denominators and every failure are recoverable without
re-running a decomposition that costs hours.

Every component-level metric is computed only over molecules that reached it, so a rate is always
reported beside it: the two stage rates over the whole sample, and `top100_split_success_rate` beside
`top100_warhead_scaffold_count`, over the same hundred. A falling split rate otherwise masquerades as
improving component quality.

**The attachment point is capped before anything is computed.** A fragment comes back carrying the
`[*:n]` dummy atoms that mark where it was cut, and fingerprinting those compares a molecule with
extra atoms against reference ligands without them: on a real predicted E3 fragment, ECFP4 Tanimoto
to thalidomide was 0.711 with the dummy and 1.000 once capped, and its SA score 3.558 against 2.597.
`warhead_representation` records which reading was used — `parent_warhead`, the default, caps with
hydrogen and gives the ligand as it exists off the degrader; `attachment_aware` keeps the dummy, so
the same ligand cut at two positions counts as two warheads — alongside
`warhead_fingerprint_radius` and `warhead_fingerprint_bits` (Morgan, radius 2, 2048 bits). Every
`warhead_*` number depends on all three, so all three are in `metrics.json`.

**`e3_recognizable_rate` is a substructure test, not a similarity threshold.** An E3 family is
defined by a pharmacophore that is present or absent, and a Tanimoto floor answers "how close is this
to some known ligand" instead; the two disagree in both directions, since a glutarimide carrying a
large unfamiliar substituent scores low while being a perfectly good CRBN binder. The cores are in
`tools/data/e3_binder_cores.smarts`, one per line with its citation: glutarimide for CRBN, the
hydroxyprolinamide of VH032 for VHL, the nutlin diaryl-imidazoline for MDM2, and the AVPI mimetic for
IAP. Each was verified to match its family's reference ligand and no other family's. **On 500 real
TPD-DB degraders, 84.2% contain a listed core**, so that is roughly the ceiling a perfectly
decomposed set of real degraders would score and a generated set near it is doing as well as the list
allows rather than well in absolute terms. The maximum ECFP4 Tanimoto to a reference ligand is still
reported as a supporting field so the verdict can be read against it.

**For a categorical conditional job the component table is also computed per class**, as
`split_success_rate_<class>`, `warhead_usable_rate_<class>`, `warhead_uniqueness_<class>`,
`warhead_diversity_<class>`, `n_unique_warheads_<class>` and `warhead_novelty_in_class_<class>`, with
`n_unique_valid_<class>` as the denominator. Only `warhead_novelty_in_class` is macro-averaged under
its bare name: the other four already have that name at the whole-sample level, computed over the
whole sample rather than as a mean over classes, and one key cannot carry two definitions. A class
with fewer than two unique warheads reports `warhead_diversity_<class>` as a categorized null rather
than 0.0, which would read as total collapse.

Measured on a twenty-target panel in earlier work, macro-averaged: split coverage around 0.95,
warhead uniqueness around 0.27, warhead diversity around 0.77 and in-class novelty around 0.89. Those
are the magnitudes to expect; a panel far from them is worth explaining before it is reported.

**The sample is subsampled.** Decomposition costs seconds per molecule, so component metrics are
computed over at most 2,000 unique valid molecules drawn with the job's own seed, recorded as
`component_sample_size` and `component_sample_seed`. A rate over a subsample is a different number
from a rate over the sample, and at 2,000 the sampling error on a rate is well under a percentage
point.

`top100_warhead_scaffold_count` is the primary reward-hacking detector for a bifunctional dataset,
because `top100_scaffold_count` cannot be one: a whole-molecule PROTAC scaffold is nearly unique per
compound, so a top 100 collapsed onto a single warhead still reports close to 100 distinct scaffolds
as long as the linkers differ. Measured on 12 real degraders, 12 distinct whole-molecule scaffolds
against 10 warhead scaffolds over the 11 that decomposed. The whole-molecule count stays reported
beside it and is expected to be uninformative, which is the evidence that the warhead count is the
one to read.

**The splitter policy.** PROTAC-Splitter's `adaptive` model is primary for every cluster and final
run; `xgboost` is selectable for development iteration, because adaptive costs 2.045 s per molecule
serially against 0.143. Decomposition runs a pool of interpreters sized from `SLURM_CPUS_PER_TASK`
and capped at eight, measured at 0.483 s per molecule — a 4.2x speedup — over a seeded 2,000-molecule
subsample, so a 10,000-molecule sample that was 5.7 hours serially is about nine minutes. The fixed
cost of an interpreter is about 21 s and inference dominates it a hundred to one, which is why this
is a pool and not a larger batch; `tools/README.md` carries the measurement. The model and version are recorded
in `metrics.json` either way, results are cached on disk by model, and the `xgboost` sensitivity pass
runs on every final generated set. On 500 real degraders adaptive decomposed 492 and xgboost 472, the
second a strict subset of the first, and the two return identical fragments for 329 of the 472 both
decomposed, which is 69.7%. **Neither model is ground truth, so that is a concordance and not an
accuracy.** It bounds how much confidence any absolute component-level number deserves: report
component results under both models as a band, restrict headline claims to the agreeing subset, state
its size, and if a conclusion flips between models it does not go in the paper.

### Goal-directed only

| Key | Definition |
|---|---|
| `best_at_<1k\|5k\|10k>`, `top10_mean_at_<...>`, `top100_mean_at_<...>` | oracle score statistics at fixed budgets |
| `auc_top10` | area under top-10 mean against oracle calls, up to 10,000 |
| `vs_baseline_top10_at_<...>` | the same top-10 mean for B molecules drawn uniformly at random from a declared screening library, same oracle, same seeds |
| `vs_delta_top10_at_<...>` | generative minus virtual screening, the headline comparison |
| `r4_baseline_top10_at_<...>` | the same top-10 mean for the mirrored REINVENT4 run, same oracle, same budget, same seeds |
| `r4_delta_top10_at_<...>` | GraphINVENT2 minus REINVENT4. Null with its reason when `reinvent_baseline_results` is unset |
| `success_rate` | fraction of unique valid molecules scoring at or above `success_threshold`; multiple property thresholds combine with AND |
| `top100_scaffold_count`, `top100_internal_diversity` | diversity among the 100 highest-scoring molecules |
| `quality_filtered_top10_mean` | top-10 mean among molecules passing structural filters and a synthesizability gate: `sa_score < 4.5` normally, and `retrotac >= 0.30` for a bifunctional dataset |
| `eval_oracle_top10_mean` | the same top-10 molecules rescored by the evaluation oracle |
| `top100_outside_domain_rate` | fraction of the top 100 whose maximum ECFP4 Tanimoto to the surrogate's training set falls below a stated floor |

**Every budget-indexed metric integrates the composite score the policy was trained on**, not any
single component. With one component those are the same thing; a six-component geometric mean is
zeroed by one failed component, so a curve over one part describes something the model was not
optimizing.

An oracle call is one *unique* molecule submitted to the oracle; cache hits are free and are not
counted, and the oracle call log written per job is the authoritative record of the budget spent.
Because the oracle is usually a surrogate costing milliseconds, the budget is a stand-in for how many
compounds a project could afford to make and test, not a real constraint. That is also why the
screening baseline matters: it spends the same notional budget in the most naive way available.

The two synthesizability gates are matched on permissiveness rather than on value, each admitting
about 94% of the real molecules of the kind it applies to: `sa_score < 4.5` passes 93.8% of drug-like
ZINC and `retrotac >= 0.30` passes 94.0% of real degraders. The RetroTAC floor is 0.30 and not the
paper's 0.7, which its authors describe as a reporting convention rather than a calibrated boundary
and which only 18.8% of real degraders reach.

**When the oracle is a learned surrogate, two are trained.** The *optimization* surrogate is fitted
on the training and validation splits only and provides the RL reward. The *evaluation* surrogate is
the same featurization and model class refitted on all data including the test split, and only scores
the final molecules. Both are reported. A large gap on optimized molecules, when the two agree on
held-out real molecules, is evidence of exploitation rather than success.
`top100_outside_domain_rate` catches the same failure from a direction neither surrogate shares,
since both have the same blind spots.

The diversity and quality rows exist to detect reward hacking, which is the default behavior of RL
against a cheap oracle. Overall diversity stays high while the top-scoring set collapses onto one
scaffold, so `top100_scaffold_count` is the metric that catches it and `internal_diversity` is not;
on a bifunctional dataset it is `top100_warhead_scaffold_count`. `quality_filtered_top10_mean` is the
headline score.

A generative method that does not beat random library screening at matched budget has not
demonstrated anything, so the virtual screening baseline runs for every optimization experiment. At
minimum the library is a random matched-size sample from the training and validation splits; an
experiment may declare additional libraries, and all declared libraries are reported.

### Model efficiency, reported for every trained model

`n_parameters`, `train_wall_clock_s` with the device recorded, `inference_s_per_10k` per device
tested, and `peak_memory_mb`.

---

## Plots

Every plot follows the AIME Lab plot style guide. The palette and the matplotlib rcParams live in
`src/metrics/plot_style.py`, copied verbatim from the guide; every figure-producing function imports
them and sets nothing else, with no per-plot colors, fonts or sizes. The palette, in cycle order, is
Deep Teal `#2D4F54`, Royal Purple `#7B539E`, Soft Lavender `#B8A0D2`, Forest Green `#5A9448`, Muted
Plum `#9E4A78` and Dark Slate `#3A3D4A`, with Dark Slate for all text, axes and annotations in place
of black. Sans serif throughout, 11 pt labels and 9.5 pt ticks, all four spines visible at 1.2 pt, no
grid, no legend frame. Never put a title inside a plot; titles belong in the caption. Panel labels go
outside the axes, left of the y-tick labels, via `ax.text(-0.18, 0.97, "(a)", transform=ax.transAxes,
fontsize=13, fontweight="bold", va="top", ha="left", color=DARK_SLATE)`. Axis labels carry units.
Every categorical distinction is carried by marker or line style as well as color, so the figure
survives greyscale. Save every figure as both `.png` at 300 dpi and `.pdf` into the job's `results/`
directory. Any inline SVG in `results/report.html` uses these same hex values, so the report and the
paper figures match.

Every figure in `src/metrics/_figures.py` passes through one grid builder, which installs the style
and allocates one axes per panel that has data. The style goes in there and nowhere else, so no
figure can be drawn without it: one drawn without it still draws, in default matplotlib blue, and
nothing raises. A panel whose input is missing is not drawn at all and the job prints a line naming
what was missing, because an empty axes with default 0-1 ticks reads as a measurement that came out
empty rather than one that was not made.

**What each job type draws.** `preprocess`: the heavy-atom distribution with `max_nodes` marked, and
the molecule count per split with the strategy and seed in the axis label. `unconditional` and
`conditional`: training and validation loss on one axes, with the from-scratch control overlaid when
`control_job_dir` names one, because reaching the same final loss in fewer epochs is the transfer
claim and one curve cannot show it. `generate`: the funnel with counts annotated, the distribution of
maximum Tanimoto to the training split, and generated against test distributions of MW, logP and SA
as z-scores against the test split. A conditional job adds the requested-against-achieved scatter
with the identity line and the decile coverage, or for a categorical panel the per-class bars with
both baselines. `goal_directed`: the score trajectory, the budget curve with the virtual screening
baseline on the same axes, and `top100_warhead_scaffold_count` against learning step where the run
recorded it.

matplotlib is the only plotting dependency. Do not add seaborn, plotly, or anything else.

---

## Tests

Keep `tests/` to these four. Each catches a failure that is silent otherwise, and all must pass on a
laptop with no GPU:

1. **Preprocessing round-trip.** SMILES → HDF5 → graph → SMILES returns the same canonical SMILES.
   Catches silent corruption in the graph representation, which otherwise shows up only as poor
   samples much later.
2. **Split disjointness.** No molecule appears in more than one split, and no subgraph in one split
   derives from a molecule in another. Catches the leakage described under [Data splits](#data-splits).
3. **Conditional tensor shapes and gradients.** The condition vector reaches the model, the virtual
   node is stripped before readout, and every parameter that should receive gradient does. Catches a
   condition signal that is silently ignored, which looks exactly like a model that failed to learn.
4. **One smoke test per job type.** A few steps on a tiny dataset, asserting the job completes and
   writes its expected outputs. Catches interface breakage.

A fifth file, `tests/test_metric_sanity.py`, is ratified: the four above exercise `src/graphinvent/`
and none imports `src/metrics/`, which is where every reported number comes from. A sixth,
`tests/test_guards_fire.py`, is ratified on the justification recorded at its top: a
behavior-selecting guard is correct code that does not run, and two shipped dead because the check
sat where a value is read from a config layer while almost every parameter is inherited rather than
set, so the check ran on the rare path and skipped the common one. Every case in it inherits the bad
value from the experiment layer, which is the path that was dead. Any further test needs a one-line
justification of the specific failure it catches, recorded at the top of the file. Delete tests that
do not earn this.

`pytest-timeout` is in the dev extra with a default timeout, because a metric that stops terminating
hangs the suite at 100% CPU with no error rather than failing. Every test that guards a fix is
mutation-tested: break the thing deliberately, confirm the test fails, revert. A test that passes
against the broken code is worse than no test.

---

## Architecture notes

Only the non-obvious parts are recorded here. Read the code for the rest.

**Data flow.** SMILES → `DataProcessor` → HDF5 (per-subgraph nodes, edges, action probabilities) →
`BlockDataLoader` → GGNN → action logits → `GraphGenerator` (add atom / add bond / terminate).

**Conditional generation.** When `condition_dim > 0`, the GGNN builds a `ConditionEncoder` (2-layer
MLP) and one extra `msg_nn` for a virtual edge type. The condition vector is encoded into a seed
embedding that initializes a virtual node prepended to the graph; every real node receives messages
from the seed at each round. The seed is stripped before readout, so the output shape is unchanged,
but the readout *input* is not: the condition embedding is also concatenated to the graph embedding,
making `graph_emb_size = gather_width + condition_embedding_dim`. That concatenation is load-bearing,
because the first action of every molecule is taken on an empty graph where no real node exists to
carry the seed's message.

`DataProcessor` stores `condition_vector` (float32, `n_subgraphs × condition_dim`) in the HDF5, built
positionally from the file's property columns. `HDFDataset` yields 4-tuples when that dataset exists
and 3-tuples otherwise.

**Oracle caching.** `src/oracles/` caches by canonical SMILES and component identity. The cache is
what makes the oracle-call accounting correct, so it is not an optimization detail; changing it
changes the reported sample efficiency.

---

## Hard constraints

- **Run from the repo root with `python submit.py`.** The editable install writes a `.pth` file that
  does not survive a repository path containing spaces, so console entry points and
  `from metrics import ...` fail. Moving the repository to a path without spaces would remove this
  constraint; until then, do not rely on the installed entry point.
- **`[Xe]` is not a molecule.** Invalid graphs must not be written to `.smi` as `[Xe]`, because RDKit
  parses that as valid xenon and it silently inflates validity. Write nothing and record the index.
- **Generated-file line alignment.** `.smi`, `.likelihood` and `.valid` line up by line number, with
  no per-batch headers.
- **The test split is not touched during development.** Not for early stopping, not for a quick
  sanity check, not for choosing a hyperparameter.
- **Evidence outlives the session.** A number a reviewer might ask about has its source committed
  under `manuscript/evidence/`. Numbers whose evidence has evaporated are not numbers.

---

## Status

Verified by running the code on 2026-10-19.

**The three features this section used to list as unbuilt are built.** The
categorical conditional metrics with their two baselines, the whole-sample
component metrics in their two-denominator form, and per-job figures for all
five job types now have callers, tests and output.

**Every key this document specifies is now emitted by a code path.** The two
that remained unwired, `eval_oracle_top10_mean` and `top100_outside_domain_rate`,
are wired: a `goal_directed` job reads `evaluation_oracle` and
`surrogate_training_set`, and records a categorized null naming the key to set
when an experiment declares neither. `protac_goal_directed` and
`protac_selectivity` declare both.

What follows is what has not been *verified*, which is a different claim from
what has not been written.

**Decisions that are not mine to take.** Each changes the scope of an experiment or the claim made
from it, so each is listed with a recommendation rather than chosen.

- **The molecular glue surrogates cannot be fitted from TPD-DB.** Measured 2026-10-19: GSPT1 has 30
  distinct compounds with a pDC50 or Dmax label, IKZF1 has 5, IKZF3 has 1 and RBM39 has 0, against a
  floor of about a hundred. So `surrogate_mg_optimization` has no usable fit and
  `mg_goal_directed` has no learned reward available. *Recommendation:* run `mg_goal_directed`
  against a property-shaping objective rather than an activity surrogate, state in the manuscript
  that no glue activity model was fittable at this data volume, and keep the fit script so the
  counts reproduce. The alternative, sourcing a larger glue activity set, is a data question rather
  than a code one.
- **No ordering is expressed as a SLURM job dependency.** `config/slurm.sh.template`, `submit.py`
  and every rendered `run.sbatch` carry no `--dependency=afterok`, so the session table in `RUN.md`
  is prose and nothing stops a job being submitted before the job that writes its input finishes.
  *Recommendation:* add an `--after <jobid>` flag to `submit.py` that renders the dependency line.
  It is listed here rather than done because this document specifies `submit.py` as one positional
  argument and one optional flag, and widening that is an interface decision.
- **BTK Dmax is marginal at 57 compounds** under a warhead-scaffold split and is expected to produce
  a held-out R² near zero. *Recommendation:* fit it, report it, and decide in advance that the zero
  is the result.

**Blocked on something I cannot obtain.**

- **The two ACS Nano PDFs.** Not in `manuscript/refs/`, which does not exist,
  and not reachable from here. Every number that would have come from them is
  either absent from the manuscript or carries its own measured source; none is
  guessed. Nothing else in the repository waits on them.
- **PubChem-120K, the battery base corpus.** Not distributed with
  `Teoroo-CMC/Batt-SLM`, and the selection procedure is in the same absent
  supporting information. `battery_prior` names it in `experiment.yaml` and all
  five of its jobs, and `battery_transfer`'s `extra_dataset` points at it, so
  `battery_prior` cannot run at all and `battery_transfer` can run its
  from-scratch arm and not its transfer arm. The only molecule count in the
  repository, 119,995 at `preprocess_pubchem120k/config.yaml:12`, is unsourced.
  In raw form it would be about 2 MB, scaled from Batt-SLM's 115,756 molecules
  in 1.88 MB of the same sub-17-heavy-atom chemistry, so this is an availability
  problem and not a storage one.

**`battp30k` is not blocked, and four places say it is.** Measured 2026-10-25:
`data/raw/battslm/Batt-P30K.h5` is on disk at 183,771,012 bytes with 29,519
molecule groups, each carrying `smiles` and a `dipole` 3-vector alongside
`homo`, `lumo`, `gap`, `ip`, `ea` and the energies, and `MANIFEST.json` records
its url, size and sha256 as retrieved on 2026-10-19. The count matches the
manuscript's 29,519 exactly, and `download_data.py:74` already asserts the true
size in a comment. So the dipole labels `battery_conditional` needs are present
and sufficient. What is missing is one extraction script — take the norm of the
dipole vector per molecule and write `battp30k_dipole.smi`, about 1 MB — and no
script in `scripts/` does it. The claim that the upstream file is a 134-byte Git
LFS pointer is stale in `experiments/battery_conditional/instructions.md:13-14`,
`experiments/battery_transfer/instructions.md:13-16`, `RUN.md:88` and
`RUN.md:929`, as is "not yet wired into `scripts/download_data.py`" in three
further places: the `battslm` source is wired at `download_data.py:83-92` and
`RUN.md` step B1 already calls it. Fix the four documents and write the script.

**`data/datasets/battslm/` has no builder.** Its 115,756 molecules were
canonicalized and deduplicated from the raw file with the rule recorded in its
`PROVENANCE.json`, but `scripts/battslm-create-dataset.py` does not exist, so
the dataset is reproducible only by hand. The other two corpora have their
builders.

**The battery tools are not inventoried.** `tools/README.md` has no entry for
any battery asset and does not mention PiNet2, which
`battery_conditional/instructions.md` names as the predictor needed to measure
achieved dipole on generated molecules. This document asserts above that the
inventory "covers PROTAC-Splitter, RetroTAC, REINVENT4 and the battery assets",
which is currently false, and a dependency used but not recorded there is a bug
by the rule in [External tools](#external-tools).

**Unverifiable without the cluster — the arrival checklist.** Run these in this
order on Arrhenius before submitting anything. They are ordered by how early a
failure would waste the most time, and each says what the failure looks like,
because several of them fail as something other than themselves.

1. **The aarch64 CUDA torch wheel, which is out of the lock.** `uv.lock`
   resolves torch to the PyPI build; on Linux aarch64 that is the **CPU-only**
   build, and the CUDA wheel goes on top of it with the line in `RUN.md` Step 3.
   *The failure looks like:* a job that runs, produces correct metrics, and
   takes the CPU walltime — not a crash. `RUN.md`'s check asserts
   `torch.version.cuda` is not `None`; run it before anything else, because
   every walltime below assumes it passed. Inside a SLURM allocation an absent
   accelerator is fatal rather than a fallback, which is the second guard on
   the same failure.
2. **Whether the Lmod `module` function survives into the batch script.** It is
   a shell function, and a non-login shell does not have it. *The failure looks
   like:* a missing shared library three steps later, which reads as a build
   problem rather than a login-shell problem. The template asserts it in its
   first seconds, so this one fails loudly and early by design.
3. **RetroTAC and PROTAC-Splitter on Linux aarch64.** Both are Python with
   compiled dependencies and neither has been run on that architecture. *The
   failure looks like:* an import error in a subprocess whose stderr the metric
   layer reports as `MissingInput`, so `retrotac_*` and every `warhead_*` key
   record an explicit null with the reason and the job still completes. A
   sparse component table is the symptom to look for.
4. **REINVENT4's platform gating.** It pins `torch==2.12.0`, which publishes
   wheels for `manylinux_2_28_aarch64` and `x86_64` and none for macOS arm64, so
   `tools/install.sh` installs it on the cluster and skips it here with a
   message. *The failure looks like:* a baseline arm that silently never ran.
   The fairness conditions are now asserted by
   `experiments/reinvent4_*/submit_reinvent.py` against the mirrored job's
   `results/config.resolved.yaml`, so what to check here is that those
   assertions pass on the cluster rather than that somebody read them.
5. **Every walltime and memory request.** They are now derived rather than
   guessed: `scripts/estimate-walltime.py` applies `RUN.md`'s procedure to the
   measured CPU per-unit costs and writes the number and its derivation into
   each job's config. **They are CPU extrapolations.** Training will finish well
   inside them on a GPU and generation will gain much less, because generation
   is a sequential autoregressive loop with a Python-level decode. *The failure
   looks like:* a job killed at 99%, which produces nothing but its last
   checkpoint. Twenty-one jobs could not be sized at all because their dataset
   is not built yet; re-run the script once it is.

Open defects are in `TODO.md`. Append to it; do not work from it without being
asked.
