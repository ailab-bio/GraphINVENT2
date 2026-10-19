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

`<experiment_name>` describes the question, not the method: `conditional_logp_zinc`, not `run3`. A
job directory is self-contained and can be copied or deleted on its own. The old `jobs/` and
`output/` directories at the repo root are gone; do not reintroduce them.

**Every program follows the same transfer-learning shape**, which is a structural feature of the
system rather than a choice an experiment makes: a prior trained on a large base corpus, fine-tuned
on the smaller task corpus with `resume_from`, against a from-scratch control trained on the task
corpus alone. The comparison between those two arms is what the transfer-learning claim rests on, so
both are always run.

---

## Job types

Five job types. These names are the only accepted values; there are no aliases.

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

**Splits are fixed per dataset and reused.** Every job in an experiment, and every experiment on the
same dataset, uses the same split files. Two models evaluated on different test sets are not
comparable. If a dataset needs a new split, it is a new dataset directory.

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
inline anywhere else. The one metric family that does not yet obey this is named in
[Status](#status).

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
  allowed: a `baseline_` prefix for the unconditional baseline of a conditional metric; a `_<class>`
  suffix for the per-class breakdown of a categorical conditional metric; the funnel counts
  `n_sampled`, `n_valid`, `n_unique_valid` and `n_unique_valid_novel`; and a **supporting field**
  named `<metric>_<field>` recording a denominator, a threshold or a parameter its metric was
  computed with. A number whose meaning depends on how many molecules reached it, or on a threshold,
  is not interpretable without them, and a reader who has to open the code to find the denominator
  will not. The supporting fields are `fcd_n_generated` and `fcd_n_reference`,
  `validity_n_generated_raw`, `property_window_rate_bounds`, `success_threshold`,
  `success_definition`, `n_scaffolds`, `n_split_successfully`, `top100_n_split`,
  `top100_split_success_rate`, `e3_similarity_floor`, `e3_max_similarities`, `ecfp4_radius`,
  `knn_reference_pool_size`, the splitter model and version beside every component metric, and the
  subsample size and seed wherever a metric is subsampled. A supporting field is not a metric and
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

### Core metrics, reported for every setting

| Key | Definition |
|---|---|
| `validity` | valid / N |
| `uniqueness` | unique canonical SMILES among valid / valid |
| `novelty` | unique valid molecules absent from the training split / unique valid |
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

| Key | Definition |
|---|---|
| `delta_nll` | for a held-out molecule of class *p*, the mean over competing classes *q* of NLL(x\|q) − NLL(x\|p), excluding legitimate alternative classes of a multi-label molecule. Positive means the molecule is more likely under its own condition |
| `knn_purity_<5\|10\|20>` | fraction of each generated molecule's *k* nearest neighbors in the held-out reference pool, by Morgan-Tanimoto, carrying the requested label |
| `class_match_median`, `class_match_rate` | median, and fraction above a stated threshold, of the maximum ECFP4 Tanimoto from each generated molecule to the training molecules carrying the requested label |
| `class_selectivity_margin` | the same maximum Tanimoto computed against the other classes in the panel, subtracted from the on-class value |
| `condition_coverage` | the per-class breakdown of all of the above |

`class_selectivity_margin` is the cleanest single number here because it is self-normalizing: an
unconditional model scores approximately zero by construction, and the metric cannot be satisfied by
generating generic chemistry. Report `novelty` next to `class_match_*`: a high match with low novelty
means the model is retrieving training molecules, not conditioning. A **nearest-neighbor retrieval
baseline**, which retrieves the training molecules closest to the requested class and scores them
with the same metric, is what makes the conditional claim falsifiable, and it runs for every
categorical conditional experiment.

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
| `retrotac_mean`, `retrotac_median` | route-derived synthesizability of the **whole** PROTAC (Ribes et al., arXiv:2610.13639), over unique valid molecules. Not clipped to [0, 1]: a value outside it means the molecule is far from the model's training distribution |
| `split_success_rate` | fraction of generated molecules successfully decomposed into components |
| `warhead_*` | the core and conditional metrics recomputed over target-binding components |
| `e3_recognizable_rate` | fraction of successfully split molecules whose predicted E3 ligand contains a known E3 ligand as an exact substructure. A validity check, not a diversity analysis |
| `top100_warhead_scaffold_count` | distinct Bemis-Murcko scaffolds among the target-binding components of the 100 highest-scoring molecules of a `goal_directed` job |

Every component-level metric is computed only over successfully split molecules, so a split rate is
always reported beside it: `split_success_rate` over the whole sample, and
`top100_split_success_rate` beside `top100_warhead_scaffold_count`, over the same hundred. A falling
split rate otherwise masquerades as improving component quality.

`top100_warhead_scaffold_count` is the primary reward-hacking detector for a bifunctional dataset,
because `top100_scaffold_count` cannot be one: a whole-molecule PROTAC scaffold is nearly unique per
compound, so a top 100 collapsed onto a single warhead still reports close to 100 distinct scaffolds
as long as the linkers differ. Measured on 12 real degraders, 12 distinct whole-molecule scaffolds
against 10 warhead scaffolds over the 11 that decomposed. The whole-molecule count stays reported
beside it and is expected to be uninformative, which is the evidence that the warhead count is the
one to read.

**The splitter policy.** PROTAC-Splitter's `adaptive` model is primary for every cluster and final
run; `xgboost` is selectable for development iteration, because adaptive costs 1.982 s per molecule
against 0.143 and a 10,000-molecule sample is 5.5 hours serially. The model and version are recorded
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
and none imports `src/metrics/`, which is where every reported number comes from. Any further test
needs a one-line justification of the specific failure it catches, recorded at the top of the file.
Delete tests that do not earn this.

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

Verified by running the code on 2026-10-19. Everything here is either unbuilt, unverifiable on a
laptop, or waiting on an asset. Nothing in this list is a defect in what exists.

**Unbuilt.** Each is a feature with no implementation, not a broken one.

- **Categorical conditional metrics have no caller.** `delta_nll`, `knn_purity_<k>`,
  `class_match_*`, `class_selectivity_margin` and `knn_reference_pool_size` exist in
  `src/metrics/_conditions.py` and no job emits them, so a categorical conditional experiment reports
  its core metrics and nothing about conditioning. The continuous keys are emitted as categorized
  nulls for a one-hot label, which is correct: they have no definition for a class.
- **Whole-sample component metrics have no caller.** `split_success_rate`, `warhead_*`,
  `e3_recognizable_rate`, `n_split_successfully`, `e3_similarity_floor`, `e3_max_similarities` and
  `ecfp4_radius` exist in `src/metrics/_components.py` and are not called.
  `top100_warhead_scaffold_count` and `top100_split_success_rate` are the wired pair. Wiring the rest
  means paying decomposition over the whole sample rather than over a hundred molecules, and
  `e3_recognizable_rate` is a similarity threshold rather than the exact-substructure check specified
  above.
- **The virtual screening baseline, the evaluation surrogate and the applicability-domain check.**
  `vs_baseline_top10_at_*`, `vs_delta_top10_at_*`, `eval_oracle_top10_mean` and
  `top100_outside_domain_rate` emit categorized nulls. No experiment declares a screening library,
  and the `vs_baseline_*` job directories are `generate` jobs, which score nothing, so they cannot
  report any `top100_*` or at-budget metric whatever else is wired.
- **The nearest-neighbor retrieval baseline** has no job directory in any conditional experiment.
- **Multi-seed aggregation.** Nothing writes `experiments/<exp>/results/aggregate.json`, and
  `submit.py` runs one job directory with one seed, so the three seeds the reporting conventions
  require are produced by hand.
- **`experiments/surrogates/`** does not exist, so no surrogate is fitted and nothing reports
  held-out performance against a mean-prediction baseline.
- **`nll_*` and `uc_jsd` are computed in `Analyzer`**, outside `src/metrics/`, which is the one
  exception to the single-implementation rule above. `src/metrics/` records the values it is handed.
- **No job writes a figure**, so the plot style guide is specified and unapplied.
  `results/report.html` carries the structure grid.
- **Decomposition is cached but neither parallelized nor subsampled.** The spawn-context pool and the
  seeded 2,000-molecule subsample for component metrics are not built; a full sample costs about 2 s
  per molecule on one core.
- **There is no `uv.lock`**, so the single-source-of-truth claim above is aspirational.
- **REINVENT4 is not installed** and `tools/install.sh` does not fetch it, while six `reinvent_*`
  jobs exist. None of the six fairness conditions in the experiment plan is met, so no REINVENT
  comparison may be reported as evidence.
- **`scripts/cluster_selftest.sbatch` is hand-written**, which is the one exception to "no
  hand-written sbatch files": it runs before a job directory exists.

**Unverifiable without the cluster.** These end as unverified rather than as done.

- The aarch64 and CUDA 13 environment. Locally `torch` resolves to 2.14.1 with `torch.version.cuda`
  of `None`, which is correct on macOS and tells us nothing about a GH200 node.
- Whether the Lmod `module` function survives into the batch script. The template now fails loudly
  in its first seconds if it does not.
- Every walltime and memory request. Nineteen jobs still carry the two-hour default, including the
  `generate` jobs that sample ten thousand molecules. Measure with the walltime procedure in
  `RUN.md`; these are measurements rather than guesses.
- RetroTAC and PROTAC-Splitter on Linux aarch64.

**Blocked on an asset the maintainer must supply.** These are the reason several parts of the system
are unbuilt, and the next pass should not rediscover them.

- **The two ACS Nano PDFs** (`acsnano-6c06255-main.pdf` and `-si.pdf`). Absent. Every battery number
  in the configs, in `RUN.md` and in the manuscript came from a specification pasted into a prompt
  rather than from the source, so none is verified in the strict sense.
- **KBS-409.** Not located. It is the reference set for one of the six multiplied terms of the
  battery objective, so without it that objective cannot be computed at all. The battery scoring
  function has no implementation in `src/` either.
- **The enumerated recombinant PROTAC library.** Not downloadable, which blocks the PROTAC base prior
  and therefore the PROTAC transfer arm.
- **The letter-grade bin edges for TPD-DB activity.** Half of the PROTAC `DC50` rows hold letter
  grades rather than numbers, and a fifth are a number with a unit, so the activity surrogate has no
  label to fit until the edges are known or those rows are dropped.

Open defects are in `TODO.md`. Append to it; do not work from it without being asked. The specific
datasets and optimization tasks are in `manuscript/experiment-plan.md`, and the operational steps are
in `RUN.md`.
