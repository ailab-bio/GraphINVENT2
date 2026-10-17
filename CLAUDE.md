# CLAUDE.md

Guidance for Claude Code when working in this repository.

GraphINVENT2 is a graph-based molecular generative model for 2D molecular generation. It extends the original GraphINVENT with conditional generation, and keeps transfer learning and RL-guided optimization but optimizes them to be both easier to use (more user-friendly) and also more efficient code. The repository is a research codebase with one maintainer. It is not a library, a product, or a framework.

**This document describes the target design.** Where the code does not yet match it, the gap is listed under [Status](#status). Build toward this document; do not preserve current behavior that contradicts it.

---

## Working principles

These exist because agent-written code drifts toward defensive complexity. Follow them literally.

- **Prefer deleting to adding.** A change that removes code and keeps behavior is a good change.
- **No new abstraction without approval.** No base classes, registries, plugin systems, factory functions, or dependency injection. If you think one is needed, say why and wait.
- **No new config parameter unless a planned experiment varies it.** A parameter that always takes the same value is a constant. Put it in the code.
- **No new test file without a stated justification.** See [Tests](#tests).
- **No backwards compatibility.** No shims, no deprecation warnings, no aliases for renamed things. Breaking changes are free here.
- **Fix bugs, do not document them.** If you find broken behavior, fix it. Do not add a note describing the breakage and route around it.
- **Fail loudly.** Do not catch an exception to continue with a default. Crash with a message that says what was wrong.
- **Inline single-use helpers** unless the name carries real meaning.
- **Do not add logging, progress bars, or output files** beyond what is specified here.
- **American spelling throughout.** In code, comments, docstrings, documentation, tutorials, and output messages: optimize, normalize, analyze, behavior, color, center, modeling, labeled, initialize, visualization. Not the British forms.

When a task is ambiguous, ask rather than building the general version that covers both readings.

---

## Setup

Python 3.11. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; that pair is the single source of truth. Nothing else declares dependencies.

```bash
# PyTorch first, matched to your platform (see pytorch.org)
uv pip install torch

uv sync                  # core
uv sync --extra oracles  # adds TDC, for RL and oracle-based evaluation
uv sync --extra dev      # adds ruff, pytest
```

Keep the dependency list short. RDKit, PyTorch, h5py, numpy, PyYAML are core. TDC is an optional extra because it is heavy and only RL jobs need it. FCD and AiZynthFinder are optional extras used by evaluation only. Adding a core dependency needs a reason.

---

## Repository layout

```
src/graphinvent/      model, training, generation
src/metrics/          evaluation (see Evaluation metrics)
src/oracles/          TDC oracle wrappers and the call-counting cache
config/defaults.yaml  every parameter and its default value
experiments/          one directory per experiment (see below)
tutorials/            one walkthrough per job type
submit.py             entry point
visualize.py          render a .smi file to a grid image
```

### Experiments

Every run lives in an experiment directory. Nothing is written outside one.

```
experiments/<experiment_name>/
    experiment.yaml              optional; settings shared by all jobs here
    jobs/
        <job_name>/
            config.yaml          this job's settings
            results/             metrics, generated molecules, oracle logs
            checkpoints/         model state
            logs/                stdout, stderr, tensorboard
```

`<experiment_name>` describes the question, not the method: `conditional_logp_zinc`, not `run3`. A job directory is self-contained and can be copied or deleted on its own.

The old `jobs/` and `output/` directories at the repo root are gone. Do not reintroduce them.

---

## Job types

Five job types. These names are the only accepted values; there are no aliases.

| job_type | Purpose | Needs | Produces |
|---|---|---|---|
| `preprocess` | Turn a SMILES file into training-ready HDF5 and create the data splits | SMILES or TSV file | HDF5 per split (nodes, edges, action probabilities; `condition_vector` if property columns are present), the three split files, `preprocessing.yaml` recording exactly how the splits were made |
| `unconditional` | Train a generative model on a dataset | preprocessed HDF5 | checkpoint, training curves |
| `conditional` | Train a model that conditions generation on property values | preprocessed HDF5 containing `condition_vector` | checkpoint, training curves |
| `goal_directed` | Optimize a trained model toward an oracle with policy-gradient RL | checkpoint, oracle name | checkpoint, oracle call log, score trajectory |
| `generate` | Sample molecules from a checkpoint and evaluate them | checkpoint | `molecules.smi`, `metrics.json` |

Notes that are easy to get wrong:

- **Transfer learning is `unconditional` with `resume_from` set.** It is not a separate job type.
- **`conditional` requires** `condition_dim > 0` and a `condition_vector` dataset in the HDF5. Fail with a clear message if either is missing.
- **`generate` has two modes.** `sample_mode: generate` writes molecules and computes the metrics below. `sample_mode: evaluate` computes NLL and UC-JSD against the test split.

---

## Data splits

The splits are the most consequential thing `preprocess` does. Every number this project reports depends on them, so they are defined once per dataset and never redefined afterward.

**Three splits, with three distinct jobs:**

| Split | Used for | Never used for |
|---|---|---|
| **train** | Fitting model parameters. The reference set for novelty. | Anything reported as a result |
| **validation** | Model selection only: choosing the best checkpoint and early stopping, in `unconditional`, `conditional` and `goal_directed` training alike | Any evaluation metric |
| **test** | Held out until final evaluation. The reference set for all distribution-matching metrics, for NLL, and for sampling condition values in conditional evaluation | Training, early stopping, or any choice made during development |

A model selected on the test set has no held-out set left, so the number it reports is not an estimate of anything. Keep the test split closed until the model is fixed.

**Splitting happens at the molecule level, before subgraph decomposition.** Preprocessing decomposes each molecule into many subgraphs, so a split applied after decomposition would scatter subgraphs of the same molecule across train, validation and test, and every reported number would be leakage. Split the SMILES file first, then decompose each split separately.

**The split strategy is recorded, not assumed.** `preprocessing.yaml` records the strategy (`random`, `scaffold`, or `custom`), the seed, the fractions, and the resulting counts. Scaffold splitting is the default, because a random split of a molecular dataset puts close analogues on both sides and overstates generalization. Use `random` only when comparing against published numbers that used it, and say so.

**Splits are fixed per dataset and reused.** Every job in an experiment, and every experiment on the same dataset, uses the same split files. Do not re-split between jobs; two models evaluated on different test sets are not comparable. If a dataset needs a new split, it is a new dataset directory.

**Conditional jobs:** at generation time, condition values are drawn from the test split's property distribution, not the training split's.

**Goal-directed jobs:** the prior is trained on train. The screening library used for the virtual screening baseline is declared separately and is not any of these three splits.

---

## Configuration

Three layers, merged key by key, later wins:

1. `config/defaults.yaml` — every parameter, with its default and a one-line comment
2. `experiments/<exp>/experiment.yaml` — optional, normally just the dataset and paths shared across the experiment's jobs
3. `experiments/<exp>/jobs/<job>/config.yaml` — this job's settings

A key left unset is inherited. There is no flag to turn inheritance on or off: omitting a key *is* inheriting it. The merged configuration is written to the job's `results/config.resolved.yaml` so every run records exactly what it ran with.

Config files are YAML, not JSON, so that parameters can carry comments. Comments in config files must match behavior; a comment that contradicts the code is a bug.

**Architecture keys come from the checkpoint.** When `resume_from` or `pretrained_model_path` is set, all architecture keys (GGNN sizes, `condition_dim`, `condition_embedding_dim`, `condition_type`, `conditioning`) are read from the checkpoint and cannot be overridden. A config that sets an architecture key to a value conflicting with the checkpoint is an error, not a silent override. Loading a model with the wrong architecture either crashes confusingly or produces a silently wrong model, so this is checked up front.

---

## Running jobs

```bash
python submit.py experiments/<exp>/jobs/<job>            # local, default
python submit.py experiments/<exp>/jobs/<job> --slurm    # submit to SLURM
```

The runner is chosen on the command line, not in the config, so the same job directory runs in both places unchanged. SLURM resource settings (partition, time, gpus, memory) live in `config/defaults.yaml` under `slurm:` and can be overridden per job. There is one SLURM template; do not add per-job template files.

```bash
tensorboard --logdir experiments/<exp>/jobs/<job>/logs/tensorboard/
```

---

## Evaluation metrics

All metrics live in `src/metrics/` and have exactly one implementation. Do not compute a metric inline anywhere else. Metrics are written to `results/metrics.json` as a flat dictionary using the key names below, so runs compare without a translation layer.

### Reporting conventions

- Sample **N = 10,000** molecules and run **3 seeds**; report mean and standard deviation.
- **Every metric names its reference set** in the output key, because train and test are used for different things: `novelty_vs_train`, `fcd_vs_test`.
- **Validity, uniqueness and novelty compose as a funnel**, computed sequentially on the surviving subset. Each stage is reported both as a conditional rate and as a fraction of N, since the two answer different questions.
- **Validity means parsed and sanitized without repair.** RDKit is not permitted to fix valences or charges. A molecule that only parses after repair is invalid.

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
| `sa_score_mean`, `sa_score_median` | synthetic accessibility over unique valid molecules |
| `filter_pass_rate` | fraction of unique valid molecules passing MCF and PAINS structural alerts |
| `Ro5_pass_rate`, `bRo5_pass_rate` | fraction of unique valid molecules passing 'Rule of 5' or 'Beyond Rule of 5' rules |

`near_duplicate_rate_vs_train` exists because binary novelty counts a molecule differing by one methyl as novel. A model with high novelty and a high near-duplicate rate is memorizing. We look at Beyond Rule of 5 rules because sometimes we might want to use GraphINVENT to generate non-small molecule drugs like PROTACs.

### Unconditional only

| Key | Definition |
|---|---|
| `fcd` | Fréchet ChemNet Distance between generated molecules and the test split |
| `nll` | mean negative log-likelihood of a 10k sample from each of the train, valid, and test splits under the model (`sample_mode: evaluate`), so 3 metrics reported |
| `uc_jsd` | uniformity-completeness JSD (`sample_mode: evaluate`) |

FCD is the headline number for unconditional generation. V.U.N. saturates on modern models and no longer separates them; distribution match does.

### Conditional only

| Key | Definition |
|---|---|
| `condition_success_rate_<property>` | fraction of generated molecules within `condition_tolerance` of the requested value |
| `condition_coverage_<property>` | per-decile success rate across the training property distribution, reported as ten numbers |

### Goal-directed only

| Key | Definition |
|---|---|
| `best_at_<1k\|5k\|10k>`, `top10_mean_at_<...>`, `top100_mean_at_<...>` | oracle score statistics at fixed budgets |
| `auc_top10` | area under top-10 mean against oracle calls, up to 10,000 |
| `vs_baseline_top10_at_<...>` | the same top-10 mean for B molecules drawn uniformly at random from the declared screening library, same oracle, same seeds |
| `vs_delta_top10_at_<...>` | generative minus virtual screening, the headline comparison |
| `success_rate` | fraction of unique valid molecules scoring at or above `success_threshold`; multiple property thresholds combine with AND |
| `top100_scaffold_count`, `top100_internal_diversity` | diversity among the 100 highest-scoring molecules |
| `quality_filtered_top10_mean` | top-10 mean among molecules passing structural filters and `sa_score < 4.5` |

An oracle call is one *unique* molecule submitted to the oracle; cache hits are free and are not counted. The oracle call log written per job is the authoritative record of the budget spent.

The last two rows exist to detect reward hacking, which is the default behavior of RL against a cheap oracle. Overall diversity stays high while the top-scoring set collapses onto one scaffold, so `top100_scaffold_count` is the metric that catches it and `internal_diversity` is not. `quality_filtered_top10_mean` is the headline score.

A generative method that does not beat random library screening at matched budget has not demonstrated anything, so the virtual screening baseline runs for every optimization experiment, where virtual screening happens by selected a random matched sample size from the training and validation sets.

### Model efficiency, reported for every trained model

`n_parameters`, `train_wall_clock_s` with the device recorded, `inference_s_per_10k` on CPU and on GPU, and `peak_memory_mb`. These go in `metrics.json` with everything else.

---

## Tutorials

One per job type in `tutorials/`, each a short markdown walkthrough pointing at a real, runnable example under `experiments/tutorial_<job_type>/`. Each must run end to end on CPU in a few minutes on a small dataset included in the repo. A tutorial that cannot be run as written is worse than no tutorial; if you change an interface, update the tutorials in the same change.

---

## Tests

Keep `tests/` to these four. Each catches a failure that is silent otherwise:

1. **Preprocessing round-trip.** SMILES → HDF5 → graph → SMILES returns the same canonical SMILES. Catches silent corruption in the graph representation, which otherwise shows up only as poor samples much later.
2. **Split disjointness.** No molecule appears in more than one split, and no subgraph in one split derives from a molecule in another. Catches the leakage described under [Data splits](#data-splits), which inflates every reported number.
3. **Conditional tensor shapes and gradients.** The condition vector reaches the model, the virtual node is stripped before readout, and every parameter that should receive gradient does. Catches a condition signal that is silently ignored, which would look exactly like a model that failed to learn.
4. **One smoke test per job type.** A few steps on a tiny dataset, asserting the job completes and writes its expected outputs. Catches interface breakage.

Any other test needs a one-line justification of the specific failure it catches, recorded at the top of the file. Delete tests that do not earn this.

---

## Architecture notes

Only the non-obvious parts are recorded here. Read the code for the rest.

**Data flow.** SMILES → `DataProcessor` → HDF5 (per-subgraph nodes, edges, action probabilities) → `BlockDataLoader` → GGNN → action logits → `GraphGenerator` (add atom / add bond / terminate).

**Conditional generation.** When `condition_dim > 0`, the GGNN builds a `ConditionEncoder` (2-layer MLP) and one extra `msg_nn` for a virtual edge type. The condition vector is encoded into a seed embedding that initializes a virtual node prepended to the graph; every real node receives messages from the seed at each round. The seed is stripped before readout, so the output shape is unchanged, but the readout *input* is not: the condition embedding is also concatenated to the graph embedding, making `graph_emb_size = gather_width + condition_embedding_dim`. That concatenation is load-bearing. The first action of every molecule is taken on an empty graph, where no real node exists to carry the seed's message.

`DataProcessor` stores `condition_vector` (float32, `n_subgraphs × condition_dim`) in the HDF5. `HDFDataset` yields 4-tuples when that dataset exists and 3-tuples otherwise.

**Oracle caching.** `src/oracles/` caches by canonical SMILES. The cache is what makes the oracle-call accounting above correct, so it is not an optimization detail; changing it changes the reported sample efficiency.

---

## Plots

Every plot in this repository follows the AIME Lab plot style guide. The palette and the matplotlib rcParams live in `src/metrics/plot_style.py`, copied verbatim from the guide; every figure-producing function imports them and sets nothing else, with no per-plot colors, fonts, or sizes. The palette, in cycle order, is Deep Teal `#2D4F54`, Royal Purple `#7B539E`, Soft Lavender `#B8A0D2`, Forest Green `#5A9448`, Muted Plum `#9E4A78` and Dark Slate `#3A3D4A`, with Dark Slate used for all text, axes and annotations in place of black. Sans serif throughout, 11 pt labels and 9.5 pt ticks, all four spines visible at 1.2 pt, no grid, no legend frame. Never put a title inside a plot; titles belong in the figure caption. Panel labels go outside the axes, left of the y-tick labels, via `ax.text(-0.18, 0.97, "(a)", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left", color=DARK_SLATE)`. Axis labels carry units. Save every figure as both `.png` at 300 dpi and `.pdf`, into the job's `results/` directory. Any inline SVG in `results/report.html` uses these same hex values, so the report and the paper figures match.

matplotlib is the only plotting dependency. Do not add seaborn, plotly, or anything else.

---

## Hard constraints

- **Run from the repo root with `python submit.py`.** The editable install writes a `.pth` file that does not survive a repository path containing spaces, so console entry points and `from metrics import ...` fail. Moving the repository to a path without spaces would remove this constraint; until then, do not rely on the installed entry point.
- **`[Xe]` is not a molecule.** Invalid graphs must not be written to `.smi` as `[Xe]`, because RDKit parses that as valid xenon and it silently inflates validity. Write nothing and record the index.
- **Generated-file line alignment.** `.smi`, `.likelihood` and `.valid` must line up by line number, with no per-batch headers.
- **The test split is not touched during development.** Not for early stopping, not for a quick sanity check, not for choosing a hyperparameter.

---

## Status

Not yet implemented, as of the last update to this file:

- `results/metrics.json`: no job writes one. The metrics in `src/metrics/` are
  not called from `src/graphinvent/`, which still logs to `logs/generation.log`
- Near-duplicate rate, scaffold diversity, structural-alert filters, Ro5 and bRo5
- Visualization and HTML reports after each job in the code
- All conditional metrics beyond MAE
- Virtual screening baseline, fixed-budget sample efficiency, AUC, top-100 diversity, quality-filtered top-k
- Model efficiency metrics
- Tutorials: `tutorials/` still describes the pre-restructure interface

Open defects are in `TODO.md`. For now just append to that file, do not execute or try to solve anything in that file.

Experiments for the paper, meaning the specific datasets and optimization tasks, are not yet chosen. Do not assume any particular benchmark.
