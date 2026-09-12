# GraphINVENT2

![cover image](./cover-image.png)

GraphINVENT2 generates molecules as graphs rather than as strings. A gated graph neural network
(GGNN) reads a partial molecular graph and predicts a probability distribution over the
possible next actions — add an atom, add a bond between two existing atoms, or stop — and
sampling from that distribution repeatedly builds a molecule one action at a time. Because the
model never emits a SMILES string, it cannot produce a syntactically malformed one; the failure
modes are chemical rather than grammatical.

The same model can be trained by maximum likelihood on a dataset, conditioned on continuous
molecular properties, or fine-tuned by policy-gradient reinforcement learning against a scoring
function.

This is the maintained successor to GraphINVENT. The method is described in
[*Graph Networks for Molecular Design*](https://iopscience.iop.org/article/10.1088/2632-2153/abcf91)
(Mercado et al., 2021), with practical notes in
[*Practical Notes on Building Molecular Graph Generative Models*](https://doi.org/10.1002/ail2.18).

---

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Job types](#job-types)
4. [Tutorials](#tutorials)
5. [Testing](#testing)
6. [Known limitations](#known-limitations)
7. [Contributing](#contributing)
8. [Changes from GraphINVENT](#changes-from-graphinvent)
9. [References](#references)
10. [License](#license)

---

## Installation

Python 3.9 or newer.

### 1. Clone

```bash
git clone https://github.com/ailab-bio/GraphINVENT2.git
cd GraphINVENT2
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows (PowerShell)
```

### 3. Install PyTorch

PyTorch has to be installed before the rest, because the correct wheel depends on your platform
and CUDA version. Get the exact command from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). The common cases:

```bash
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU — substitute your CUDA version for cu121
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon — the standard wheel uses MPS
pip install torch torchvision torchaudio
```

### 4. Install GraphINVENT2

```bash
pip install -e .
```

Editable mode means changes to the sources under `src/` take effect without reinstalling.

Optional extras:

```bash
pip install -e ".[docking]"   # vina and meeko, for the AutoDock Vina scoring oracle
pip install -e ".[dev]"       # black, ruff, mypy, pytest, pre-commit
```

### Verify

```bash
python -c "import torch, rdkit, h5py; print('PyTorch', torch.__version__)"
python submit.py --help
```

### Conda alternative

Useful on Windows, where RDKit pip wheels are occasionally unavailable:

```bash
conda create -n graphinvent python=3.11 -y
conda activate graphinvent
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 -y
conda install -c conda-forge rdkit h5py tqdm scikit-learn matplotlib tensorboard -y
```

### HPC

`docker/graphinvent.def` is a Singularity definition file for clusters where neither Conda nor
a pip virtualenv is practical.

---

## Quick start

### 1. Preprocess a dataset

The `jobs/*/params.json` files are templates. Copy one before editing, so the original stays
intact and each experiment keeps its own config:

```bash
cp jobs/preprocess/params.json jobs/preprocess/my_experiment.json
python submit.py --config jobs/preprocess/my_experiment.json
```

The feature vocabulary — `atom_types`, `formal_charge`, `imp_H`, `max_n_nodes` — is detected by
scanning the SMILES, so only the encoding flags such as `use_chirality` and
`use_aromatic_bonds` need setting.

Four dataset directories ship with the repository under `data/datasets/`. `debug` and `test`
are small enough to run the whole pipeline in minutes and are already preprocessed, so you can
skip straight to training; `DRD2_actives` is a focused set suitable as a transfer-learning
target; `unit_testing` holds a fixture for the test suite rather than a trainable dataset.

### 2. Train

```bash
python submit.py --config jobs/unconditional/params.json
```

Progress goes to `output/<dataset>/unconditional/<job_name>/convergence.log`, and a nine-panel
`progress.png` is regenerated at each evaluation epoch. With `"use_tensorboard": true`:

```bash
tensorboard --logdir output/<dataset>/unconditional/<job_name>/tensorboard/
```

### 3. Generate

```bash
python submit.py --config jobs/generate/params.json
```

Generated molecules are written to `output/<dataset>/generate/<job_name>/` as
`<n_samples>_samples.smi`, with matching `.likelihood` and `.valid` files.

### Looking at the output

```bash
python visualize.py path/to/molecules.smi                                # 25 at random, 5 columns
python visualize.py path/to/molecules.smi --n 50 --ncols 10
python visualize.py path/to/molecules.smi --first                        # first N instead of random
python visualize.py path/to/molecules.smi --size 300x200 --out grid.png
```

The PNG is written next to the input as `<filename>_grid.png` unless `--out` says otherwise.

`cleanup.py` removes stale outputs, preprocessed data, and `_previous_run_*` backup directories,
with a confirmation step.

---

## Job types

```bash
python submit.py --config jobs/<job_type>/params.json
```

Each config has a `"submission"` block (how to run) and a `"job"` block (what to run). Anything
omitted from the job block falls back to `src/graphinvent/parameters/defaults.py`. Output is
written to `output/<dataset>/<job_type>/<job_name>/`, with `job_name` taken from the submission
block.

| Job type | Config | What it does |
|----------|--------|--------------|
| `preprocess` | `jobs/preprocess/params.json` | SMILES to HDF5; detects the feature vocabulary; splits a single file if asked |
| `unconditional` | `jobs/unconditional/params.json` | Supervised training. `resume_from: null` trains from scratch; a checkpoint path makes it transfer learning |
| `conditional` | `jobs/conditional/params.json` | Supervised training with a property-conditioned model; requires TSV input with property columns |
| `goal_directed` | `jobs/goal_directed/params.json` | Policy-gradient RL against a scoring function; `oracle_budget` caps the run by oracle calls instead of steps |
| `generate` | `jobs/generate/params.json` | Sample from a checkpoint, or evaluate it on the test set with `sample_mode: "evaluate"` |

`submit.py` validates the config before creating any directories and reports every problem it
finds at once. The pre-refactor names `pretrain`, `transfer`, `rl`, `constrained_rl`, `sample`,
and `test` are rejected by that validator.

### Typical sequences

```
preprocess → unconditional → generate
preprocess (new data) → unconditional with resume_from → generate      # transfer learning
unconditional → goal_directed → generate
preprocess (TSV with properties) → conditional → generate with sample_conditions
```

---

## Tutorials

| Document | Topic |
|----------|-------|
| [01 Preprocessing](./tutorials/01_preprocessing.md) | SMILES to HDF5, vocabularies, splits |
| [02 Pretraining](./tutorials/02_pretraining.md) | Supervised training from scratch |
| [03 Transfer learning](./tutorials/03_transfer_learning.md) | Continuing from a checkpoint |
| [04 Reinforcement learning](./tutorials/04_reinforcement_learning.md) | Goal-directed optimisation, policy-gradient fine-tuning |
| [05 Sampling](./tutorials/05_sampling.md) | Generating and cleaning molecules |
| [05 Conditional generation](./tutorials/05_conditional_generation.md) | Property-conditioned models |
| [06 Custom oracles](./tutorials/06_custom_oracles.md) | Defining your own scoring objectives, transforms, uncertainty modulation |
| [Evaluation](./tutorials/evaluation.md) | Every metric and which job types compute it |

`experiments/` holds end-to-end configs and scripts for the paper's four experiments, each with
its own README.

---

## Testing

Run everything from the repository root.

Most of the suite is self-contained. `tests/test_preprocessing.py` is the exception: it verifies
that a *completed* preprocessing job produced correct output, so `tests/config.py` has to point
at a dataset directory that already contains the `.smi` and `.h5` files.

```python
# tests/config.py
DATASET_DIR = Path("data/datasets/debug")
SMILES_FILE = Path("data/datasets/debug/debug.smi")   # None for pre-split (Mode B) datasets
```

```bash
pytest tests/ -v
pytest tests/test_preprocessing.py -v
```

| Test class | Checks |
|------------|--------|
| `TestSplitFileCounts` | The three `.smi` files exist, their counts sum to the original, and the splits do not overlap |
| `TestHDFFileCounts` | Each `.h5` contains `nodes`, `edges`, and `action_probs`; molecule counts match the `.smi` files; subgraph count is at least the molecule count |
| `TestSMILESReconstruction` | Every graph decodes to a valid SMILES and the reconstructed set matches the `.smi` file exactly |

The remaining files cover the model (`test_model.py`), the scoring function (`test_scoring.py`),
the oracle wrappers (`test_oracles.py`), conditioning (`test_conditioning.py`), and the metrics
packages (`test_metrics.py`, `test_metrics_module.py`).

---

## Known limitations

- **The `graphinvent-submit` console script does not work.** `submit.py` lives at the repository
  root and setuptools only packages `src/`, so the entry point cannot import it. Use
  `python submit.py --config ...`.
- **The editable install may not put `src/` on the import path.** When the repository lives under
  a path containing spaces — an iCloud Drive directory, for instance — the `.pth` file setuptools
  writes is not honoured, and `from metrics import ...` fails. Run from the repository root and
  add `sys.path.insert(0, "src")` in scripts that import the `metrics` or `oracles` packages.
- **`data/surrogates/QSAR_model_example.pickle` is a zero-byte placeholder**, not a model.
  Referencing it from `score_components` fails at startup. Train a surrogate of your own with
  `src/graphinvent/tools/train-surrogate.py` and declare it as an oracle instead.
- **`generation.log` statistics come from the first generation batch only**, so with
  `n_samples` larger than `batch_size` they describe a subset of the run.
- **Invalid graphs are written to `.smi` as `[Xe]`**, which RDKit parses successfully as a xenon
  atom. Downstream validity counts must use the `.valid` file or filter `[Xe]` first.

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Issues and pull requests are welcome; a bug report is
most useful with the failing job's `params.json` and the `run_info` block from its
`params_all.json`.

---

## Changes from GraphINVENT

GraphINVENT2 is a rewrite. **Existing config files and preprocessed datasets are not compatible
and must be recreated.**

### Job configuration

The original `submit.py` was a script with a hardcoded `Config` class you edited in place. It is
now a JSON-driven CLI:

```bash
# old
python submit.py           # edit the Config class inside the file first

# new
python submit.py --config jobs/unconditional/params.json
```

### `job_type` values

| Old | New | Notes |
|-----|-----|-------|
| `"train"` | `"unconditional"` | With `resume_from: null` |
| `"fine-tune"` | `"unconditional"` | With `resume_from` set to a checkpoint |
| — | `"goal_directed"` | RL; `oracle_budget` caps the run by oracle calls |
| — | `"conditional"` | Property-conditioned generation, new in GraphINVENT2 |
| — | `"generate"` | Sampling or evaluation, selected by `sample_mode` |

The intermediate names `pretrain`, `transfer`, `rl`, `constrained_rl`, `sample`, and `test` are
rejected by `submit.py`'s validator. `main.py` still remaps them with a deprecation warning if
invoked directly, but that bypasses validation.

### HDF5 dataset key: `"APDs"` → `"action_probs"`

All previously preprocessed `.h5` files must be regenerated.

### Preprocessing parameter file: `.csv` → `.json`

The old `preprocessing_params.csv` is not read by the new code, which writes and reads
`preprocessing_params.json`.

### Specifying a checkpoint

The `generation_epoch` plus `pretrained_model_dir` pair is replaced by a single path:

```json
// old
{ "generation_epoch": 100, "pretrained_model_dir": "output/debug/pretrain/run/" }

// new
{ "pretrained_model_path": "output/debug/unconditional/run/model_restart_100.pth" }
```

The GGNN architecture is then read from the `params_all.json` beside the checkpoint — but only
for keys absent from your job config, since anything you write explicitly wins. Delete the
architecture block from a config that sets `resume_from` or `pretrained_model_path`.

### Data directory layout

| Old | New |
|-----|-----|
| `data/pre-training/<dataset>/` | `data/datasets/<dataset>/` |
| `data/fine-tuning/<dataset>/` | `data/datasets/<dataset>/` |

### Output layout

TensorBoard logs now live inside the job directory, so a run is one self-contained folder:

| Old | New |
|-----|-----|
| `output/<dataset>/<job_type>/tensorboard/<job_name>/` | `output/<dataset>/<job_type>/<job_name>/tensorboard/` |

### Generation output

Per-batch files are concatenated into one trio at the job root and the temporary directory is
removed:

```
# old
output/<dataset>/generate/<job_name>/generation/epoch_GEN100_batch_0.smi
output/<dataset>/generate/<job_name>/generation/epoch_GEN100_batch_1.smi

# new
output/<dataset>/generate/<job_name>/<n_samples>_samples.smi
output/<dataset>/generate/<job_name>/<n_samples>_samples.likelihood
output/<dataset>/generate/<job_name>/<n_samples>_samples.valid
```

### Internal renames

Relevant if you import GraphINVENT2 modules directly.

| Old | New |
|-----|-----|
| `DataProcesser` | `DataProcessor` |
| `APDReadout` | `ActionProbReadout` |
| `get_decoding_APD()` | `get_action_probs()` |
| `get_final_decoding_APD()` | `get_final_action_probs()` |
| HDF5 key `"APDs"` | `"action_probs"` |

### Python version

Python 3.6/3.8 → **3.9 or newer**.

### What is new

| Feature | Description |
|---------|-------------|
| Automatic feature detection | `atom_types`, `formal_charge`, `imp_H`, and `max_n_nodes` are scanned from the SMILES rather than specified by hand |
| Built-in splitting | A single `smiles_file` is deduplicated and split into train/valid/test by a random, Butina-cluster, or user-defined strategy |
| Multi-dataset preprocessing | Several datasets in one run, sharing a union feature vocabulary so their encodings stay compatible |
| Conditional generation | Continuous property conditioning via a virtual seed node; see [the tutorial](./tutorials/05_conditional_generation.md) |
| User-defined oracles | Scoring objectives declared in the job config: a pickled scikit-learn surrogate, any importable Python callable, or AutoDock Vina docking; each with its own transform onto [0, 1] and a direction, so an anti-target is the same machinery as a target. See [the tutorial](./tutorials/06_custom_oracles.md) |
| Uncertainty-aware RL | An oracle that reports predictive spread can damp the reward or the gradient it contributes, keeping the agent inside the surrogate's applicability domain (Medina and Janet, [arXiv:2606.24990](https://arxiv.org/abs/2606.24990)) |
| Apple Silicon (MPS) support | Device selection covers MPS as well as CUDA and CPU |
| Reproducibility logging | `params_all.json` records the seed, Python/PyTorch/RDKit/NumPy versions, CUDA version, device, and git commit for every run |
| Seed control | `"seed": <int>` fixes Python, NumPy, and PyTorch RNGs; 0 leaves them unseeded |
| Backup on re-run | Re-running into an existing output directory moves the previous results to `_previous_run_<timestamp>/` |
| Extended metrics | Novelty, SA score, internal diversity, and test-set nearest-neighbour similarity are logged during training; see [the evaluation reference](./tutorials/evaluation.md) |
| `visualize.py`, `cleanup.py` | Molecule grid rendering and output housekeeping |
| Test suite | `tests/` covers preprocessing round-trips, model shapes and gradients, scoring, oracles, conditioning, and both metrics packages |

---

## References

```bibtex
@article{mercado2020graph,
  author  = {Roc{\'{i}}o Mercado and Tobias Rastemo and Edvard Lindel{\"{o}}f
             and G{\"{u}}nter Klambauer and Ola Engkvist and Hongming Chen
             and Esben Jannik Bjerrum},
  title   = {Graph Networks for Molecular Design},
  journal = {Machine Learning: Science and Technology},
  year    = {2021},
  doi     = {10.1088/2632-2153/abcf91}
}

@article{mercado2020practical,
  author  = {Roc{\'{i}}o Mercado and Tobias Rastemo and Edvard Lindel{\"{o}}f
             and G{\"{u}}nter Klambauer and Ola Engkvist and Hongming Chen
             and Esben Jannik Bjerrum},
  title   = {Practical Notes on Building Molecular Graph Generative Models},
  journal = {Applied AI Letters},
  year    = {2021},
  doi     = {10.1002/ail2.18}
}
```

---

## License

MIT, provided as-is.

GitHub: https://github.com/ailab-bio/GraphINVENT2
