# GraphINVENT2

![cover image](./cover-image.png)

GraphINVENT2 is a platform for graph-based molecular generation and optimization.
It uses a tiered deep neural network architecture (GGNN) to probabilistically
generate new molecules one bond at a time, and reinforcement learning to guide
the model towards molecules with user-defined properties.

This is the actively maintained successor to GraphINVENT.  The methods are
described in [*Graph Networks for Molecular Design*](https://iopscience.iop.org/article/10.1088/2632-2153/abcf91)
(Mercado et al., 2021).

---

## Table of contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Job types](#job-types)
5. [Tutorials](#tutorials)
6. [Testing](#testing)
7. [Contributing](#contributing)
8. [References](#references)
9. [License](#license)

---

## Features

- **Autoregressive graph generation** — builds molecules atom-by-atom using a gated graph neural network (GGNN).
- **Pretraining** — learn a prior distribution over a large molecular dataset.
- **Transfer learning** — adapt a pretrained model to a new chemical space with a smaller, focused dataset.
- **Reinforcement learning** — optimize toward user-defined scoring criteria (QED, QSAR activity, target size, or custom).
- **Flexible preprocessing** — auto-detects molecular feature vocabulary; supports random, Butina-cluster, or custom dataset splits.
- **GPU and CPU support** — runs on CUDA GPUs or CPU; SLURM submission included.

---

## Installation

**Requirements:** Python 3.9+, pip.

### 1. Clone the repository

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

Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to
get the exact command for your platform and CUDA version.  Common cases:

```bash
# CPU only (any platform)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU — replace cu121 with your CUDA version (e.g. cu118, cu124)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (M1/M2/M3) — standard pip install uses MPS automatically
pip install torch torchvision torchaudio
```

### 4. Install GraphINVENT2

Install the package and all remaining dependencies in one command:

```bash
pip install -e .
```

This installs GraphINVENT2 in **editable mode** — changes to the source files in
`graphinvent/` take effect immediately without reinstalling.  It also registers
the `graphinvent-submit` command (equivalent to `python submit.py`).

### Verify

```bash
python -c "import torch, rdkit, h5py; print('Setup complete. PyTorch', torch.__version__)"
```

### Optional extras

```bash
pip install -e ".[tdc]"   # adds PyTDC for tools/tdc-create-dataset.py
pip install -e ".[dev]"   # adds ruff + pyright for development
```

### Conda alternative

If you prefer Conda (e.g. on Windows where RDKit pip wheels are sometimes unavailable):

```bash
conda create -n graphinvent python=3.11 -y
conda activate graphinvent
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 -y
conda install -c conda-forge rdkit h5py tqdm scikit-learn matplotlib tensorboard -y
```

### HPC / Singularity

A Singularity definition file is available at `docker/graphinvent.def` for
cluster environments where Conda or pip virtualenvs are not practical.

---

## Quick start

### 1. Preprocess a dataset

Edit `jobs/preprocess/params.json` to point at your SMILES file, then run:

```bash
python submit.py --config jobs/preprocess/params.json
# or, after pip install -e .:
graphinvent-submit --config jobs/preprocess/params.json
```

Feature parameters (`atom_types`, `formal_charge`, `imp_H`, `max_n_nodes`) are
**auto-detected** from the SMILES data — you only need to set encoding flags
(`use_chirality`, `use_aromatic_bonds`, etc.).

A small preprocessed example dataset is at `data/gdb13_1K/` if you want to
skip preprocessing and go straight to training.

### 2. Train a model

```bash
python submit.py --config jobs/pretrain/params.json
```

Training progress is logged to `output/<dataset>/pretrain/job_0/convergence.log`.
If `use_tensorboard: true`, launch the dashboard with:

```bash
tensorboard --logdir output/<dataset>/pretrain/tensorboard/
```

### 3. Generate molecules

```bash
python submit.py --config jobs/sample/params.json
```

Generated SMILES are written to `output/<dataset>/generate/job_0/generation/`.

---

## Job types

All jobs are launched with:

```bash
python submit.py --config jobs/<job_type>/params.json
```

Each config has two sections: `"submission"` (how to run) and `"job"` (what to
run).  Output is always written to `output/<dataset>/<job_type>/job_<idx>/`.

| Job type | Config | Description |
|----------|--------|-------------|
| `preprocess` | `jobs/preprocess/params.json` | Convert SMILES to HDF5; auto-detects feature vocabulary |
| `pretrain` | `jobs/pretrain/params.json` | Train a generative model from random initialization |
| `transfer` | `jobs/transfer/params.json` | Fine-tune a pretrained model on a new dataset |
| `rl` | `jobs/rl/params.json` | Optimize a model for molecular properties with RL |
| `generate` | `jobs/sample/params.json` | Sample new molecules from a trained model |

### Typical workflows

```
Preprocess → Pretrain → Generate
Preprocess (new data) → Transfer learning → Generate
Pretrain → Reinforcement learning → Generate
```

---

## Tutorials

| Tutorial | Topic |
|----------|-------|
| [01 Preprocessing](./tutorials/01_preprocessing.md) | Convert SMILES to HDF5 |
| [02 Pretraining](./tutorials/02_pretraining.md) | Train from scratch |
| [03 Transfer learning](./tutorials/03_transfer_learning.md) | Fine-tune on a new dataset |
| [04 Reinforcement learning](./tutorials/04_reinforcement_learning.md) | Property optimization |
| [05 Sampling](./tutorials/05_sampling.md) | Generate molecules |

---

## Testing

The test suite verifies that a completed preprocessing job produced correct output.
All commands should be run from the **repository root**.

### 1. Configure the target dataset

Edit `tests/config.py` to point at the dataset you want to verify:

```python
# tests/config.py
DATASET_DIR = Path("data/datasets/debug")   # must contain .smi and .h5 files
SMILES_FILE = Path("data/datasets/debug/debug.smi")  # original input; set to None for Mode B
```

### 2. Run the tests

```bash
pytest tests/ -v
```

### What is tested

| Test class | Checks |
|------------|--------|
| `TestSplitFileCounts` | `train/valid/test.smi` exist; molecule counts sum to the original file; no overlap between splits |
| `TestHDFFileCounts` | `train/valid/test.h5` exist and contain `nodes`, `edges`, `action probabilities`; HDF5 molecule count matches `.smi` count; subgraph count ≥ molecule count |
| `TestSMILESReconstruction` | Every graph in each HDF5 decodes to a valid SMILES; reconstructed SMILES set matches the `.smi` file (lossless round-trip) |

---

## Contributing

Contributions are welcome as issues or pull requests.  To report a bug, please
open an issue on GitHub.

---

## References

If you use GraphINVENT2 in your research, please cite:

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

GraphINVENT2 is licensed under the MIT license and is provided as-is.

GitHub: https://github.com/ailab-bio/GraphINVENT2
