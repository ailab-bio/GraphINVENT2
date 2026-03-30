# GraphINVENT2 Tutorials

Step-by-step guides for every job type in GraphINVENT2.

---

## Tutorials

| # | Tutorial | Description |
|---|----------|-------------|
| 1 | [Preprocessing](./01_preprocessing.md) | Convert SMILES files to HDF5 format for training |
| 2 | [Pretraining](./02_pretraining.md) | Train a generative model from random initialisation |
| 3 | [Transfer Learning](./03_transfer_learning.md) | Supervised fine-tuning on a new dataset |
| 4 | [Reinforcement Learning](./04_reinforcement_learning.md) | Optimise for molecular properties with RL |
| 5 | [Sampling](./05_sampling.md) | Generate new molecules from a trained model |

---

## Typical workflows

### Train from scratch and generate
```
01_preprocessing → 02_pretraining → 05_sampling
```

### Domain adaptation
```
01_preprocessing (new dataset) → 03_transfer_learning → 05_sampling
```

### Property optimisation
```
02_pretraining → 04_reinforcement_learning → 05_sampling
```

---

## Quick reference

All jobs are launched via `submit.py`:

```bash
python submit.py --config jobs/<job_type>/params.json
```

Each `params.json` has two sections:

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "gdb13-debug",
    "n_jobs": 1,
    "jobdir_start_idx": 0,
    "use_slurm": false
  },
  "job": {
    "job_type": "pretrain",
    "..."
  }
}
```

Output is always written to `output/<dataset>/<job_type>/job_<idx>/`.

---

## Author

Rocío Mercado
