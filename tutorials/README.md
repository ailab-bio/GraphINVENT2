# GraphINVENT2 Tutorials

Step-by-step guides for every job type in GraphINVENT2.

---

## Tutorials

| # | Tutorial | Description |
|---|----------|-------------|
| 1 | [Preprocessing](./01_preprocessing.md) | Convert SMILES files to HDF5 format for training |
| 2 | [Pretraining](./02_pretraining.md) | Train a generative model from random initialisation (`unconditional`) |
| 3 | [Transfer Learning](./03_transfer_learning.md) | Supervised fine-tuning on a new dataset (`unconditional` + `resume_from`) |
| 4 | [Reinforcement Learning](./04_reinforcement_learning.md) | Goal-directed property optimisation (`goal_directed`) |
| 5 | [Sampling](./05_sampling.md) | Generate new molecules from a trained model (`generate`) |
| 5 | [Conditional Generation](./05_conditional_generation.md) | Train and sample a property-conditioned model (`conditional`) |

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
    "graphinvent_path": "./src/graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "gdb13-debug",
    "job_name": "run",
    "use_slurm": false
  },
  "job": {
    "job_type": "unconditional",
    "..."
  }
}
```

Output is always written to `output/<dataset>/<job_type>/<job_name>/`.

---

## Author

Rocío Mercado
