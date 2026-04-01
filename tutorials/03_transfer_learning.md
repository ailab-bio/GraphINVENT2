# Tutorial 3: Transfer Learning

Transfer learning (also called supervised fine-tuning or domain adaptation) takes a model
already pretrained on a large general dataset and continues supervised training on a smaller,
more focused dataset.  The idea is that the pretrained model has already learned general
structural patterns, so the fine-tuning dataset needs far fewer examples and fewer epochs
to push the model's distribution towards the target chemical space.

Common use cases:

- Adapting a general drug-like-molecule model to a specific target class (e.g. kinase
  inhibitors, natural products).
- Adapting a model trained on a synthetic-accessible space to a bioactive-compound
  database.

---

## Prerequisites

1. A **pretrained model** from [Tutorial 2: Pretraining](./02_pretraining.md).  You need
   the checkpoint file `model_restart_<N>.pth` from the pretrain job directory.
2. A **new dataset** (train/valid/test SMILES) that has been **preprocessed** using the
   same feature parameters as the original pretraining dataset.  See
   [Tutorial 1: Preprocessing](./01_preprocessing.md).

---

## Dataset compatibility

The new (fine-tuning) dataset and the original pretraining dataset **must share the same
feature encoding**:

| Must match | Why |
|-----------|-----|
| `atom_types` | Node feature dimension |
| `formal_charge` | Node feature dimension |
| `imp_H` | Node feature dimension |
| `chirality` | Node feature dimension |
| `max_n_nodes` | Model input/output tensor shapes |
| `use_aromatic_bonds` | Edge feature dimension |
| `use_chirality` | Node feature dimension |
| `use_explicit_H` / `ignore_H` | Node feature dimension |
| `decoding_route` | Determines subgraph ordering |

The model architecture parameters (`message_passes`, `hidden_node_features`, all MLP
parameters) must also be identical to those used during pretraining — you cannot change
the model capacity during transfer learning.

---

## How transfer learning works

Transfer learning uses the **same supervised training loop** as pretraining (KL-divergence
loss, one-cycle LR schedule, gradient accumulation), with two differences:

1. **Initialisation**: the model weights are loaded from the pretrained checkpoint
   instead of being randomly initialised.
2. **Learning rate**: a lower initial learning rate (`init_lr`) is recommended to avoid
   overwriting the pretrained representations too aggressively.

`generation_epoch` specifies **which pretrain checkpoint to load** (i.e., the epoch
number N in `model_restart_<N>.pth`).  If you trained for 100 epochs and want to start
from the final checkpoint, set `generation_epoch: 100`.

---

## Parameters

### New parameters (not in pretraining)

| Parameter | Description |
|-----------|-------------|
| `pretrained_model_dir` | Path to the directory containing `model_restart_<N>.pth` |
| `generation_epoch` | Checkpoint to load: loads `model_restart_<generation_epoch>.pth` |

### Recommended changes from pretraining defaults

| Parameter | Pretraining | Transfer Learning | Reason |
|-----------|------------|-------------------|--------|
| `init_lr` | `1e-4` | `1e-5` | Lower LR to preserve pretrained features |
| `epochs` | `100` | `50` | Fewer epochs needed; risk of forgetting otherwise |
| `max_rel_lr` | `10` | `5` | Smaller LR swing |

---

## Directory layout

For this tutorial we assume:

```
data/
  datasets/
    gdb13-debug/          ← original pretraining dataset (already preprocessed)
      train.h5, valid.h5, test.h5
    new-dataset/          ← fine-tuning target dataset
      train.smi, valid.smi, test.smi   ← preprocess this first!
      train.h5, valid.h5, test.h5      ← produced by preprocessing

output/
  gdb13-debug/
    pretrain/
      job_0/
        model_restart_100.pth          ← pretrained model we want to load
```

---

## Configuration file

> **Tip:** `jobs/transfer/params.json` is a template — copy it before editing
> so the original stays intact and each experiment has its own config file:
> ```bash
> cp jobs/transfer/params.json jobs/transfer/my_experiment.json
> python submit.py --config jobs/transfer/my_experiment.json
> ```

Edit your copy of `jobs/transfer/params.json`:

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "new-dataset",
    "n_jobs": 1,
    "jobdir_start_idx": 0,
    "use_slurm": false,
    "slurm": {
      "account": "XXXXXXXXXX",
      "run_time": "0-06:00:00",
      "gpus_per_node": "T4:1"
    }
  },
  "job": {
    "job_type": "transfer",
    "atom_types": ["C", "N", "O", "S", "Cl"],
    "formal_charge": [-1, 0, 1],
    "imp_H": [0, 1, 2, 3],
    "chirality": ["None", "R", "S"],
    "max_n_nodes": 13,
    "use_aromatic_bonds": false,
    "use_canon": true,
    "use_chirality": false,
    "use_explicit_H": false,
    "ignore_H": false,
    "device": "cuda",
    "batch_size": 1000,
    "block_size": 100000,
    "accumulation_steps": 256,
    "epochs": 50,
    "init_lr": 1e-5,
    "max_rel_lr": 5,
    "min_rel_lr": 0.0001,
    "sample_every": 10,
    "n_samples": 2000,
    "n_workers": 0,
    "restart": false,
    "generation_epoch": 100,
    "pretrained_model_dir": "./output/gdb13-debug/pretrain/job_0/",
    "decoding_route": "bfs",
    "use_tensorboard": true,
    "enn_depth": 4,
    "enn_dropout_p": 0.0,
    "enn_hidden_dim": 250,
    "mlp1_depth": 4,
    "mlp1_dropout_p": 0.0,
    "mlp1_hidden_dim": 500,
    "mlp2_depth": 4,
    "mlp2_dropout_p": 0.0,
    "mlp2_hidden_dim": 500,
    "gather_att_depth": 4,
    "gather_att_dropout_p": 0.0,
    "gather_att_hidden_dim": 250,
    "gather_emb_depth": 4,
    "gather_emb_dropout_p": 0.0,
    "gather_emb_hidden_dim": 250,
    "gather_width": 100,
    "hidden_node_features": 100,
    "message_passes": 3,
    "message_size": 100
  }
}
```

Key fields to change for your use case:

- `"dataset"`: name of your fine-tuning dataset directory
- `"data_path"`: parent directory of the fine-tuning dataset
- `"generation_epoch"`: which pretrain checkpoint to load
- `"pretrained_model_dir"`: path to the pretrain job output directory
- All feature parameters: must match preprocessing of the fine-tuning dataset

---

## Running the job

First preprocess the fine-tuning dataset if you have not already:

```bash
# Adjust jobs/preprocess/params.json to point at your new dataset
python submit.py --config jobs/preprocess/params.json
```

Then run transfer learning:

```bash
python submit.py --config jobs/transfer/params.json
```

---

## Output files

Output is written to `output/<dataset>/transfer/job_0/`, with the same structure as
pretraining:

| File | Description |
|------|-------------|
| `params_all.json` | All resolved parameters |
| `convergence.log` | Epoch, LR, train loss, validation loss, UC-JSD |
| `generation.log` | Per-epoch molecule quality metrics |
| `validation.log` | Per-epoch NLL statistics |
| `model_restart_<N>.pth` | Checkpoints saved at evaluation epochs |
| `generation/` | Generated SMILES, likelihoods, validity flags |

The epoch counter resets to 1 for the new training run, so `model_restart_50.pth` is
the final checkpoint after 50 epochs of transfer learning.

---

## Monitoring and tips

Monitor the same metrics as pretraining (`convergence.log`, `generation.log`).

- If `fraction_valid_pt` drops sharply at the start of training, reduce `init_lr`
  further or reduce `max_rel_lr`.
- If the model does not converge to the target distribution, try more `epochs` or a
  slightly higher `init_lr`.
- A `fraction_valid_pt` substantially above the pretraining baseline suggests the
  model is successfully specialising.

---

## Next steps

- Generate molecules from the fine-tuned model: [Tutorial 5: Sampling](./05_sampling.md).
  Set `generation_epoch` to the transfer-learning epoch you want to sample from and
  `pretrained_model_dir` (or `job_dir`) to the transfer job output directory.
- Apply RL on top of transfer learning: [Tutorial 4: Reinforcement Learning](./04_reinforcement_learning.md).
