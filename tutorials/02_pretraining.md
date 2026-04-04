# Tutorial 2: Pretraining

Pretraining trains the GGNN generative model **from random weight initialisation** using
supervised learning.  At each step the model receives a partial molecular graph and must
predict the probability of every possible next action (the Action Probability Distribution,
or action probabilities).  The loss is the KL divergence between the predicted action probabilities and the target action probabilities
derived from the training data during preprocessing.

---

## Prerequisites

- The dataset directory contains `train.h5`, `valid.h5`, and `test.h5` produced by
  [Tutorial 1: Preprocessing](./01_preprocessing.md).
- All feature parameters (`atom_types`, `formal_charge`, etc.) **must match** those used
  during preprocessing — the code checks this automatically.

---

## How training works

1. **Data loading** — training data is loaded in large blocks (controlled by `block_size`)
   from the HDF5 file into RAM, then served as mini-batches (`batch_size`) to the GPU.
2. **Forward pass** — the GGNN runs message passing on the partial graph, pools node
   embeddings via attention, and predicts action probabilities logits.
3. **Loss** — KL divergence between log-softmax of the model output and the normalised
   target action probabilities.
4. **Gradient accumulation** — gradients are accumulated over `accumulation_steps` batches
   before the optimiser step, allowing effective batch sizes larger than GPU memory permits.
5. **Learning rate schedule** — a one-cycle scheduler ramps the LR up from
   `init_lr / max_rel_lr` to `init_lr * max_rel_lr` and then down to
   `init_lr * min_rel_lr` over the full training run.
6. **Sampling** — every `sample_every` epochs the model is evaluated by generating
   `n_samples` new molecules and computing validity, uniqueness, and UC-JSD.
7. **Checkpointing** — a model checkpoint (`model_restart_<epoch>.pth`) is saved after
   each evaluation epoch.

---

## Parameters

### Required (must match preprocessing)

| Parameter | Description |
|-----------|-------------|
| `atom_types` | Same as preprocessing |
| `formal_charge` | Same as preprocessing |
| `imp_H` | Same as preprocessing |
| `chirality` | Same as preprocessing |
| `max_n_nodes` | Same as preprocessing |
| `use_aromatic_bonds` | Same as preprocessing |
| `use_canon` | Same as preprocessing |
| `use_chirality` | Same as preprocessing |
| `use_explicit_H` | Same as preprocessing |
| `ignore_H` | Same as preprocessing |
| `decoding_route` | Same as preprocessing |

### Training settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `"cuda"` | `"cuda"` for GPU, `"cpu"` for CPU |
| `epochs` | `100` | Total number of training epochs |
| `batch_size` | `1000` | Mini-batch size (number of subgraphs per gradient step before accumulation) |
| `block_size` | `100000` | Number of subgraphs loaded from disk into RAM at once |
| `accumulation_steps` | `256` | Gradient accumulation steps; effective batch = `batch_size × accumulation_steps` |
| `init_lr` | `1e-4` | Base learning rate |
| `max_rel_lr` | `10` | Peak LR = `init_lr × max_rel_lr` |
| `min_rel_lr` | `0.0001` | Final LR = `init_lr × min_rel_lr` |
| `n_workers` | `0` | DataLoader worker processes (0 = main process) |
| `restart` | `false` | Set to `true` to resume from the last saved checkpoint |

### Evaluation settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_every` | `10` | Generate molecules every N epochs for evaluation |
| `n_samples` | `2000` | Number of molecules to sample per evaluation |

### TensorBoard

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_tensorboard` | `false` | Enable TensorBoard logging |

### GGNN architecture

The model architecture is a Gated Graph Neural Network (GGNN).  The defaults below work
well for drug-like molecules with up to ~30 heavy atoms.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `message_passes` | `3` | Number of message-passing rounds |
| `message_size` | `100` | Dimensionality of the messages and hidden node states |
| `hidden_node_features` | `100` | Node embedding dimension |
| `enn_depth` | `4` | Layers in the edge network (message) MLP |
| `enn_hidden_dim` | `250` | Width of the edge network MLP |
| `enn_dropout_p` | `0.0` | Dropout in the edge network MLP |
| `mlp1_depth` | `4` | Layers in the tier-1 action probabilities readout MLP |
| `mlp1_hidden_dim` | `500` | Width of the tier-1 action probabilities readout MLP |
| `mlp1_dropout_p` | `0.0` | Dropout in the tier-1 action probabilities readout MLP |
| `mlp2_depth` | `4` | Layers in the tier-2 action probabilities readout MLP |
| `mlp2_hidden_dim` | `500` | Width of the tier-2 action probabilities readout MLP |
| `mlp2_dropout_p` | `0.0` | Dropout in the tier-2 action probabilities readout MLP |
| `gather_att_depth` | `4` | Layers in the attention MLP (graph pooling) |
| `gather_att_hidden_dim` | `250` | Width of the attention MLP |
| `gather_att_dropout_p` | `0.0` | Dropout in the attention MLP |
| `gather_emb_depth` | `4` | Layers in the embedding MLP (graph pooling) |
| `gather_emb_hidden_dim` | `250` | Width of the embedding MLP |
| `gather_emb_dropout_p` | `0.0` | Dropout in the embedding MLP |
| `gather_width` | `100` | Output size of the graph-level pooling block |

---

## Configuration file

> **Tip:** `jobs/pretrain/params.json` is a template — copy it before editing
> so the original stays intact and each experiment has its own config file:
> ```bash
> cp jobs/pretrain/params.json jobs/pretrain/my_experiment.json
> python submit.py --config jobs/pretrain/my_experiment.json
> ```

Edit your copy of `jobs/pretrain/params.json`:

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "gdb13-debug",
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
    "job_type": "pretrain",
    "atom_types": ["C", "N", "O", "S", "Cl"],
    "formal_charge": [-1, 0, 1],
    "imp_H": [0, 1, 2, 3],
    "chirality": ["None", "R", "S"],
    "max_n_nodes": 13,
    "use_aromatic_bonds": true,
    "use_canon": true,
    "use_chirality": false,
    "use_explicit_H": false,
    "ignore_H": false,
    "device": "cuda",
    "batch_size": 1000,
    "block_size": 100000,
    "accumulation_steps": 256,
    "epochs": 100,
    "init_lr": 1e-4,
    "max_rel_lr": 10,
    "min_rel_lr": 0.0001,
    "sample_every": 10,
    "n_samples": 2000,
    "n_workers": 0,
    "restart": false,
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

---

## Running the job

```bash
python submit.py --config jobs/pretrain/params.json
```

---

## Output files

Everything is written to `output/<dataset>/pretrain/job_0/`.

| File | Description |
|------|-------------|
| `params_all.json` | All resolved parameters |
| `convergence.log` | Epoch, LR, training loss, validation loss, UC-JSD score per evaluation epoch |
| `generation.log` | Per-epoch generation metrics: fraction valid, fraction unique, avg nodes, property histograms |
| `validation.log` | Per-epoch NLL statistics for validation, training, and generated sets |
| `model_restart_<N>.pth` | Model checkpoint saved after evaluation epoch N |
| `generation/` | Directory containing generated SMILES (`.smi`), likelihoods (`.likelihood`), and validity flags (`.valid`) for each evaluation epoch |

If TensorBoard is enabled, the TensorBoard data is written to
`output/<dataset>/pretrain/<job_name>/tensorboard/`.

---

## Monitoring training

### `convergence.log`

```
epoch, lr, avg_train_loss, avg_valid_loss, model_score
Epoch 10, 0.00100000, 1.23456789, 1.34567890, 0.05
Epoch 20, 0.00080000, 1.10234567, 1.19876543, 0.12
...
```

- `avg_train_loss` / `avg_valid_loss`: KL divergence averaged over mini-batches.
  Both should decrease and converge.  A large gap indicates overfitting.
- `model_score`: The UC-JSD between the NLL distributions of the training, validation,
  and generated sets.  Lower is better; values near 0 indicate the model has learned
  the training distribution well.

### `generation.log`

```
set, fraction_valid, fraction_valid_pt, fraction_pt, run_time, avg_n_nodes, ...
Epoch 10, 0.523, 0.412, 0.789, 45.2, 9.3, ...
```

- `fraction_valid`: Fraction of generated graphs that pass RDKit sanitisation.
- `fraction_pt` (fraction properly terminated): Fraction of graphs that emitted an
  explicit terminate action (as opposed to hitting `max_n_nodes`).
- `fraction_valid_pt`: Fraction of properly-terminated graphs that are also valid.
  This is the most meaningful quality metric during training.

### TensorBoard

```bash
tensorboard --logdir output/<dataset>/pretrain/<job_name>/tensorboard/
```

---

## Restarting training

If training is interrupted, set `"restart": true` and rerun `submit.py`.
The script automatically finds the latest `model_restart_<N>.pth` in the job
directory and resumes from epoch N+1.

---

## Tips

- **Underfitting**: increase `epochs`, decrease `init_lr`, or increase model capacity
  (`message_passes`, `hidden_node_features`, MLP widths).
- **Overfitting**: add dropout (`enn_dropout_p`, `mlp1_dropout_p`, etc.) or reduce
  model capacity.
- **GPU out of memory**: reduce `batch_size` and increase `accumulation_steps` by the
  same factor to keep the effective batch size constant.
- **Slow preprocessing IO**: increase `block_size` so fewer disk reads are needed.

---

## Next steps

- Use the trained model to generate molecules: [Tutorial 5: Sampling](./05_sampling.md).
- Fine-tune on a new dataset: [Tutorial 3: Transfer Learning](./03_transfer_learning.md).
- Optimise for a specific property: [Tutorial 4: Reinforcement Learning](./04_reinforcement_learning.md).
