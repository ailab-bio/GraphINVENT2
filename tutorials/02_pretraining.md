# Tutorial 2: Pretraining

Pretraining fits the GGNN to a dataset by supervised learning on the decoding routes produced
during preprocessing. Each training example is a partial molecular graph paired with a target
distribution over the next action, and the model is asked to reproduce that distribution. The
loss is the KL divergence between the log-softmax of the model's output and the row-normalised
target, averaged over the batch.

Nothing about this objective rewards chemistry directly. The model learns which action follows
which subgraph in the training data, and chemical validity is a consequence of having learned
that mapping well rather than something the loss enforces. This is why the validity of sampled
molecules is tracked separately during training and why it can lag the loss curve.

---

## Prerequisites

- `train.h5`, `valid.h5`, and `test.h5` in the dataset directory, produced by
  [Tutorial 1: Preprocessing](./01_preprocessing.md).
- Feature and encoding parameters matching that preprocessing run. They are read back from
  `preprocessing_params.json` and checked, so a mismatch is reported rather than silently
  training a model with the wrong tensor shapes.

---

## What a training run does

Training data is read from HDF5 in blocks of `block_size` subgraphs and served as mini-batches
of `batch_size`. Gradients are accumulated over `accumulation_steps` batches before each
optimiser step, which gives an effective batch of `batch_size × accumulation_steps` without
holding that many graphs on the device at once.

The learning rate follows a one-cycle schedule sized to the number of optimiser steps rather
than the number of batches, so it starts at `init_lr`, peaks at `init_lr × max_rel_lr`, and
anneals to `init_lr × min_rel_lr` across the whole run. Sizing it by batch count instead would
advance the schedule only a fraction of the way and the peak would never be reached; this is
worth knowing if you change `accumulation_steps`, because doing so changes how many optimiser
steps a run contains.

Every `sample_every` epochs the model generates `n_samples` molecules, the resulting validity,
uniqueness, novelty, diversity and similarity statistics are appended to `generation.log`, the
UC-JSD is written to `validation.log`, and a checkpoint is saved. Checkpoints exist only at
these evaluation epochs, so `sample_every` controls checkpoint granularity as well as
evaluation cost. Each checkpoint stores the optimiser and scheduler state alongside the
weights so that a restart resumes Adam's moments and the LR cycle instead of resetting both.

---

## Parameters

### Fixed by preprocessing

`atom_types`, `formal_charge`, `imp_H`, `chirality`, `max_n_nodes`, `use_aromatic_bonds`,
`use_canon`, `use_chirality`, `use_explicit_H`, `ignore_H`, and `decoding_route` must match the
values recorded in `preprocessing_params.json`. You do not normally set them in the training
config at all — leaving them out lets them be inherited.

### Training

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `device` | `"cuda"` | `"cuda"`, `"mps"`, or `"cpu"` |
| `epochs` | `100` | Number of passes over the training set |
| `batch_size` | `1000` | Subgraphs per forward/backward pass |
| `block_size` | `100000` | Subgraphs read from HDF5 into RAM at a time |
| `accumulation_steps` | `10` | Batches accumulated per optimiser step |
| `init_lr` | `1e-4` | Learning rate at the start of the cycle |
| `max_rel_lr` | `10` | Peak LR as a multiple of `init_lr` |
| `min_rel_lr` | `0.0001` | Final LR as a multiple of `init_lr` |
| `n_workers` | `0` | DataLoader worker processes; 0 means the main process |
| `restart` | `false` | Resume from the last checkpoint in the job directory |
| `seed` | `0` | 0 leaves RNG unseeded; any positive integer fixes Python, NumPy, and PyTorch |

### Evaluation

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `sample_every` | `10` | Epochs between evaluation, sampling, and checkpointing |
| `n_samples` | `2000` | Molecules generated per evaluation |
| `use_tensorboard` | `false` | Write TensorBoard event files to `<job_dir>/tensorboard/` |

The shipped `jobs/unconditional/params.json` sets `n_samples` to 100, which keeps a smoke test
fast but is far too few molecules for the uniqueness and diversity numbers to mean anything.
Raise it before you read those columns seriously.

### GGNN architecture

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `message_passes` | `3` | Message-passing rounds |
| `message_size` | `100` | Width of a message and of the GRU input |
| `hidden_node_features` | `100` | Node hidden state dimension |
| `enn_depth` | `2` | Layers in the edge network that produces messages |
| `enn_hidden_dim` | `128` | Width of the edge network |
| `enn_dropout_p` | `0.0` | Dropout in the edge network |
| `mlp1_depth` | `2` | Layers in the first-tier readout MLP |
| `mlp1_hidden_dim` | `256` | Width of the first-tier readout MLP |
| `mlp1_dropout_p` | `0.0` | Dropout in the first-tier readout MLP |
| `mlp2_depth` | `2` | Layers in the second-tier readout MLP |
| `mlp2_hidden_dim` | `256` | Width of the second-tier readout MLP |
| `mlp2_dropout_p` | `0.0` | Dropout in the second-tier readout MLP |
| `gather_att_depth` | `2` | Layers in the attention MLP used for graph pooling |
| `gather_att_hidden_dim` | `128` | Width of the attention MLP |
| `gather_att_dropout_p` | `0.0` | Dropout in the attention MLP |
| `gather_emb_depth` | `2` | Layers in the embedding MLP used for graph pooling |
| `gather_emb_hidden_dim` | `128` | Width of the embedding MLP |
| `gather_emb_dropout_p` | `0.0` | Dropout in the embedding MLP |
| `gather_width` | `100` | Output size of the graph-level pooling block |

These are the values in `src/graphinvent/parameters/defaults.py` and in
`jobs/unconditional/params.json`, and they are sized for small graphs — the defaults were
tuned on GDB-13-scale molecules of at most around 13 heavy atoms. For drug-like molecules the
`experiments/chembl_pretrain/pretrain_params.json` config is a better starting point; it widens
`hidden_node_features` and `message_size` to 256 and uses four message-passing rounds.

Any architecture change makes existing checkpoints unloadable, since `load_state_dict` requires
matching shapes.

---

## Configuration file

Copy the template rather than editing it:

```bash
cp jobs/unconditional/params.json jobs/unconditional/my_experiment.json
python submit.py --config jobs/unconditional/my_experiment.json
```

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./src/graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "debug",
    "job_name": "run",
    "use_slurm": false,
    "slurm": {
      "account": "XXXXXXXXXX",
      "run_time": "0-06:00:00",
      "gpus_per_node": "T4:1"
    }
  },
  "job": {
    "job_type": "unconditional",
    "resume_from": null,

    "device": "cuda",
    "restart": false,
    "use_tensorboard": true,
    "decoding_route": "bfs",
    "use_aromatic_bonds": true,

    "epochs": 100,
    "batch_size": 1000,
    "block_size": 100000,
    "accumulation_steps": 10,
    "init_lr": 1e-4,
    "max_rel_lr": 10,
    "min_rel_lr": 0.0001,
    "sample_every": 10,
    "n_samples": 2000,
    "n_workers": 0,

    "hidden_node_features": 100,
    "message_passes": 3,
    "message_size": 100,
    "enn_depth": 2,
    "enn_dropout_p": 0.0,
    "enn_hidden_dim": 128,
    "mlp1_depth": 2,
    "mlp1_dropout_p": 0.0,
    "mlp1_hidden_dim": 256,
    "mlp2_depth": 2,
    "mlp2_dropout_p": 0.0,
    "mlp2_hidden_dim": 256,
    "gather_att_depth": 2,
    "gather_att_dropout_p": 0.0,
    "gather_att_hidden_dim": 128,
    "gather_emb_depth": 2,
    "gather_emb_dropout_p": 0.0,
    "gather_emb_hidden_dim": 128,
    "gather_width": 100
  }
}
```

`resume_from: null` means train from random initialisation. Setting it to a checkpoint path
turns the same job type into transfer learning, which is [Tutorial 3](./03_transfer_learning.md).

`use_aromatic_bonds` appears here because it determines the edge feature dimension and so must
agree with preprocessing; the remaining feature parameters are inherited from
`preprocessing_params.json`.

---

## Running the job

```bash
python submit.py --config jobs/unconditional/params.json
```

---

## Output

Everything is written to `output/<dataset>/unconditional/<job_name>/`, where `job_name` comes
from the `submission` block. Re-running a job into a directory that already has results moves
the old ones into a timestamped `_previous_run_<stamp>/` rather than overwriting them.

| File | Contents |
|------|----------|
| `params.json` | The job block as submitted |
| `params_all.json` | All resolved parameters, library versions, device, git hash, seed |
| `convergence.log` | `epoch, lr, avg_train_loss, avg_valid_loss, model_score` |
| `generation.log` | One row per evaluation epoch: validity, uniqueness, novelty, SA scores, diversity and similarity statistics, and property histograms |
| `validation.log` | Per-molecule NLL on the validation, training, and generated sets, plus UC-JSD |
| `model_restart_<N>.pth` | Weights, optimiser state, and scheduler state at evaluation epoch N |
| `generation/` | `epoch_<N>_batch_<B>.smi`, `.likelihood`, and `.valid` for each evaluation epoch |
| `progress.png` | Regenerated at every evaluation epoch from `generation.log` and `convergence.log` |
| `tensorboard/` | Event files when `use_tensorboard` is true |

---

## Reading the logs

### `convergence.log`

```
epoch, lr, avg_train_loss, avg_valid_loss, model_score
Epoch 1, 0.00004068, 10.06695271, 10.00380898, NA
Epoch 2, 0.00004272, 9.94658470, 9.90818405, NA
```

`avg_train_loss` and `avg_valid_loss` are the KL divergence averaged over mini-batches. Both
should fall and converge; a widening gap between them is the usual overfitting signature.
`model_score` is the UC-JSD and is `NA` on epochs where no evaluation ran.

### `validation.log`

```
set, avg_likelihood_per_molecule_val, avg_likelihood_per_molecule_train, avg_likelihood_per_molecule_gen, uc_jsd
Epoch 10, 109.40362, 102.80915, 0.00009, 0.3918432
```

UC-JSD is the Jensen–Shannon divergence between the NLL distributions of generated and
training molecules. It approaches zero as the model's own samples become as likely under the
model as the training data is, which makes it a convergence diagnostic rather than a quality
measure — a model that has memorised the training set scores well on it.

### `generation.log`

The first row is the training set itself, which gives the reference distribution the generated
sets are compared against. Subsequent rows are evaluation epochs.

| Column | Meaning |
|--------|---------|
| `fraction_valid` | Generated graphs that pass RDKit sanitisation |
| `fraction_pt` | Graphs that terminated by sampling the terminate action rather than hitting `max_n_nodes` |
| `fraction_valid_pt` | Graphs that are both valid and properly terminated |
| `avg_n_nodes`, `avg_n_edges` | Size of the generated molecules |
| `fraction_unique` | Distinct canonical SMILES among the valid molecules |
| `novelty` | Unique valid molecules absent from the training set |
| `sa_score_mean`, `sa_score_median`, `sa_score_std` | Synthetic accessibility (Ertl & Schuffenhauer), 1 easy to 10 hard |
| `internal_diversity` and similarity columns | See [the evaluation reference](./evaluation.md) |

`fraction_valid_pt` is the most informative single column during training, because a model can
reach high `fraction_valid` while rarely deciding to stop, in which case most molecules are
truncated at `max_n_nodes` and the size distribution is an artefact of the cap. Validity and
uniqueness are necessary conditions, not evidence that the model has learned the training
distribution; the histogram columns and the similarity statistics are what show whether the
generated set occupies the same region of chemical space.

### TensorBoard

```bash
tensorboard --logdir output/<dataset>/unconditional/<job_name>/tensorboard/
```

---

## Restarting

Set `"restart": true` and rerun. The job finds the highest `model_restart_<N>.pth` in the job
directory and continues from epoch N+1 with the saved optimiser and scheduler state. Note that
a restart appends to the existing logs rather than backing them up, which is the intended
behaviour but differs from a fresh run.

---

## When training does not converge

- Loss still falling at the end of the run: increase `epochs`, or raise `init_lr` if the curve
  is close to linear rather than flattening.
- Loss flat from the start: the LR peak may be too high; reduce `max_rel_lr`.
- Train and validation losses diverging: add dropout through `enn_dropout_p`, `mlp1_dropout_p`
  and the other dropout parameters, or reduce capacity.
- Out of memory: halve `batch_size` and double `accumulation_steps` to keep the effective batch
  constant. Remember that this also halves the number of optimiser steps per epoch, which
  changes the LR schedule.
- Data loading dominating runtime: increase `block_size` so fewer HDF5 reads are needed, or
  raise `n_workers` above 0.

---

## Next steps

- Sample from the trained model: [Tutorial 5: Sampling](./05_sampling.md).
- Adapt it to a narrower dataset: [Tutorial 3: Transfer learning](./03_transfer_learning.md).
- Optimise it toward a property: [Tutorial 4: Reinforcement learning](./04_reinforcement_learning.md).
