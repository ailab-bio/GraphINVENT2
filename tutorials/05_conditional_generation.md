# Tutorial 5: Conditional generation

A conditional model is trained on molecules labelled with continuous property values and
learns to condition its action distribution on a target vector of those properties. At
generation time you supply the target and the model samples molecules intended to have it.

The mechanism is a virtual seed node. A property vector is encoded by a small MLP into an
embedding, that embedding initialises the hidden state of an extra node prepended to every
graph, and each real atom receives a message from the seed at every message-passing round via
a dedicated virtual edge type. The seed is stripped before the readout, so the action
probability distribution keeps the same shape as in the unconditional model.

One detail that is easy to miss: the seed alone would be insufficient, because the first
action of every molecule is taken on an *empty* graph where there are no real atoms to carry
the signal. The condition embedding is therefore also concatenated to the graph embedding
before the readout, which widens the readout's input by `condition_embedding_dim`. The output
size is unchanged, but the model is not architecturally identical to its unconditional
counterpart and their checkpoints are not interchangeable.

---

## Architecture

```
condition_vector (batch, condition_dim)
        │
        ▼
ConditionEncoder (2-layer MLP)
        │
        ├──────────────────────────────► concatenated to the graph embedding
        ▼                                  before the readout
initialises the virtual seed node (index 0)
┌───────────────────────────────────────┐
│  GGNN message passing (T rounds)      │
│  seed node → real nodes (virtual edge)│
│  real nodes ↔ real nodes (bond edges) │
└───────────────────────────────────────┘
        │ strip seed node
        ▼
readout → action probability distribution
```

---

## Step 1 — Prepare a labelled dataset

The input file must be tab-separated with a header row whose first column is named `SMILES`.
Every subsequent column is a property.

```
SMILES	pLogS	MW
CC(=O)O	-0.51	60.05
c1ccccc1	0.23	78.11
```

The header is required, every value must parse as a float, and every row must have the same
number of columns; anything else raises a `ValueError` during preprocessing rather than being
skipped.

Scale the properties before you write the file. The `ConditionEncoder` is a two-layer MLP
receiving raw values, so a column in the hundreds and a column in the units contribute
gradients of very different magnitude and the larger one dominates. Zero-mean/unit-variance or
a [0, 1] rescaling both work; what matters is that the same transformation is applied at
preprocessing, training, and generation time, since nothing in the pipeline records or
re-applies it for you.

---

## Step 2 — Preprocess

Preprocessing is the standard `preprocess` job with `condition_dim` set to the number of
property columns. Every property column in the file is used, in file order, and preprocessing
fails if the count does not match `condition_dim`.

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./src/graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "my_cond_dataset",
    "smiles_file": "./data/raw/my_molecules.tsv",
    "job_name": "run",
    "use_slurm": false
  },
  "job": {
    "job_type": "preprocess",
    "condition_dim": 2,
    "conditioning": {
      "properties": ["pLogS", "MW"],
      "source": "smiles_file"
    },
    "auto_detect_features": true,
    "split_type": "random",
    "train_frac": 0.8,
    "valid_frac": 0.1,
    "use_aromatic_bonds": true,
    "use_canon": true,
    "batch_size": 1000,
    "block_size": 100000,
    "decoding_route": "bfs"
  }
}
```

```bash
python submit.py --config jobs/preprocess/my_cond_dataset.json
```

The `conditioning` block is not read during preprocessing — column selection and ordering come
from the TSV header alone. It is worth writing anyway, because the same block *is* read at
generation time to order the condition vector, and keeping one authoritative record of the
column order avoids getting it wrong later.

The run writes `train.smi`, `valid.smi`, and `test.smi` in tab-separated form with the property
columns preserved, plus the HDF5 files, each carrying a `condition_vector` dataset of shape
`(n_subgraphs, condition_dim)`.

Conditional preprocessing produces a larger dataset than the unconditional equivalent, because
identical subgraphs arising from different molecules can no longer be merged: each needs its
own condition vector. Budget disk space and preprocessing time accordingly.

---

## Step 3 — Train

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./src/graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "my_cond_dataset",
    "job_name": "run",
    "use_slurm": false
  },
  "job": {
    "job_type": "conditional",
    "resume_from": null,
    "condition_dim": 2,
    "condition_embedding_dim": 100,
    "conditioning": {
      "properties": ["pLogS", "MW"],
      "source": "smiles_file"
    },
    "sample_conditions": {"pLogS": -1.5, "MW": 250.0},

    "device": "cuda",
    "use_tensorboard": true,
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
    "n_workers": 0
  }
}
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `condition_dim` | `0` | Number of property inputs; must match the preprocessed dataset. Required by `submit.py` for this job type |
| `condition_embedding_dim` | `100` | Width of the `ConditionEncoder`'s hidden and output layers, and therefore of the seed embedding and of the extra readout input |
| `conditioning` | `null` | `{"properties": [...], "source": "smiles_file"}`; read only when ordering `sample_conditions` |
| `sample_conditions` | `null` | Property name to target value, used whenever molecules are generated |
| `resume_from` | `null` | Checkpoint to start from |

`sample_conditions` is not optional here even though this is a training job. Every evaluation
epoch generates molecules, and generation from a model with `condition_dim > 0` raises a
`ValueError` if `sample_conditions` is unset — rather than silently sampling with a neutral
condition and reporting metrics that do not describe the conditional model. The shipped
`jobs/conditional/params.json` does not include the key, so add it before the first evaluation
epoch or the run will stop at epoch `sample_every`.

The default `condition_embedding_dim` of 100 equals the default `hidden_node_features`, which
is a reasonable starting point since the seed embedding then has the same width as a node
hidden state. There is no automatic fallback: setting it to 0 builds a zero-width layer rather
than inheriting `hidden_node_features`.

```bash
python submit.py --config jobs/conditional/my_experiment.json
```

Training is the same supervised loop as the unconditional case; the only difference is that the
data loader yields a fourth tensor per batch and the model consumes it. See
[Tutorial 2](./02_pretraining.md) for the training parameters and the output files.

### Starting from an unconditional checkpoint

Setting `resume_from` to a pretrained unconditional checkpoint gives the message-passing layers
a warm start while the `ConditionEncoder`, the virtual-edge MLP, and the widened readout begin
from random weights. This is usually worth doing, since the chemical grammar is the expensive
part to learn and is unaffected by conditioning.

Note that the architecture inheritance in `src/graphinvent/parameters/config.py` treats
`condition_dim`, `condition_embedding_dim`, and `condition_type` as architecture keys, so they
are inherited from the checkpoint only when absent from your job config. Since the checkpoint
is unconditional and records `condition_dim: 0`, they must be present in your config — which
they are in the example above — or the model will be built unconditional and the conditioning
will be silently dropped.

---

## Step 4 — Generate at a target condition

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./src/graphinvent/",
    "job_name": "run",
    "use_slurm": false
  },
  "job": {
    "job_type": "generate",
    "sample_mode": "generate",
    "device": "cuda",
    "batch_size": 1000,
    "n_samples": 1000,
    "n_workers": 0,
    "pretrained_model_path": "./output/my_cond_dataset/conditional/run/model_restart_100.pth",
    "conditioning": {
      "properties": ["pLogS", "MW"],
      "source": "smiles_file"
    },
    "sample_conditions": {
      "pLogS": -2.5,
      "MW": 350.0
    }
  }
}
```

`pretrained_model_path` is required; `submit.py` rejects a `generate` job that has neither it
nor a valid `pretrained_model_dir` plus `generation_epoch` pair. `condition_dim` and
`condition_embedding_dim` are inherited from the checkpoint's `params_all.json`, as is the rest
of the architecture, and `dataset`/`data_path` are derived from it too.

Include `conditioning` here as well. The condition vector is assembled in the order given by
`conditioning["properties"]`; when that key is absent the code falls back to the key order of
the `sample_conditions` JSON object, which happens to work if you wrote the keys in column
order and silently permutes the vector if you did not.

The same condition vector is broadcast across the whole generation batch, so one job samples at
one target. Sweeping a property means running several jobs with different `sample_conditions`
and different `job_name` values.

```bash
python submit.py --config jobs/generate/my_cond_run.json
```

Output goes to `output/<dataset>/generate/<job_name>/` as `<n_samples>_samples.smi`,
`.likelihood`, and `.valid`, exactly as for an unconditional generate job. See
[Tutorial 5: Sampling](./05_sampling.md#output) for the file layout and the header caveat.

During *training*, by contrast, the per-epoch samples are written inside the job directory as
`generation/epoch_<N>_batch_<B>.smi`.

---

## Step 5 — Look at the results

```bash
python visualize.py output/my_cond_dataset/generate/run/1000_samples.smi
```

Whether conditioning worked is not answered by validity or by the loss. The question is
whether the property you asked for actually shifted, which means computing that property on the
generated molecules and comparing its distribution against the target and against an
unconditional baseline from the same model family. A conditional model that ignores its input
still produces valid, unique, novel molecules; it produces the same distribution whatever
you ask for.

Two checks are worth running before drawing conclusions:

- Sample at two well-separated targets and confirm the property distributions separate. If they
  do not, `condition_dim` and the column order are the first things to verify.
- Compare the achieved distribution against the training set's distribution at that condition.
  A model can appear to condition well by reproducing the region of chemical space where most
  of its training data with that property value lives, which is a weaker claim than steering.

The nearest-neighbour similarity statistics described in [the evaluation
reference](./evaluation.md) are automatically restricted to test molecules whose condition
values fall within ±0.3 of `sample_conditions`, so the reported `sim_mean` for a conditional
run compares against the matching part of the test set rather than all of it.

---

## Summary of the conditioning parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `condition_dim` | `0` | Number of property inputs; 0 means unconditional |
| `condition_embedding_dim` | `100` | Seed-embedding width, and the extra width added to the readout input |
| `condition_type` | `"virtual_node"` | Injection mechanism; the only value implemented |
| `conditioning` | `null` | Property names and source; consulted only to order `sample_conditions` |
| `sample_conditions` | `null` | Property name to target value, required whenever a conditional model generates |
