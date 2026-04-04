# Tutorial 5: Conditional Molecular Generation

Conditional generation lets the model learn to produce molecules with specific property values, such as a target solubility, bioactivity, or molecular weight.  A property (condition) vector is injected into the GGNN at every message-passing round via a **virtual seed node**, so the model can condition on any continuous scalar properties you supply.

---

## Architecture overview

```
condition_vector (batch, condition_dim)
        │
        ▼
ConditionEncoder (2-layer MLP)
        │
        ▼
condition_embedding (batch, condition_embedding_dim)
        │
        ▼ initialises hidden state of virtual seed node (index 0)
┌───────────────────────────────────────┐
│  GGNN message passing (T rounds)      │
│  seed node → real nodes (virtual edge)│
│  real nodes ↔ real nodes (bond edges) │
└───────────────────────────────────────┘
        │ strip seed node
        ▼
readout → action probability distribution
```

The seed node is prepended to every molecular graph.  At each message-passing step every real atom receives a message from it, so the condition influences atom representations throughout the network.  The seed node is stripped before the readout, so the action probability distribution has the same size as for unconditional models — no other code needs to change.

---

## Step 1 — Prepare a conditioned dataset

Your input SMILES file must be **tab-separated** with a header row.  The first column must be named `SMILES`; subsequent columns are property values.

```
SMILES	pLogS	MW
CC(=O)O	-0.51	60.05
c1ccccc1	0.23	78.11
...
```

- Header line is required.
- Property values must be numeric (float or int).
- All rows must have the same number of columns.

> **Full path required** when the file is referenced from a params.json.  See Tutorial 01 for notes on path conventions.

---

## Step 2 — Preprocess

Configure `jobs/preprocess/params.json`:

```json
{
  "submission": {
    "dataset_dir": "data/datasets/my_cond_dataset/",
    "smiles_file": "/absolute/path/to/my_molecules.tsv"
  },
  "job": {
    "job_type": "preprocess",
    "condition_dim": 2,
    "conditioning": {
      "properties": ["pLogS", "MW"],
      "source": "smiles_file"
    }
  }
}
```

- `condition_dim` must equal the number of property columns in the input file.
- Set `conditioning.properties` to the column names in the same order.

Run:

```bash
python submit.py --config jobs/preprocess/params.json
```

This writes `train.smi`, `valid.smi`, `test.smi` (tab-separated, preserving property columns) and the corresponding HDF5 files.  Each subgraph in the HDF5 carries a `condition_vector` dataset of shape `(n_subgraphs, condition_dim)`.

> **Note:** When conditioning is enabled, identical subgraphs from different molecules are stored separately (no deduplication) so that each carries its own condition vector.  This means the preprocessed dataset will be larger than the unconditional equivalent.

---

## Step 3 — Train a conditional model

Configure `jobs/conditional/params.json`:

```json
{
  "submission": {
    "dataset_dir": "data/datasets/my_cond_dataset/",
    "job_dir": "output/my_cond_dataset/conditional/run1/"
  },
  "job": {
    "job_type": "conditional",
    "condition_dim": 2,
    "condition_embedding_dim": 100,
    "n_epochs": 100,
    "batch_size": 1000,
    "accumulation_steps": 10
  }
}
```

Key parameters:

| Parameter | Description |
|---|---|
| `condition_dim` | Number of property inputs (must match preprocessing). |
| `condition_embedding_dim` | Hidden size of the ConditionEncoder MLP and the seed node embedding.  Defaults to `hidden_node_features` if 0. |
| `resume_from` | Path to a pretrained unconditional checkpoint to fine-tune from.  Null = train from scratch. |

Run:

```bash
python submit.py --config jobs/conditional/params.json
```

Training is identical to an unconditional job except that at each step the condition vectors from the batch are encoded and injected into the GGNN.

---

## Step 4 — Generate with target properties

Configure `jobs/sample/params.json`:

```json
{
  "submission": {
    "dataset_dir": "data/datasets/my_cond_dataset/",
    "job_dir": "output/my_cond_dataset/conditional/run1/"
  },
  "job": {
    "job_type": "sample",
    "sample_mode": "generate",
    "condition_dim": 2,
    "condition_embedding_dim": 100,
    "sample_conditions": {
      "pLogS": -2.5,
      "MW": 350.0
    },
    "n_samples": 1000,
    "generation_epoch": 100
  }
}
```

- `sample_conditions` maps property names to target values.  The order must match the column order used during preprocessing.
- The same condition vector is broadcast across the entire generation batch.

Run:

```bash
python submit.py --config jobs/sample/params.json
```

Generated SMILES will be written to `output/.../generation/`.

---

## Step 5 — Visualise

```bash
python visualize.py output/my_cond_dataset/conditional/run1/generation/epoch_100_0.smi
```

---

## Tips

**Normalise your properties.**  Scale each property to a similar range (e.g. zero mean, unit variance or [0, 1]) before preprocessing.  The ConditionEncoder is a small MLP, so large-magnitude inputs or very different scales between properties make optimisation harder.

**Start from a pretrained unconditional model.**  Set `resume_from` to a pretrained checkpoint.  The ConditionEncoder and virtual-edge MLP start from random weights; the rest of the network benefits from a warm start.

**condition_embedding_dim.**  A value equal to `hidden_node_features` (the default) works well.  Smaller values reduce the capacity of the condition signal.

**Batch size and accumulation.**  Because subgraphs are not deduplicated, conditional datasets tend to be larger.  Increase `accumulation_steps` rather than `batch_size` if memory is tight.

---

## Summary of new parameters

| Parameter | Default | Description |
|---|---|---|
| `condition_dim` | `0` | Number of property inputs.  0 = unconditional. |
| `condition_embedding_dim` | `hidden_node_features` | Size of seed-node embedding. |
| `conditioning` | `null` | Dict with `properties` list and `source` key.  Required for preprocessing. |
| `sample_conditions` | `null` | Dict mapping property name → target value, used at generation time. |
