# Tutorial 1: Preprocessing

Before training any model, your raw SMILES data must be converted into HDF5 format.
This step encodes each molecule as a sequence of subgraphs (its **decoding route**) and
stores the node features, edge features, and target Action Probability Distributions (action probabilities)
in a compact binary format that the data loader can stream efficiently during training.

---

## SMILES file format

All `.smi` files (whether a single input file or pre-split train/valid/test files) must follow this format:

- One molecule per line: `<SMILES> [optional_identifier]`
- The identifier (name, ID, etc.) is separated by a space and is ignored during preprocessing
- A header line is detected automatically if it contains the word `SMILES` and is skipped
- Lines that cannot be parsed by RDKit are silently skipped

Example:
```
CCO ethanol
c1ccccc1 benzene
CC(=O)O acetic_acid
```

A bare SMILES-only file (no identifiers, no header) is equally valid.

---

## Dataset input modes

There are two ways to provide data.  Choose the one that fits your workflow.

### Mode A — single SMILES file (automatic splitting)

Set `"smiles_file"` in the `submission` block to the **full (absolute) path** of a `.smi`
file containing all your molecules (one SMILES per line, optional space-separated
identifier ignored).  Relative paths may work when `submit.py` is run from the repo root,
but an absolute path is safer and always unambiguous.

`submit.py` will read the file, split it into train / valid / test, write the three
`.smi` files into the dataset directory, then launch the HDF5 conversion.

```json
"submission": {
  "data_path": "./data/datasets/",
  "dataset":   "my-dataset",
  "smiles_file": "./data/datasets/my_molecules.smi"
}
```

### Mode B — pre-split directory (default)

Leave `"smiles_file"` absent or set to `null`.  The dataset directory must already
contain exactly these three files; an error is raised if any are missing:

| File | Content |
|------|---------|
| `train.smi` | Training set — one SMILES per line |
| `valid.smi` | Validation set — one SMILES per line |
| `test.smi`  | Test set — one SMILES per line |

```json
"submission": {
  "data_path": "./data/datasets/",
  "dataset":   "gdb13-debug",
  "smiles_file": null
}
```

---

## Preprocessing multiple datasets simultaneously

Both `"dataset"` and `"data_path"` accept either a single string or a **list of strings**.
When you provide a list, all datasets are preprocessed in a single run with a **shared,
union vocabulary** — the feature vocabulary (atom types, formal charges, implicit H counts,
and maximum node count) is computed across all datasets together before any HDF5 file is
written.  Each dataset still produces its own separate set of HDF5 files.

This is useful when you intend to pretrain on one corpus and later fine-tune or transfer
to another, since a shared vocabulary guarantees that both datasets are encoded with
identical feature dimensions.

`"smiles_file"` can be a list matching `"dataset"` in length — use a path for datasets
that need splitting (Mode A) and `null` for datasets that are already pre-split (Mode B).
Mixed Mode A and Mode B datasets in the same run are fully supported.

```json
"submission": {
  "python_path": "python",
  "graphinvent_path": "./graphinvent/",
  "data_path": "./data/datasets/",
  "dataset":     ["new-dataset",          "pretrained-set"],
  "smiles_file": ["./data/raw/new.smi",   null],
  "job_name": "run",
  "use_slurm": false
}
```

Here `new-dataset` will be split automatically from `new.smi`, while `pretrained-set`
is expected to already contain `train.smi`, `valid.smi`, and `test.smi`.

To use Mode B for all datasets (all already pre-split), simply set `"smiles_file": null`:

```json
"dataset":     ["dataset_1", "dataset_2"],
"smiles_file": null
```

A single `"data_path"` string is broadcast to all datasets.  If each dataset lives in a
different root directory, provide a matching list:

```json
"data_path": ["./data/internal/", "./data/external/"],
"dataset":   ["internal-set",     "external-set"]
```

---

## Splitting strategies (Mode A only)

Set `"split_type"` in the `job` block to one of the following.

| `split_type` | Description |
|--------------|-------------|
| `"random"` | Shuffle with a fixed seed, then split by count.  Fast and the default. |
| `"butina"` | Cluster molecules with the Butina algorithm (ECFP4 / Tanimoto distance ≤ 0.4).  The largest clusters go to train; remaining small clusters go to valid / test.  Requires RDKit. |
| `"custom"` | Calls `_custom_split()` in `submit.py`.  Replace the placeholder body with your own logic (e.g., scaffold split, temporal split, property-stratified split). |

### Splitting ratios

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_frac` | `0.8` | Fraction of molecules assigned to the training set |
| `valid_frac` | `0.1` | Fraction assigned to the validation set |
| (implicit) test | `0.1` | Remainder: `1 - train_frac - valid_frac` |

### Implementing a custom split

Open `submit.py` and edit `_custom_split()`:

```python
def _custom_split(smiles, train_frac, valid_frac):
    # 1. Compute a property or similarity score per molecule.
    scores = [my_score_fn(smi) for smi in smiles]

    # 2. Sort or partition.
    sorted_smiles = [s for _, s in sorted(zip(scores, smiles))]

    # 3. Slice into splits.
    n = len(sorted_smiles)
    n_train = int(round(n * train_frac))
    n_valid = int(round(n * valid_frac))
    train = sorted_smiles[:n_train]
    valid = sorted_smiles[n_train:n_train + n_valid]
    test  = sorted_smiles[n_train + n_valid:]
    return train, valid, test
```

---

## Choosing preprocessing parameters

These parameters are **baked into the HDF5 files** and must match exactly for every
subsequent job (training, generation, RL).  They are checked automatically at runtime.

### Molecular feature parameters

The following parameters describe the chemical vocabulary of the dataset.

| Parameter | What it controls |
|-----------|-----------------|
| `atom_types` | Allowed element symbols (e.g. `["C", "N", "O", "F"]`) |
| `formal_charge` | Allowed formal charges (e.g. `[-1, 0, 1]`) |
| `imp_H` | Allowed implicit H counts (e.g. `[0, 1, 2, 3]`); omitted when `use_explicit_H` or `ignore_H` |
| `max_n_nodes` | Maximum number of heavy atoms in any generated molecule |
| `chirality` | Fixed as `["None", "R", "S"]` when `use_chirality` is `true`; omitted entirely when `false` |

#### `auto_detect_features` (default: `true`)

When `true`, all of the parameters above are **automatically detected** by scanning
your SMILES files before the HDF5 conversion starts — you do not need to specify them
in `params.json`.

When `false`, the values you provide in `params.json` are used directly.  Any parameter
left as an empty list (`[]`) or `0` is still auto-detected individually, so you can
hard-code some parameters and auto-detect others:

```json
"job": {
  "auto_detect_features": false,
  "atom_types":    ["C", "N", "O", "F", "S", "Cl", "Br"],
  "formal_charge": [],
  "imp_H":         [],
  "max_n_nodes":   0
}
```

Here `atom_types` is fixed to the listed elements (useful for transfer learning, to
ensure a larger vocabulary than the fine-tuning set alone contains), while
`formal_charge`, `imp_H`, and `max_n_nodes` are still auto-detected.

The detected (and/or provided) values are printed at the start of the preprocessing run
and written to `preprocessing_params.json` in the dataset directory so that subsequent
jobs can verify they are using a compatible feature encoding.

#### `extra_dataset`

Set `"extra_dataset"` to the path of an additional `.smi` file or a directory
containing `.smi` files that should be **scanned for vocabulary** but **not
preprocessed**.  This is useful when you want the vocabulary to be large enough to
cover molecules you plan to generate or fine-tune on later, without including those
molecules in the training set.

```json
"extra_dataset": "./data/datasets/future-finetune-set"
```

Set to `null` (the default) to disable.

### Encoding options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_aromatic_bonds` | `true` | Include an aromatic bond type (see below) |
| `use_canon` | `true` | Use RDKit canonical atom ordering (recommended) |
| `use_chirality` | `false` | Encode chirality in node features |
| `use_explicit_H` | `false` | Treat all H atoms explicitly (not recommended) |
| `ignore_H` | `false` | Ignore H atoms entirely |

> `use_explicit_H` and `ignore_H` are mutually exclusive.

**Kekulé vs aromatic bonds (`use_aromatic_bonds`):**
By default (`false`), molecules are Kekulized before graph construction: aromatic
rings are represented as alternating single and double bonds (Kekulé form), giving
a bond vocabulary of three types (SINGLE, DOUBLE, TRIPLE).  This is the
recommended setting — it is more robust because the model cannot generate
an invalid aromatic system.

Setting `use_aromatic_bonds: true` adds a fourth bond type (AROMATIC) and skips
Kekulization.  This can be a more compact representation for aromatic-heavy
datasets, but molecules generated with misplaced aromatic bonds will fail RDKit
sanitization and be discarded as invalid.  **This flag must match between
preprocessing and all subsequent training/generation jobs.**

### Decoding route

| Parameter | Options | Description |
|-----------|---------|-------------|
| `decoding_route` | `"bfs"` / `"dfs"` | Traversal order used to build the subgraph sequence.  BFS is the default and generally recommended. |

### Performance parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `1000` | Subgraphs processed per group during preprocessing |
| `block_size` | `100000` | Subgraphs loaded into RAM per block during training |

---

## Configuration file

> **Tip:** `jobs/preprocess/params.json` is a template — copy it before editing
> so the original stays intact and each experiment has its own config file:
> ```bash
> cp jobs/preprocess/params.json jobs/preprocess/my_dataset.json
> python submit.py --config jobs/preprocess/my_dataset.json
> ```

Edit your copy of `jobs/preprocess/params.json`.

### Mode A example (single file, random split)

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "my-dataset",
    "smiles_file": "./data/datasets/my_molecules.smi",
    "job_name": "run",
    "use_slurm": false,
    "slurm": {
      "account": "XXXXXXXXXX",
      "run_time": "0-02:00:00",
      "gpus_per_node": "T4:1"
    }
  },
  "job": {
    "job_type": "preprocess",
    "auto_detect_features": true,
    "extra_dataset": null,
    "split_type": "random",
    "train_frac": 0.8,
    "valid_frac": 0.1,
    "use_aromatic_bonds": true,
    "use_canon": true,
    "use_chirality": false,
    "use_explicit_H": false,
    "ignore_H": false,
    "batch_size": 1000,
    "block_size": 100000,
    "decoding_route": "bfs"
  }
}
```

To use the Butina split instead, change `"split_type"` to `"butina"` (and adjust
fractions if desired).  Everything else stays the same.

### Mode B example (pre-split directory)

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "gdb13-debug",
    "smiles_file": null,
    "job_name": "run",
    "use_slurm": false,
    "slurm": {
      "account": "XXXXXXXXXX",
      "run_time": "0-02:00:00",
      "gpus_per_node": "T4:1"
    }
  },
  "job": {
    "job_type": "preprocess",
    "auto_detect_features": true,
    "extra_dataset": null,
    "use_aromatic_bonds": true,
    "use_canon": true,
    "use_chirality": false,
    "use_explicit_H": false,
    "ignore_H": false,
    "batch_size": 1000,
    "block_size": 100000,
    "decoding_route": "bfs"
  }
}
```

### Multi-dataset example (shared vocabulary)

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": ["dataset_1", "dataset_2"],
    "smiles_file": null,
    "job_name": "run",
    "use_slurm": false,
    "slurm": {
      "account": "XXXXXXXXXX",
      "run_time": "0-02:00:00",
      "gpus_per_node": "T4:1"
    }
  },
  "job": {
    "job_type": "preprocess",
    "auto_detect_features": true,
    "extra_dataset": null,
    "use_aromatic_bonds": true,
    "use_canon": true,
    "use_chirality": false,
    "use_explicit_H": false,
    "ignore_H": false,
    "batch_size": 1000,
    "block_size": 100000,
    "decoding_route": "bfs"
  }
}
```

Both `dataset_1` and `dataset_2` must be pre-split Mode B directories.  The run produces
`dataset_1/train.h5` (and `valid.h5`, `test.h5`) and likewise for `dataset_2`, all encoded
with the same union vocabulary.

### Fixing the vocabulary for transfer learning

If you plan to fine-tune on a dataset that contains atom types not present in the
pretraining set, set `auto_detect_features: false` and list all atom types explicitly
so that the pretraining HDF5 uses a vocabulary large enough to cover the fine-tuning
molecules:

```json
"job": {
  "job_type": "preprocess",
  "auto_detect_features": false,
  "atom_types": ["C", "N", "O", "F", "S", "Cl", "Br", "I"],
  "formal_charge": [],
  "imp_H": [],
  "max_n_nodes": 0,
  ...
}
```

Empty lists (`[]`) and `0` fall back to auto-detection for those individual fields.
Alternatively, point `extra_dataset` at the fine-tuning SMILES file and leave
`auto_detect_features: true` — the extra molecules will be scanned for vocabulary
but will not be included in the pretraining HDF5.

---

## Running the job

```bash
python submit.py --config jobs/preprocess/params.json
```

`submit.py` will:
1. If `smiles_file` is set: split the file and write `train.smi` / `valid.smi` / `test.smi` into the dataset directory.
2. If `smiles_file` is null: verify that all three `.smi` files exist in the dataset directory (error if any are missing).
3. If `dataset` is a list: scan all datasets (plus `extra_dataset` if set) to compute the union vocabulary, then preprocess each dataset separately using that shared vocabulary.
4. Create `output/<dataset>/preprocess/job_0/`
5. Write a resolved `params.json` into that directory.
6. Launch `graphinvent/main.py --job-dir output/<dataset>/preprocess/job_0/`

`main.py` will then:
7. Scan the `.smi` files to detect `atom_types`, `formal_charge`, `imp_H`, and `max_n_nodes` (skipped when `auto_detect_features` is `false` and values are fully specified).
8. Run the HDF5 conversion using the feature vocabulary.
9. Write `preprocessing_params.json` to the dataset directory.

---

## Output files

All output is written to `output/<dataset>/preprocess/job_0/` and to the **dataset directory** itself.

### In the job directory

| File | Description |
|------|-------------|
| `params_all.json` | Record of all resolved parameters |

### In the dataset directory

| File | Description |
|------|-------------|
| `train.h5` | HDF5 file for the training set |
| `valid.h5` | HDF5 file for the validation set |
| `test.h5`  | HDF5 file for the test set |
| `train.csv` | Training-set property statistics used as the reference distribution during model evaluation |
| `preprocessing_params.json` | Snapshot of the parameters used; loaded by subsequent jobs to verify consistency |

If `smiles_file` was specified, the split `.smi` files are also written here before HDF5 conversion.

---

## Restarting an interrupted preprocessing job

If preprocessing is interrupted, set `"restart": true` in the `job` section and rerun.
The script detects which HDF5 files have been partially created and resumes from the
correct point.

---

## Next step

Once all three `.h5` files exist in the dataset directory, proceed to
[Tutorial 2: Pretraining](./02_pretraining.md).
