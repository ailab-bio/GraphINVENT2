# Tutorial 1: Preprocessing

Before training any model, your raw SMILES data must be converted into HDF5 format.
This step encodes each molecule as a sequence of subgraphs (its **decoding route**) and
stores the node features, edge features, and target Action Probability Distributions (APDs)
in a compact binary format that the data loader can stream efficiently during training.

---

## Dataset input modes

There are two ways to provide data.  Choose the one that fits your workflow.

### Mode A — single SMILES file (automatic splitting)

Set `"smiles_file"` in the `submission` block to the path of a `.smi` file containing
all your molecules (one SMILES per line, optional space-separated identifier ignored).

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

### Molecular feature parameters — auto-detected

The following parameters are **automatically detected** by scanning your SMILES files
before the HDF5 conversion starts.  You do not need to specify them in `params.json`:

| Parameter | What is detected |
|-----------|-----------------|
| `atom_types` | All unique element symbols present in the dataset |
| `formal_charge` | All unique formal charges present |
| `imp_H` | All unique implicit H counts present (omitted when `use_explicit_H` or `ignore_H`) |
| `max_n_nodes` | Maximum number of heavy atoms in any molecule |
| `chirality` | Fixed as `["None", "R", "S"]` when `use_chirality` is `true`; omitted entirely when `false` |

The detected values are printed at the start of the preprocessing run and written to
`preprocessing_params.json` in the dataset directory so that subsequent jobs can verify
they are using a compatible feature encoding.

### Encoding options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_aromatic_bonds` | `false` | Include an aromatic bond type |
| `use_canon` | `true` | Use RDKit canonical atom ordering (recommended) |
| `use_chirality` | `false` | Encode chirality in node features |
| `use_explicit_H` | `false` | Treat all H atoms explicitly (not recommended) |
| `ignore_H` | `false` | Ignore H atoms entirely |

> `use_explicit_H` and `ignore_H` are mutually exclusive.

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

Edit `jobs/preprocess/params.json`.

### Mode A example (single file, random split)

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "my-dataset",
    "smiles_file": "./data/datasets/my_molecules.smi",
    "n_jobs": 1,
    "jobdir_start_idx": 0,
    "use_slurm": false,
    "slurm": {
      "account": "XXXXXXXXXX",
      "run_time": "0-02:00:00",
      "gpus_per_node": "T4:1"
    }
  },
  "job": {
    "job_type": "preprocess",
    "split_type": "random",
    "train_frac": 0.8,
    "valid_frac": 0.1,
    "use_aromatic_bonds": false,
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
    "n_jobs": 1,
    "jobdir_start_idx": 0,
    "use_slurm": false,
    "slurm": {
      "account": "XXXXXXXXXX",
      "run_time": "0-02:00:00",
      "gpus_per_node": "T4:1"
    }
  },
  "job": {
    "job_type": "preprocess",
    "use_aromatic_bonds": false,
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

---

## Running the job

```bash
python submit.py --config jobs/preprocess/params.json
```

`submit.py` will:
1. If `smiles_file` is set: split the file and write `train.smi` / `valid.smi` / `test.smi` into the dataset directory.
2. If `smiles_file` is null: verify that all three `.smi` files exist in the dataset directory (error if any are missing).
3. Create `output/<dataset>/preprocess/job_0/`
4. Write a resolved `params.json` into that directory.
5. Launch `graphinvent/main.py --job-dir output/<dataset>/preprocess/job_0/`

`main.py` will then:
6. Scan all three `.smi` files to auto-detect `atom_types`, `formal_charge`, `imp_H`, and `max_n_nodes`.
7. Run the HDF5 conversion using the detected feature vocabulary.
8. Write `preprocessing_params.json` to the dataset directory.

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
