# Tutorial 1: Preprocessing

Training data for GraphINVENT2 is not a list of molecules but a list of *decisions*. A
preprocessing job takes each SMILES string, converts it to a molecular graph, traverses that
graph in a fixed order, and records the sequence of partial subgraphs the traversal passes
through together with the action that extends each one. The model is then trained to predict
the action given the subgraph, which is why the stored target is a probability distribution
over all possible next actions (add an atom, connect two existing atoms, or terminate) rather
than a molecule.

Because the traversal order is fixed by `decoding_route`, one molecule yields one decoding
route and therefore a deterministic set of training examples. The whole set is written to
HDF5 so that the data loader can stream it in blocks instead of holding it in memory, which
matters as soon as the dataset is larger than a few tens of thousands of molecules.

---

## SMILES file format

Every `.smi` file, whether it is a single input file or one of the pre-split
train/valid/test files, is read the same way:

- One molecule per line, `<SMILES> [optional_identifier]`, whitespace-separated.
- The identifier is ignored during preprocessing.
- A first line containing the word `SMILES` is treated as a header and skipped.
- Lines RDKit cannot parse are skipped without failing the job.

```
CCO ethanol
c1ccccc1 benzene
CC(=O)O acetic_acid
```

A file of bare SMILES with no identifiers and no header is equally valid.

For conditional training the input is instead tab-separated with a header whose first column
is `SMILES`; see [Tutorial 5: Conditional generation](./05_conditional_generation.md).

---

## Two ways to supply data

### Mode A — one SMILES file, split automatically

Set `"smiles_file"` in the `submission` block to the path of a file containing all your
molecules. The preprocessing job reads that file, removes duplicates by canonical SMILES,
splits the remainder into train/valid/test, writes the three `.smi` files into the dataset
directory, and then converts them to HDF5.

Deduplication happens before the split and is not optional: a molecule appearing twice in the
input would otherwise land in two different splits, leaking test molecules into training and
inflating every novelty and similarity number computed later.

Relative paths are resolved against the directory `submit.py` is run from, which is normally
the repository root. An absolute path is unambiguous and safer if you run jobs from elsewhere.

```json
"submission": {
  "data_path": "./data/datasets/",
  "dataset":   "my-dataset",
  "smiles_file": "./data/raw/my_molecules.smi"
}
```

### Mode B — a directory that is already split

Leave `"smiles_file"` absent or `null`. The dataset directory must then contain `train.smi`,
`valid.smi`, and `test.smi`; `submit.py` refuses the job before launching anything if any of
the three is missing.

```json
"submission": {
  "data_path": "./data/datasets/",
  "dataset":   "debug",
  "smiles_file": null
}
```

The repository ships four dataset directories under `data/datasets/`: `debug` and `test` are
small sets for smoke-testing the pipeline, `DRD2_actives` is a focused set usable as a
transfer-learning target, and `unit_testing` holds a fixture file for the test suite rather
than a trainable dataset.

---

## Preprocessing several datasets at once

Both `"dataset"` and `"data_path"` accept a list as well as a single string. Given a list,
all datasets are scanned together to compute one union feature vocabulary — the atom types,
formal charges, implicit hydrogen counts, and maximum node count found across all of them —
and each dataset is then encoded separately against that shared vocabulary.

This matters because the node feature vector length is determined by the vocabulary. A model
pretrained on a dataset encoded with one vocabulary cannot load into a job whose dataset was
encoded with another, so if you intend to pretrain on one corpus and fine-tune on a second,
preprocessing them together is the reliable way to guarantee compatible tensors.

`"smiles_file"` may also be a list of the same length: a path selects Mode A for that dataset,
`null` selects Mode B. The two modes can be mixed freely within one run.

```json
"submission": {
  "python_path": "python",
  "graphinvent_path": "./src/graphinvent/",
  "data_path": "./data/datasets/",
  "dataset":     ["new-dataset",          "pretrained-set"],
  "smiles_file": ["./data/raw/new.smi",   null],
  "job_name": "run",
  "use_slurm": false
}
```

A single `"data_path"` string is broadcast over all datasets. If the datasets live under
different roots, give a list of the same length:

```json
"data_path": ["./data/internal/", "./data/external/"],
"dataset":   ["internal-set",     "external-set"]
```

One limitation: the union-vocabulary scan reads only the datasets named in `"dataset"`. The
`extra_dataset` parameter described below is honoured only in single-dataset runs, so it
cannot be used to widen a multi-dataset vocabulary.

---

## Splitting strategies (Mode A only)

`"split_type"` in the `job` block selects one of three strategies.

| `split_type` | Behaviour |
|--------------|-----------|
| `"random"` | Shuffle, then slice by count. The default. |
| `"butina"` | Cluster with the Butina algorithm on ECFP4 fingerprints at Tanimoto distance 0.4, then fill train with whole clusters until `train_frac` is reached, valid likewise, test with the remainder. |
| `"custom"` | Calls `_custom_split()` in `src/graphinvent/DataProcessor.py`, which raises `NotImplementedError` until you replace its body. |

A random split will place close analogues of test molecules into the training set, so held-out
performance measured against it says more about interpolation than about generalisation. The
Butina split assigns whole clusters to a single split, which makes the test set structurally
dissimilar to the training set and gives a harder, more honest estimate. It is also
O(n²) in the number of molecules, since it computes a full pairwise distance matrix, so it
becomes impractical somewhere in the low hundreds of thousands of molecules.

Note that clusters are assigned in the order `Butina.ClusterData` returns them, largest first,
which means the training split receives the densest regions of chemical space and the test
split the sparsest. That is a defensible choice for measuring extrapolation but it is a
heuristic, not a principled scaffold split.

### Splitting ratios

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `train_frac` | `0.8` | Fraction assigned to training |
| `valid_frac` | `0.1` | Fraction assigned to validation |
| (implicit) test | `0.1` | `1 - train_frac - valid_frac` |

`submit.py` rejects the job if `train_frac + valid_frac` exceeds 1.0.

### Implementing a custom split

Edit `_custom_split()` in `src/graphinvent/DataProcessor.py`. It receives the deduplicated
SMILES list and the two fractions, and must return three lists of SMILES:

```python
def _custom_split(smiles, train_frac, valid_frac):
    scores = [my_score_fn(smi) for smi in smiles]
    sorted_smiles = [s for _, s in sorted(zip(scores, smiles))]

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

These parameters are baked into the HDF5 files. Every later job reads
`preprocessing_params.json` from the dataset directory and checks its own settings against it,
so a mismatch is caught rather than silently producing a model with the wrong tensor shapes.

### Molecular feature vocabulary

| Parameter | What it fixes |
|-----------|--------------|
| `atom_types` | Element symbols the node features can encode |
| `formal_charge` | Formal charges the node features can encode |
| `imp_H` | Implicit hydrogen counts; omitted when `use_explicit_H` or `ignore_H` is set |
| `max_n_nodes` | Largest graph the model can represent, and therefore the shape of every node and edge tensor |
| `chirality` | Fixed to `["None", "R", "S"]` when `use_chirality` is true; unused otherwise |

`max_n_nodes` deserves attention because it is not only a data property. It sets the width of
the action probability distribution and so the size of the readout layer, and during
generation it is the point at which an unfinished graph is force-terminated. A value chosen
from the training set alone will truncate anything larger the model tries to build.

#### `auto_detect_features` (default `true`)

With auto-detection on, the SMILES files are scanned before conversion and the five parameters
above are set from what is actually present, so you do not specify them at all.

With it off, the values in `params.json` are used as given, except that an empty list or `0`
still falls back to auto-detection for that individual field. This lets you pin some fields
and detect the rest:

```json
"job": {
  "auto_detect_features": false,
  "atom_types":    ["C", "N", "O", "F", "S", "Cl", "Br"],
  "formal_charge": [],
  "imp_H":         [],
  "max_n_nodes":   0
}
```

The reason to pin `atom_types` is transfer learning: if the fine-tuning set contains bromine
and the pretraining set does not, a vocabulary detected from the pretraining set alone gives a
model that cannot represent the fine-tuning data at all.

The resolved values are printed at the start of the run and written to
`preprocessing_params.json` in the dataset directory.

#### `extra_dataset`

Points at an additional `.smi` file, or a directory containing `train.smi`/`valid.smi`/`test.smi`,
that should be scanned for vocabulary but not preprocessed. It is the lighter alternative to
pinning `atom_types` by hand when you already have the future fine-tuning set on disk.

```json
"extra_dataset": "./data/datasets/future-finetune-set"
```

It is read only when `auto_detect_features` is `true` and only in single-dataset runs. Set it
to `null` (the default) to disable.

### Encoding options

| Parameter | Default | Effect |
|-----------|---------|--------|
| `use_aromatic_bonds` | `true` | Adds AROMATIC as a fourth bond type (see below) |
| `use_canon` | `true` | Use RDKit canonical atom ordering |
| `use_chirality` | `false` | Encode chirality in node features |
| `use_explicit_H` | `false` | Treat hydrogens as explicit graph nodes |
| `ignore_H` | `false` | Drop hydrogens from the representation entirely |

`use_explicit_H` and `ignore_H` are mutually exclusive; setting both raises an error.

#### Aromatic bonds versus Kekulé structures

The default, `use_aromatic_bonds: true`, keeps RDKit's aromatic perception and gives a
four-type bond vocabulary (single, double, triple, aromatic). Aromatic rings are then a single
edge label rather than an alternating pattern the model has to reproduce.

Setting it to `false` calls `Chem.Kekulize(mol, clearAromaticFlags=True)` before graph
construction, so rings become alternating single and double bonds and the vocabulary has three
types. Each choice moves the failure mode rather than removing it: with aromatic bonds the
model can emit an aromatic ring that fails RDKit's sanitisation and is discarded as invalid,
while with Kekulé structures it must instead learn the alternating pattern, and a ring with
the wrong parity is equally invalid. Which one produces higher validity is an empirical
question for a given dataset, and both are in use — the shipped `jobs/preprocess/params.json`
uses aromatic bonds, the ChEMBL experiment config under `experiments/` uses Kekulé.

The flag must match between preprocessing and every job that reads the resulting HDF5, since
it changes the edge feature dimension.

### Decoding route

| Parameter | Options | Effect |
|-----------|---------|--------|
| `decoding_route` | `"bfs"` / `"dfs"` | Traversal order used to enumerate subgraphs. BFS is the default. |

### Throughput parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `batch_size` | `1000` | Molecules converted per group during preprocessing |
| `block_size` | `100000` | Subgraphs held in RAM per block when the data is later read for training |

---

## Configuration file

`jobs/preprocess/params.json` is a template. Copy it rather than editing it, so that the
original stays intact and every experiment has a config file you can point at afterwards:

```bash
cp jobs/preprocess/params.json jobs/preprocess/my_dataset.json
python submit.py --config jobs/preprocess/my_dataset.json
```

### Mode A (single file, random split)

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./src/graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "my-dataset",
    "smiles_file": "./data/raw/my_molecules.smi",
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
    "restart": false,
    "conditioning": null,
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

Switching to the Butina split is a one-word change to `"split_type"`.

### Mode B (pre-split directory)

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./src/graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "debug",
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
    "restart": false,
    "conditioning": null,
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

`split_type`, `train_frac`, and `valid_frac` are ignored in Mode B.

### Multiple datasets, shared vocabulary

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./src/graphinvent/",
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
    "restart": false,
    "conditioning": null,
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

Both datasets must be pre-split Mode B directories here. The run produces
`dataset_1/{train,valid,test}.h5` and the same trio for `dataset_2`, all encoded against the
union vocabulary. Note that `submit.py` sets `auto_detect_features` to `false` internally for
these per-dataset jobs and injects the union values, which is what keeps the two encodings
identical.

---

## Running the job

```bash
python submit.py --config jobs/preprocess/params.json
```

`submit.py` validates the config, refuses it with a list of specific problems if anything is
wrong, creates `output/<dataset>/preprocess/<job_name>/`, writes the resolved parameters there
as `params.json`, and launches `src/graphinvent/main.py --job-dir <that directory>/`. The
`job_name` comes from the `submission` block and defaults to `job`; the shipped templates set
it to `run`.

`main.py` then resolves the feature vocabulary (scanning the SMILES files when
`auto_detect_features` is on), splits the input file if you are in Mode A, converts each split
to HDF5, and writes `preprocessing_params.json`.

---

## Output

### In the job directory, `output/<dataset>/preprocess/<job_name>/`

| File | Contents |
|------|----------|
| `params.json` | The job block as submitted |
| `params_all.json` | Every resolved parameter, plus library versions, device, git hash, and seed |

### In the dataset directory, `<data_path>/<dataset>/`

| File | Contents |
|------|----------|
| `train.h5`, `valid.h5`, `test.h5` | Node features, edge features, and target action probabilities per subgraph |
| `train.csv` | Property distributions of the training set, used as the reference when generated molecules are evaluated |
| `preprocessing_params.json` | The vocabulary and encoding flags, plus split sizes; read back by every later job to verify compatibility |

In Mode A the split `.smi` files are written here too. Files left over from an earlier run are
moved into a timestamped `_previous_run_<stamp>/` directory rather than overwritten.

---

## Restarting an interrupted job

Set `"restart": true` and rerun. The job first compares the current parameters against the
saved `preprocessing_params.json`; if they differ it says so and starts fresh instead of
producing a dataset encoded two different ways. Otherwise it inspects which HDF5 files are
complete and resumes at the first incomplete one.

---

## Next step

Once `train.h5`, `valid.h5`, and `test.h5` exist in the dataset directory, continue to
[Tutorial 2: Pretraining](./02_pretraining.md).
