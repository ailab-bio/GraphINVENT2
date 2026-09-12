# Tools

Standalone utilities for dataset preparation, inspection, and analysis.
All scripts are located in `src/graphinvent/tools/` and should be run from the
repository root.

---

## Dataset feature scanning

### `scan_features.py`

Scans one or more SMILES files and reports the molecular feature vocabulary
(`atom_types`, `formal_charge`, `imp_H`, `max_n_nodes`) needed for preprocessing.

> **Note:** During a normal preprocessing run, GraphINVENT2 detects these values
> automatically by scanning your SMILES files — you do not need to run this script
> first.  Use it to inspect or verify vocabulary ahead of time, or to check that
> two datasets share a compatible feature space before transfer learning.

```bash
python src/graphinvent/tools/scan_features.py --smi path/to/train.smi path/to/valid.smi path/to/test.smi
```

Optional flags match the corresponding preprocessing parameters:

| Flag | Effect |
|------|--------|
| `--use_explicit_H` | Add explicit Hs before scanning atoms (use when `use_explicit_H: true`) |
| `--ignore_H` | Omit implicit H count report (use when `ignore_H: true`) |

---

## Dataset creation

### `tpddb-create-dataset.py`

Builds a dataset of targeted protein degraders from
[TPDdb](https://tpddb.idrblab.net), which publishes its release as static
tab-separated files, so the download needs no login or API key.

```bash
python src/graphinvent/tools/tpddb-create-dataset.py \
    --modality both --n-molecules 10 --output data/datasets/tpddb_small/
```

Molecules are kept only if RDKit sanitizes them and they are a single fragment,
since the BFS/DFS decoding route cannot order a disconnected graph. Selection
is deterministic: candidates are sorted by heavy-atom count and then by
canonical SMILES, and the smallest are taken first. Preferring small molecules
matters more here than in most datasets, because a PROTAC routinely has 60 to
120 heavy atoms and GraphINVENT sizes its action-probability tensor and readout
MLPs from `max_n_nodes`, so cost grows steeply with the largest molecule in the
set. Each run writes a `PROVENANCE.json` recording the source URL, the
retrieval time, and a SHA-256 of every raw file, so two runs can be compared
rather than assumed identical.

---

## Surrogate models for goal-directed generation

### `train-surrogate.py`

Trains a random forest over Morgan fingerprints from a table of SMILES and
labels, and pickles it in the form the `sklearn` oracle loads. This is the
intended route to a target-specific objective: the model is trained on data you
supply, so its quality and provenance are yours to report.

```bash
python src/graphinvent/tools/train-surrogate.py \
    --input data/assays/egfr.csv --smiles-column smiles --label-column pIC50 \
    --threshold 6.0 --split scaffold --output data/surrogates/egfr_rf.pkl
```

Passing `--threshold` binarises the label and trains a classifier, whose oracle
`output` is then `"proba"`; omitting it trains a regressor, whose `output` is
`"predict"` and which needs a transform to map its native scale onto a
desirability. The script prints the config block to paste into the `oracles`
section of a job, and the held-out metrics that say whether the model is worth
optimising against at all. A surrogate that cannot predict its own test set
will still drive an RL run perfectly happily, producing molecules that score
well and mean nothing.

A random forest is the default partly because its per-tree spread is a free
uncertainty estimate, which is what the uncertainty modulation described in
`tutorials/06_custom_oracles.md` consumes. The default scaffold split is
pessimistic relative to a random split; that is the point, since a random split
of congeneric series measures memorisation.

---

## Large-dataset preprocessing

### `submit-split-preprocessing-supercloud.py`

For very large datasets that are more efficiently preprocessed in parallel across
multiple compute nodes.

**Step 1 — split the dataset** (run in an interactive session):
```bash
python src/graphinvent/tools/submit-split-preprocessing-supercloud.py --type split
```

**Step 2 — submit the preprocessing jobs**:
```bash
python src/graphinvent/tools/submit-split-preprocessing-supercloud.py --type submit
```

**Step 3 — aggregate the resulting HDF files**:
```bash
python src/graphinvent/tools/submit-split-preprocessing-supercloud.py --type aggregate
```

---

## Combining HDF files

### `combine_HDFs.py`

Combines multiple preprocessed HDF files into one.  Useful when a large dataset
was preprocessed in chunks.  Edit the variables at the bottom of the script to
set the dataset name, feature dimensions, and number of splits, then run:

```bash
python src/graphinvent/tools/combine_HDFs.py
```

---

## Utilities

### `utils.py`

Shared helper used by the deprecated individual scripts (`load_molecules`).

---

## Deprecated individual scripts

`atom_types.py`, `formal_charges.py`, and `max_n_nodes.py` are superseded by
`scan_features.py`, which collects all three in a single pass.  They are kept
for backwards compatibility.
