# Tools

Standalone utilities for dataset preparation, inspection, and analysis.
All scripts are located in `graphinvent/tools/` and should be run from the
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
python graphinvent/tools/scan_features.py --smi path/to/train.smi path/to/valid.smi path/to/test.smi
```

Optional flags match the corresponding preprocessing parameters:

| Flag | Effect |
|------|--------|
| `--use_explicit_H` | Add explicit Hs before scanning atoms (use when `use_explicit_H: true`) |
| `--ignore_H` | Omit implicit H count report (use when `ignore_H: true`) |

---

## Dataset creation

### `tdc-create-dataset.py`

Downloads a dataset (ChEMBL, MOSES, or ZINC) from the
[Therapeutics Data Commons](https://tdcommons.ai/) and applies basic filters
(maximum heavy-atom count, formal charge range).

```bash
python graphinvent/tools/tdc-create-dataset.py --dataset MOSES
```

Edit the script to adjust the filters.

---

## Large-dataset preprocessing

### `submit-split-preprocessing-supercloud.py`

For very large datasets that are more efficiently preprocessed in parallel across
multiple compute nodes.

**Step 1 — split the dataset** (run in an interactive session):
```bash
python graphinvent/tools/submit-split-preprocessing-supercloud.py --type split
```

**Step 2 — submit the preprocessing jobs**:
```bash
python graphinvent/tools/submit-split-preprocessing-supercloud.py --type submit
```

**Step 3 — aggregate the resulting HDF files**:
```bash
python graphinvent/tools/submit-split-preprocessing-supercloud.py --type aggregate
```

---

## Combining HDF files

### `combine_HDFs.py`

Combines multiple preprocessed HDF files into one.  Useful when a large dataset
was preprocessed in chunks.  Edit the variables at the bottom of the script to
set the dataset name, feature dimensions, and number of splits, then run:

```bash
python graphinvent/tools/combine_HDFs.py
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
