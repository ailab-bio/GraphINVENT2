# Preprocessing

*Last updated: October 18th, 2026.*

Preprocessing turns a SMILES file into the HDF5 files training reads, and it decides the data splits every number
this project reports depends on. It is the one job type whose output lives in the dataset directory rather than in
the job directory, because a dataset is shared by every job that uses it.

## Run it

```bash
python submit.py experiments/tutorial_preprocess/jobs/debug
```

About five seconds on 30 molecules. The configuration is two lines:

```yaml
job_type: preprocess
smiles_file: data/datasets/debug/debug.smi
```

Everything else is inherited. `experiments/tutorial_preprocess/experiment.yaml` sets `dataset_dir: debug`, and
`config/defaults.yaml` supplies the rest.

## What it prints

```
* cuda not available; using device 'mps'.
  atom_types    : ['C', 'N', 'O']
  formal_charge : [0]
  imp_H         : [0, 1, 2, 3]
  max_n_nodes   : 13
* Run mode: 'preprocess'
* Reading SMILES from data/datasets/debug/debug.smi …
  30 molecules read.
* Splitting (scaffold): 80% train / 10% valid / 10% test
```

The first line is the interactive device fallback. The default device is `cuda`, this laptop has none, so the job
drops to MPS and says so. Inside a SLURM allocation that fallback is fatal instead, because a job that asked for a
GPU and silently got a CPU spends its whole walltime on the wrong device and still writes plausible metrics.

The four lines after it are the feature vocabulary, scanned from the SMILES unless you set it yourself. They are
baked into the HDF5 and into every checkpoint trained on it, which is why a checkpoint cannot be loaded against a
dataset with a different vocabulary.

`max_n_nodes` is the 99th-percentile heavy-atom count, not the maximum. Every stored tensor is padded to it and
the edges array grows with its square, so taking the maximum would cost a great deal of disk for the few largest
molecules in the set.

## What it writes

Into `data/datasets/debug/`:

```
train.h5  valid.h5  test.h5        the preprocessed graphs
train.smi valid.smi test.smi       the split molecule lists
preprocessing.yaml                 how the splits were made
```

Into the job directory, `results/config.resolved.yaml` and two figures: `results/preprocessing.png` and the
matching `.pdf`. Panel (a) is the heavy-atom distribution with `max_nodes` marked, which is the figure that
justifies the value, since a tail past the line is molecules the representation cannot hold. Panel (b) is the
molecule count per split, with the strategy and seed in the axis label. A preprocess job computes no metrics,
so it writes no `metrics.json`.

Each HDF5 holds one row per construction action rather than one per molecule: `nodes`, `edges` and
`action_probs`, plus `condition_vector` when the input carried property columns. A molecule with *n* heavy atoms
decomposes into one subgraph per action, so the training set is far larger than the molecule count and both its
size and the epoch time grow superlinearly with molecule size. Measure on a subset before committing to a full
run on anything large.

## The splits, which are the consequential part

`preprocessing.yaml` records them:

```yaml
split:
  strategy: scaffold
  seed: 1
  train_frac: 0.8
  valid_frac: 0.1
  n_train: 24
  n_valid: 3
  n_test: 3
  n_dropped_oversized: 0
```

Three splits with three distinct jobs. **train** fits the parameters and is the reference set for novelty.
**validation** chooses the best checkpoint and nothing else. **test** stays closed until final evaluation, where
it is the reference set for distribution-matching metrics and for NLL. A model selected on the test split has no
held-out set left, so the number it reports is not an estimate of anything.

Splitting happens at the molecule level, before decomposition. If it happened after, subgraphs of the same
molecule would land in different splits and every reported number would be leakage.

Scaffold splitting is the default because a random split of a molecular dataset puts close analogues on both
sides and overstates generalization. Use `random` only to compare against published numbers that used it, and say
so when you do.

`n_dropped_oversized` counts molecules above `max_n_nodes`, removed before splitting. The dataset is not the input
file without that number, which is why it is recorded rather than printed and forgotten.

## Splits are fixed per dataset

Every job in an experiment, and every experiment on the same dataset, reads the same split files. Do not
re-split between jobs: two models evaluated on different test sets are not comparable. If a dataset needs a new
split it is a new dataset directory.

Re-running preprocessing on a dataset that already has splits moves the old files aside rather than overwriting
them:

```
* Found 6 leftover file(s) from a previous preprocessing run.
  Backup location: data/datasets/debug/_previous_run_20261018_204135/
```

With the same `seed` and fractions the new split reproduces the old one exactly, so a re-run is safe. Changing
either produces a different dataset wearing the same name, which is the failure the backup exists to make
visible.

## Conditioning on properties

A tab-separated input with a header and numeric property columns:

```
SMILES	logp	mol_weight
O=C(CCCN1C(=O)CCC1=O)NCC1(c2ccccc2)CCC1	2.1537	328.412
```

The columns are not picked up automatically. Ask for them:

```yaml
condition_dim: 2
conditioning:
  properties: ["logp", "mol_weight"]
  source: smiles_file
```

Then the HDF5 gains a `condition_vector` of shape `n_subgraphs × condition_dim`, float32, and a `conditional`
job can read it. Without those keys the property columns are dropped and you get an unconditional dataset: the
file parses either way, so the only signal is `condition_dim: 0` in `preprocessing.yaml`. Check it.

A categorical label is expressed as one numeric column per class holding 0 or 1. There is no separate categorical
code path, and none is needed: a one-hot label is numeric columns, and the float32 `condition_vector` carries it.

[Conditional generation](./conditional.md) continues from here.

## When it goes wrong

A molecule RDKit cannot parse is skipped and counted. A disconnected molecule raises, because the graph
representation cannot encode one and silently keeping the largest fragment would change the chemistry.

If `n_train` is far below your molecule count, look at `n_dropped_oversized` first: a `max_n_nodes` set by hand
and set too low discards the tail of the distribution without comment.
