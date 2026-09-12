# Tutorial 5: Sampling

A `generate` job loads a checkpoint and samples molecular graphs from it. Generation is
autoregressive: starting from an empty graph, the model predicts a distribution over all
possible next actions, one action is drawn from it, the graph is updated, and the process
repeats until the terminate action is drawn or the graph reaches `max_n_nodes`.

Sampling is multinomial rather than greedy, so the same checkpoint produces a different set
every run unless `seed` is fixed. This is deliberate: the model defines a distribution and the
point of sampling is to draw from it, but it does mean that any statistic computed from a
single small sample carries sampling noise that the logs do not report.

---

## Prerequisites

A `model_restart_<N>.pth` checkpoint with its `params_all.json` in the same directory, from
[Tutorial 2](./02_pretraining.md), [Tutorial 3](./03_transfer_learning.md), or
[Tutorial 4](./04_reinforcement_learning.md).

---

## How a graph is built

At each step the model outputs logits over the concatenated action space: add a new atom with
a given element, charge, hydrogen count and bond type attached to a specified existing atom;
connect two existing atoms with a given bond type; or terminate. One action per graph in the
batch is sampled from the softmax of those logits and applied.

Termination happens in two ways, and the distinction matters when reading the output:

- **Proper termination** — the model drew the terminate action, meaning it decided the molecule
  was finished.
- **Forced termination** — the graph hit `max_n_nodes`, or the sampled action was invalid.
  Force-terminated graphs are often still chemically valid, which is why validity and proper
  termination are reported as separate columns rather than combined.

`n_samples` molecules are generated in batches of `min(batch_size, n_samples)`, and the
per-batch files are concatenated at the end.

---

## Parameters

| Parameter | Default in `defaults.py` | Meaning |
|-----------|--------------------------|---------|
| `pretrained_model_path` | `""` | Path to the `.pth` checkpoint; required |
| `sample_mode` | `"generate"` | `"generate"` writes SMILES; `"evaluate"` computes NLL and UC-JSD on the test set instead |
| `n_samples` | `2000` | Total molecules to generate |
| `batch_size` | `1000` | Molecules generated in parallel per batch |
| `n_workers` | `0` | Unused during generation |
| `sample_conditions` | `null` | Required when the checkpoint is a conditional model; see [Tutorial 5: Conditional generation](./05_conditional_generation.md) |

The feature vocabulary, `max_n_nodes`, and the full GGNN architecture are read from
`params_all.json` next to the checkpoint, so none of them belong in the generation config. As
elsewhere, an architecture key written into the job block overrides the inherited value, which
will usually mean `load_state_dict` fails.

---

## Configuration file

```bash
cp jobs/generate/params.json jobs/generate/my_run.json
```

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./src/graphinvent/",
    "job_name": "run",
    "use_slurm": false,
    "slurm": {
      "account": "XXXXXXXXXX",
      "run_time": "0-01:00:00",
      "gpus_per_node": "T4:1"
    }
  },
  "job": {
    "job_type": "generate",
    "sample_mode": "generate",
    "device": "cuda",
    "batch_size": 1000,
    "n_samples": 10000,
    "n_workers": 0,
    "pretrained_model_path": "./output/debug/unconditional/run/model_restart_100.pth",
    "sample_conditions": null
  }
}
```

`dataset` and `data_path` may be omitted: `submit.py` reads `dataset_dir` from the checkpoint's
`params_all.json` and derives both, which also fixes which dataset the generated molecules are
compared against. Set them explicitly only to compare against a different dataset.

`pretrained_model_path` is not optional in practice. Without it the config must supply both
`pretrained_model_dir` and `generation_epoch`, and `submit.py` rejects the job otherwise.

Setting `"sample_mode": "evaluate"` runs the test-set evaluation instead of writing SMILES.
That path loads `train.h5`, `valid.h5`, and `test.h5`, so the dataset must be preprocessed and
present even though no training happens.

---

## Running the job

```bash
python submit.py --config jobs/generate/my_run.json
```

---

## Output

Written to `output/<dataset>/generate/<job_name>/`. Results from a previous run in the same
directory are moved into a timestamped `_previous_run_<stamp>/` first.

| File | Contents |
|------|----------|
| `params_all.json` | Resolved parameters, library versions, git hash, seed |
| `generation.log` | One row of statistics, labelled `Epoch GEN<N>` where N is the epoch number parsed from the checkpoint filename. Computed from the **first batch only**, not the full sample |
| `<n_samples>_samples.smi` | Generated SMILES |
| `<n_samples>_samples.likelihood` | Total log-likelihood per molecule |
| `<n_samples>_samples.valid` | `1.0` or `0.0` per molecule |
| `generation/features.png` | Feature histograms, moved to the job root as `features.png` when the batch files are concatenated |

The three `*_samples.*` files list molecules in the same order, but they do not line up by
line number. The `.smi` file is written by RDKit's `SmilesWriter`, which emits a `SMILES Name`
header, and the per-batch files are concatenated verbatim, so one header line appears at the
start of every batch's block — eleven of them in a 1000-molecule run generated in batches of
100. The `.likelihood` and `.valid` files have no headers. Skip lines equal to `SMILES Name`
when pairing the `.smi` file with the other two rather than assuming a constant offset.

`.valid` holds `1.0` or `0.0` per molecule, written as floats.

---

## The output metrics

`generation.log` has the same columns as during training (see
[Tutorial 2](./02_pretraining.md#generationlog) and [the evaluation reference](./evaluation.md)):

| Column | Meaning |
|--------|---------|
| `fraction_valid` | Graphs passing RDKit sanitisation |
| `fraction_pt` | Graphs that terminated by choosing to |
| `fraction_valid_pt` | Graphs that are both |
| `avg_n_nodes`, `avg_n_edges` | Size of the generated molecules |
| `fraction_unique` | Distinct canonical SMILES among the valid ones |
| `novelty`, similarity and diversity columns | Relation to the training and test sets |

These statistics are computed from the first generation batch only, so with
`n_samples = 10000` and `batch_size = 1000` they describe 1000 molecules rather than 10 000.
Recompute them over the concatenated `.smi` file if you need numbers for the whole sample.

A well-trained model on drug-like data typically reaches `fraction_valid` above 0.7 and
`fraction_valid_pt` above 0.8, but these numbers are dataset-dependent and are floors rather
than targets. High validity and uniqueness are necessary for a generated set to be worth
looking at and are not sufficient for it to be interesting: a model can be perfectly valid,
perfectly unique, and confined to a small region of chemical space. The property histograms in
`generation.log`, the internal-diversity columns, and the nearest-neighbour similarity to the
test set are what show where in chemical space the samples actually fall.

---

## Cleaning the SMILES output

Graphs that fail sanitisation are written as the placeholder `[Xe]` so that line order is
preserved across the three files. This placeholder is itself a parseable SMILES — RDKit reads
it as a xenon atom — so any downstream code that measures validity by counting successful
`MolFromSmiles` calls on the raw file will overcount badly. Use the `.valid` file, or filter
`[Xe]` out first.

Filter into a new file rather than in place: `sed -i` takes a mandatory backup-suffix argument
on BSD/macOS and the GNU and BSD forms are not compatible.

```bash
grep -v -e 'Xe' -e '^SMILES' output/debug/generate/run/10000_samples.smi > molecules_clean.smi
```

Or, keeping the correspondence with the likelihood file:

```python
from rdkit import Chem

smi_file = "output/debug/generate/run/10000_samples.smi"

valid_smiles = []
with open(smi_file) as f:
    for line in f:
        parts = line.split()
        if not parts:
            continue
        smi = parts[0]
        if smi == "SMILES" or "Xe" in smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(Chem.MolToSmiles(mol))

print(f"{len(valid_smiles)} valid molecules")
```

---

## Looking at the molecules

```bash
python visualize.py output/debug/generate/run/10000_samples.smi
python visualize.py output/debug/generate/run/10000_samples.smi --n 50 --ncols 10
python visualize.py output/debug/generate/run/10000_samples.smi --first
python visualize.py output/debug/generate/run/10000_samples.smi --size 300x200 --out grid.png
```

By default it draws 25 molecules picked at random in a 5-column grid and writes
`<filename>_grid.png` beside the input; `--first` takes the first N instead. Random selection
is the sensible default, because the first N molecules of a batch are not a random sample of
the run.

Or directly with RDKit:

```python
import math, random
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolsToGridImage

mols = [MolFromSmiles(s) for s in valid_smiles]
mols = [m for m in mols if m is not None]
sample = random.sample(mols, min(100, len(mols)))

img = MolsToGridImage(sample, molsPerRow=int(math.sqrt(len(sample))))
img.save("generated_molecules.png")
```

---

## Practical notes

- Throughput scales with `batch_size` until device memory runs out; every graph in a batch is
  extended in lockstep, so a batch runs for as many steps as its largest molecule needs.
- A highly repetitive sample points at overfitting for a supervised model, or at too large a
  `sigma` for an RL model. Neither is diagnosed by validity, which stays high in both cases.
- Which checkpoint to sample from is best chosen from `validation.log` for supervised runs, or
  from `score.log` read alongside the diversity columns of `generation.log` for RL runs. The
  final epoch is not automatically the best one.
