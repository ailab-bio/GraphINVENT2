# Tutorial 5: Sampling (Generation)

A generation job loads a trained model checkpoint and autoregressively samples a large
batch of new molecular graphs.  Each graph is built one action at a time (add atom,
connect atoms, terminate) until the terminate action is sampled or the maximum node count
is reached.

---

## Prerequisites

A trained model checkpoint (`model_restart_<N>.pth`) from one of:

- [Tutorial 2: Pretraining](./02_pretraining.md)
- [Tutorial 3: Transfer Learning](./03_transfer_learning.md)
- [Tutorial 4: Reinforcement Learning](./04_reinforcement_learning.md)

---

## How generation works

1. A batch of empty graphs is initialised.
2. At each generation step the GGNN predicts action probabilities for every graph in the batch.
3. One action is sampled per graph (multinomial sampling over the action probabilities).
4. The action is applied: a node is added, a bond is added, or the graph is terminated.
5. Terminated graphs are moved to the output buffer; generation continues until
   `batch_size` graphs have been collected.
6. If `n_samples > batch_size`, the above loop is repeated in multiple batches.

**Termination** can be:
- **Proper**: the model samples the terminate action explicitly.
- **Forced**: the graph reaches `max_n_nodes` or samples an invalid action.
  Force-terminated graphs can still be chemically valid, but the model did not
  explicitly decide to finish them.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pretrained_model_path` | — | Direct path to the `.pth` checkpoint to load (e.g. `"./output/debug/pretrain/job/model_restart_100.pth"`).  Architecture and dataset are auto-loaded from `params_all.json` in the same directory. |
| `n_samples` | `2000` | Total number of molecules to generate. |
| `batch_size` | `1000` | Number of molecules generated in parallel per batch.  Larger is faster (up to GPU memory limits). |
| `n_workers` | `0` | DataLoader workers (not used during generation). |

All molecular feature parameters (`atom_types`, `max_n_nodes`, architecture, etc.) are
loaded automatically from the `params_all.json` file in the same directory as the `.pth`
checkpoint.  You do not need to specify them manually.

---

## Configuration file

> **Tip:** `jobs/sample/params.json` is a template — copy it before editing
> so the original stays intact and each generation run has its own config file:
> ```bash
> cp jobs/sample/params.json jobs/sample/my_run.json
> python submit.py --config jobs/sample/my_run.json
> ```

Edit your copy of `jobs/sample/params.json`:

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./graphinvent/",
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
    "device": "cuda",
    "batch_size": 1000,
    "n_samples": 10000,
    "n_workers": 0,
    "pretrained_model_path": "./output/pretrain/run/model_restart_100.pth"
  }
}
```

> **`dataset` and `data_path` are optional.** They are inferred automatically from the
> pretrained model's `params_all.json`.  Set them explicitly only if you want to use a
> different dataset directory.

> **Model architecture** is loaded automatically from `params_all.json` in the same
> directory as the `.pth` file.  You do not need to repeat these in your generation config.

---

## Running the job

```bash
python submit.py --config jobs/sample/params.json
```

---

## Output files

Output is written to `output/<dataset>/generate/<job_name>/`.

| File / Directory | Description |
|-----------------|-------------|
| `params_all.json` | All resolved parameters |
| `generation.log` | Summary statistics: fraction valid, fraction valid & properly terminated, fraction properly terminated, avg nodes, property histograms |
| `generation/` | Temporary per-batch SMILES files (cleaned up after concatenation) |
| `<n_samples>_samples.smi` | All generated SMILES, one per line |
| `<n_samples>_samples.likelihood` | Per-molecule total log-likelihood (sum of log action probabilities) |
| `<n_samples>_samples.valid` | Binary validity vector (1 = valid, 0 = invalid), same line order as `.smi` |

The three output files (`*.smi`, `*.likelihood`, `*.valid`) share the same line order —
line *i* in each file refers to the same generated molecule.

---

## Understanding the output metrics

`generation.log` contains one row per generation batch:

```
set, fraction_valid, fraction_valid_pt, fraction_pt, run_time, avg_n_nodes, ...
Epoch GEN100, 0.731, 0.684, 0.935, 18.4, 9.7, ...
```

| Column | Meaning |
|--------|---------|
| `fraction_valid` | Fraction of generated graphs that pass RDKit sanitisation |
| `fraction_pt` | Fraction of graphs that terminated via the explicit terminate action (not force-terminated) |
| `fraction_valid_pt` | Of the properly-terminated graphs, what fraction are chemically valid |
| `avg_n_nodes` | Average number of heavy atoms in generated molecules |

A well-trained model typically produces `fraction_valid > 0.7` and
`fraction_valid_pt > 0.8` after sufficient training.

---

## Post-processing the SMILES output

The `.smi` file contains one molecule per line.  Invalid graphs are written as the
placeholder `[Xe]`, and empty graphs are written as a bare molecule identifier.
Clean the output before further analysis:

```bash
# Remove [Xe] placeholders (invalid molecules)
sed -i '/Xe/d' output/debug/generate/run/10000_samples.smi

# Remove empty-graph entries
sed -i '/^ [0-9]\+$/d' output/debug/generate/run/10000_samples.smi
```

Or, in Python:

```python
from rdkit import Chem

smi_file = "output/debug/generate/run/10000_samples.smi"

valid_smiles = []
with open(smi_file) as f:
    for line in f:
        smi = line.strip().split()[0] if line.strip() else ""
        if not smi or "Xe" in smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(Chem.MolToSmiles(mol))

print(f"{len(valid_smiles)} valid molecules")
```

---

## Visualising generated molecules

```python
import math, random
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolsToGridImage

mols = [MolFromSmiles(s) for s in valid_smiles if MolFromSmiles(s) is not None]
sample = random.sample(mols, min(100, len(mols)))

n_per_row = int(math.sqrt(len(sample)))
img = MolsToGridImage(
    mols=sample,
    molsPerRow=n_per_row,
    legends=[str(i) for i in range(len(sample))],
)
img.save("generated_molecules.png")
```

Or use the built-in visualisation tool:

```bash
python visualize.py output/debug/generate/run/10000_samples.smi
```

---

## Tips

- **Speed**: larger `batch_size` is faster (up to GPU memory limits).  On a modern GPU,
  `batch_size = 1000` generates roughly 1 000–5 000 molecules per second depending on
  molecule size.
- **Diversity**: if the generated set is highly repetitive, the model may have overfit.
  Consider stopping training earlier or adjusting the `sigma` parameter (for RL models).
- **Best checkpoint**: identify the best checkpoint from `score.log` (RL) or
  `validation.log` (pretrain/transfer) and set `pretrained_model_path` accordingly.
