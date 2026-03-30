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

You also need the preprocessed dataset in the same molecular feature space (used only for
parameter validation; no actual loading of training data occurs during generation).

---

## How generation works

1. A batch of empty graphs is initialised.
2. At each generation step the GGNN predicts an APD for every graph in the batch.
3. An action is sampled from the APD for each graph (multinomial sampling).
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
| `generation_epoch` | `30` | Epoch/step of the checkpoint to load: reads `model_restart_<generation_epoch>.pth` from the job directory |
| `n_samples` | `2000` | Total number of molecules to generate.  If `n_samples > 100 000`, molecules are generated in batches of 100 000. |
| `batch_size` | `1000` | Number of molecules generated in parallel per batch.  Larger is faster (up to GPU memory limits). |
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `n_workers` | `0` | DataLoader workers (not used during generation, but must be present) |

All feature parameters (`atom_types`, `max_n_nodes`, architecture parameters, etc.) must
match the checkpoint exactly.

---

## Configuration file

Edit `jobs/sample/params.json`.  The key fields are:

```json
{
  "submission": {
    "python_path": "python",
    "graphinvent_path": "./graphinvent/",
    "data_path": "./data/datasets/",
    "dataset": "gdb13-debug",
    "n_jobs": 1,
    "jobdir_start_idx": 0,
    "use_slurm": false,
    "slurm": {
      "account": "XXXXXXXXXX",
      "run_time": "0-01:00:00",
      "gpus_per_node": "T4:1"
    }
  },
  "job": {
    "job_type": "generate",
    "atom_types": ["C", "N", "O", "S", "Cl"],
    "formal_charge": [-1, 0, 1],
    "imp_H": [0, 1, 2, 3],
    "chirality": ["None", "R", "S"],
    "max_n_nodes": 13,
    "use_aromatic_bonds": false,
    "use_canon": true,
    "use_chirality": false,
    "use_explicit_H": false,
    "ignore_H": false,
    "device": "cuda",
    "batch_size": 1000,
    "n_samples": 10000,
    "n_workers": 0,
    "generation_epoch": 100,
    "enn_depth": 4,
    "enn_dropout_p": 0.0,
    "enn_hidden_dim": 250,
    "mlp1_depth": 4,
    "mlp1_dropout_p": 0.0,
    "mlp1_hidden_dim": 500,
    "mlp2_depth": 4,
    "mlp2_dropout_p": 0.0,
    "mlp2_hidden_dim": 500,
    "gather_att_depth": 4,
    "gather_att_dropout_p": 0.0,
    "gather_att_hidden_dim": 250,
    "gather_emb_depth": 4,
    "gather_emb_dropout_p": 0.0,
    "gather_emb_hidden_dim": 250,
    "gather_width": 100,
    "hidden_node_features": 100,
    "message_passes": 3,
    "message_size": 100
  }
}
```

The `"job_type"` must be `"generate"`.  The model checkpoint is loaded from the job
directory itself — which means `submit.py` must point at the **same job directory** that
was used for training.

### Pointing at the right model

`submit.py` creates the job directory as `output/<dataset>/<job_type>/job_<idx>/`.  The
generation job looks for `model_restart_<generation_epoch>.pth` **inside that same
directory**.  Therefore:

- To generate from a pretrain checkpoint, set:
  ```json
  "dataset": "gdb13-debug",
  "data_path": "./data/datasets/"
  ```
  and re-use the same `jobdir_start_idx` that was used for pretraining.  Set
  `"job_type"` to `"generate"` — the script will write into a new
  `output/gdb13-debug/generate/job_0/` directory but will **read** the model from
  whichever directory `params.json` specifies as `job_dir`.

  Because `submit.py` sets `job_dir` automatically based on dataset and job_type, the
  simplest approach is to run a generation job from the same config and just change
  `"job_type"` to `"generate"`, add `"generation_epoch"`, and run.

> **Tip — generating from a specific checkpoint**: `submit.py` places the model
> checkpoint in the *training* job directory (e.g. `output/gdb13-debug/pretrain/job_0/`).
> The generation job directory is separate (`output/gdb13-debug/generate/job_0/`).
> The generation job reads its `params.json` from its own directory, which must contain
> the correct `dataset_dir` and `job_dir`.  The easiest workflow is to run generation
> via `submit.py` using the same dataset/submission block as training and just change
> the `job` section to `job_type: "generate"` with the correct `generation_epoch`.
> `submit.py` will write the resolved `params.json` (including the correct `job_dir`)
> into the new generation job directory automatically.

---

## Running the job

```bash
python submit.py --config jobs/sample/params.json
```

---

## Output files

Output is written to `output/<dataset>/generate/job_0/`.

| File / Directory | Description |
|-----------------|-------------|
| `params_all.json` | All resolved parameters |
| `generation.log` | Summary statistics: fraction valid, fraction valid & properly terminated, fraction properly terminated, avg nodes, property histograms |
| `generation/` | Directory of generated molecule files (one set of files per generation batch) |
| `generation/epoch_GEN<N>_batch_<B>.smi` | SMILES for batch B |
| `generation/epoch_GEN<N>_batch_<B>.likelihood` | Per-molecule total log-likelihood (sum of log probabilities over all actions) |
| `generation/epoch_GEN<N>_batch_<B>.valid` | Binary validity vector (1 = valid, 0 = invalid) |

---

## Understanding the output metrics

`generation.log` contains one row per generation batch:

```
set, fraction_valid, fraction_valid_pt, fraction_pt, run_time, avg_n_nodes, ...
Epoch GEN100 batch_0, 0.731, 0.684, 0.935, 18.4, 9.7, ...
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

The `.smi` files contain one molecule per line.  Invalid graphs are written as the
placeholder `[Xe]`, and empty graphs (force-terminated before adding any atom) are
written as a bare molecule identifier.  Clean the output before further analysis:

```bash
# Remove [Xe] placeholders (invalid molecules)
sed -i '/Xe/d' output/gdb13-debug/generate/job_0/generation/epoch_GEN100_batch_0.smi

# Remove empty-graph entries (lines that are just a number with no SMILES)
sed -i '/^ [0-9]\+$/d' output/gdb13-debug/generate/job_0/generation/epoch_GEN100_batch_0.smi
```

Or, in Python:

```python
from rdkit import Chem

smi_file = "output/gdb13-debug/generate/job_0/generation/epoch_GEN100_batch_0.smi"

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

# load and sample
mols = [MolFromSmiles(s) for s in valid_smiles if MolFromSmiles(s) is not None]
sample = random.sample(mols, min(100, len(mols)))

n_per_row = int(math.sqrt(len(sample)))
img = MolsToGridImage(mols=sample,
                      molsPerRow=n_per_row,
                      legends=[str(i) for i in range(len(sample))])
img.save("generated_molecules.png")
```

---

## Tips

- **Generating large numbers of molecules**: `n_samples > 100 000` is automatically
  handled in batches of 100 000.  For very large runs (millions), launch multiple
  generation jobs with different `jobdir_start_idx` values and combine the output.
- **Speed**: larger `batch_size` is faster (up to GPU memory limits).  On a modern GPU,
  `batch_size = 1000` generates roughly 1 000–5 000 molecules per second depending on
  molecule size.
- **Diversity**: if the generated set is highly repetitive, the model may have overfit.
  Consider stopping training earlier or adjusting the `sigma` parameter (for RL models).
