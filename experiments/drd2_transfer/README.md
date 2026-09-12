# Experiment 2: DRD2 transfer learning

Fine-tune the ChEMBL-pretrained GGNN on a focused set of dopamine receptor D2 (DRD2) actives, and measure how far the generated distribution moves toward that chemistry without losing the validity the prior provides.

---

## What this tests

Supervised fine-tuning is the cheapest way to specialise a generative model, and it is the baseline any RL result should be compared against. The question is whether a model trained on broad chemical space can be pulled toward a focused active series using a few thousand examples and a low learning rate, and what it costs in validity and diversity to do so. A fine-tuned model that reproduces the actives it was shown has failed; the useful outcome is enrichment in DRD2-relevant chemistry combined with novelty against both the fine-tuning set and the ChEMBL prior.

---

## Step 1 — Obtain DRD2 actives

No DRD2 model or dataset ships with this repository, so the actives have to come from measured data.

**Option A: a public bioactivity set.** ChEMBL target CHEMBL217 or ExCAPE-DB ([solr.ideaconsult.net/search/excape](https://solr.ideaconsult.net/search/excape/), UniProt P14416), filtered to pXC50 ≥ 5. These are experimental measurements, which is what makes any later enrichment claim meaningful. The cost is a manual download and a filtering step that is only as reproducible as you make it, so record the query, the release, and the filter alongside the file.

**Option B: expand a small set with a surrogate you train.** A few thousand molecules is roughly the floor for fine-tuning, and a curated active set is often smaller than that. Train a classifier on what you have and use it to label a larger library, keeping the high-scoring tail:

```bash
python src/graphinvent/tools/train-surrogate.py \
    --input data/assays/drd2.csv --smiles-column smiles --label-column pXC50 \
    --threshold 5.0 --split scaffold --output data/surrogates/drd2_rf.pkl
```

```python
# run from repository root
import sys
sys.path.insert(0, "src")
from pathlib import Path
from oracles import OracleFactory

library = [
    ln.split()[0]
    for ln in Path("data/raw/chembl_v34_filtered.smi").read_text().splitlines()
    if ln.strip()
]

drd2 = OracleFactory.create_cached(
    "DRD2", {"type": "sklearn", "path": "data/surrogates/drd2_rf.pkl"}
)
scores = drd2(library)

actives = [smi for smi, s in zip(library, scores) if s >= 0.5]
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/raw/drd2_actives.smi").write_text("\n".join(actives) + "\n")
print(f"{len(actives)} of {len(library)} molecules scored >= 0.5")
```

Circularity is the problem with this route. Fine-tuning on molecules a surrogate called active and then reporting that same surrogate's opinion of the output measures agreement with the surrogate, not activity. Hold out a scaffold-disjoint slice of the labelled data from surrogate training, or keep a second model trained on different data, and evaluate against that instead. Scoring 1.5M molecules is also not quick, so truncating the library or starting from a smaller screening set is reasonable.

Either way the result is `data/raw/drd2_actives.smi`, one SMILES per line, which is what `preprocess_params.json` expects.

---

## Step 2 — Preprocess with a shared vocabulary

```bash
python submit.py --config experiments/drd2_transfer/preprocess_params.json
```

The config passes both datasets in one call:

```json
"dataset": ["chembl_v34", "drd2_actives"],
"smiles_file": [null, "./data/raw/drd2_actives.smi"]
```

`submit.py` scans every dataset first and computes the union of atom types, formal charges, implicit-hydrogen counts and `max_n_nodes`, then preprocesses each dataset separately against that shared vocabulary. This matters because the node feature vector is a concatenation of one-hot segments whose widths come from those lists: a DRD2 set containing an element ChEMBL lacks would otherwise produce HDF5 files with a different feature dimension, and the pretrained checkpoint would not load.

The `null` entry puts ChEMBL in Mode B (its `train.smi`/`valid.smi`/`test.smi` already exist from Experiment 1) while the DRD2 file is split automatically. Because ChEMBL is re-encoded here, its existing HDF5 files are backed up and rebuilt; that is intended, since the union vocabulary may be wider than the one Experiment 1 detected.

The config also sets `"use_aromatic_bonds": false`, matching Experiment 1. A mismatch here changes the edge feature dimension and the checkpoint will not load.

Output: `data/datasets/drd2_actives/{train,valid,test}.{smi,h5}`, encoded compatibly with ChEMBL. Expect under 30 minutes for a set of a few thousand actives, dominated by the ChEMBL re-encoding rather than by DRD2.

---

## Step 3 — Fine-tune

Point `resume_from` in `transfer_params.json` at the ChEMBL checkpoint chosen in Experiment 1, then:

```bash
python submit.py --config experiments/drd2_transfer/transfer_params.json
```

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `resume_from` | `./output/chembl_v34/unconditional/run/model_restart_<N>.pth` | Loads the pretrained weights; the epoch counter restarts at 1 |
| `init_lr` | `1e-5` | An order of magnitude below pretraining, to move the distribution without overwriting the grammar |
| `max_rel_lr` | `5` | Peak LR of 5e-5 under the one-cycle schedule |
| `epochs` | `100` | An upper bound; a few thousand actives usually converge in 20–50 |
| `accumulation_steps` | `10` | Matches pretraining, keeping the gradient scale comparable |
| `sample_every` | `5` | Frequent enough to catch a validity collapse early |

This config deliberately contains **no architecture keys**. `src/graphinvent/parameters/config.py` copies the GGNN dimensions from the pretrained run's `params_all.json`, but only for keys the job config does not set itself. Adding `hidden_node_features` or the MLP dimensions here would override the inherited values and break `load_state_dict` unless they happened to match exactly.

Watch `output/drd2_actives/unconditional/run/convergence.log`. Validation loss should drop quickly given the warm start; a rise indicates overfitting to a small active set, which is the expected failure mode here and an argument for stopping early rather than for more epochs. Expect 1–4 hours on one GPU.

---

## Step 4 — Generate

Set `pretrained_model_path` in `generate_params.json` to the checkpoint you selected, then:

```bash
python submit.py --config experiments/drd2_transfer/generate_params.json
```

This samples 10,000 molecules into `output/drd2_actives/generate/run/10000_samples.smi`, with `10000_samples.likelihood` and `10000_samples.valid` alongside.

Three details about these files matter when you read them:

- The `.smi` file carries a `SMILES Name` header line **for every generation batch**, not only at the top, because the per-batch files are concatenated verbatim.
- Invalid graphs are written as the placeholder `[Xe]`. RDKit parses `[Xe]` perfectly well as a xenon atom, so any validity computed by feeding the raw file to RDKit will be badly inflated. The `.valid` file is the authoritative validity vector.
- After the header lines are dropped, the `.smi`, `.likelihood` and `.valid` files are aligned line for line.

`generation.log` in a generate job records only the **first** batch, since the analyser writes its row when the batch index is zero. Whole-run numbers have to be recomputed from the concatenated files.

---

## Step 5 — Evaluate

```python
# run from repository root
import sys
sys.path.insert(0, "src")          # the editable install's .pth does not resolve here
from pathlib import Path
from metrics import evaluate_unconditional
from oracles import OracleFactory


def read_generated(job_dir: str, stem: str):
    """Return (all_smiles, valid_flags) for one generation run, correctly aligned."""
    d = Path(job_dir)
    smiles = [
        ln.split()[0]
        for ln in (d / f"{stem}.smi").read_text().splitlines()
        if ln.strip() and ln.split()[0] != "SMILES"
    ]
    flags = [
        float(ln) == 1.0
        for ln in (d / f"{stem}.valid").read_text().splitlines()
        if ln.strip()
    ]
    assert len(smiles) == len(flags), "header stripping left the files misaligned"
    return smiles, flags


def read_reference(path: str):
    return [
        ln.split()[0]
        for ln in Path(path).read_text().splitlines()
        if ln.strip() and ln.split()[0].upper() != "SMILES"
    ]


smiles, flags = read_generated("output/drd2_actives/generate/run", "10000_samples")
validity = sum(flags) / len(flags)
valid_smiles = [s for s, ok in zip(smiles, flags) if ok]
print(f"validity = {validity:.3f}  ({len(valid_smiles)}/{len(smiles)})")

# Novelty is measured against the ChEMBL prior's training set; the function
# canonicalises both sides internally.  reference_mols is only consumed by FCD,
# so the DRD2 test split is the meaningful reference there.
results = evaluate_unconditional(
    valid_smiles,
    read_reference("data/datasets/drd2_actives/test.smi"),
    training_smiles=set(read_reference("data/datasets/chembl_v34/train.smi")),
    include_fcd=False,
)
print({k: results[k] for k in ("uniqueness", "novelty", "diversity", "sa_mean")})
```

`evaluate_unconditional` returns `validity`, `uniqueness`, `novelty`, `vun`, `diversity`, `sa_mean`, `sa_median`, `sa_std` and `fcd`. Its `validity` field reads 1.0 here by construction, because only already-valid molecules were passed in; the number to report is the one computed from the `.valid` file. `diversity` is computed on a subsample of 1,000 unique molecules by default, adjustable through the `subsample` argument, and `fcd` is `None` unless `include_fcd=True` and `fcd_torch` is installed.

For the DRD2 signal:

```python
drd2 = OracleFactory.create_cached(
    "DRD2", {"type": "sklearn", "path": "data/surrogates/drd2_rf.pkl"}
)
scores = drd2(valid_smiles)
print(f"mean DRD2 score        : {sum(scores) / len(scores):.3f}")
print(f"fraction scoring >= 0.5: {sum(s >= 0.5 for s in scores) / len(scores):.3f}")
```

---

## Output files

| File | Contents |
|------|----------|
| `output/drd2_actives/unconditional/run/convergence.log` | `epoch, lr, avg_train_loss, avg_valid_loss, model_score` |
| `output/drd2_actives/unconditional/run/validation.log` | Per-epoch likelihoods and UC-JSD |
| `output/drd2_actives/unconditional/run/generation.log` | Per-evaluation-epoch generation metrics |
| `output/drd2_actives/unconditional/run/model_restart_<N>.pth` | Checkpoints at evaluation epochs |
| `output/drd2_actives/generate/run/10000_samples.{smi,likelihood,valid}` | Generated molecules, log-likelihoods, validity flags |

`aggregate_results.py` does not read any of these, so Experiment 2's numbers have to be collected by hand.

---

## Reading the results

Validity should stay in the region the prior established; a pronounced drop means the learning rate was high enough to disturb the generation grammar, and the fix is a lower `init_lr` or fewer epochs rather than more training.

Novelty against the ChEMBL training set says how much of what the model produces is new, but it is a binary exact-match criterion and a molecule differing from a training compound by one methyl counts as fully novel. The nearest-neighbour similarity statistics in `generation.log` are the continuous version of the same question and are more informative; a model with high novelty and a mean nearest-neighbour similarity near 1.0 is producing analogues, not new chemistry.

The fraction of generated molecules the DRD2 oracle scores above 0.5 is the enrichment signal, read against the same fraction for the untuned ChEMBL prior rather than against an absolute threshold — the baseline rate is what makes the number interpretable, and it should be measured rather than assumed. Under Option B labelling this comparison is partly circular, as noted above, and the scoring model should at least be one the fine-tuning set did not come from.

Internal diversity and the SA distribution are the guardrails. Fine-tuning on a small active set can collapse the model onto one series while every headline metric still looks acceptable, and a falling internal diversity together with a rising mean nearest-neighbour similarity is what that collapse looks like before it becomes obvious in the samples.

---

## Next

- [Experiment 3: Goal-directed optimisation](../goal_directed/README.md), which starts from the same ChEMBL prior and optimises against oracles directly rather than through a labelled set.
