# Experiment 1: ChEMBL v34 pretraining

Train a GGNN generative model from random initialisation on ChEMBL v34, producing the prior that every downstream experiment starts from.

---

## Why pretrain first

Transfer learning, RL and conditional fine-tuning all modify an existing distribution rather than learning one from scratch. Starting from a prior over drug-like chemistry gives the RL agent a usable action distribution from step one, which is what keeps the augmented log-likelihood objective from collapsing onto degenerate high-scoring structures, and it gives the conditional and transfer models a chemical grammar they no longer have to spend capacity learning. The quality of everything downstream is bounded by this run, so it is worth spending the convergence checks on it.

---

## Step 1 — Download and filter ChEMBL v34

```bash
mkdir -p data/raw
cd data/raw
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34/chembl_34_chemreps.txt.gz
gunzip chembl_34_chemreps.txt.gz
```

The download is a tab-separated file with the columns `chembl_id`, `canonical_smiles`, `standard_inchi`, `standard_inchi_key`.

ChEMBL contains a great deal that a graph generator has no business modelling: salts, peptides, metal complexes, and molecules far outside the size range the model can represent. The filters below cut it to drug-like small molecules. The property cutoffs are conventional Lipinski-adjacent thresholds rather than principled ones, and the heavy-atom ceiling matters most, because `max_n_nodes` is detected from the data and drives the size of every tensor in the model.

```python
# filter_chembl.py — run from repository root
import csv
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

input_file  = Path("data/raw/chembl_34_chemreps.txt")
output_file = Path("data/raw/chembl_v34_filtered.smi")

filters = {
    "max_mw": 700,
    "max_hba": 10,
    "max_hbd": 5,
    "max_rotbonds": 10,
    "min_atoms": 5,
    "max_atoms": 50,
}

written = 0
with open(input_file) as f_in, open(output_file, "w") as f_out:
    reader = csv.DictReader(f_in, delimiter="\t")
    for row in reader:
        smi = row.get("canonical_smiles", "").strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        # Charges beyond ±1 and any metal would enlarge the formal-charge and
        # atom-type vocabularies for a handful of molecules.
        if any(abs(a.GetFormalCharge()) > 1 for a in mol.GetAtoms()):
            continue
        if any(a.GetAtomicNum() not in
               {1,5,6,7,8,9,14,15,16,17,34,35,53} for a in mol.GetAtoms()):
            continue
        mw   = Descriptors.ExactMolWt(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n    = mol.GetNumHeavyAtoms()
        if (mw > filters["max_mw"] or hba > filters["max_hba"] or
                hbd > filters["max_hbd"] or rotb > filters["max_rotbonds"] or
                n < filters["min_atoms"] or n > filters["max_atoms"]):
            continue
        f_out.write(Chem.MolToSmiles(mol) + "\n")
        written += 1

print(f"Wrote {written} molecules to {output_file}")
```

```bash
python filter_chembl.py
```

Roughly 1.5–2 million molecules survive, depending on where the cutoffs land.

---

## Step 2 — Preprocess

```bash
python submit.py --config experiments/chembl_pretrain/preprocess_params.json
```

The config requests `"split_type": "butina"`, which clusters molecules by ECFP4 Tanimoto distance with a cutoff of 0.4 and assigns whole clusters to train, then valid, then test. The intent is a split where the test set is structurally dissimilar to the training set, which is a harder and more informative evaluation than a random split. This is a fingerprint-similarity clustering, not a Murcko scaffold split, so it groups analogues rather than exact shared scaffolds.

**The Butina split does not scale to the full ChEMBL set.** `_butina_split_indices` in `src/graphinvent/DataProcessor.py` materialises the complete lower-triangular distance list before clustering, which is n(n−1)/2 entries — on the order of 10¹² floats for 1.5M molecules, and far beyond available memory. In practice you have one of three options: run the Butina split on a subsample (a few hundred thousand molecules is already slow but feasible), switch `split_type` to `"random"` and accept the easier evaluation, or split externally and supply pre-split `train.smi`/`valid.smi`/`test.smi` in Mode B. Choose deliberately and report which you used, because the choice moves novelty and test-similarity numbers substantially.

Preprocessing writes into `data/datasets/chembl_v34/`:

- `train.smi` / `valid.smi` / `test.smi`
- `train.h5` / `valid.h5` / `test.h5`
- `train.csv`, the training-set property statistics used as the reference distribution during evaluation
- `preprocessing_params.json`, the feature vocabulary that every later job is checked against

The config sets `"use_aromatic_bonds": false`, so molecules are Kekulised and the bond vocabulary is single/double/triple. This removes an entire failure mode, since the model cannot emit an aromatic system that fails to sanitise. This flag is baked into the HDF5 files and must match in every downstream job.

---

## Step 3 — Pretrain

```bash
python submit.py --config experiments/chembl_pretrain/pretrain_params.json
```

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `epochs` | 200 | An upper bound rather than a target; stop when validation loss plateaus |
| `hidden_node_features` | 256 | ChEMBL's chemical diversity needs more capacity than the debug default of 100 |
| `message_passes` | 4 | Four rounds propagate information across most drug-sized graphs |
| `accumulation_steps` | 10 | Effective batch ≈ 10,000 subgraphs per optimiser step |
| `sample_every` | 5 | 1,000 molecules sampled every 5 epochs to track validity and UC-JSD |
| `n_workers` | 4 | HDF5 block loading overlaps with the GPU step |
| `seed` | 42 | Fixes Python, NumPy and PyTorch RNGs; 0 would leave the run non-deterministic |
| `use_tensorboard` | true | The scalar curves are easier to read than the logs |

```bash
tensorboard --logdir output/chembl_v34/unconditional/run/tensorboard/
```

Three things indicate convergence, and they are worth watching together rather than individually. Validation loss (the KL divergence between predicted and target action distributions) should flatten. UC-JSD, the Jensen–Shannon divergence between the model's negative-log-likelihood distributions on training and generated molecules, should fall and stabilise; it measures whether the model assigns generated molecules likelihoods resembling those it assigns training molecules, so it detects a model that has learned the data's typical set rather than only its mode. Validity of sampled molecules should rise and then plateau.

A validity plateau above roughly 85% is a reasonable expectation for a Kekulised drug-like dataset at this scale, but it is an expectation drawn from comparable runs rather than a guarantee, and a lower plateau is worth investigating before assuming the model is broken.

Expect on the order of 48–72 hours on one A100 for 200 epochs over ~1.5M molecules.

Once the run has converged, take the epoch with the lowest `avg_valid_loss` in `convergence.log` — usually well before epoch 200 — and use that checkpoint downstream:

```bash
CKPT="./output/chembl_v34/unconditional/run/model_restart_180.pth"
```

Checkpoints exist only at evaluation epochs, so with `sample_every: 5` the available epochs are multiples of 5.

---

## Output files

Everything lands in `output/chembl_v34/unconditional/run/`.

| File | Contents |
|------|----------|
| `convergence.log` | `epoch, lr, avg_train_loss, avg_valid_loss, model_score` — one row per epoch, where `model_score` is the UC-JSD at evaluation epochs and `NA` otherwise |
| `validation.log` | Mean per-molecule likelihood for the validation, training and generated sets, plus `uc_jsd`, one row per evaluation epoch |
| `generation.log` | Per-evaluation-epoch generation metrics: fraction valid, fraction properly terminated, fraction unique, novelty, SA statistics, internal diversity, test-set similarity, and feature histograms |
| `generation/` | `epoch_<N>_batch_<B>.smi`, `.likelihood` and `.valid` for each evaluation epoch |
| `model_restart_<N>.pth` | Checkpoint at evaluation epoch N, including optimiser and scheduler state |
| `progress.png` | Nine-panel plot regenerated at every evaluation epoch |
| `params_all.json` | Resolved parameters plus git hash, library versions, device and seed |
| `tensorboard/` | Scalar time series |

The three metric logs answer different questions and are easy to confuse. `convergence.log` is about optimisation, `validation.log` is about likelihood calibration, and `generation.log` is about what the model actually produces when sampled.

---

## Reading the results

Validity is a floor, not a result. A model can reach high validity by collapsing onto a few small, trivially valid fragments, which is why `generation.log` also carries `fraction_unique`, `novelty`, `internal_diversity` and the nearest-neighbour similarity statistics against the test set. Read them together: high validity with low uniqueness is mode collapse; high validity and uniqueness with near-1.0 test similarity is memorisation; high novelty with very low test similarity may mean the model is exploring genuinely new space, or that it is producing unrealistic structures, and the SA score distribution is what separates those two readings.

The `avg_n_nodes` and feature histograms in `generation.log` are worth comparing against the training-set row written at the top of the file, since a model matching aggregate validity but generating systematically smaller molecules has not learned the distribution.

---

## Next

- [Experiment 2: DRD2 transfer learning](../drd2_transfer/README.md)
- [Experiment 3: Goal-directed optimisation](../goal_directed/README.md)
- [Experiment 4: Conditional generation](../conditional/README.md)
