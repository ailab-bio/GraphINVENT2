# Experiment 1: ChEMBL v34 Pretraining

Train a GGNN generative model from scratch on a large, diverse drug-like chemical space using ChEMBL v34 as the reference dataset.  This pretrained model serves as the starting point for all downstream experiments (transfer learning, goal-directed RL, and conditional generation).

---

## Purpose

A pretrained model learns a broad prior distribution over drug-like chemistry.  All subsequent experiments start from this prior, which prevents mode collapse in RL and provides a warm start for transfer learning and conditional fine-tuning.

---

## Step 1 — Download and filter ChEMBL v34

ChEMBL v34 is available from the EMBL-EBI FTP server.  Download the SMILES export file and apply standard drug-likeness filters to produce a training set of clean, processable SMILES strings.

```bash
mkdir -p data/raw
cd data/raw

# Download ChEMBL v34 SMILES export (~600 MB compressed)
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34/chembl_34_chemreps.txt.gz
gunzip chembl_34_chemreps.txt.gz
```

The file has the format:
```
chembl_id   canonical_smiles   standard_inchi   standard_inchi_key
```

Filter to drug-like small molecules.  A recommended filtering script:

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
        # No charged atoms beyond +1/-1
        if any(abs(a.GetFormalCharge()) > 1 for a in mol.GetAtoms()):
            continue
        # No metals
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
        can = Chem.MolToSmiles(mol)
        f_out.write(can + "\n")
        written += 1

print(f"Wrote {written} molecules to {output_file}")
```

```bash
python filter_chembl.py
```

Expected output: approximately 1.5–2 million molecules depending on exact filter settings.

---

## Step 2 — Preprocess (scaffold split)

```bash
python submit.py --config experiments/chembl_pretrain/preprocess_params.json
```

This runs a **Butina scaffold split** (80/10/10) so that training, validation, and test sets contain molecules from different chemical scaffolds.  The Butina clustering uses ECFP4 fingerprints with a Tanimoto distance threshold of 0.4.

Output files are written to `data/datasets/chembl_v34/`:
- `train.smi` / `valid.smi` / `test.smi`
- `train.h5` / `valid.h5` / `test.h5`
- `preprocessing_params.json` (records the feature vocabulary)

**Expected runtime:** 2–6 hours on a single CPU (Butina clustering is O(n²)).

---

## Step 3 — Pretrain

```bash
python submit.py --config experiments/chembl_pretrain/pretrain_params.json
```

Key configuration choices (see `pretrain_params.json`):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `epochs` | 200 | Sufficient for convergence on ChEMBL; monitor validation NLL |
| `hidden_node_features` | 256 | Larger than debug default to capture ChEMBL's chemical diversity |
| `message_passes` | 4 | Four rounds of message passing for richer graph representations |
| `accumulation_steps` | 10 | Effective batch ≈ 10,000 subgraphs per optimizer step |
| `sample_every` | 5 | Generate 1,000 molecules every 5 epochs to track validity/UC-JSD |
| `use_tensorboard` | true | Monitor NLL, UC-JSD, and validity curves in real time |

**Convergence check:**
```bash
tensorboard --logdir output/chembl_v34/unconditional/run/tensorboard/
```

Watch for:
1. Validation NLL plateauing (< 0.5% improvement over 10 epochs).
2. Validity > 85% on sampled molecules.
3. UC-JSD decreasing and stabilising.

**Expected runtime:** 48–72 hours on a single A100 GPU for 200 epochs over ~1.5M molecules.

After convergence, **record the best epoch** (lowest validation NLL in `convergence.log`) and update the `pretrained_model_path` in all downstream experiment configs:

```bash
# Example: best epoch is 180
CKPT="./output/chembl_v34/unconditional/run/model_restart_180.pth"
```

---

## Output files

| File | Description |
|------|-------------|
| `output/chembl_v34/unconditional/run/convergence.log` | Epoch-by-epoch training and validation NLL |
| `output/chembl_v34/unconditional/run/model_restart_<N>.pth` | Model checkpoint at epoch N |
| `output/chembl_v34/unconditional/run/tensorboard/` | TensorBoard logs |
| `output/chembl_v34/unconditional/run/params_all.json` | Full resolved parameters including git commit, library versions, and device |

---

## How to interpret the outputs

- **Convergence log:** columns are epoch, train NLL, valid NLL, and (every `sample_every` epochs) validity, uniqueness, novelty, and UC-JSD of sampled molecules.
- **UC-JSD:** lower is better; values below 0.3 indicate the model's property distribution closely matches the training set.
- **Validity:** fraction of generated SMILES that parse correctly under RDKit.  Should reach > 85% after ~50 epochs.

---

## Next step

Once pretraining is complete, proceed to:
- [Experiment 2: DRD2 Transfer Learning](../drd2_transfer/README.md)
- [Experiment 3: Goal-Directed Optimization](../goal_directed/README.md)
- [Experiment 4: Conditional Generation](../conditional/README.md)
