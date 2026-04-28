# Experiment 2: DRD2 Transfer Learning

Fine-tune the ChEMBL-pretrained GGNN on a focused dataset of dopamine receptor D2 (DRD2) active compounds to evaluate supervised domain adaptation.

---

## Purpose

Transfer learning (domain adaptation) tests whether a model pretrained on broad chemical space can be adapted to a focused active compound library.  Starting from the ChEMBL prior, the model learns to favor DRD2-relevant scaffolds while retaining chemical validity.

---

## Step 1 — Obtain DRD2 active compounds

**Option A: TDC (recommended — easy, reproducible)**

```python
# run from repository root
from tdc.single_pred import HTS
import pandas as pd
from pathlib import Path

data = HTS(name="drd2")
df = data.get_data()

# Keep only active compounds (label == 1) and extract SMILES
actives = df[df["Y"] == 1][["Drug"]].rename(columns={"Drug": "SMILES"})

Path("data/raw").mkdir(parents=True, exist_ok=True)
actives["SMILES"].to_csv("data/raw/drd2_actives.smi", index=False, header=False)
print(f"Saved {len(actives)} DRD2 active SMILES to data/raw/drd2_actives.smi")
```

Install TDC if needed:
```bash
pip install -e ".[tdc]"
```

**Option B: ExCAPE-DB**

Download the ExCAPE-DB dataset from [https://solr.ideaconsult.net/search/excape/](https://solr.ideaconsult.net/search/excape/) and filter for DRD2 actives (target accession P14416, activity threshold pXC50 ≥ 5).

---

## Step 2 — Preprocess (union vocabulary)

```bash
python submit.py --config experiments/drd2_transfer/preprocess_params.json
```

This step processes both `chembl_v34` (already pre-split in `data/datasets/chembl_v34/`) and `drd2_actives` (will be split from `data/raw/drd2_actives.smi`) together to compute a **union vocabulary**.  Both datasets are then encoded with the same feature dimensions, which is required because the model was pretrained on ChEMBL's vocabulary.

> **Why a union vocabulary?**  The DRD2 dataset may contain atom types or structural features not present in ChEMBL alone.  The union vocabulary ensures the transfer-learned model can represent all features present in either dataset.

Output: `data/datasets/drd2_actives/{train,valid,test}.{smi,h5}` — encoded with the ChEMBL vocabulary.

**Expected runtime:** < 30 minutes.

---

## Step 3 — Fine-tune

Update `resume_from` in `transfer_params.json` to the best ChEMBL checkpoint from Experiment 1, then:

```bash
python submit.py --config experiments/drd2_transfer/transfer_params.json
```

Key configuration choices:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `resume_from` | `output/chembl_v34/.../model_restart_<N>.pth` | Start from pretrained ChEMBL weights |
| `init_lr` | `1e-5` | 10× lower than pretraining to avoid catastrophic forgetting |
| `max_rel_lr` | `5` | Peak LR = 5e-5; one-cycle schedule keeps updates small |
| `epochs` | `100` | DRD2 dataset is small; convergence typically reached in 20–50 epochs |

**Convergence check:**
Monitor `output/drd2_actives/unconditional/run/convergence.log`.  Expect validation NLL to drop quickly given the warm start.  Stop if validation NLL starts increasing (overfitting sign).

**Expected runtime:** 1–4 hours on a single GPU.

---

## Step 4 — Generate molecules

Update `pretrained_model_path` in `generate_params.json` to the best checkpoint from Step 3, then:

```bash
python submit.py --config experiments/drd2_transfer/generate_params.json
```

This generates 10,000 molecules.  Output: `output/drd2_actives/generate/run/<N>_samples.smi`.

---

## Step 5 — Evaluate generated molecules

Compute the standard metrics and DRD2 activity of generated molecules:

```python
# Quick evaluation script — run from repository root
from pathlib import Path
import sys
sys.path.insert(0, "./src")

from metrics import evaluate_unconditional
from oracles import OracleFactory

# Generated SMILES file
smi_path = Path("output/drd2_actives/generate/run/10000_samples.smi")
smiles = [l.split()[0] for l in smi_path.read_text().splitlines() if l.strip()]

# Reference SMILES (ChEMBL training set for novelty)
ref_path = Path("data/datasets/chembl_v34/train.smi")
ref_smiles = set(l.split()[0] for l in ref_path.read_text().splitlines() if l.strip())

# Standard metrics
results = evaluate_unconditional(smiles, reference_smiles=ref_smiles)
print(results)

# DRD2 activity of valid, unique, novel molecules
drd2 = OracleFactory.create_cached("DRD2")
valid_novel = [s for s in results["unique_smiles"] if s not in ref_smiles]
scores = drd2(valid_novel)
print(f"Mean DRD2 activity: {sum(scores)/len(scores):.3f}")
print(f"Fraction active (≥0.5): {sum(s>=0.5 for s in scores)/len(scores):.3f}")
```

---

## Output files

| File | Description |
|------|-------------|
| `output/drd2_actives/unconditional/run/convergence.log` | Fine-tuning NLL per epoch |
| `output/drd2_actives/unconditional/run/model_restart_<N>.pth` | Checkpoints |
| `output/drd2_actives/generate/run/<N>_samples.smi` | Generated SMILES |

---

## How to interpret the outputs

- **Validity:** should remain > 80% (a drop below 80% indicates the pretrained grammar is being disrupted; lower the learning rate).
- **Novelty vs ChEMBL training set:** measures how much new chemical space the fine-tuned model explores beyond the pretraining set.
- **DRD2 activity fraction:** fraction of generated molecules predicted as active (score ≥ 0.5) by the TDC oracle.  Higher is better; a good fine-tuned model should show enrichment over the ~3–5% baseline expected from random drug-like molecules.

---

## Next step

- [Experiment 3: Goal-Directed Optimization](../goal_directed/README.md) — use the ChEMBL prior for RL optimization toward DRD2 and other targets.
