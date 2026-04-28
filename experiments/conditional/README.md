# Experiment 4: Conditional Generation

Train a property-conditioned GGNN and evaluate its ability to generate molecules with user-specified property profiles.

---

## Purpose

Conditional generation tests whether the model can steer sampling toward molecules with target physicochemical or biological properties.  The virtual seed node mechanism injects the condition vector at every message-passing round without changing the model's output dimensionality.

This experiment conditions on four properties simultaneously: QED, SA (normalised), LogP (normalised), and GSK3B activity.

---

## Architecture overview

```
condition_vector (batch × 4)
        │
        ▼
ConditionEncoder (2-layer MLP → 256-dim)
        │
        ▼ initialises virtual seed node (index 0)
┌──────────────────────────────────────────┐
│  GGNN (4 message passing rounds)         │
│  seed → all real atoms (virtual edge)   │
│  real atoms ↔ real atoms (bond edges)   │
└──────────────────────────────────────────┘
        │ strip seed node
        ▼
readout → action probability distribution
```

---

## Property definitions and normalisation

| Property | Raw range | Normalised range | Formula |
|----------|-----------|------------------|---------|
| `QED` | [0, 1] | [0, 1] | Raw RDKit QED (no change) |
| `SA_norm` | [1, 10] raw SA | [0, 1] | `(10 - SA) / 9` |
| `LogP_norm` | [−∞, +∞] | [0, 1] clipped | `(clip(LogP, −3, 7) + 3) / 10` |
| `GSK3B` | [0, 1] | [0, 1] | TDC oracle score (no change) |

> **Important:** always use the same normalisation during preprocessing, training, and generation.  The `compute_properties.py` script applies this normalisation automatically.

---

## Step 1 — Create the labelled TSV

```bash
python experiments/conditional/compute_properties.py \
    --smiles data/raw/chembl_v34_filtered.smi \
    --out data/raw/chembl_v34_cond.tsv \
    --gsk3b
```

This creates a tab-separated file with columns: `SMILES`, `QED`, `SA_norm`, `LogP_norm`, `GSK3B`.

The `--gsk3b` flag calls the TDC GSK3B oracle (requires `pip install -e "[tdc]"`).  Expect ~1 hour for 1.5M molecules.

To skip GSK3B (faster):

```bash
python experiments/conditional/compute_properties.py \
    --smiles data/raw/chembl_v34_filtered.smi \
    --out data/raw/chembl_v34_cond.tsv
# Then update preprocess_params.json: condition_dim=3, properties=["QED","SA_norm","LogP_norm"]
```

---

## Step 2 — Preprocess

```bash
python submit.py --config experiments/conditional/preprocess_params.json
```

The preprocessing step reads the TSV file, splits it into train/valid/test, and stores the property values as `condition_vector` tensors in the HDF5 files alongside the molecular subgraphs.

Output: `data/datasets/chembl_v34_cond/{train,valid,test}.{smi,h5}`

**Expected runtime:** 4–8 hours (similar to Experiment 1 preprocessing).

---

## Step 3 — Train the conditional model

Update `resume_from` in `train_params.json` to the best ChEMBL pretrained checkpoint from Experiment 1:

```bash
# Edit train_params.json: "resume_from": "./output/chembl_v34/unconditional/run/model_restart_<N>.pth"
python submit.py --config experiments/conditional/train_params.json
```

> **Training tip:** starting from a pretrained unconditional model dramatically speeds up convergence.  The ConditionEncoder and virtual-edge MLP start from random weights; the rest of the GGNN benefits from the chemical grammar already learned during pretraining.

**Expected runtime:** 12–24 hours on a single A100 GPU.

---

## Step 4 — Evaluate conditional generation

```bash
python experiments/conditional/evaluate_conditional.py \
    --checkpoint output/chembl_v34_cond/conditional/run/model_restart_100.pth
```

This script generates 1,000 molecules for each of the condition ranges defined in `PROPERTY_RANGES` (12 ranges across 4 properties) and reports:

| Metric | Definition |
|--------|-----------|
| **Validity** | Fraction of SMILES that pass RDKit sanitisation |
| **Conditional accuracy** | Fraction of valid molecules whose computed property falls within the target range |
| **Internal diversity** | Mean pairwise Tanimoto distance (ECFP4) within the generated set |
| **Mean ± std** | Mean and standard deviation of the actual computed property |

Output: `experiments/conditional/results/conditional_results.csv`

---

## Step 5 — Generate molecules with a specific condition (manual)

```bash
# Edit generate_params.json: set sample_conditions and pretrained_model_path
python submit.py --config experiments/conditional/generate_params.json
```

Example: generate molecules targeting high QED (0.85), easy SA (0.8 normalised ≈ raw SA 2.8), neutral LogP (0.5 normalised ≈ LogP 2.0), and no GSK3B requirement (0.0):

```json
"sample_conditions": {
  "QED":      0.85,
  "SA_norm":  0.80,
  "LogP_norm":0.50,
  "GSK3B":    0.0
}
```

---

## Property ranges evaluated

### QED
| Range | Normalised target | Raw QED |
|-------|------|---------|
| Low | [0.00, 0.30] | Unlikely drug-like |
| Medium | [0.40, 0.60] | Moderately drug-like |
| High | [0.70, 1.00] | Drug-like |

### SA score (normalised)
| Range | Normalised target | Raw SA |
|-------|------|--------|
| Easy | [0.78, 1.00] | SA ≤ 3.0 |
| Medium | [0.44, 0.78] | SA 3.0–5.0 |
| Hard | [0.00, 0.44] | SA ≥ 5.0 |

### LogP (normalised)
| Range | Normalised target | LogP |
|-------|------|------|
| Low | [0.00, 0.20] | −3 to −1 |
| Medium | [0.20, 0.50] | −1 to 2 |
| High | [0.50, 0.80] | 2 to 5 |

### GSK3B activity
| Range | Target | Interpretation |
|-------|--------|----------------|
| Inactive | [0.0, 0.3] | Not predicted active |
| Active | [0.5, 1.0] | Predicted active |

---

## How to interpret the outputs

- **Conditional accuracy > 0.5** indicates the model is successfully steering generation toward the target range.  Values above 0.7 are considered strong for a single property.
- **Conditional accuracy ≈ baseline frequency** (i.e., matching random drug-like molecules) indicates the model is ignoring the condition signal — check that `condition_dim` and `sample_conditions` are configured consistently.
- **Diversity within a condition** should remain reasonably high (> 0.5 Tanimoto distance) — a well-conditioned model does not collapse to a single molecule.
