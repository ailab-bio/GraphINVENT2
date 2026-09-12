# Experiment 4: Conditional generation

Train a property-conditioned GGNN on ChEMBL and measure how reliably a requested property value shows up in what it generates.

---

## What this tests

Conditioning is worth having only if the model responds to the condition vector rather than ignoring it, and the honest way to establish that is to sweep the requested value across its range and check whether the realised property distribution follows. A model that reproduces the training distribution regardless of what is asked will still post good validity and diversity, so those metrics cannot distinguish a working conditional model from an expensive unconditional one. Conditional accuracy against a baseline rate is what separates them.

This experiment conditions on three properties simultaneously: QED, normalised synthetic accessibility, and normalised LogP. All three are computed directly by RDKit, so the labelled dataset needs nothing but a SMILES file. A fourth column predicted by a surrogate you trained — activity against a target of interest — can be added, and is the more interesting version of the experiment, since it asks the model to steer a property RDKit cannot see. Requesting several properties at once is harder than any single-property version, because they are correlated in ChEMBL and some combinations barely exist in the training data.

The configuration files under this directory are written for the four-property version and declare `"condition_dim": 4` with `GSK3B` as the fourth column. Running the three-property version means changing `condition_dim` to 3 in `preprocess_params.json` and `train_params.json`, dropping the fourth entry from `conditioning.properties`, and dropping the corresponding key from `sample_conditions` in `generate_params.json`.

---

## How the conditioning works

```
condition_vector (batch × n properties)
        │
        ▼
ConditionEncoder (2-layer MLP → 256-dim)
        │
        ▼ initialises the hidden state of a virtual seed node at index 0
┌──────────────────────────────────────────┐
│  GGNN, 4 message-passing rounds          │
│  seed → real atoms  (virtual edge type)  │
│  real atoms ↔ real atoms  (bond edges)   │
└──────────────────────────────────────────┘
        │ seed node stripped
        ▼
readout → action probability distribution
```

The condition vector is encoded once into an embedding that initialises a virtual node prepended to every graph. That node sends messages to every real atom at every round through an extra edge type reserved for it, so the condition reaches atom representations throughout the network rather than being concatenated at the output. Stripping the seed before readout keeps the action distribution the same size as in the unconditional model, which is why nothing downstream of the readout needs to change.

The cost is in the data. When conditioning is enabled, `DataProcessor` stops deduplicating identical subgraphs, because two occurrences of the same partial graph now carry different condition vectors and different targets. Conditional HDF5 files are therefore substantially larger than their unconditional equivalents for the same molecules.

---

## Property definitions and normalisation

| Property | Raw range | Transform | Normalised |
|----------|-----------|-----------|------------|
| `QED` | [0, 1] | none | [0, 1] |
| `SA_norm` | raw SA 1–10 | `(10 − SA) / 9` | [0, 1], 1 = easiest |
| `LogP_norm` | unbounded | `(clip(LogP, −3, 7) + 3) / 10` | [0, 1] |
| surrogate column | classifier probability, [0, 1] | none | [0, 1] |

Normalisation is not cosmetic. The `ConditionEncoder` is a small MLP, so inputs on very different scales make its optimisation harder and let the largest-magnitude property dominate the embedding. The clip bounds on LogP cover the overwhelming majority of drug-like molecules, at the cost of making the model unable to distinguish anything beyond them.

The same transform has to be applied during preprocessing, training and generation. `compute_properties.py` and `evaluate_conditional.py` share these definitions; a hand-written `sample_conditions` block does not, so values there must already be normalised.

---

## Step 1 — Build the labelled TSV

```bash
python experiments/conditional/compute_properties.py \
    --smiles data/raw/chembl_v34_filtered.smi \
    --out data/raw/chembl_v34_cond.tsv \
    --surrogate GSK3B=data/surrogates/gsk3b_rf.pkl
```

Required flags are `--smiles` and `--out`; those two alone give the three RDKit properties. `--surrogate NAME=PATH` is repeatable and appends one column per surrogate, where `PATH` is a pickled scikit-learn model over Morgan fingerprints such as `src/graphinvent/tools/train-surrogate.py` produces. `--batch-size` (default 1000) controls how many molecules are sent to a surrogate at a time, and `--max-mols` caps the number processed, which is useful for a trial run.

The output is tab-separated with a header, columns `SMILES`, `QED`, `SA_norm`, `LogP_norm`, then the surrogate columns in the order the flags were given. Molecules RDKit cannot parse are dropped and SMILES are canonicalised on the way through. The three RDKit properties take a few minutes for 1.5M molecules; a surrogate column adds roughly an hour, dominated by fingerprinting and prediction.

A surrogate that fails to load raises rather than filling the column with zeros, which is deliberate: a column of zeros would be used as a conditioning target and would quietly train the model to associate every molecule with the same value.

Whatever columns you produce, three files have to agree with them: `conditioning.properties` and `condition_dim` in `preprocess_params.json`, `condition_dim` in `train_params.json`, and `PROPERTY_COLUMN_ORDER` in `evaluate_conditional.py`, which is hard-coded to `["QED", "SA_norm", "LogP_norm"]` and needs the surrogate names appended in the same order.

---

## Step 2 — Preprocess

```bash
python submit.py --config experiments/conditional/preprocess_params.json
```

The config declares the conditioning explicitly:

```json
"condition_dim": 4,
"conditioning": {
  "properties": ["QED", "SA_norm", "LogP_norm", "GSK3B"],
  "source": "smiles_file"
}
```

This is the four-property version; drop `GSK3B` and set `condition_dim` to 3 if the TSV has no surrogate column. `condition_dim` must equal the number of property columns, and `properties` must list them in file order, because the condition vector is built positionally. The split files written to `data/datasets/chembl_v34_cond/` keep the TSV format and header, so property values survive into `train.smi`, `valid.smi` and `test.smi`, and each subgraph in the HDF5 files carries a `condition_vector` row of shape `(n_subgraphs, condition_dim)`.

A random split is used here rather than the Butina split of Experiment 1, on the reasoning that the conditional dataset should stay comparable to the unconditional ChEMBL run; that is a pragmatic choice, and a cluster split would make the conditional accuracy numbers harder but more meaningful.

Expect 4–8 hours, longer than the unconditional equivalent because subgraphs are no longer deduplicated.

---

## Step 3 — Train

Set `resume_from` in `train_params.json` to the ChEMBL checkpoint from Experiment 1, then:

```bash
python submit.py --config experiments/conditional/train_params.json
```

Starting from an unconditional prior is worth doing because only the `ConditionEncoder` and the virtual-edge message MLP begin from random weights, while the rest of the network arrives with the chemical grammar already learned. The model spends its training budget learning to respond to the condition rather than relearning valence rules.

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `condition_dim` | 4 | Must match the preprocessed dataset; 3 without a surrogate column |
| `condition_embedding_dim` | 256 | Equal to `hidden_node_features`, so the seed node matches real nodes in width |
| `init_lr` | `5e-5` | Below pretraining, since this is fine-tuning |
| `max_rel_lr` | `5` | Peak LR of 2.5e-4 under the one-cycle schedule |
| `epochs` | 100 | Upper bound; watch validation loss |
| `sample_every` | 5 | 500 molecules sampled per evaluation |

**The `_architecture` comment in this config is wrong.** It claims the architecture block is ignored when `resume_from` is set, but `src/graphinvent/parameters/config.py` inherits an architecture key from the checkpoint's `params_all.json` only when that key is *absent* from the job config. Every key spelled out here overrides the checkpoint. The block currently works because its values match `chembl_pretrain/pretrain_params.json` exactly; change one of them, or point `resume_from` at a differently-shaped checkpoint, and `load_state_dict` will fail.

`condition_dim` and `condition_embedding_dim` are part of that same inheritance set, and here the override is load-bearing rather than incidental: the unconditional prior records `condition_dim: 0`, so leaving it out would inherit zero and build a model with no `ConditionEncoder` at all.

Expect 12–24 hours on one A100.

---

## Step 4 — Evaluate across property ranges

```bash
python experiments/conditional/evaluate_conditional.py \
    --checkpoint output/chembl_v34_cond/conditional/run/model_restart_100.pth
```

Options: `--checkpoint` (required), `--dataset` (default `chembl_v34_cond`), `--n-samples` (default 1000), `--out` (default `experiments/conditional/results/`), `--device` (default `cuda`), and a repeatable `--property` filter.

The script walks the nine ranges in `PROPERTY_RANGES`, three each for QED, `SA_norm` and `LogP_norm`, and for each one sets the target property to the midpoint of the range while holding the others at fixed neutral values (QED 0.7, `SA_norm` 0.75, `LogP_norm` 0.35). It writes a temporary generate config, calls `submit.py`, then recomputes the actual property values of the generated molecules with the same transforms used at preprocessing time.

A surrogate column is not evaluated here. `PROPERTY_RANGES` covers only the three RDKit properties, and the script recomputes realised values with its own RDKit transforms, so measuring conditional accuracy on a predicted-activity column means adding a range entry, a neutral value, and a scoring branch that loads the same surrogate.

Results go to `experiments/conditional/results/conditional_results.csv` with the columns `property`, `range`, `target_lo`, `target_hi`, `target_val`, `n_generated`, `validity`, `cond_accuracy`, `internal_diversity`, `mean_prop`, `std_prop`.

Two caveats about the numbers this script produces. Its `load_smiles` helper does not skip the per-batch `SMILES Name` header lines that GraphINVENT writes into `.smi` files, so a handful of unparseable entries are counted in the validity denominator. It also does not filter the `[Xe]` placeholder that marks an invalid graph, and since RDKit parses `[Xe]` as a xenon atom, invalid graphs are counted as valid molecules and then contribute a meaningless property value. Both effects push `validity` up and `cond_accuracy` around; the `.valid` file written alongside each `.smi` is the authoritative validity vector if you need exact numbers.

---

## Step 5 — Generate for one condition

```bash
python submit.py --config experiments/conditional/generate_params.json
```

```json
"sample_conditions": {
  "QED":       0.85,
  "SA_norm":   0.80,
  "LogP_norm": 0.50,
  "GSK3B":     0.0
}
```

This asks for a drug-like molecule (QED 0.85), readily synthesisable (`SA_norm` 0.80, raw SA ≈ 2.8), of moderate lipophilicity (`LogP_norm` 0.50, LogP ≈ 2), with no activity requirement on the surrogate column. Remove the fourth key for a three-property model; the number of keys has to match `condition_dim`.

The key order matters. `Workflow.sample_molecules` builds the condition vector in the order given by `constants.conditioning["properties"]`, and falls back to the order of the keys in `sample_conditions` when `conditioning` is absent. A generate job does not inherit `conditioning` from the checkpoint, so in practice the fallback applies and the JSON key order determines the vector. Writing the keys in a different order from the preprocessing columns silently permutes the condition rather than raising an error. Either keep the order identical to `conditioning.properties`, or repeat the `conditioning` block in the generate config.

---

## Property ranges evaluated

Ranges are in normalised units; the raw column shows what they correspond to.

### QED
| Range | Normalised | Interpretation |
|-------|-----------|----------------|
| low | [0.00, 0.30] | Unlikely to be drug-like |
| medium | [0.40, 0.60] | Moderately drug-like |
| high | [0.70, 1.00] | Drug-like |

### SA_norm
| Range | Normalised | Raw SA |
|-------|-----------|--------|
| easy | [0.78, 1.00] | ≤ 3.0 |
| medium | [0.44, 0.78] | 3.0 – 6.0 |
| hard | [0.00, 0.44] | ≥ 6.0 |

### LogP_norm
| Range | Normalised | LogP |
|-------|-----------|------|
| low | [0.00, 0.20] | −3 to −1 |
| medium | [0.20, 0.50] | −1 to 2 |
| high | [0.50, 0.80] | 2 to 5 |

A surrogate column has no entry in `PROPERTY_RANGES`. Adding one, with ranges such as [0.00, 0.30] for "not predicted active" and [0.50, 1.00] for "predicted active", also requires teaching the script to recompute the realised value from the same pickled model, since it currently recomputes only the RDKit properties.

---

## Reading the results

`cond_accuracy` is only interpretable against the rate the same range would achieve without conditioning. The high-QED range sits where much of ChEMBL already sits, so an unconditional model scores well on it by default and a conditional accuracy of 0.6 there demonstrates very little. The low-QED and hard-SA ranges are the informative ones, because they ask for molecules the training distribution is thin in, and accuracy there is what shows the condition is doing work. Measure the baseline by generating from the unconditional prior and running the same range check; without it the numbers have no scale.

Accuracy alone can also be satisfied degenerately. A model that answers every request in a range with the same molecule scores perfectly, which is why `internal_diversity` is reported per range and should stay high; a diversity that falls as the requested value moves toward the tails means the model has memorised one solution rather than learned to steer.

`mean_prop` and `std_prop` are more diagnostic than the in-range fraction, since they show whether a miss is a systematic offset — the model consistently overshooting the target, which suggests the condition is being attenuated — or a wide spread centred correctly, which suggests the signal is present but weak.

Reasonable failure modes to check before concluding the method does not work: `condition_dim` disagreeing between preprocessing and training, `sample_conditions` keys in a different order from `conditioning.properties`, and un-normalised values passed at generation time.
