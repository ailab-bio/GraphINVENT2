"""
Trains a scikit-learn surrogate from SMILES and labels, ready for the sklearn oracle.

The goal-directed workflow expects the objective to be a model the user owns.
This produces one in the format `oracles.SklearnOracle` loads: a random forest
over Morgan fingerprints, pickled together with the featurisation settings it
was trained with.

A random forest is the default for a reason beyond convenience: its per-tree
predictions give an ensemble spread at no extra cost, which is what the
uncertainty-modulated RL described in `oracles._uncertainty` consumes. A single
model would score just as well and offer nothing to modulate with.

The held-out metrics printed at the end are the honest part of this script. A
surrogate that cannot predict its own test set will still happily drive an RL
run, producing molecules that score well and mean nothing, so the numbers are
printed prominently and a scaffold split is offered because a random split of
congeneric series flatters a model that has merely memorised scaffolds.

Input is a CSV or TSV with a SMILES column and a label column. Classification
labels may be 0/1 or a continuous activity thresholded with --threshold.

Usage:
    python src/graphinvent/tools/train-surrogate.py \
        --input data/assays/egfr.csv --smiles-column smiles --label-column pIC50 \
        --threshold 6.0 --split scaffold --output data/surrogates/egfr_rf.pkl
"""

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)


def read_table(path: Path, smiles_column: str, label_column: str):
    """Read SMILES and labels from a CSV/TSV, choosing the delimiter by suffix."""
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    smiles, labels = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} appears to be empty.")
        for column in (smiles_column, label_column):
            if column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{column}' not in {path}; found {reader.fieldnames}."
                )
        for row in reader:
            value = row[label_column]
            if value is None or value == "":
                continue
            smiles.append(row[smiles_column])
            labels.append(float(value))
    return smiles, np.asarray(labels, dtype=float)


def featurize(smiles, radius: int, n_bits: int):
    """Morgan fingerprints for the parseable molecules, with their indices."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows, kept = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        rows.append(np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=np.float64))
        kept.append(i)
    if not rows:
        raise ValueError("No parseable molecules in the input.")
    return np.vstack(rows), kept


def scaffold_split(smiles, test_fraction: float, seed: int):
    """
    Split by Bemis-Murcko scaffold, largest groups into training first.

    A random split over a dataset built from congeneric series leaves near
    duplicates on both sides, so the test score measures memorisation rather
    than generalisation. Scaffold splitting is the cheap standard correction;
    it is pessimistic relative to a random split, which is the point.
    """
    groups = defaultdict(list)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
        except (ValueError, RuntimeError):
            scaffold = smi
        groups[scaffold].append(i)

    ordered = sorted(groups.values(), key=len, reverse=True)
    n_test_target = int(round(test_fraction * sum(len(g) for g in ordered)))
    train_idx, test_idx = [], []
    for group in ordered:
        if len(test_idx) < n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)
    return sorted(train_idx), sorted(test_idx)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binarise the label at this value. Omit to train a regressor.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--split",
        choices=["scaffold", "random"],
        default="scaffold",
        help="Scaffold splitting gives a more honest estimate for congeneric data.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    smiles, labels = read_table(args.input, args.smiles_column, args.label_column)
    print(f"* Read {len(smiles)} rows from {args.input}", flush=True)

    features, kept = featurize(smiles, args.radius, args.n_bits)
    labels = labels[kept]
    kept_smiles = [smiles[i] for i in kept]
    if len(kept) < len(smiles):
        print(f"  {len(smiles) - len(kept)} unparseable SMILES dropped", flush=True)

    is_classification = args.threshold is not None
    if is_classification:
        labels = (labels >= args.threshold).astype(int)
        n_positive = int(labels.sum())
        print(
            f"  {n_positive} active / {len(labels) - n_positive} inactive "
            f"at threshold {args.threshold}",
            flush=True,
        )
        if n_positive == 0 or n_positive == len(labels):
            print(
                "One class is empty; a model trained on this cannot discriminate.",
                file=sys.stderr,
            )
            return 1

    if args.split == "scaffold":
        train_idx, test_idx = scaffold_split(kept_smiles, args.test_fraction, args.seed)
    else:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(kept_smiles))
        n_test = int(round(args.test_fraction * len(order)))
        test_idx, train_idx = sorted(order[:n_test]), sorted(order[n_test:])

    print(
        f"* {args.split} split: {len(train_idx)} train / {len(test_idx)} test",
        flush=True,
    )

    model_cls = RandomForestClassifier if is_classification else RandomForestRegressor
    model = model_cls(n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1)
    model.fit(features[train_idx], labels[train_idx])

    print("* Held-out performance", flush=True)
    if test_idx:
        if is_classification:
            probabilities = model.predict_proba(features[test_idx])[:, 1]
            if len(set(labels[test_idx])) > 1:
                print(
                    f"  ROC-AUC : {roc_auc_score(labels[test_idx], probabilities):.3f}",
                    flush=True,
                )
            else:
                print("  ROC-AUC : undefined (test set has one class)", flush=True)
            accuracy = (model.predict(features[test_idx]) == labels[test_idx]).mean()
            print(f"  accuracy: {accuracy:.3f}", flush=True)
        else:
            predictions = model.predict(features[test_idx])
            print(
                f"  R^2     : {r2_score(labels[test_idx], predictions):.3f}", flush=True
            )
            print(
                f"  MAE     : {mean_absolute_error(labels[test_idx], predictions):.3f}",
                flush=True,
            )
    else:
        print("  no test set; performance unknown", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(model, f)

    oracle_type = "proba" if is_classification else "predict"
    print(f"* Wrote {args.output}", flush=True)
    print(
        "\nUse it by adding this to the 'oracles' block of a goal_directed config:\n"
        f'  "{args.output.stem}": {{\n'
        f'      "type": "sklearn",\n'
        f'      "path": "{args.output}",\n'
        f'      "radius": {args.radius}, "n_bits": {args.n_bits},\n'
        f'      "output": "{oracle_type}",\n'
        f'      "direction": "maximize"\n'
        "  }\n"
        "and naming it in 'score_components'. The fingerprint settings must "
        "match the ones above, since nothing can verify them at load time.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
