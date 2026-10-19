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
    python scripts/train-surrogate.py \
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


DESCRIPTORS = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
)


def featurize(
    smiles,
    radius: int,
    n_bits: int,
    counts: bool = False,
    categorical: dict | None = None,
):
    """
    Morgan fingerprints, physicochemical descriptors and one-hot categoricals.

    Fingerprint bits alone were the whole feature set, which leaves out the
    block the experiment plan calls load-bearing: the descriptors that carry
    size and polarity, and the assay context. Pooling measurements across cell
    lines and ligases without telling the model which is which asks it to fit a
    label that means different things in different rows.

    `counts` uses the count fingerprint rather than the binary one. Two real
    degraders differing by one linker methylene score a binary Tanimoto of
    1.000 even at 65,536 bits, because they contain the same set of radius-2
    environments and differ only in their multiplicities; linker length is the
    most tuned variable in degrader chemistry.

    `categorical` is {column name: [value per row]}; each column becomes one
    indicator per observed value, with the vocabulary pickled alongside.
    """
    from rdkit.Chem import Descriptors

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    vocab = {}
    for name, values in (categorical or {}).items():
        vocab[name] = sorted({str(v).strip().lower() for v in values if str(v).strip()})

    rows, kept = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        if counts:
            fp = generator.GetCountFingerprintAsNumPy(mol)
        else:
            fp = generator.GetFingerprintAsNumPy(mol)
        parts = [np.asarray(fp, dtype=np.float64)]
        parts.append(
            np.asarray(
                [getattr(Descriptors, d)(mol) for d in DESCRIPTORS], dtype=np.float64
            )
        )
        for name, values in (categorical or {}).items():
            observed = str(values[i]).strip().lower()
            parts.append(
                np.asarray(
                    [1.0 if observed == v else 0.0 for v in vocab[name]],
                    dtype=np.float64,
                )
            )
        rows.append(np.concatenate(parts))
        kept.append(i)
    if not rows:
        raise ValueError("No parseable molecules in the input.")
    return np.vstack(rows), kept, vocab


def _warhead_key(smiles_list):
    """
    The Bemis-Murcko scaffold of each molecule's target-binding component.

    A whole-molecule scaffold of a PROTAC contains every ring system and is
    nearly unique per compound: measured on the curated degrader corpus, 14,406
    scaffolds over 21,429 molecules with 55% singletons, so grouping by it
    leaves near analogues on both sides and the split is a random one with
    extra steps.
    """
    repo = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(repo / "src")]
    from metrics._components import decompose, warhead_of

    print(
        "* Decomposing for a warhead_scaffold split (cached under tools/)", flush=True
    )
    keys = []
    for smi, fragments in zip(smiles_list, decompose(list(smiles_list))):
        warhead = warhead_of(fragments["poi"])[0] if fragments else None
        mol = Chem.MolFromSmiles(warhead) if warhead else None
        scaffold = None
        if mol is not None:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
            except (ValueError, RuntimeError):
                scaffold = None
        # Its own group, so a molecule with no usable warhead cannot silently
        # join another warhead's group.
        keys.append(scaffold or f"__no_warhead__{smi}")
    return keys


def scaffold_split(smiles, test_fraction: float, seed: int, kind: str = "scaffold"):
    """
    Split by scaffold group, filling TRAIN first from the largest groups.

    The direction matters and was backwards: filling test first held out the
    dominant congeneric series and trained on the singleton tail, so the
    reported score was pessimistic for a reason that had nothing to do with
    generalization. `DataProcessor._assign_groups` fills train first and this
    now matches it, including the shuffle that stops equal-sized groups from
    being ordered by their scaffold string.

    `kind` is "scaffold" for a whole-molecule Bemis-Murcko split or
    "warhead_scaffold" for the target-binding component, which is the only one
    that means anything on a bifunctional corpus.
    """
    import random as _random

    if kind == "warhead_scaffold":
        keys = _warhead_key(smiles)
    else:
        keys = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                keys.append(None)
                continue
            try:
                keys.append(
                    MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                )
            except (ValueError, RuntimeError):
                keys.append(smi)

    groups = defaultdict(list)
    for i, key in enumerate(keys):
        if key is not None:
            groups[key].append(i)

    ordered = sorted(groups.items(), key=lambda kv: kv[0])
    _random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda kv: -len(kv[1]))

    n_total = sum(len(g) for _, g in ordered)
    n_train_target = int(round((1.0 - test_fraction) * n_total))
    train_idx, test_idx = [], []
    for _, group in ordered:
        if len(train_idx) < n_train_target:
            train_idx.extend(group)
        else:
            test_idx.extend(group)

    # Whole groups are assigned, so a corpus with a few very large groups can
    # exhaust the training target before any group is left over. Reporting "no
    # held-out performance" without saying why is the failure this guards: the
    # kill criterion for the goal-directed arm is a held-out number.
    if not test_idx:
        raise SystemExit(
            f"The {kind} split put all {n_total} molecules in train and left "
            f"the test set empty: {len(ordered)} groups, the largest holding "
            f"{len(ordered[0][1])}.\nWhole groups are assigned, so a corpus "
            "this concentrated cannot be split this way. Use --split random "
            "and say so, or raise --test-fraction."
        )
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
        choices=["scaffold", "warhead_scaffold", "random"],
        default="scaffold",
        help="Scaffold splitting gives a more honest estimate for congeneric "
        "data. Use warhead_scaffold for a bifunctional corpus, where a "
        "whole-molecule scaffold is nearly unique per compound and the "
        "split degenerates into a random one.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--count-fingerprint",
        action="store_true",
        help="use the count fingerprint rather than the binary one, which is "
        "the only form that can distinguish linker homologues",
    )
    parser.add_argument(
        "--categorical",
        action="append",
        default=[],
        metavar="COLUMN",
        help="a column to one-hot encode, e.g. --categorical cell_line "
        "--categorical ligase. Repeatable. Pooling across assay contexts "
        "without telling the model which is which fits a label that means "
        "different things in different rows.",
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    smiles, labels = read_table(args.input, args.smiles_column, args.label_column)
    print(f"* Read {len(smiles)} rows from {args.input}", flush=True)

    categorical = {}
    if args.categorical:
        import csv as _csv

        with args.input.open() as handle:
            sample = handle.read(4096)
            handle.seek(0)
            delimiter = "\t" if "\t" in sample.splitlines()[0] else ","
            table = list(_csv.DictReader(handle, delimiter=delimiter))
        for column in args.categorical:
            if column not in (table[0] if table else {}):
                print(
                    f"--categorical {column} is not a column in {args.input}; "
                    f"columns are {sorted(table[0]) if table else '[]'}",
                    file=sys.stderr,
                )
                return 1
            categorical[column] = [row.get(column, "") for row in table]
        print(f"  one-hot columns: {list(categorical)}", flush=True)

    features, kept, vocabulary = featurize(
        smiles,
        args.radius,
        args.n_bits,
        counts=args.count_fingerprint,
        categorical=categorical or None,
    )
    print(
        f"  feature width {features.shape[1]}: "
        f"{args.n_bits} fingerprint "
        f"({'count' if args.count_fingerprint else 'binary'}) + "
        f"{len(DESCRIPTORS)} descriptors + "
        f"{features.shape[1] - args.n_bits - len(DESCRIPTORS)} indicators",
        flush=True,
    )
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

    if args.split in ("scaffold", "warhead_scaffold"):
        train_idx, test_idx = scaffold_split(
            kept_smiles, args.test_fraction, args.seed, kind=args.split
        )
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
            truth = labels[test_idx]
            # The baseline the model has to beat: predicting the training mean
            # for everything. An R^2 near zero means exactly this, and reporting
            # R^2 alone hides how close to it the model is.
            baseline = np.full_like(truth, labels[train_idx].mean(), dtype=float)
            rmse = float(np.sqrt(np.mean((truth - predictions) ** 2)))
            rmse_baseline = float(np.sqrt(np.mean((truth - baseline) ** 2)))
            try:
                from scipy.stats import spearmanr

                rho = float(spearmanr(truth, predictions).statistic)
            except Exception:
                rho = float("nan")
            print(
                f"  RMSE    : {rmse:.3f}   (mean-prediction baseline "
                f"{rmse_baseline:.3f})",
                flush=True,
            )
            print(f"  Spearman: {rho:.3f}", flush=True)
            print(
                f"  beats the mean-prediction baseline: "
                f"{'YES' if rmse < rmse_baseline else 'NO'}",
                flush=True,
            )
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
        # A dict, not a bare estimator: the featurization is part of the
        # model. Pickling only the estimator meant the oracle had to be told
        # the fingerprint settings by hand and nothing could check them, and
        # once the feature block grew past the fingerprint the two silently
        # disagreed on width. The oracle reads these and reproduces them.
        from collections import Counter

        pickle.dump(
            {
                "model": model,
                "featurization": {
                    "radius": args.radius,
                    "n_bits": args.n_bits,
                    "use_counts": bool(args.count_fingerprint),
                    "descriptors": list(DESCRIPTORS),
                    "categorical_vocab": vocabulary,
                    # A generated molecule has no assay context, so inference fills
                    # the indicator block with the modal training value and the
                    # prediction is for that context. Recorded here so a reader
                    # knows which one rather than guessing.
                    "categorical_defaults": {
                        name: Counter(
                            str(v).strip().lower()
                            for v in categorical[name]
                            if str(v).strip()
                        ).most_common(1)[0][0]
                        for name in vocabulary
                    },
                },
                "label_column": args.label_column,
                "n_features": int(features.shape[1]),
            },
            f,
        )

    oracle_type = "proba" if is_classification else "predict"
    print(f"* Wrote {args.output}", flush=True)
    print(
        "\nUse it by adding this to the 'oracles' block of a goal_directed config:\n"
        f'  "{args.output.stem}": {{\n'
        f'      "type": "sklearn",\n'
        f'      "path": "{args.output}",\n'
        f'      "output": "{oracle_type}",\n'
        f'      "direction": "maximize"\n'
        "  }\n"
        "and naming it in 'score_components'. The featurization is stored in "
        "the pickle and the oracle reads it, so there is nothing to keep in "
        "sync by hand.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
