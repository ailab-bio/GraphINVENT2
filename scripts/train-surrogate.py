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
from collections import Counter, defaultdict
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


def read_table(
    path: Path, smiles_column: str, label_column: str, extra=(), optional=()
):
    """
    Read SMILES, labels and any extra columns from a CSV/TSV, in ONE pass.

    One pass is the point. The one-hot columns used to be read by a second
    `DictReader` over the same file, which indexed them by raw file row while
    the featurizer indexed them by position in the label-filtered `smiles`
    list. Any row with a blank label shifted every later molecule's assay
    context by one, silently, and a group column read the same way would have
    inherited it. Everything the fit needs now comes out of the same loop and
    the same skip.

    `extra` columns must exist; `optional` ones are read when the table has
    them and omitted from the result when it does not, which is how the noise
    floor gets a target column from the tables that carry one without
    breaking the fits whose tables have no target at all.
    """
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    smiles, labels = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} appears to be empty.")
        for column in (smiles_column, label_column, *extra):
            if column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{column}' not in {path}; found {reader.fieldnames}."
                )
        # Deduplicated: a column named in both `extra` and `optional` would
        # otherwise be appended twice per row, so every value after the first
        # would belong to a different molecule than the index says.
        present = list(
            dict.fromkeys([*extra, *(c for c in optional if c in reader.fieldnames)])
        )
        extras: dict[str, list] = {name: [] for name in present}
        for row in reader:
            value = row[label_column]
            if value is None or value == "":
                continue
            smiles.append(row[smiles_column])
            labels.append(float(value))
            for name in present:
                extras[name].append(row.get(name, ""))
    return smiles, np.asarray(labels, dtype=float), extras


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


GROUP_KINDS = ("scaffold", "warhead_scaffold", "poi", "compound")
CV_MODES = ("none", "grouped")

# The measured noise floor of the degrader pDC50 labels: the pooled
# within-compound standard deviation across cell lines, over the compounds
# measured in more than one line. It caps what any model of this label can
# reach, so an R^2 is reported against this ceiling and not against 1.0.
# Derivation and the per-corpus numbers are in each surrogate's PERFORMANCE.md.


def label_noise_floor(labels, compound_keys):
    """
    Pooled within-compound SD, and the R^2 ceiling it implies.

    A compound measured in three cell lines carries three labels that the model
    sees as one feature vector plus an assay one-hot. The spread among them is
    irreducible for a model that does not know more than the context columns
    say, so it bounds achievable performance: R^2_max = 1 - (noise/total)^2.
    Returns (noise_sd, total_sd, r2_ceiling, n_compounds_with_replicates).

    `compound_keys` must identify the compound AND the target. On a pooled
    multi-target table, keying on the compound alone pools a degrader's
    SMARCA2 and SMARCA4 potencies into one "replicate" group, so a real
    difference between two proteins is counted as assay noise and the ceiling
    comes out far too low -- measured at 0.410 instead of 0.774 on the pooled
    pDC50 table, which would have made a usable model look impossible.
    """
    by_compound = defaultdict(list)
    for key, value in zip(compound_keys, labels):
        by_compound[key].append(float(value))
    repeated = [v for v in by_compound.values() if len(v) > 1]
    total_sd = float(np.std(np.asarray(labels, dtype=float), ddof=1))
    if not repeated:
        return None, total_sd, None, 0
    # Pooled within-group variance, weighted by degrees of freedom.
    numerator = sum(np.var(np.asarray(v), ddof=1) * (len(v) - 1) for v in repeated)
    denominator = sum(len(v) - 1 for v in repeated)
    noise_sd = float(np.sqrt(numerator / denominator))
    ceiling = float(1.0 - (noise_sd / total_sd) ** 2) if total_sd > 0 else None
    return noise_sd, total_sd, ceiling, len(repeated)


def regression_report(truth, predictions, train_labels):
    """
    The held-out numbers for one test set, against the mean-prediction baseline.

    `r2_train_mean` is the stated kill criterion -- variance explained against
    predicting the TRAINING mean, which is what a model at fit time could
    actually do. `r2_test_mean` is sklearn's `r2_score`, against the test mean,
    which no model can know. They differ by 0.075 on the SMARCA2 fit, so naming
    either one "the" held-out R^2 misreports whichever criterion is applied.
    """
    truth = np.asarray(truth, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    baseline = np.full_like(truth, float(np.mean(train_labels)))
    rmse = float(np.sqrt(np.mean((truth - predictions) ** 2)))
    rmse_baseline = float(np.sqrt(np.mean((truth - baseline) ** 2)))
    try:
        from scipy.stats import spearmanr

        rho = float(spearmanr(truth, predictions).statistic)
    except Exception:
        rho = float("nan")
    return {
        "n": int(len(truth)),
        "rmse": rmse,
        "rmse_baseline": rmse_baseline,
        "margin": rmse_baseline - rmse,
        "r2_train_mean": (
            float(1.0 - (rmse**2) / (rmse_baseline**2))
            if rmse_baseline > 0
            else float("nan")
        ),
        "r2_test_mean": float(r2_score(truth, predictions)),
        "spearman": rho,
        "mae": float(mean_absolute_error(truth, predictions)),
    }


def clustered_bootstrap_margin(truth, predictions, baseline, clusters, seed, n=2000):
    """
    Bootstrap interval for the RMSE margin, resampling COMPOUNDS not rows.

    The earlier interval resampled residual pairs independently, and said so in
    a comment: it does not correct for one compound appearing in several rows.
    On the SMARCA2 table 186 compounds supply 241 rows, so roughly 55 are
    repeated and an iid resample treats correlated rows as independent
    evidence, which makes the interval too narrow in the direction that matters
    -- the lower bound is the whole decision. Resampling whole compounds is the
    right unit: the cluster, not the measurement, is what was sampled.
    """
    truth = np.asarray(truth, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    by_cluster = defaultdict(list)
    for position, key in enumerate(clusters):
        by_cluster[key].append(position)
    blocks = [np.asarray(v) for v in by_cluster.values()]
    rng = np.random.default_rng(seed)
    margins = []
    for _ in range(n):
        pick = np.concatenate(
            [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        )
        boot_rmse = float(np.sqrt(np.mean((truth[pick] - predictions[pick]) ** 2)))
        boot_base = float(np.sqrt(np.mean((truth[pick] - baseline[pick]) ** 2)))
        margins.append(boot_base - boot_rmse)
    lo, hi = (float(v) for v in np.percentile(margins, [2.5, 97.5]))
    return lo, hi, len(blocks)


def group_keys(smiles, kind: str, poi=None):
    """
    One group key per molecule, for a grouped split or a grouped k-fold.

    The same vector serves both, which is why it lives here rather than inside
    `scaffold_split`: a held-out fifth and a held-out fold have to be grouped
    the same way or the two estimates are not of the same quantity.

      scaffold          whole-molecule Bemis-Murcko
      warhead_scaffold  Bemis-Murcko of the target-binding component
      compound          canonical SMILES, which groups the replicate rows of
                        one compound measured in several cell lines
      poi               the protein, as the CONNECTED COMPONENT over shared
                        canonical symbols

    The `poi` case is the one with a decision in it. A composite target such as
    `KRAS/CRAF` carries two symbols, so grouping on the string alone would put
    it in a different fold from plain `KRAS` and the model would see KRAS
    chemistry while "holding out" KRAS. Taking connected components over shared
    symbols makes the hold-out strict: no protein named in a test fold is named
    anywhere in the training folds. The cost is coarseness, since a chain of
    composites can merge two families into one group, so the component sizes
    are printed rather than left implicit.
    """
    if kind not in GROUP_KINDS:
        raise SystemExit(
            f"--group-by {kind} is not implemented; implemented kinds are "
            f"{', '.join(GROUP_KINDS)}."
        )
    if kind == "warhead_scaffold":
        return _warhead_key(smiles)
    if kind == "compound":
        keys = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi) if smi else None
            keys.append(Chem.MolToSmiles(mol) if mol is not None else smi)
        return keys
    if kind == "poi":
        if poi is None:
            raise SystemExit(
                "--group-by poi needs the target column, which this table does "
                "not have. Rebuild the labels with "
                "`scripts/tpddb-activity-labels.py`, which writes a `poi` "
                "column, and pass --group-column if it is named something else."
            )
        parent: dict[str, str] = {}

        def find(symbol: str) -> str:
            parent.setdefault(symbol, symbol)
            while parent[symbol] != symbol:
                parent[symbol] = parent[parent[symbol]]
                symbol = parent[symbol]
            return symbol

        for value in poi:
            symbols = [s for s in str(value).split(";") if s]
            for other in symbols[1:]:
                root_a, root_b = find(symbols[0]), find(other)
                if root_a != root_b:
                    parent[root_a] = root_b
        keys = []
        for value in poi:
            symbols = [s for s in str(value).split(";") if s]
            if not symbols:
                raise SystemExit(
                    "A row has an empty target column. POI-grouped folds "
                    "cannot place it, and giving it a group of its own would "
                    "be a fold that holds out nothing identifiable."
                )
            keys.append(find(symbols[0]))
        return keys
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

    keys = group_keys(smiles, kind)

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


ESTIMATES = {
    "warhead_scaffold": "the error on novel molecules for a target the model "
    "has seen, which is what the reward does",
    "scaffold": "the error on a novel whole-molecule scaffold",
    "compound": "the error on a compound not measured in training, with its "
    "analogues possibly present",
    "poi": "the error for a target the model has NEVER seen; for an unseen POI "
    "the one-hot column is all zeros, so this estimate is honest and "
    "pessimistic",
}


def run_grouped_cv(
    args,
    features,
    labels,
    kept_smiles,
    extras,
    is_classification: bool,
    model_cls,
    r2_ceiling,
):
    """
    Grouped k-fold, reported with what it estimates and kept as an ensemble.

    Two things come out of this and they are not the same thing. The reported
    numbers are a generalization estimate, labeled with the quantity they
    estimate, because a warhead-scaffold-grouped error and a POI-grouped error
    answer different questions and a single "held-out R^2" hides which was
    asked. The fold MODELS are the reward-hacking detector: their per-molecule
    disagreement is evidence that RL has driven a molecule outside the data,
    which the spread across one forest's trees cannot show, since those trees
    saw the same rows.
    """
    from sklearn.model_selection import GroupKFold

    keys = group_keys(
        kept_smiles,
        args.group_by,
        poi=extras.get(args.group_column) if args.group_by == "poi" else None,
    )
    # A molecule the grouping could not place gets its own group rather than a
    # shared None group, which would be a fold holding out an arbitrary mixture.
    keys = [k if k else f"__ungrouped__{i}" for i, k in enumerate(keys)]
    n_groups = len(set(keys))
    folds = min(args.folds, n_groups)
    if folds < 2:
        raise SystemExit(
            f"--group-by {args.group_by} produced {n_groups} group(s) over "
            f"{len(keys)} rows, so there is nothing to hold out. A grouped "
            "k-fold needs at least two groups; pick a different grouping or "
            "drop --cv grouped and say in the write-up that no grouped "
            "estimate exists for this table."
        )
    if folds < args.folds:
        print(
            f"* Only {n_groups} groups, so {folds} folds rather than " f"{args.folds}",
            flush=True,
        )

    sizes = sorted(Counter(keys).values(), reverse=True)
    print(
        f"* {folds}-fold CV grouped by {args.group_by}: {n_groups} groups over "
        f"{len(keys)} rows, largest {sizes[0]}, median {sizes[len(sizes) // 2]}",
        flush=True,
    )
    print(f"  this estimates {ESTIMATES[args.group_by]}", flush=True)

    fold_models, rows, per_group = [], [], defaultdict(list)
    splitter = GroupKFold(n_splits=folds)
    for number, (train_idx, test_idx) in enumerate(
        splitter.split(features, labels, groups=keys), start=1
    ):
        fold = model_cls(
            n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1
        )
        fold.fit(features[train_idx], labels[train_idx])
        fold_models.append(fold)
        held_out = sorted({keys[i] for i in test_idx})
        # The guarantee the whole scheme rests on, asserted rather than assumed.
        leaked = {keys[i] for i in train_idx} & set(held_out)
        if leaked:
            raise SystemExit(
                f"Fold {number} has {len(leaked)} group(s) on both sides of the "
                f"split, so it holds out nothing: {sorted(leaked)[:5]}. That is "
                "a bug in the grouping, not a result."
            )
        if is_classification:
            probabilities = fold.predict_proba(features[test_idx])[:, 1]
            auc = (
                float(roc_auc_score(labels[test_idx], probabilities))
                if len(set(labels[test_idx])) > 1
                else float("nan")
            )
            report = {"n": int(len(test_idx)), "roc_auc": auc}
            print(
                f"  fold {number}: n={report['n']:5d}  ROC-AUC {auc:.3f}  "
                f"({len(held_out)} groups held out)",
                flush=True,
            )
        else:
            predictions = fold.predict(features[test_idx])
            report = regression_report(labels[test_idx], predictions, labels[train_idx])
            print(
                f"  fold {number}: n={report['n']:5d}  RMSE {report['rmse']:.3f} "
                f"(baseline {report['rmse_baseline']:.3f})  "
                f"R^2 {report['r2_train_mean']:+.3f}  "
                f"rho {report['spearman']:+.3f}  "
                f"({len(held_out)} groups held out)",
                flush=True,
            )
            for i, prediction in zip(test_idx, predictions):
                per_group[str(extras.get(args.group_column, keys)[i])].append(
                    (float(labels[i]), float(prediction))
                )
        report["groups_held_out"] = len(held_out)
        rows.append(report)

    summary = {
        "scheme": f"{folds}-fold GroupKFold",
        "group_by": args.group_by,
        "estimates": ESTIMATES[args.group_by],
        "n_groups": n_groups,
        "n_rows": len(keys),
        "folds": rows,
    }

    if not is_classification:
        for field in ("rmse", "rmse_baseline", "margin", "r2_train_mean", "spearman"):
            values = [r[field] for r in rows if not np.isnan(r[field])]
            summary[f"{field}_mean"] = float(np.mean(values)) if values else None
            # Sample SD. Five folds is a small n and a population SD would
            # understate the spread by about 10% at that size.
            summary[f"{field}_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        print(
            f"  across folds: RMSE {summary['rmse_mean']:.3f} "
            f"({summary['rmse_sd']:.3f})   "
            f"margin {summary['margin_mean']:.3f} ({summary['margin_sd']:.3f})   "
            f"R^2 {summary['r2_train_mean_mean']:+.3f} "
            f"({summary['r2_train_mean_sd']:.3f})",
            flush=True,
        )
        beaten = sum(1 for r in rows if r["rmse"] < r["rmse_baseline"])
        print(
            f"  folds beating the mean-prediction baseline: {beaten} of "
            f"{len(rows)}"
            + (
                ""
                if beaten == len(rows)
                else "   <- a fold that does not beat its baseline is a FAILED "
                "FIT on that fold and is reported as one, not averaged away"
            ),
            flush=True,
        )
        if r2_ceiling is not None and summary["r2_train_mean_mean"] is not None:
            summary["r2_ceiling"] = r2_ceiling
            mean_r2 = summary["r2_train_mean_mean"]
            if r2_ceiling <= 0:
                # noise_sd == total_sd: the label is all noise and no model of
                # it can explain anything. A fraction of zero is undefined, so
                # it is said rather than divided.
                print(
                    "  the label-noise ceiling is zero or below, so this label "
                    "carries no explainable variance and no fraction of it is "
                    "defined",
                    flush=True,
                )
                summary["fraction_of_achievable"] = None
            elif mean_r2 < 0:
                # A negative mean R^2 is worse than predicting the training
                # mean. Printing it as a percentage of the achievable variance
                # read as "-20% of achievable" -- a fraction of a quantity it
                # does not have any of.
                print(
                    f"  the {r2_ceiling:.3f} label-noise ceiling is not the "
                    f"binding constraint here: a mean R^2 of {mean_r2:.3f} is "
                    "below zero, so the fit does not beat predicting the "
                    "training mean and explains none of the achievable variance",
                    flush=True,
                )
                summary["fraction_of_achievable"] = None
            else:
                fraction = mean_r2 / r2_ceiling
                print(
                    f"  against the {r2_ceiling:.3f} label-noise ceiling, that "
                    f"is {fraction:.0%} of the achievable variance",
                    flush=True,
                )
                if fraction > 1.0:
                    # Exceeding the ceiling is not a better-than-perfect fit:
                    # it means the ceiling is underestimated. The within-group
                    # SD underestimates independent assay error on a bounded,
                    # saturating label like Dmax, where many replicates sit at
                    # the same clipped value and agree for that reason rather
                    # than because the assay is precise.
                    print(
                        "  a fraction above 100% means the CEILING is wrong, "
                        "not that the fit beat it: on a bounded saturating "
                        "label the within-group SD understates assay error, so "
                        "read this fit against the fold spread instead",
                        flush=True,
                    )
                summary["fraction_of_achievable"] = float(fraction)

        # Per-TARGET performance, because a pooled R^2 can look healthy while
        # one target sits at zero. Only for a POI grouping: on a scaffold
        # grouping the groups are mostly singletons, and a per-group table of
        # hundreds of n=1 RMSEs is noise that buries the fold summary above it.
        if args.group_by == "poi" and len(per_group) > 1:
            print("  per group, pooled over the folds that held it out:", flush=True)
            summary["per_group"] = {}
            for name, pairs in sorted(per_group.items(), key=lambda kv: -len(kv[1])):
                truth = np.asarray([p for p, _ in pairs])
                predicted = np.asarray([q for _, q in pairs])
                rmse = float(np.sqrt(np.mean((truth - predicted) ** 2)))
                spread = float(np.std(truth, ddof=1)) if len(truth) > 1 else None
                summary["per_group"][name] = {
                    "n": len(pairs),
                    "rmse": rmse,
                    "label_sd": spread,
                }
                print(
                    f"    {name:22s} n={len(pairs):5d}  RMSE {rmse:.3f}"
                    + (f"  (its own label SD {spread:.3f})" if spread else ""),
                    flush=True,
                )
    else:
        values = [r["roc_auc"] for r in rows if not np.isnan(r["roc_auc"])]
        summary["roc_auc_mean"] = float(np.mean(values)) if values else None
        summary["roc_auc_sd"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else None
        )

    return fold_models, summary


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
        "--cv",
        choices=list(CV_MODES),
        default="none",
        help="grouped runs a grouped k-fold in addition to the single held-out "
        "split, reports the across-fold spread, and keeps the fold models in "
        "the pickle. Those fold models are the ensemble-disagreement "
        "reward-hacking detector: disagreement between models fitted on "
        "different data is evidence of extrapolation, which the spread across "
        "the trees of one forest is not.",
    )
    parser.add_argument(
        "--group-by",
        choices=list(GROUP_KINDS),
        default="warhead_scaffold",
        help="what a fold holds out. warhead_scaffold estimates the error on "
        "novel molecules for a target the model has seen, which is what the "
        "reward actually does. poi estimates the error for a target it has "
        "never seen, which is what a held-out-target experiment needs. The two "
        "are different quantities and both are labeled with what they estimate.",
    )
    parser.add_argument(
        "--group-column",
        default="poi",
        help="the column holding the canonical target symbols, for " "--group-by poi",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--no-store-folds",
        action="store_true",
        help="compute the grouped-CV numbers but keep the fold models out of "
        "the pickle. The fold models exist to give an oracle its "
        "per-molecule disagreement signal, and they multiply the artifact's "
        "size by the fold count: the pooled pDC50 fit is 123 MB with them and "
        "20 MB without. A fit that is reported rather than used as a reward "
        "needs the numbers, which go in the log, and not the models.",
    )
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

    # The group column is read whenever the table has one, not only when POI
    # grouping asks for it: the label noise floor keys replicates on
    # (compound, target), and requesting the column only for `--group-by poi`
    # silently degraded that key to compound alone for every other scheme. On
    # the pooled table that reported the achievable R^2 ceiling as 0.410 when
    # it is 0.755, because one compound measured against two proteins became
    # its own replicate pair.
    wanted = list(dict.fromkeys(args.categorical))
    if args.cv == "grouped" and args.group_by == "poi":
        wanted = list(dict.fromkeys([*wanted, args.group_column]))
    try:
        smiles, labels, extras = read_table(
            args.input,
            args.smiles_column,
            args.label_column,
            extra=wanted,
            optional=(args.group_column,),
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"* Read {len(smiles)} rows from {args.input}", flush=True)

    categorical = {column: extras[column] for column in args.categorical}
    if categorical:
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
    # Every parallel column follows the same `kept` subset as the labels, or
    # the assay context and the group key belong to a different molecule from
    # the row they are used with.
    extras = {name: [values[i] for i in kept] for name, values in extras.items()}
    categorical = {name: extras[name] for name in categorical}
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
            report = regression_report(truth, predictions, labels[train_idx])
            # The clustered interval. A compound measured in three cell lines
            # is three rows and one piece of evidence, so the resample is over
            # compounds. The decision to spend an optimization allocation rests
            # on this lower bound.
            baseline = np.full_like(truth, labels[train_idx].mean(), dtype=float)
            compounds = group_keys([kept_smiles[i] for i in test_idx], "compound")
            lo, hi, n_clusters = clustered_bootstrap_margin(
                truth, predictions, baseline, compounds, args.seed
            )
            print(
                f"  RMSE    : {report['rmse']:.3f}   (mean-prediction baseline "
                f"{report['rmse_baseline']:.3f})",
                flush=True,
            )
            print(f"  Spearman: {report['spearman']:.3f}", flush=True)
            print(
                f"  margin over baseline: {report['margin']:.3f} "
                f"[95% bootstrap {lo:.3f}, {hi:.3f}] over {n_clusters} held-out "
                f"compounds in {report['n']} rows (clustered on compound)",
                flush=True,
            )
            # The verdict reads the INTERVAL, not the point estimate. It used
            # to print an unqualified YES whenever the point margin was
            # positive and then parenthesize "(interval includes zero)" after
            # it, so a fit whose advantage was indistinguishable from nothing
            # was recorded as beating its baseline.
            if report["rmse"] >= report["rmse_baseline"]:
                verdict = "NO -- worse than predicting the training mean"
            elif lo > 0:
                verdict = "YES"
            else:
                verdict = (
                    "NOT ESTABLISHED -- the point margin is positive but the "
                    "95% interval includes zero, so this fit does not clear "
                    "the kill criterion on this split"
                )
            print(f"  beats the mean-prediction baseline: {verdict}", flush=True)
            print(
                f"  R^2     : {report['r2_train_mean']:.3f} against the "
                f"mean-prediction baseline (the stated kill criterion); "
                f"{report['r2_test_mean']:.3f} against the test mean "
                "(sklearn r2_score)",
                flush=True,
            )
            print(f"  MAE     : {report['mae']:.3f}", flush=True)
    else:
        print("  no test set; performance unknown", flush=True)

    # The label noise floor, which caps what any model of this label can reach.
    # The replicate group is (compound, target): two measurements of one
    # compound against two proteins are not replicates of each other.
    compound_of = group_keys(kept_smiles, "compound")
    targets = extras.get(args.group_column)
    noise_keys = (
        [f"{c}|{t}" for c, t in zip(compound_of, targets)]
        if targets
        else list(compound_of)
    )
    unit = "(compound, target)" if targets else "compound"
    noise_sd, total_sd, ceiling, n_repeated = label_noise_floor(labels, noise_keys)
    if not is_classification and noise_sd is not None:
        print(
            f"* Label noise floor: within-{unit} SD {noise_sd:.3f} "
            f"against a total label SD of {total_sd:.3f}, over {n_repeated} "
            f"{unit} group(s) measured more than once",
            flush=True,
        )
        if not targets:
            # Naming the omission, because on a multi-target table keyed on
            # compound alone the ceiling is wrong in the flattering direction:
            # a genuine biological difference between two proteins is counted
            # as assay noise, which inflates the apparent noise and lowers the
            # apparent ceiling.
            print(
                f"  NOTE: no '{args.group_column}' column in this table, so "
                "replicates are keyed on compound alone. If the table pools "
                "several targets, this ceiling is not trustworthy.",
                flush=True,
            )
        print(
            f"  so R^2 for this label is capped at {ceiling:.3f}, not 1.0; read "
            "the held-out R^2 against that ceiling",
            flush=True,
        )

    fold_models, cv_summary = [], None
    if args.cv == "grouped":
        fold_models, cv_summary = run_grouped_cv(
            args,
            features,
            labels,
            kept_smiles,
            extras,
            is_classification,
            model_cls,
            ceiling,
        )
        # The shipped model is refitted on EVERY row. CLAUDE.md specifies an
        # oracle "fitted on all available data", and with grouped CV in hand
        # the held-out error estimate no longer comes from the single split,
        # so there is nothing left for the 20% to do but be withheld from the
        # reward. Before this it was fitted on train_idx alone while the fold
        # models each saw 80% of ALL rows, so the reward came from a narrower
        # fit than the disagreement reported beside it.
        print(
            f"* Refitting on all {len(labels)} rows for the shipped model; the "
            "error estimate above comes from the grouped folds, not from the "
            "single split",
            flush=True,
        )
        model = model_cls(
            n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1
        )
        model.fit(features, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        # A dict, not a bare estimator: the featurization is part of the
        # model. Pickling only the estimator meant the oracle had to be told
        # the fingerprint settings by hand and nothing could check them, and
        # once the feature block grew past the fingerprint the two silently
        # disagreed on width. The oracle reads these and reproduces them.
        pickle.dump(
            {
                "model": model,
                # The fold models, in fold order. `model` is what predicts:
                # the all-rows fit under --cv grouped, the train-split fit
                # under --cv none, where the split is the only held-out
                # estimate there is. These exist so the oracle can report
                # per-molecule disagreement between models fitted on DIFFERENT
                # data, which is the reward-hacking detector CLAUDE.md
                # specifies. Empty when --cv none, and the oracle then falls
                # back to one forest's trees and records which source it used.
                "models": [] if args.no_store_folds else fold_models,
                "cv": cv_summary,
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
    if is_classification:
        transform_advice = (
            "A classifier's probability is already a desirability in [0, 1], so "
            "no transform is needed."
        )
        transform_block = ""
    else:
        # The advice used to omit the transform, and this text is copied
        # verbatim into the fit log and from there into the oracle block, so
        # the omission propagated: with no transform the identity clamps every
        # prediction into [0, 1] and a label scale above 1.0 collapses the
        # reward to a constant. The bounds printed here are this fit's own
        # label range, as a starting point, not a recommendation.
        low, high = float(np.min(labels)), float(np.max(labels))
        transform_advice = (
            "THE TRANSFORM IS REQUIRED for a regression output and the oracle "
            "raises without one: the raw value is not a desirability, and the "
            f"identity transform clamps to [0, 1] while this fit's labels span "
            f"{low:.2f} to {high:.2f}. Choose low and high on the label scale "
            "as a modeling decision and state them in the write-up; the "
            "bounds below are this fit's label range, not advice. Then "
            "re-derive success_threshold on the transformed [0, 1] scale, "
            "because success_rate compares the transformed composite."
        )
        transform_block = (
            f'      "transform": {{"type": "clipped_linear", '
            f'"low": {low:.1f}, "high": {high:.1f}}},\n'
        )
    print(
        "\nUse it by adding this to the 'oracles' block of a goal_directed config:\n"
        f'  "{args.output.stem}": {{\n'
        f'      "type": "sklearn",\n'
        f'      "path": "{args.output}",\n'
        f'      "output": "{oracle_type}",\n'
        f"{transform_block}"
        f'      "direction": "maximize"\n'
        "  }\n"
        "and naming it in 'score_components'. The featurization is stored in "
        "the pickle and the oracle reads it, so there is nothing to keep in "
        f"sync by hand.\n\n{transform_advice}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
