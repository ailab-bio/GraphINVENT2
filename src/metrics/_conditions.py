"""
Conditional generation metrics, continuous and categorical.

Every function here takes the conditional sample **and** the unconditional
baseline sample, because CLAUDE.md requires each conditional number to be
reported next to the baseline: the same model trained without conditioning and
sampled the same way. A conditional number with no baseline cannot be read,
since nothing separates the conditioning from the base rate of the corpus. The
baseline is a second sample, so it is a second argument, not a flag.
"""

from __future__ import annotations

import numpy as np

from ._core import ecfp4

# CLAUDE.md: condition_coverage_<property> is a per-decile success rate across
# the training property distribution, reported as ten numbers.
N_DECILES = 10


class MissingInput(Exception):
    """A metric's input is absent. The message says which and where from."""


# ---------------------------------------------------------------------------
# Continuous properties
# ---------------------------------------------------------------------------


def condition_success_rate(
    requested: list, achieved: list, tolerance: float
) -> float | None:
    """
    Fraction of generated molecules whose achieved property is within
    `tolerance` of the value that was requested of them.

    The denominator is every molecule for which both a request and an achieved
    value exist, which excludes the invalid ones: a molecule that is not a
    molecule has no property to compare.
    """
    pairs = [
        (r, a)
        for r, a in zip(requested, achieved)
        if r is not None and a is not None and np.isfinite(a)
    ]
    if not pairs:
        return None
    return sum(1 for r, a in pairs if abs(a - r) <= tolerance) / len(pairs)


def condition_mae(requested: list, achieved: list) -> float | None:
    """
    `condition_mae_<property>`: the mean absolute difference between the value
    requested of a molecule and the value it achieved, in the property's own
    units.

    This is reported beside `condition_success_rate_<property>` because that
    rate is a step function of `condition_tolerance`. Two models whose typical
    error differs threefold score the same rate when both errors sit inside the
    tolerance, and a model that halves its error moves the rate not at all
    until it crosses the threshold. The rate answers whether the target was
    hit; this answers by how much it was missed, and no choice of tolerance
    after the fact can flatter it.

    The denominator is the same as the rate's: molecules with both a request and
    a finite achieved value. Comparable across models conditioned on the same
    property, not across properties, since the units are the property's own.
    """
    errors = [
        abs(a - r)
        for r, a in zip(requested, achieved)
        if r is not None and a is not None and np.isfinite(a)
    ]
    if not errors:
        return None
    return float(np.mean(errors))


def condition_coverage(
    requested: list,
    achieved: list,
    tolerance: float,
    training_values: list,
) -> list:
    """
    `condition_coverage_<property>`: the success rate in each decile of the
    training property distribution, as ten numbers.

    A model can look good on `condition_success_rate` while only working near
    the middle of the training distribution, which is where most requests land.
    Splitting by training decile is what exposes that. A decile with no
    requests in it records None rather than zero, so an untested decile is not
    read as a failed one.
    """
    finite_training = [v for v in training_values if v is not None and np.isfinite(v)]
    if len(finite_training) < N_DECILES:
        raise MissingInput(
            f"condition_coverage needs the training property distribution, and "
            f"got {len(finite_training)} usable values for {N_DECILES} deciles."
        )

    # Edges from the training distribution, not from the requests: the deciles
    # are a property of the dataset, so they stay fixed across runs.
    edges = np.quantile(finite_training, np.linspace(0, 1, N_DECILES + 1))

    coverage: list = []
    for i in range(N_DECILES):
        low, high = edges[i], edges[i + 1]
        bucket = [
            (r, a)
            for r, a in zip(requested, achieved)
            if r is not None and a is not None and np.isfinite(a)
            # The last decile has to include its upper edge or the single
            # largest training value falls outside every bucket.
            and (low <= r < high or (i == N_DECILES - 1 and r == high))
        ]
        if not bucket:
            coverage.append(None)
        else:
            coverage.append(
                sum(1 for r, a in bucket if abs(a - r) <= tolerance) / len(bucket)
            )
    return coverage


def compute_continuous_condition_metrics(
    property_name: str,
    requested: list,
    achieved: list,
    tolerance: float,
    training_values: list,
    *,
    baseline_requested: list | None = None,
    baseline_achieved: list | None = None,
) -> dict:
    """
    The continuous-property conditional table for one property, with its
    unconditional baseline.

    `baseline_requested` is the set of values the baseline sample is *scored
    against* -- the baseline model was not conditioned, so nothing was requested
    of it, and the comparison is made by asking the same questions of it.
    """
    # The success rate and the coverage are computed separately because they
    # need different things: the rate needs only the requests and what was
    # achieved, while the coverage also needs the training distribution to cut
    # into deciles. A dataset too small for ten deciles must not take the rate
    # down with it.
    results: dict = {
        f"condition_success_rate_{property_name}": condition_success_rate(
            requested, achieved, tolerance
        ),
        f"condition_mae_{property_name}": condition_mae(requested, achieved),
    }
    coverage_error: str | None = None
    try:
        results[f"condition_coverage_{property_name}"] = condition_coverage(
            requested, achieved, tolerance, training_values
        )
    except MissingInput as error:
        results[f"condition_coverage_{property_name}"] = None
        coverage_error = " ".join(str(error).split())

    if baseline_requested is not None and baseline_achieved is not None:
        results[f"baseline_condition_success_rate_{property_name}"] = (
            condition_success_rate(baseline_requested, baseline_achieved, tolerance)
        )
        results[f"baseline_condition_mae_{property_name}"] = condition_mae(
            baseline_requested, baseline_achieved
        )
        if coverage_error is None:
            results[f"baseline_condition_coverage_{property_name}"] = (
                condition_coverage(
                    baseline_requested, baseline_achieved, tolerance, training_values
                )
            )
    if coverage_error is not None:
        results["_coverage_unavailable_reason"] = coverage_error
    return results


# ---------------------------------------------------------------------------
# Categorical labels
# ---------------------------------------------------------------------------


def class_match_metrics(
    generated_smiles: list,
    requested_labels: list,
    training_by_label: dict,
    *,
    threshold: float,
) -> dict:
    """
    `class_match_median`, `class_match_rate` and `class_selectivity_margin`.

    Parameters
    ----------
    training_by_label
        {label: [training SMILES carrying that label]}. **The training split**,
        not the test split: these metrics ask how close the generated molecule
        is to the chemistry the model was taught for that class. Using held-out
        molecules instead would measure something else entirely while still
        producing plausible numbers.
    threshold
        `class_match_rate` is the fraction above this maximum Tanimoto.
    """
    from rdkit import Chem, DataStructs

    if not training_by_label:
        raise MissingInput(
            "class_match_* needs training molecules grouped by label, and got "
            "an empty mapping. Supply {label: [training smiles]} from the "
            "training split."
        )

    fps_by_label = {}
    for label, smiles in training_by_label.items():
        fps = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fps.append(ecfp4(mol))
        fps_by_label[label] = fps

    on_class: list = []
    margins: list = []
    for smi, label in zip(generated_smiles, requested_labels):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None or label not in fps_by_label:
            continue
        fp = ecfp4(mol)

        own = fps_by_label[label]
        own_max = max(DataStructs.BulkTanimotoSimilarity(fp, own)) if own else None
        if own_max is None:
            continue
        on_class.append(own_max)

        # The same maximum against the other classes in the panel, subtracted
        # from the on-class value. Self-normalizing: an unconditional model
        # scores about zero here by construction, and the metric cannot be
        # satisfied by generating generic chemistry.
        others = [f for other, fs in fps_by_label.items() if other != label for f in fs]
        if others:
            off_max = max(DataStructs.BulkTanimotoSimilarity(fp, others))
            margins.append(own_max - off_max)

    if not on_class:
        return {
            "class_match_median": None,
            "class_match_rate": None,
            "class_selectivity_margin": None,
        }
    return {
        "class_match_median": float(np.median(on_class)),
        "class_match_rate": sum(1 for v in on_class if v > threshold) / len(on_class),
        "class_selectivity_margin": (float(np.mean(margins)) if margins else None),
    }


def knn_purity(
    generated_smiles: list,
    requested_labels: list,
    reference_pool: list,
    *,
    k_values: tuple = (5, 10, 20),
) -> dict:
    """
    `knn_purity_<k>`: of each generated molecule's k nearest neighbors in the
    held-out reference pool, by Morgan-Tanimoto, the fraction carrying the
    requested label.

    Parameters
    ----------
    reference_pool
        (smiles, label) pairs from **the test split**, which is what CLAUDE.md
        means by the held-out reference pool. Its size is returned as
        `knn_reference_pool_size`, because a purity at k = 20 computed against
        a pool of 25 molecules is not measuring what the name suggests and a
        later reader needs to be able to see that.
    """
    from rdkit import Chem, DataStructs

    if not reference_pool:
        raise MissingInput(
            "knn_purity needs a labelled reference pool of (smiles, label) "
            "pairs from the test split, and got none."
        )

    pool_fps = []
    pool_labels = []
    for smi, label in reference_pool:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            pool_fps.append(ecfp4(mol))
            pool_labels.append(label)

    results: dict = {"knn_reference_pool_size": len(pool_fps)}
    if not pool_fps:
        for k in k_values:
            results[f"knn_purity_{k}"] = None
        return results

    per_k: dict = {k: [] for k in k_values}
    for smi, label in zip(generated_smiles, requested_labels):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(ecfp4(mol), pool_fps)
        order = np.argsort(sims)[::-1]
        for k in k_values:
            top = order[:k]
            if len(top) == 0:
                continue
            per_k[k].append(sum(1 for i in top if pool_labels[i] == label) / len(top))

    for k in k_values:
        results[f"knn_purity_{k}"] = float(np.mean(per_k[k])) if per_k[k] else None
    return results


def delta_nll(
    per_condition_nll: dict,
    true_labels: list,
    *,
    alternative_labels: dict | None = None,
) -> float | None:
    """
    `delta_nll`: for a held-out molecule of class p, the mean over competing
    classes q of NLL(x|q) - NLL(x|p). Positive means the molecule is more
    likely under its own condition.

    This is a **scoring** path, not a sampling one: it needs the model's NLL for
    a molecule it did not generate, evaluated once per candidate class. The
    caller supplies those numbers; this function only aggregates them.

    Parameters
    ----------
    per_condition_nll
        {molecule_index: {label: nll}} -- one NLL per molecule per class.
    true_labels
        The class each molecule actually belongs to, indexed as above.
    alternative_labels
        {molecule_index: set of labels that are also legitimate for it}.
        Excluded from the competing set, so a multi-label molecule is not
        penalized for being likely under a class it genuinely belongs to.
    """
    if not per_condition_nll:
        raise MissingInput(
            "delta_nll needs NLL(x|condition) for each held-out molecule under "
            "each class, which requires scoring a given molecule under a given "
            "condition. Supply {index: {label: nll}}."
        )

    deltas: list = []
    for index, by_label in per_condition_nll.items():
        true_label = true_labels[index] if index < len(true_labels) else None
        if true_label is None or true_label not in by_label:
            continue
        excluded = {true_label}
        if alternative_labels and index in alternative_labels:
            excluded |= set(alternative_labels[index])
        competing = [nll for label, nll in by_label.items() if label not in excluded]
        if not competing:
            continue
        deltas.append(float(np.mean(competing)) - by_label[true_label])
    return float(np.mean(deltas)) if deltas else None


# ---------------------------------------------------------------------------
# Categorical labels: the whole block, with its baselines
# ---------------------------------------------------------------------------
#
# CLAUDE.md reports the headline as a macro-average across the classes in the
# panel, with the per-class breakdown kept flat under a `_<class>` suffix and
# written in full to `results/condition_per_class.json`. Macro rather than
# micro, because a panel is almost never balanced: the classes differ by an
# order of magnitude in how many training molecules they have, and a
# micro-average reports the largest class's number under a name that reads as
# the panel's.
#
# Two baselines, because the metrics alone are not evidence:
#
#   LOOKUP      the training molecules carrying the requested label, sampled to
#               the same N. This is the retrieval ceiling. A conditional model
#               cannot beat it on `class_match_*` by construction -- the lookup
#               set *is* training chemistry -- so the comparison that matters is
#               novelty: matching as well as retrieval while being novel is the
#               claim, and matching as well while not being novel is retrieval.
#
#   POST-HOC    the unconditional model sampled the same way, then reranked per
#               class by similarity to that class's training warheads and cut to
#               the same N. This is the one that can falsify the claim. If an
#               unconditional model plus a similarity filter reaches the same
#               `class_selectivity_margin`, the conditioning bought nothing:
#               the same molecules were in the unconditional sample and a
#               one-line reranking found them.


def requested_label(one_hot: dict, class_names: list) -> str | None:
    """
    The class a one-hot request asks for, or None when it asks for no class.

    A categorical condition is a one-hot over numeric columns, which is what
    the HDF5 carries and what the model was trained on; the class is the column
    set to one. More than one nonzero is a multi-label request, which the
    categorical metrics have no definition for, so it is None rather than a
    guess at which was meant.
    """
    hot = [name for name in class_names if float(one_hot.get(name, 0.0) or 0.0) > 0.5]
    return hot[0] if len(hot) == 1 else None


def _macro(per_class: dict, key: str) -> float | None:
    """The mean over the classes that produced a number for `key`."""
    values = [row[key] for row in per_class.values() if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def categorical_block(
    generated_smiles: list,
    requested_labels: list,
    class_names: list,
    training_by_label: dict,
    reference_pool: list,
    *,
    threshold: float,
    training_smiles: list | None = None,
    isomeric: bool = False,
) -> tuple:
    """
    One macro-averaged categorical metric block, with its per-class rows.

    Returns `(metrics, unavailable, per_class)`. The same function computes the
    conditional sample and both baselines, so the three have identical shape by
    construction rather than by a reviewer checking.
    """
    from ._core import canonical_smiles

    per_class: dict = {}
    metrics: dict = {}
    unavailable: dict = {}

    by_label: dict = {name: [] for name in class_names}
    for smi, label in zip(generated_smiles, requested_labels):
        if label in by_label and smi:
            by_label[label].append(smi)

    reference = (
        {s for s in canonical_smiles(training_smiles, isomeric=isomeric) if s}
        if training_smiles
        else set()
    )

    for name in class_names:
        molecules = by_label[name]
        row: dict = {
            "n_unique_valid": len(set(molecules)),
            "class_match_median": None,
            "class_match_rate": None,
            "class_selectivity_margin": None,
            "novelty": None,
        }
        if molecules:
            row.update(
                class_match_metrics(
                    molecules,
                    [name] * len(molecules),
                    training_by_label,
                    threshold=threshold,
                )
            )
            unique = sorted(set(molecules))
            if reference:
                row["novelty"] = sum(1 for s in unique if s not in reference) / len(
                    unique
                )
            purity = knn_purity(molecules, [name] * len(molecules), reference_pool)
            row.update({k: v for k, v in purity.items() if k.startswith("knn_purity")})
        per_class[name] = row

    k_keys = sorted(
        {k for row in per_class.values() for k in row if k.startswith("knn_purity_")}
    )
    for key in (
        "class_match_median",
        "class_match_rate",
        "class_selectivity_margin",
        "novelty",
        *k_keys,
    ):
        metrics[key] = _macro(per_class, key)
        if metrics[key] is None:
            unavailable[key] = (
                "insufficient_sample: no class in the panel produced a value "
                f"for {key}; every class needs at least one valid molecule and "
                "training molecules carrying its label"
            )
        for name, row in per_class.items():
            metrics[f"{key}_{name}"] = row.get(key)

    metrics["knn_reference_pool_size"] = len(reference_pool)
    metrics["class_match_threshold"] = threshold
    metrics["class_panel"] = list(class_names)
    for name, row in per_class.items():
        metrics[f"n_unique_valid_{name}"] = row["n_unique_valid"]
    return metrics, unavailable, per_class


def posthoc_filter_baseline(
    unconditional_smiles: list,
    class_names: list,
    training_by_label: dict,
    counts: dict,
) -> tuple:
    """
    The unconditional sample reranked per class, cut to the same N per class.

    Returns `(smiles, labels)` in the shape `categorical_block` takes. Each
    class takes the `counts[class]` unconditional molecules most similar to
    that class's training molecules, so the baseline spends the same sample the
    conditional model did and answers the question the conditional claim has to
    beat: could a filter have found these without conditioning.

    A molecule may be selected for more than one class, which is correct. The
    baseline is not a partition of the unconditional sample; it is what the
    best post-hoc filter for each class would return.
    """
    from rdkit import Chem, DataStructs

    fps: list = []
    kept: list = []
    for smi in unconditional_smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            fps.append(ecfp4(mol))
            kept.append(smi)
    if not kept:
        raise MissingInput(
            "the post-hoc filter baseline needs an unconditional sample to "
            "rerank, and the declared baseline .smi parsed to nothing."
        )

    out_smiles: list = []
    out_labels: list = []
    for name in class_names:
        want = int(counts.get(name, 0))
        if want <= 0:
            continue
        reference = []
        for smi in training_by_label.get(name, []):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                reference.append(ecfp4(mol))
        if not reference:
            continue
        scores = [max(DataStructs.BulkTanimotoSimilarity(fp, reference)) for fp in fps]
        order = np.argsort(scores)[::-1][:want]
        for index in order:
            out_smiles.append(kept[index])
            out_labels.append(name)
    return out_smiles, out_labels
