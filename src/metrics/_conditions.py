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
