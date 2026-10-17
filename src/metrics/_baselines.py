"""
The comparisons a generative result has to survive to mean anything: the
virtual screening baseline, the evaluation surrogate, and the applicability
domain check.

Each of these needs an input that does not exist in the repository yet, so each
raises `MissingInput` naming exactly what it is waiting for. The interfaces are
here so that whoever supplies the input does not also have to design them.
"""

from __future__ import annotations

from ._budget import BUDGETS, _top_mean
from ._core import ecfp4


class MissingInput(Exception):
    """A metric's input is absent. The message says which and where from."""


def screening_baseline_metrics(
    library_scores_at_budget: dict | None,
    generative_top10_at_budget: dict,
) -> dict:
    """
    `vs_baseline_top10_at_<...>` and `vs_delta_top10_at_<...>`.

    A generative method that does not beat random library screening at matched
    budget has not demonstrated anything, so this runs for every optimization
    experiment. `vs_delta_*` is the headline comparison.

    Parameters
    ----------
    library_scores_at_budget
        {'1k': [scores], '5k': [...], '10k': [...]} -- the oracle scores of B
        molecules drawn uniformly at random from a declared screening library,
        with B the matching budget, scored by the **same** oracle under the
        **same** seeds as the generative run. At minimum the library is a
        random matched-size sample from the training and validation splits; an
        experiment may declare more, and all declared libraries are reported.
    generative_top10_at_budget
        {'1k': top10_mean, ...} from the generative run, for the subtraction.

    Raises
    ------
    MissingInput
        When no library has been declared.
    """
    if not library_scores_at_budget:
        raise MissingInput(
            "vs_baseline_top10_at_* needs a declared screening library: B "
            "molecules drawn uniformly at random from it, scored by the same "
            "oracle under the same seeds, for each budget in "
            f"{sorted(BUDGETS)}.\n"
            "No experiment declares one yet. Supply "
            "{'1k': [scores], '5k': [...], '10k': [...]}."
        )

    results: dict = {}
    for label in BUDGETS:
        scores = library_scores_at_budget.get(label)
        baseline = _top_mean(list(scores), 10) if scores else None
        results[f"vs_baseline_top10_at_{label}"] = baseline
        generative = generative_top10_at_budget.get(label)
        results[f"vs_delta_top10_at_{label}"] = (
            generative - baseline
            if baseline is not None and generative is not None
            else None
        )
    return results


def evaluation_surrogate_metrics(
    top10_smiles: list,
    evaluation_oracle=None,
) -> dict:
    """
    `eval_oracle_top10_mean`: the same top-10 molecules rescored by the
    evaluation oracle.

    CLAUDE.md trains two surrogates. The *optimization* surrogate is fitted on
    train and validation only and provides the RL reward; the *evaluation*
    surrogate is the same featurization and model class refitted on all data
    including test, and only scores the final molecules. A large gap on
    optimized molecules, when the two agree on held-out real molecules, is
    evidence of exploitation rather than success.

    Parameters
    ----------
    evaluation_oracle
        A callable taking a list of SMILES and returning scores. This is the
        second surrogate, not the one that produced the reward.
    """
    if evaluation_oracle is None:
        raise MissingInput(
            "eval_oracle_top10_mean needs the evaluation surrogate: the same "
            "featurization and model class as the optimization surrogate, "
            "refitted on all data including the test split, used only to score "
            "final molecules.\n"
            "No surrogate pair is trained yet. Supply a callable taking a list "
            "of SMILES and returning scores."
        )
    if not top10_smiles:
        return {"eval_oracle_top10_mean": None}
    scores = list(evaluation_oracle(list(top10_smiles)))
    return {"eval_oracle_top10_mean": _top_mean(scores, 10)}


def applicability_domain_metrics(
    top100_smiles: list,
    surrogate_training_smiles: list | None,
    *,
    similarity_floor: float,
) -> dict:
    """
    `top100_outside_domain_rate`: the fraction of the top 100 whose maximum
    ECFP4 Tanimoto to the surrogate's training set falls below `similarity_floor`.

    This catches the same failure as the surrogate pair from a direction neither
    surrogate shares, since both have the same blind spots: a molecule far from
    everything the surrogate ever saw has a score that is an extrapolation.

    Parameters
    ----------
    surrogate_training_smiles
        The molecules the **optimization** surrogate was fitted on. Not the
        generative model's training split, which is a different set.
    similarity_floor
        The stated floor. Recorded in the output, since the rate is meaningless
        without it.
    """
    if not surrogate_training_smiles:
        raise MissingInput(
            "top100_outside_domain_rate needs the optimization surrogate's own "
            "training set, to measure how far the top-scoring molecules sit "
            "from anything it has seen.\n"
            "No surrogate is trained yet. Supply its training SMILES."
        )
    from rdkit import Chem, DataStructs

    reference_fps = []
    for smi in surrogate_training_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            reference_fps.append(ecfp4(mol))
    if not reference_fps:
        raise MissingInput(
            "top100_outside_domain_rate: none of the surrogate's training "
            "SMILES could be parsed."
        )

    outside = 0
    compared = 0
    for smi in top100_smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        compared += 1
        sims = DataStructs.BulkTanimotoSimilarity(ecfp4(mol), reference_fps)
        if not sims or max(sims) < similarity_floor:
            outside += 1
    return {
        "top100_outside_domain_rate": outside / compared if compared else None,
        "outside_domain_similarity_floor": similarity_floor,
    }
