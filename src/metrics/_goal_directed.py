"""
Goal-directed molecular generation metrics.

Extends evaluate_conditional with oracle-efficiency metrics.
"""

from __future__ import annotations

from typing import Callable

from ._conditional import evaluate_conditional
from ._criteria import SuccessCriterion


def evaluate_goal_directed(
    mols: list,
    reference_mols: list,
    criteria: list[SuccessCriterion],
    oracle_calls: int,
    *,
    training_smiles: set[str] | None = None,
    subsample: int = 1000,
    include_fcd: bool = False,
    property_fns: dict[str, Callable] | None = None,
) -> dict[str, float | None]:
    """
    Compute goal-directed molecular generation metrics.

    Extends :func:`evaluate_conditional` with oracle-efficiency metrics.

    Parameters
    ----------
    mols : list
        Generated molecules (SMILES strings or RDKit Mol objects).
    reference_mols : list
        Reference molecules (SMILES strings or RDKit Mol objects) for FCD.
    criteria : list of SuccessCriterion
        Success criteria (all must be satisfied for a molecule to succeed).
    oracle_calls : int
        Number of oracle (scoring function) calls made to generate mols.
    training_smiles : set of str, optional
        Training-set canonical SMILES for novelty computation.
    subsample : int
        Max molecules used for diversity computation (default 1000).
    include_fcd : bool
        Whether to compute FCD (requires fcd_torch). Default False.
    property_fns : dict, optional
        Extra property functions {name: callable(Mol) -> float}.

    Returns
    -------
    dict — all keys from evaluate_conditional, plus:
        'oracle_calls'      : int
        'sample_efficiency' : None  (not yet implemented)

    Notes
    -----
    ``'sample_efficiency'`` is not yet implemented.  It is intended to measure
    how many oracle calls are needed to find a given number of successful
    molecules.  Currently returns None.

    Examples
    --------
    >>> from metrics import evaluate_goal_directed, SuccessCriterion
    >>> criteria = [SuccessCriterion(property='qed', type='threshold',
    ...                              value=0.6, direction='greater')]
    >>> results = evaluate_goal_directed(mols, ref_mols, criteria,
    ...                                  oracle_calls=10000)
    """
    results: dict[str, float | None] = evaluate_conditional(
        mols,
        reference_mols,
        criteria,
        training_smiles=training_smiles,
        subsample=subsample,
        include_fcd=include_fcd,
        property_fns=property_fns,
    )

    results["oracle_calls"] = oracle_calls  # type: ignore[assignment]
    # sample_efficiency: stub — not yet implemented
    results["sample_efficiency"] = None  # TODO: implement efficiency metric

    return results
