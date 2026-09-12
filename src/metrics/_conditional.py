"""
Conditional molecular generation metrics.

Extends evaluate_unconditional with success-rate metrics based on a list of
SuccessCriterion objects.
"""

from __future__ import annotations

from typing import Callable

from ._criteria import SuccessCriterion, molecule_passes
from ._unconditional import evaluate_unconditional
from ._utils import to_mols, to_smiles


def evaluate_conditional(
    mols: list,
    reference_mols: list,
    criteria: list[SuccessCriterion],
    *,
    training_smiles: set[str] | None = None,
    subsample: int = 1000,
    include_fcd: bool = False,
    property_fns: dict[str, Callable] | None = None,
) -> dict[str, float | None]:
    """
    Compute conditional molecular generation metrics.

    Extends :func:`evaluate_unconditional` with success-rate metrics.

    Parameters
    ----------
    mols : list
        Generated molecules (SMILES strings or RDKit Mol objects).
    reference_mols : list
        Reference molecules (SMILES strings or RDKit Mol objects) for FCD.
    criteria : list of SuccessCriterion
        Success criteria (all must be satisfied for a molecule to succeed).
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
    dict — all keys from evaluate_unconditional, plus:
        'success_rate'    : float in [0, 1]  (fraction of valid mols passing all criteria)
        'conditional_vun' : float in [0, 1] or None
                            (validity x success_rate x uniqueness x novelty)
        'rediscovery_rate': float or None (None if reference_mols is empty)

    Examples
    --------
    >>> from metrics import evaluate_conditional, SuccessCriterion
    >>> criteria = [SuccessCriterion(property='qed', type='threshold',
    ...                              value=0.5, direction='greater')]
    >>> results = evaluate_conditional(generated_smiles, ref_smiles, criteria)
    """
    # Base unconditional metrics
    results: dict[str, float | None] = evaluate_unconditional(
        mols,
        reference_mols,
        training_smiles=training_smiles,
        subsample=subsample,
        include_fcd=include_fcd,
    )

    # Resolve valid Mol objects
    rdkit_mols = to_mols(mols)
    valid_mols = [m for m in rdkit_mols if m is not None]

    # ------------------------------------------------------------------
    # Success rate: fraction of valid mols passing all criteria
    # ------------------------------------------------------------------
    if not valid_mols:
        success_rate = 0.0
    else:
        n_success = sum(
            1
            for mol in valid_mols
            if molecule_passes(mol, criteria, property_fns=property_fns)
        )
        success_rate = n_success / len(valid_mols)

    results["success_rate"] = success_rate

    # ------------------------------------------------------------------
    # Conditional VUN = success_rate x uniqueness x novelty
    # ------------------------------------------------------------------
    validity = results.get("validity") or 0.0
    uniqueness = results.get("uniqueness") or 0.0
    novelty = results.get("novelty")
    if novelty is None:
        results["conditional_vun"] = None
    else:
        # `success_rate` is a fraction *of the valid molecules*, so validity has
        # to be reinstated explicitly -- otherwise it cancels out and a 25%-valid
        # model scores the same conditional VUN as a 100%-valid one.
        results["conditional_vun"] = validity * success_rate * uniqueness * novelty

    # ------------------------------------------------------------------
    # Rediscovery rate
    # ------------------------------------------------------------------
    ref_smiles_canonical = {s for s in to_smiles(reference_mols) if s is not None}
    if not ref_smiles_canonical:
        results["rediscovery_rate"] = None
    else:
        from rdkit import Chem

        gen_smiles = set()
        for mol in valid_mols:
            try:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    gen_smiles.add(smi)
            except Exception:
                pass
        found = gen_smiles & ref_smiles_canonical
        results["rediscovery_rate"] = len(found) / len(ref_smiles_canonical)

    return results
