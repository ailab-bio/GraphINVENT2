"""
Utility helpers for the src.metrics package.

Provides type-normalisation functions that accept either SMILES strings or
RDKit Mol objects and return a uniform list of Mol / canonical-SMILES / None.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Mol


def to_mols(inputs: list) -> list[Mol | None]:
    """
    Convert a mixed list of SMILES strings and/or Mol objects to a list of Mol.

    Parameters
    ----------
    inputs : list
        Each element may be a SMILES string, an RDKit Mol object, or None.

    Returns
    -------
    list of Mol or None
        None at positions where the input was None or where SMILES parsing
        failed.
    """
    result: list[Mol | None] = []
    for item in inputs:
        if item is None:
            result.append(None)
        elif isinstance(item, Mol):
            result.append(item)
        else:
            # treat as SMILES string
            try:
                mol = Chem.MolFromSmiles(str(item))
            except Exception:
                mol = None
            result.append(mol)
    return result


def to_smiles(inputs: list) -> list[str | None]:
    """
    Convert a mixed list of SMILES strings and/or Mol objects to canonical SMILES.

    Parameters
    ----------
    inputs : list
        Each element may be a SMILES string, an RDKit Mol object, or None.

    Returns
    -------
    list of str or None
        Canonical SMILES where valid, None otherwise.
    """
    result: list[str | None] = []
    for item in inputs:
        if item is None:
            result.append(None)
            continue
        if isinstance(item, Mol):
            mol = item
        else:
            try:
                mol = Chem.MolFromSmiles(str(item))
            except Exception:
                mol = None
        if mol is None:
            result.append(None)
        else:
            try:
                result.append(Chem.MolToSmiles(mol))
            except Exception:
                result.append(None)
    return result
