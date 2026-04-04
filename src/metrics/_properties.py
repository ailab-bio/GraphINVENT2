"""
Built-in molecular property computers for the src.metrics package.

Each function accepts an RDKit Mol and returns a float.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

from rdkit.Chem import Mol

# ---------------------------------------------------------------------------
# Module-level cache for the lazily-loaded sascorer
# ---------------------------------------------------------------------------
_sascorer = None


def _get_sascorer():
    """Lazily import sascorer from RDKit contrib (mirrors graphinvent/metrics.py)."""
    global _sascorer
    if _sascorer is None:
        from rdkit import RDConfig

        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer as _sa

        _sascorer = _sa
    return _sascorer


# ---------------------------------------------------------------------------
# Property functions
# ---------------------------------------------------------------------------


def qed(mol: Mol) -> float:
    """Quantitative Estimate of Drug-likeness in [0, 1]."""
    from rdkit.Chem import QED as _QED

    return float(_QED.qed(mol))


def sa_score(mol: Mol) -> float:
    """Synthetic Accessibility score in [1, 10] (lower is more accessible)."""
    sascorer = _get_sascorer()
    return float(sascorer.calculateScore(mol))


def mol_weight(mol: Mol) -> float:
    """Molecular weight (Da)."""
    from rdkit.Chem import Descriptors

    return float(Descriptors.MolWt(mol))


def logp(mol: Mol) -> float:
    """Wildman-Crippen LogP."""
    from rdkit.Chem import Descriptors

    return float(Descriptors.MolLogP(mol))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROPERTY_REGISTRY: dict[str, Callable[[Mol], float]] = {
    "qed": qed,
    "sa_score": sa_score,
    "mol_weight": mol_weight,
    "logp": logp,
}
