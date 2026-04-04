"""
TDC oracle wrapper for GraphINVENT2.

Supported named oracles (PMO benchmark, Gao et al. 2022):
    SA                   -- Synthetic Accessibility (RDKit; normalised to [0, 1])
    DRD2                 -- Dopamine Receptor D2 (SVM, ECFP6)
    GSK3B                -- Glycogen Synthase Kinase 3beta (RF, ECFP6)
    JNK3                 -- c-Jun N-terminal Kinase 3 (RF, ECFP6)
    celecoxib_rediscovery -- Tanimoto similarity to Celecoxib (GuacaMol)

Any other TDC oracle can be used by passing its TDC name directly to
``TDCOracle``.  The factory will also try arbitrary TDC names for
unrecognised oracle names.
"""

from __future__ import annotations

import os
from pathlib import Path

from ._base import BaseOracle

# ---------------------------------------------------------------------------
# Registry: GraphINVENT2 name -> TDC oracle name
# ---------------------------------------------------------------------------

#: Maps the oracle name used in params.json to the exact name expected by
#: ``tdc.Oracle``.  Oracle names in this registry are treated as first-class
#: citizens in :class:`OracleFactory`; all other names are forwarded to TDC
#: verbatim so that the full TDC oracle catalogue is available without code
#: changes.
ORACLE_REGISTRY: dict = {
    "SA": "SA",
    "DRD2": "DRD2",
    "GSK3B": "GSK3B",
    "JNK3": "JNK3",
    "celecoxib_rediscovery": "Celecoxib_Rediscovery",
}

# Canonical path for TDC surrogate model downloads
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SURROGATES_DIR = _REPO_ROOT / "data" / "surrogates"


class TDCOracle(BaseOracle):
    """
    Wraps a Therapeutics Data Commons (TDC) oracle.

    Surrogate model files are stored in ``data/surrogates/`` (the same
    directory used for user-provided QSAR models) by setting the
    ``TDC_HOME`` environment variable before TDC is imported.

    Parameters
    ----------
    name : str
        Oracle name as used in the GraphINVENT2 config.  Must be a key in
        :data:`ORACLE_REGISTRY`, or any valid TDC oracle name (passed through
        verbatim).

    Raises
    ------
    ImportError
        If ``PyTDC`` is not installed.
    ValueError
        If TDC raises an error while loading the oracle (e.g. unknown name).

    Examples
    --------
    >>> oracle = TDCOracle("DRD2")
    >>> scores = oracle(["CCO", "c1ccccc1"])
    >>> assert all(0.0 <= s <= 1.0 for s in scores)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        tdc_name = ORACLE_REGISTRY.get(name, name)

        # Point TDC to the project-managed surrogate cache
        _SURROGATES_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TDC_HOME", str(_SURROGATES_DIR))

        try:
            from tdc import Oracle as _TDCOracle  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyTDC is required for TDC oracle integration. "
                "Install with: pip install -e '.[tdc]'"
            ) from exc

        self._oracle = _TDCOracle(name=tdc_name)

    # ------------------------------------------------------------------

    def __call__(self, smiles: list) -> list:
        """
        Score a list of SMILES strings with the TDC oracle.

        Parameters
        ----------
        smiles : list of str or None
            Input SMILES.  None entries receive 0.0 without a model call.

        Returns
        -------
        list of float
            Scores in [0, 1].
        """
        results: list = []
        for smi in smiles:
            if smi is None:
                results.append(0.0)
                continue
            try:
                score = float(self._oracle(smi))
                # Clamp to [0, 1] in case of numerical noise
                score = max(0.0, min(1.0, score))
            except Exception:
                score = 0.0
            results.append(score)
        return results

    @property
    def name(self) -> str:
        return self._name
