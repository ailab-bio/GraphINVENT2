"""Cached oracle wrapper with deduplication and call counting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._base import BaseOracle


class CachedOracle:
    """
    Wraps any oracle with result caching, deduplication, and call counting.

    Oracle calls are only made for SMILES strings not seen before; previously
    evaluated molecules are returned from the in-memory cache without
    incrementing the call counter.  This matches the PMO benchmark protocol of
    counting unique molecule evaluations only.

    Parameters
    ----------
    oracle : BaseOracle
        The oracle to wrap.

    Examples
    --------
    >>> from oracles import OracleFactory
    >>> cached = OracleFactory.create_cached("DRD2")
    >>> scores = cached(["CCO", "c1ccccc1"])
    >>> print(cached.call_count)
    2
    >>> scores_again = cached(["CCO"])   # cached — call_count stays at 2
    >>> print(cached.call_count)
    2
    """

    def __init__(self, oracle: "BaseOracle") -> None:
        self._oracle = oracle
        self._cache: dict[str, float] = {}
        self._call_count: int = 0
        # Chronological log of (cumulative_call_count, score) for newly evaluated molecules
        self._optimization_log: list[tuple[int, float]] = []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def __call__(self, smiles: list) -> list:
        """
        Score a list of SMILES, using the cache for previously seen molecules.

        Parameters
        ----------
        smiles : list of str or None
            Input SMILES.  None entries receive score 0.0 without oracle call.

        Returns
        -------
        list of float
            Scores in [0, 1].
        """
        # Identify new, unique, non-None SMILES that need oracle calls
        new_smiles = [s for s in smiles if s is not None and s not in self._cache]
        # deduplicate while preserving order for the oracle call
        seen: set = set()
        unique_new: list = []
        for s in new_smiles:
            if s not in seen:
                unique_new.append(s)
                seen.add(s)

        if unique_new:
            new_scores = self._oracle(unique_new)
            for smi, score in zip(unique_new, new_scores):
                self._cache[smi] = float(score)
                self._call_count += 1
                self._optimization_log.append((self._call_count, float(score)))

        return [self._cache.get(s, 0.0) if s is not None else 0.0 for s in smiles]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total number of unique oracle calls made so far."""
        return self._call_count

    @property
    def optimization_log(self) -> list:
        """
        Chronological record of oracle evaluations.

        Returns
        -------
        list of (cumulative_call_count, score)
            One entry per unique molecule evaluated so far.
        """
        return list(self._optimization_log)

    @property
    def name(self) -> str:
        """Delegate to the wrapped oracle's name."""
        return self._oracle.name

    def reset(self) -> None:
        """Clear the cache and reset call counters (useful between runs)."""
        self._cache.clear()
        self._call_count = 0
        self._optimization_log.clear()
