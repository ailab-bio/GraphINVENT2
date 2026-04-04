"""Abstract base class for all oracles."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseOracle(ABC):
    """
    Abstract base class for all scoring oracles.

    Implement this to add a custom oracle that is not available in TDC.

    Parameters
    ----------
    name : str
        Human-readable oracle name returned by the ``name`` property.

    Examples
    --------
    >>> class MyOracle(BaseOracle):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_oracle"
    ...     def __call__(self, smiles: list[str]) -> list[float]:
    ...         return [0.5] * len(smiles)  # dummy implementation
    """

    @abstractmethod
    def __call__(self, smiles: list[str]) -> list[float]:
        """
        Score a list of SMILES strings.

        Parameters
        ----------
        smiles : list of str
            Input SMILES.  Invalid or None entries should return 0.0.

        Returns
        -------
        list of float
            Scores in [0, 1], one per input SMILES.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable oracle name (used as a cache key and log label)."""
