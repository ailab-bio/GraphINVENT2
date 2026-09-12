"""Abstract base class for scoring oracles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

from ._transform import apply_direction, build_transform


class BaseOracle(ABC):
    """
    Base class for anything that scores molecules during goal-directed generation.

    An oracle reports its *native* quantity from :meth:`predict` -- a docking
    energy in kcal/mol, a predicted pIC50, a classifier probability -- and the
    base class converts that to the bounded desirability the RL objective needs,
    via the configured transform and direction.  Keeping the two apart means a
    surrogate can be swapped between a "maximise affinity" and an "avoid
    off-target" role without retraining or rewriting it.

    Subclasses implement :meth:`predict`, and optionally
    :meth:`predict_with_uncertainty` when the underlying model can report a
    predictive spread (an ensemble, a Gaussian process, repeated docking runs).
    Uncertainty is used to damp the reward in regions where the surrogate is
    not trustworthy; an oracle that cannot estimate it simply does not
    implement the method.

    Parameters
    ----------
    name
        Label used in logs, as a cache key, and to reference this oracle from
        ``score_components``.
    transform
        Transform spec (see :mod:`._transform`).  ``None`` means the oracle
        already returns a value in [0, 1].
    direction
        ``"maximize"`` (default) or ``"minimize"``.  Minimising expresses an
        anti-target: the score is inverted after the transform.

    Examples
    --------
    >>> class ConstantOracle(BaseOracle):
    ...     def predict(self, smiles):
    ...         return [7.5] * len(smiles)
    >>> oracle = ConstantOracle(
    ...     name="demo", transform={"type": "clipped_linear", "low": 5, "high": 9}
    ... )
    >>> oracle(["CCO"])
    [0.625]
    """

    def __init__(
        self,
        name: str,
        transform: Optional[dict] = None,
        direction: str = "maximize",
    ) -> None:
        self._name = name
        self._transform = build_transform(transform)
        self._direction = direction
        # Fail now rather than mid-run on the first scored batch.
        apply_direction(0.0, direction)

    # ------------------------------------------------------------------
    # To implement in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def predict(self, smiles: Sequence[Optional[str]]) -> List[float]:
        """
        Return the oracle's native value for each input SMILES.

        Implementations must return one value per input, in order, and must
        tolerate ``None`` and unparseable entries: return the worst plausible
        value for those rather than raising, since invalid molecules are a
        normal occurrence during generation.
        """

    def predict_with_uncertainty(
        self, smiles: Sequence[Optional[str]]
    ) -> Tuple[List[float], List[float]]:
        """
        Return native values and a per-molecule uncertainty on the same scale.

        Raises
        ------
        NotImplementedError
            When the underlying model provides no predictive spread.  Callers
            should treat this as "no uncertainty information available" rather
            than as an error.
        """
        raise NotImplementedError(
            f"Oracle '{self.name}' does not provide uncertainty estimates."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, smiles: Sequence[Optional[str]]) -> List[float]:
        """Score molecules, returning desirabilities in [0, 1]."""
        return [self.score_from_raw(v) for v in self.predict(smiles)]

    def score_from_raw(self, value: float) -> float:
        """Apply the configured transform and direction to one native value."""
        return apply_direction(self._transform(value), self._direction)

    @property
    def name(self) -> str:
        """Oracle label, used in logs and as a cache key."""
        return self._name

    @property
    def direction(self) -> str:
        """``"maximize"`` or ``"minimize"``."""
        return self._direction

    @property
    def supports_uncertainty(self) -> bool:
        """
        Whether :meth:`predict_with_uncertainty` is implemented.

        Checked by identity against the base implementation so a subclass only
        has to override the method, with nothing else to keep in sync.
        """
        return (
            type(self).predict_with_uncertainty
            is not BaseOracle.predict_with_uncertainty
        )
