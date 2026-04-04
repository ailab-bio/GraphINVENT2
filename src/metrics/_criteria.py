"""
Success criteria system for goal-directed and conditional generation evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rdkit.Chem import Mol

from ._properties import PROPERTY_REGISTRY


@dataclass
class SuccessCriterion:
    """
    A single success criterion for a molecular property.

    Parameters
    ----------
    property : str
        Name of the property.  Must be a key in PROPERTY_REGISTRY, or the
        caller must supply the function via the ``property_fns`` argument
        of :func:`is_satisfied` / :func:`molecule_passes`.
    type : str
        One of ``'threshold'``, ``'range'``, or ``'target'``.

    Threshold fields (used when type == 'threshold')
    -------------------------------------------------
    value : float
        The threshold value.
    direction : str
        ``'greater'`` — mol must have property > value.
        ``'less'``    — mol must have property < value.

    Range fields (used when type == 'range')
    ----------------------------------------
    min : float
        Lower bound (inclusive).
    max : float
        Upper bound (inclusive).

    Target fields (used when type == 'target')
    ------------------------------------------
    value : float
        The target value.
    tolerance : float
        Acceptable absolute deviation from the target (|prop - value| <= tolerance).
    """

    property: str
    type: str
    # threshold / target
    value: float | None = None
    direction: str | None = None  # 'greater' | 'less'
    # range
    min: float | None = None
    max: float | None = None
    # target
    tolerance: float | None = None

    def _get_property_fn(
        self, property_fns: dict[str, Callable] | None
    ) -> Callable[[Mol], float]:
        """Resolve the property function from the registry or the override dict."""
        if property_fns and self.property in property_fns:
            return property_fns[self.property]
        if self.property in PROPERTY_REGISTRY:
            return PROPERTY_REGISTRY[self.property]
        raise KeyError(
            f"Unknown property '{self.property}'. "
            f"Available: {list(PROPERTY_REGISTRY.keys())}. "
            "Pass custom functions via property_fns."
        )

    def is_satisfied(
        self,
        mol: Mol,
        property_fns: dict[str, Callable] | None = None,
    ) -> bool:
        """
        Return True if *mol* satisfies this criterion.

        Parameters
        ----------
        mol : Mol
            The molecule to evaluate.
        property_fns : dict, optional
            Extra / override property functions keyed by name.
        """
        if mol is None:
            return False

        fn = self._get_property_fn(property_fns)
        try:
            prop_value = fn(mol)
        except Exception:
            return False

        if self.type == "threshold":
            if self.direction == "greater":
                return prop_value > self.value  # type: ignore[operator]
            elif self.direction == "less":
                return prop_value < self.value  # type: ignore[operator]
            else:
                raise ValueError(
                    f"direction must be 'greater' or 'less', got '{self.direction}'"
                )

        elif self.type == "range":
            lo = self.min if self.min is not None else float("-inf")
            hi = self.max if self.max is not None else float("inf")
            return lo <= prop_value <= hi

        elif self.type == "target":
            if self.value is None or self.tolerance is None:
                raise ValueError(
                    "SuccessCriterion with type='target' requires 'value' and 'tolerance'."
                )
            return abs(prop_value - self.value) <= self.tolerance

        else:
            raise ValueError(
                f"Unknown criterion type '{self.type}'. "
                "Must be 'threshold', 'range', or 'target'."
            )


def load_criteria_from_config(config: dict) -> list[SuccessCriterion]:
    """
    Parse a list of SuccessCriterion objects from a job config dict.

    Expects ``config["job"]["success_criteria"]`` to be a list of dicts, each
    containing at minimum ``property`` and ``type`` keys.

    Parameters
    ----------
    config : dict
        Top-level job config (as loaded from params.json).

    Returns
    -------
    list of SuccessCriterion
    """
    raw_list: list[dict[str, Any]] = config.get("job", {}).get("success_criteria", [])
    criteria: list[SuccessCriterion] = []
    for entry in raw_list:
        criteria.append(
            SuccessCriterion(
                property=entry["property"],
                type=entry["type"],
                value=entry.get("value"),
                direction=entry.get("direction"),
                min=entry.get("min"),
                max=entry.get("max"),
                tolerance=entry.get("tolerance"),
            )
        )
    return criteria


def molecule_passes(
    mol: Mol,
    criteria: list[SuccessCriterion],
    property_fns: dict[str, Callable] | None = None,
) -> bool:
    """
    Return True if *mol* satisfies ALL criteria (logical AND).

    Parameters
    ----------
    mol : Mol
        The molecule to evaluate.
    criteria : list of SuccessCriterion
        All criteria must be satisfied.
    property_fns : dict, optional
        Extra / override property functions keyed by name.
    """
    return all(c.is_satisfied(mol, property_fns=property_fns) for c in criteria)
