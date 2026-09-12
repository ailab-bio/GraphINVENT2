"""
Transforms mapping a raw oracle output onto a [0, 1] desirability score.

Oracles return whatever their underlying model produces: AutoDock Vina reports
a binding free energy in kcal/mol where roughly -11 is excellent and -4 is
negligible, a regression surrogate might report pIC50, and a classifier reports
a probability.  The reinforcement-learning objective, on the other hand,
requires a bounded score, because the augmented log-likelihood adds
``sigma * score`` to the prior log-likelihood and an unbounded score would let a
single component dominate the loss without limit.

Separating the transform from the oracle keeps the oracle honest -- it reports
its native quantity -- and puts the value judgement ("how good is -8.5 kcal/mol")
in the job configuration, where it belongs and can be varied per target.

Choosing transform parameters is a modelling decision, not a technical one.
A ``clipped_linear`` from -4 to -11 kcal/mol encodes an assumption that scores
below -11 are not meaningfully better, which is defensible for docking because
Vina's scoring function is not accurate enough to rank very strong binders, but
it is an assumption and should be stated in any write-up that uses it.
"""

from __future__ import annotations

import math
from typing import Callable, Optional


def _clipped_linear(value: float, low: float, high: float) -> float:
    """
    Linear ramp from 0 at *low* to 1 at *high*, flat outside that interval.

    ``high`` may be numerically smaller than ``low``, which is how a
    "lower is better" quantity such as a docking energy is expressed.
    """
    if high == low:
        return 1.0 if value == low else 0.0
    fraction = (value - low) / (high - low)
    return max(0.0, min(1.0, fraction))


def _sigmoid(value: float, low: float, high: float, k: float = 1.0) -> float:
    """
    Smooth ramp centred midway between *low* and *high*.

    Preferred over ``clipped_linear`` when the objective should keep providing
    gradient signal outside the nominal window: a molecule slightly worse than
    *low* still scores above zero, so the agent is not left with a flat reward
    surface early in training when nothing yet reaches the target range.
    """
    if high == low:
        return 1.0 if value >= low else 0.0
    midpoint = (low + high) / 2.0
    # Scale so that k=1 puts roughly the full transition inside [low, high].
    steepness = 10.0 * k / (high - low)
    exponent = -steepness * (value - midpoint)
    # Guard the exponential: |exponent| beyond ~700 overflows float.
    if exponent > 700:
        return 0.0
    if exponent < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(exponent))


def _step(value: float, threshold: float, above: bool = True) -> float:
    """
    Hard 0/1 indicator.

    Gives no gradient between molecules on the same side of the threshold, so
    it suits a constraint that must simply be satisfied (a hard property
    filter) rather than an objective to be improved.
    """
    if above:
        return 1.0 if value >= threshold else 0.0
    return 1.0 if value <= threshold else 0.0


def build_transform(spec: Optional[dict]) -> Callable[[float], float]:
    """
    Build a raw-value -> [0, 1] callable from a configuration dict.

    Parameters
    ----------
    spec
        ``None`` or ``{}`` yields the identity transform, clamped to [0, 1];
        use it for oracles that already return a probability or a normalised
        score.  Otherwise ``spec["type"]`` selects the transform:

        ``clipped_linear``
            Requires ``low`` and ``high``.  Set ``high`` below ``low`` for
            quantities where smaller is better, e.g. ``{"low": -4.0,
            "high": -11.0}`` for a Vina energy.
        ``sigmoid``
            Requires ``low`` and ``high``; optional ``k`` (default 1.0)
            controls steepness.
        ``step``
            Requires ``threshold``; optional ``above`` (default True).
        ``identity``
            Clamps to [0, 1] and does nothing else.

    Returns
    -------
    Callable mapping one raw float to one score in [0, 1].

    Raises
    ------
    ValueError
        For an unknown transform type or missing required parameters, rather
        than silently falling back to the identity, which would misreport an
        unbounded quantity as a desirability.
    """
    if not spec:
        return lambda v: max(0.0, min(1.0, float(v)))

    kind = spec.get("type", "identity")

    if kind == "identity":
        return lambda v: max(0.0, min(1.0, float(v)))

    if kind == "clipped_linear":
        if "low" not in spec or "high" not in spec:
            raise ValueError(
                "clipped_linear transform requires both 'low' and 'high', "
                f"got {sorted(spec)}."
            )
        low, high = float(spec["low"]), float(spec["high"])
        return lambda v: _clipped_linear(float(v), low, high)

    if kind == "sigmoid":
        if "low" not in spec or "high" not in spec:
            raise ValueError(
                f"sigmoid transform requires both 'low' and 'high', got {sorted(spec)}."
            )
        low, high = float(spec["low"]), float(spec["high"])
        k = float(spec.get("k", 1.0))
        return lambda v: _sigmoid(float(v), low, high, k)

    if kind == "step":
        if "threshold" not in spec:
            raise ValueError(
                f"step transform requires 'threshold', got {sorted(spec)}."
            )
        threshold = float(spec["threshold"])
        above = bool(spec.get("above", True))
        return lambda v: _step(float(v), threshold, above)

    raise ValueError(
        f"Unknown transform type '{kind}'. "
        "Choose from: identity, clipped_linear, sigmoid, step."
    )


def apply_direction(score: float, direction: str) -> float:
    """
    Flip a desirability score when the objective is to *avoid* a property.

    Selectivity objectives are the reason this exists: designing a molecule
    that binds one target while avoiding another is expressed as two oracles
    over the same kind of quantity, one maximised and one minimised, rather
    than as two differently-parameterised transforms.

    Raises
    ------
    ValueError
        For any direction other than "maximize" or "minimize", so a typo
        cannot silently invert an objective.
    """
    if direction == "maximize":
        return score
    if direction == "minimize":
        return 1.0 - score
    raise ValueError(f"Unknown direction '{direction}'. Use 'maximize' or 'minimize'.")
