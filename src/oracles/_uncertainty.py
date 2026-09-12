"""
Uncertainty-aware reward and loss shaping for RL fine-tuning.

A scoring function trained on finite data is not an oracle, but the RL
objective treats it as one: the agent is rewarded for the *predicted* property,
so it will happily migrate into regions where the surrogate is extrapolating
and its high predictions mean nothing.  The fix is to let the surrogate's own
predictive uncertainty damp the signal it produces, so that a confident
prediction of 0.8 counts for more than an unsupported prediction of 0.9.

The implementation follows Medina and Janet, *Uncertainty-aware reinforcement
learning for chemical language models* (arXiv:2606.24990), which introduces two
complementary strategies for REINVENT.  Both are provided here because they act
at different points and can be combined:

**Score modulation (SM)** treats reliability as an additional objective, folded
into the aggregate score alongside the real components (their Eq. 6).  Under
GraphINVENT's product aggregation this multiplies the score by a reliability
factor, so an uncertain molecule is worth less *as a molecule* and the agent is
steered toward the surrogate's applicability domain.

**Loss modulation (LM)** leaves the score alone and instead reweights how much
each sampled molecule contributes to the gradient (their Eq. 8):

    L = (1/N) * sum_j [ w_j / ((1/N) * sum_l w_l) ] * L_j

The division by the batch-mean weight is the part that matters and is easy to
omit.  Without it, a batch of uniformly uncertain molecules would shrink the
whole loss, which is indistinguishable from lowering the learning rate; with it,
only the *relative* contribution of molecules changes and the average gradient
magnitude is preserved.

The two differ in what they do to a molecule that scores well but is not
trusted.  Score modulation tells the agent that molecule is worse than it looks,
which changes the optimum being sought.  Loss modulation says nothing about the
molecule's quality and merely declines to learn much from it, leaving the
objective intact.  Which is appropriate depends on whether unreliability is a
property you want to design against or merely a reason to be cautious.

Reported effect in the paper: on an EGFR task with a deliberately noisy
component, the true hit rate rose from 0.5 to 0.75 and the number of true hits
roughly doubled.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

#: Modulation applied when a component names no method.  Off by default: the
#: modulation only makes sense where an uncertainty estimate is actually
#: available, and silently damping every component would quietly change the
#: objective of every existing run.
DEFAULT_METHOD = "none"


# ---------------------------------------------------------------------------
# Uncertainty -> reliability weight
# ---------------------------------------------------------------------------


def reliability_weight(
    uncertainty: float,
    method: str = DEFAULT_METHOD,
    **params: float,
) -> float:
    """
    Map one raw uncertainty onto a reliability weight in [0, 1].

    Larger uncertainty always yields a smaller weight; a weight of 1 means
    "trust this prediction fully" and 0 means "ignore it".

    Methods
    -------
    ``none``
        Constant 1.0.  The default, so a component with no configured
        modulation is untouched.
    ``linear``
        ``1 - u / max_uncertainty``, clipped to [0, 1].  This is the paper's
        Eq. 14 (``w = 1 - d``) with an explicit scale, since their distance was
        already normalised to [0, 1] and a raw surrogate uncertainty is not.
        Requires ``max_uncertainty``: the uncertainty at which a prediction
        becomes worthless.
    ``sigmoid``
        ``1 - 1/(1 + exp(-alpha * (u - beta)))``, the paper's Eq. 13 read as a
        reliability rather than a distance.  ``beta`` is the uncertainty at
        which reliability is 0.5 and ``alpha`` how sharply it falls off.
        Preferable to ``linear`` when there is a meaningful threshold between
        "in domain" and "out of domain" rather than a gradual decay.
    ``inverse``
        ``1 / (1 + u / scale)``.  The paper uses ``w = 1/uncertainty`` for its
        MVE models; that form is unbounded and diverges as ``u -> 0``, so the
        bounded variant is used here.  It matches the paper's ordering and
        approaches it up to a constant for ``u >> scale``, but it is not the
        identical function, and the difference is absorbed by the batch-mean
        normalisation in :func:`modulate_loss_weights`.
    ``exponential``
        ``exp(-beta * u)``.  Decays smoothly with no hard cutoff; useful when
        no principled ``max_uncertainty`` can be named.

    Raises
    ------
    ValueError
        For an unknown method or a missing required parameter, rather than
        defaulting silently to no modulation, which would leave the run
        unprotected while appearing configured.
    """
    if method in ("none", None):
        return 1.0

    u = float(uncertainty)
    if not math.isfinite(u) or u < 0.0:
        # A non-finite or negative uncertainty means the estimator failed;
        # distrust the prediction completely rather than propagate a NaN.
        return 0.0

    if method == "linear":
        if "max_uncertainty" not in params:
            raise ValueError(
                "reliability_weight(method='linear') requires 'max_uncertainty'."
            )
        max_u = float(params["max_uncertainty"])
        if max_u <= 0:
            raise ValueError("'max_uncertainty' must be positive.")
        return float(max(0.0, min(1.0, 1.0 - u / max_u)))

    if method == "sigmoid":
        if "beta" not in params:
            raise ValueError(
                "reliability_weight(method='sigmoid') requires 'beta' "
                "(the uncertainty at which reliability is 0.5)."
            )
        beta = float(params["beta"])
        alpha = float(params.get("alpha", 10.0))
        exponent = -alpha * (u - beta)
        # reliability = 1 - 1/(1 + exp(exponent)).  A very negative exponent
        # means u is far ABOVE beta, i.e. no reliability at all; a very
        # positive one means u is far below beta and the prediction is fully
        # trusted.  Getting these the wrong way round inverts the whole
        # objective for out-of-domain molecules.
        if exponent < -700:
            return 0.0
        if exponent > 700:
            return 1.0
        return float(1.0 - 1.0 / (1.0 + math.exp(exponent)))

    if method == "inverse":
        scale = float(params.get("scale", 1.0))
        if scale <= 0:
            raise ValueError("'scale' must be positive.")
        return float(1.0 / (1.0 + u / scale))

    if method == "exponential":
        beta = float(params.get("beta", 1.0))
        if beta < 0:
            raise ValueError("'beta' must be non-negative.")
        exponent = -beta * u
        if exponent < -700:
            return 0.0
        return float(math.exp(exponent))

    raise ValueError(
        f"Unknown uncertainty method '{method}'. Choose from: "
        "none, linear, sigmoid, inverse, exponential."
    )


def reliability_weights(
    uncertainties: Sequence[float],
    method: str = DEFAULT_METHOD,
    **params: float,
) -> List[float]:
    """Vectorised :func:`reliability_weight` over a batch."""
    return [reliability_weight(u, method, **params) for u in uncertainties]


# ---------------------------------------------------------------------------
# Per-component configuration
# ---------------------------------------------------------------------------


class UncertaintyModulation:
    """
    Resolves and applies per-component uncertainty modulation.

    Per-component configuration is the point rather than a convenience: in a
    multi-objective run each oracle reports uncertainty in its own units -- a
    docking spread in kcal/mol, an ensemble standard deviation in probability,
    a conformal p-value -- and a single global threshold cannot be meaningful
    for all of them at once.  Each component therefore names its own method and
    parameters, which is also what puts the differing scales onto a common
    footing: after the mapping every component yields a weight in [0, 1].
    Nothing here calibrates those weights against each other, so a "0.5" from
    one component does not represent the same degree of doubt as a "0.5" from
    another; the mapping is a modelling choice per component, not a
    principled cross-component normalisation.

    Configuration
    -------------
    ::

        "uncertainty_modulation": {
            "mode": "loss",
            "components": {
                "EGFR":  {"method": "sigmoid", "beta": 0.4, "alpha": 10.0},
                "hERG":  {"method": "linear", "max_uncertainty": 0.25},
                "QED":   {"method": "none"}
            }
        }

    ``mode`` selects where modulation acts:

    ``"none"``
        Disabled.
    ``"score"``
        Score modulation: the aggregate score is multiplied by the combined
        reliability, so uncertainty becomes part of the objective.
    ``"loss"``
        Loss modulation: per-molecule loss contributions are reweighted,
        leaving the objective unchanged.
    ``"both"``
        Both, as in the paper's combined setting.

    Components absent from ``components`` are unmodulated.
    """

    VALID_MODES = ("none", "score", "loss", "both")

    def __init__(self, config: Optional[dict] = None) -> None:
        config = dict(config or {})
        config = {k: v for k, v in config.items() if not k.startswith("_")}

        self.mode = config.get("mode", "none")
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown uncertainty_modulation mode '{self.mode}'. "
                f"Choose from: {list(self.VALID_MODES)}."
            )

        raw_components = config.get("components", {}) or {}
        self.components: Dict[str, dict] = {
            name: {k: v for k, v in spec.items() if not k.startswith("_")}
            for name, spec in raw_components.items()
            if not name.startswith("_")
        }

        # Validate every spec now: a bad parameter should surface at job start,
        # not on the first batch that happens to carry an uncertainty estimate.
        for name, spec in self.components.items():
            spec = dict(spec)
            method = spec.pop("method", DEFAULT_METHOD)
            try:
                reliability_weight(0.0, method, **spec)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid uncertainty_modulation for component '{name}': {exc}"
                ) from exc

    @property
    def enabled(self) -> bool:
        """Whether any modulation is active."""
        return self.mode != "none" and bool(self.components)

    def modulates_score(self) -> bool:
        return self.mode in ("score", "both")

    def modulates_loss(self) -> bool:
        return self.mode in ("loss", "both")

    def is_configured(self, component: str) -> bool:
        """Whether *component* has a modulation method other than "none"."""
        spec = self.components.get(component)
        return bool(spec) and spec.get("method", DEFAULT_METHOD) != "none"

    def weights_for(
        self, component: str, uncertainties: Sequence[float]
    ) -> List[float]:
        """
        Reliability weights for one component's per-molecule uncertainties.

        Returns all-ones for a component with no configured modulation, so the
        caller can apply this unconditionally.
        """
        spec = dict(self.components.get(component, {}))
        method = spec.pop("method", DEFAULT_METHOD)
        return reliability_weights(uncertainties, method, **spec)


# ---------------------------------------------------------------------------
# Aggregation across components
# ---------------------------------------------------------------------------


def combine_loss_weights(
    per_component: Dict[str, Sequence[float]], n_molecules: int
) -> np.ndarray:
    """
    Combine per-component reliability weights into one weight per molecule.

    Uses the arithmetic mean, following the paper: a product would let a single
    distrusted component drive the weight to near zero and effectively delete
    the molecule from the update, and the paper adopts the mean specifically
    "to prevent extreme values from dominating the policy update".

    Components with no uncertainty estimate should simply be absent from
    *per_component* rather than passed as ones, since including them would
    dilute the signal from the components that do report uncertainty.
    """
    if not per_component:
        return np.ones(n_molecules, dtype=np.float64)
    stacked = np.vstack(
        [np.asarray(w, dtype=np.float64) for w in per_component.values()]
    )
    return stacked.mean(axis=0)


def combine_score_weights(
    per_component: Dict[str, Sequence[float]], n_molecules: int
) -> np.ndarray:
    """
    Combine per-component reliability weights into one score multiplier.

    Uses the geometric mean, which is what the paper's Eq. 2 reduces to for
    equal weights: the exponent ``w_i / sum_l w_l`` normalises by the number of
    components, so the aggregate does not shrink simply because more of them
    were added.  A raw product does shrink that way, and the effect is severe
    in practice -- two components at reliability 0.18 and 0.08 multiply to
    0.014 but average geometrically to 0.12 -- which would suppress the reward
    for every molecule rather than discriminating between them.

    A single zero still forces the aggregate to zero.  That is intended: under
    score modulation reliability is an objective the molecule must satisfy, and
    a prediction with no support at all should not be rewarded whatever the
    other components say.
    """
    if not per_component:
        return np.ones(n_molecules, dtype=np.float64)
    stacked = np.vstack(
        [np.asarray(w, dtype=np.float64) for w in per_component.values()]
    )
    # Geometric mean via logs would need a zero guard; with a handful of
    # components in [0, 1] the direct root is simpler and cannot underflow.
    return np.prod(stacked, axis=0) ** (1.0 / stacked.shape[0])


def modulate_loss_weights(weights: Sequence[float]) -> np.ndarray:
    """
    Normalise per-molecule loss weights to preserve the mean gradient scale.

    Implements the ``w_j / ((1/N) sum_l w_l)`` factor of the paper's Eq. 8.
    Without this division a uniformly-distrusted batch would simply shrink the
    loss, which acts as an unintended learning-rate change rather than as a
    reweighting; dividing by the batch mean leaves the average contribution at
    1.0 so only the relative weighting of molecules changes.

    A batch whose weights are all zero -- every molecule maximally distrusted --
    returns ones rather than dividing by zero.  That deliberately declines to
    express "learn nothing from this batch": doing so is the learning-rate
    change this normalisation exists to avoid, and a batch in which nothing is
    trusted carries no information about which molecules are better anyway.
    """
    array = np.asarray(weights, dtype=np.float64)
    if array.size == 0:
        return array
    mean_weight = float(array.mean())
    if mean_weight <= 0.0:
        return np.ones_like(array)
    return array / mean_weight
