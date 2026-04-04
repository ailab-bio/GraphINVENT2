"""
src.metrics — Molecular generation evaluation metrics for GraphINVENT2.

Public API
----------
evaluate_unconditional
    Validity, uniqueness, novelty, diversity, SA scores, and FCD for an
    unconditional generation run.

evaluate_conditional
    All unconditional metrics plus success rate, conditional VUN, and
    rediscovery rate against a reference set.

evaluate_goal_directed
    All conditional metrics plus oracle-call count and a (stub)
    sample-efficiency metric.

SuccessCriterion
    Dataclass describing a single property-based success criterion
    (threshold / range / target).
"""

from ._conditional import evaluate_conditional
from ._criteria import SuccessCriterion
from ._goal_directed import evaluate_goal_directed
from ._unconditional import evaluate_unconditional

__all__ = [
    "evaluate_unconditional",
    "evaluate_conditional",
    "evaluate_goal_directed",
    "SuccessCriterion",
]
