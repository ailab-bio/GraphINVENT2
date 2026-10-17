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

compute_test_set_similarity
    Nearest-neighbor Tanimoto similarity between generated molecules and a
    hold-out test set, with aggregate statistics and optional condition
    filtering for conditional generation evaluation.

compute_core_metrics
    The core metric table from CLAUDE.md for one generated sample: the
    validity/uniqueness/novelty funnel and its composite, near-duplicate rate,
    internal and scaffold diversity, and SA score.

compute_internal_diversity
    Pairwise Tanimoto similarity within a set of generated molecules, with
    aggregate statistics including the MOSES-standard internal diversity score
    (1 − mean pairwise similarity).

SuccessCriterion
    Dataclass describing a single property-based success criterion
    (threshold / range / target).
"""

from ._conditional import evaluate_conditional
from ._core import compute_core_metrics
from ._criteria import SuccessCriterion
from ._goal_directed import evaluate_goal_directed
from ._internal_diversity import compute_internal_diversity
from ._similarity import compute_test_set_similarity
from ._unconditional import evaluate_unconditional

__all__ = [
    "compute_core_metrics",
    "evaluate_unconditional",
    "evaluate_conditional",
    "evaluate_goal_directed",
    "compute_test_set_similarity",
    "compute_internal_diversity",
    "SuccessCriterion",
]
