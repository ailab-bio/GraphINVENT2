"""
src.metrics — molecular generation evaluation metrics for GraphINVENT2.

Every metric CLAUDE.md defines has exactly one implementation here. The entry
points a job uses are the four in `_run.py`, which `src/graphinvent/Workflow.py`
calls; the names below are the reusable pieces those are built from.

compute_core_metrics
    The core metric table from CLAUDE.md for one generated sample: the
    validity, uniqueness and novelty funnel and its composite, near-duplicate
    rate, internal and scaffold diversity, SA score and the structural filters.

compute_internal_diversity
    Pairwise Tanimoto similarity within a set of generated molecules, with the
    aggregate statistics including internal diversity (1 - mean pairwise
    similarity).

compute_test_set_similarity
    Nearest-neighbor Tanimoto similarity between generated molecules and a
    held-out set, with optional condition filtering.

SuccessCriterion
    Dataclass describing a single property-based success criterion.
"""

from ._core import compute_core_metrics
from ._criteria import SuccessCriterion
from ._internal_diversity import compute_internal_diversity
from ._similarity import compute_test_set_similarity

__all__ = [
    "compute_core_metrics",
    "compute_internal_diversity",
    "compute_test_set_similarity",
    "SuccessCriterion",
]
