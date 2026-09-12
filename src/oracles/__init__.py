"""
src.oracles — user-defined scoring oracles for goal-directed generation.

Oracles are declared in the job configuration under an ``oracles`` block and
referenced by name from ``score_components``.  Nothing here depends on a
curated third-party oracle collection: the objective is whatever model the user
trains or supplies, which keeps responsibility for its quality and provenance
where it belongs.

Available oracle types
----------------------
``sklearn``
    A pickled scikit-learn estimator over molecular fingerprints.
``python``
    Any importable ``f(list[str]) -> list[float]`` callable.
``vina``
    AutoDock Vina docking against a prepared receptor.

Public API
----------
BaseOracle
    Base class; subclass it and register in ``ORACLE_TYPES`` to add a type.
CachedOracle
    Deduplicates queries and counts unique molecules evaluated.
SklearnOracle, PythonOracle, VinaOracle
    The built-in oracle types.
OracleFactory
    Builds oracles from configuration.
build_transform, apply_direction
    Map a native oracle value onto a [0, 1] desirability.
compute_auc_top_k
    AUC Top-k over an oracle-call budget.
UncertaintyModulation
    Per-component uncertainty-aware reward and loss shaping.
"""

from ._auc import compute_auc_top_k
from ._base import BaseOracle
from ._cache import CachedOracle
from ._factory import ORACLE_TYPES, OracleFactory
from ._surrogate import PythonOracle, SklearnOracle
from ._transform import apply_direction, build_transform
from ._uncertainty import (
    UncertaintyModulation,
    combine_loss_weights,
    combine_score_weights,
    modulate_loss_weights,
    reliability_weight,
    reliability_weights,
)
from ._vina import VinaOracle

__all__ = [
    "BaseOracle",
    "CachedOracle",
    "SklearnOracle",
    "PythonOracle",
    "VinaOracle",
    "OracleFactory",
    "ORACLE_TYPES",
    "build_transform",
    "apply_direction",
    "compute_auc_top_k",
    "UncertaintyModulation",
    "reliability_weight",
    "reliability_weights",
    "combine_loss_weights",
    "combine_score_weights",
    "modulate_loss_weights",
]
