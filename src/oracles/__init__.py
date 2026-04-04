"""
src.oracles — TDC oracle integration for GraphINVENT2 goal-directed generation.

Public API
----------
BaseOracle
    Abstract base class for all oracles.
CachedOracle
    Wraps any oracle with deduplication and call counting.
TDCOracle
    Wraps a Therapeutics Data Commons oracle.
OracleFactory
    Creates oracle instances by name from config or directly.
compute_auc_top_k
    PMO-style AUC Top-k metric over the full optimization curve.
"""

from ._auc import compute_auc_top_k
from ._base import BaseOracle
from ._cache import CachedOracle
from ._factory import OracleFactory
from ._tdc import TDCOracle

__all__ = [
    "BaseOracle",
    "CachedOracle",
    "TDCOracle",
    "OracleFactory",
    "compute_auc_top_k",
]
