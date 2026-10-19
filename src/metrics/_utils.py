"""
Utility helpers for the src.metrics package.

Provides type-normalization functions that accept either SMILES strings or
RDKit Mol objects and return a uniform list of Mol / canonical-SMILES / None.
"""

from __future__ import annotations
