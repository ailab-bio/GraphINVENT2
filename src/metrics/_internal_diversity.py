"""
Internal diversity metric for molecular generation evaluation.
"""

from __future__ import annotations

import random
import warnings
from typing import Optional

import numpy as np


def compute_internal_diversity(
    generated_smiles: list[str],
    *,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    max_mols: Optional[int] = 10000,
) -> dict:
    """
    Compute pairwise Tanimoto similarity within a set of generated molecules.

    All pairwise similarities in the upper triangle (excluding the diagonal)
    are computed using Morgan fingerprints via
    ``DataStructs.BulkTanimotoSimilarity``, which evaluates each row of the
    pairwise matrix in a single vectorised call.

    SMILES are canonicalised and deduplicated before fingerprinting; the number
    of duplicates removed is reported as a diversity signal in its own right.

    The primary output ``internal_diversity`` follows the MOSES convention:

        internal_diversity = 1 − mean(pairwise Tanimoto similarities)

    where 1.0 means all molecules are maximally different and 0.0 means all
    are identical.  For a set with fewer than two unique valid molecules,
    ``internal_diversity`` is 0.0 and a warning is emitted.

    Parameters
    ----------
    generated_smiles : list[str]
        SMILES strings of generated molecules.  ``None`` entries and
        unparseable strings are skipped.
    fp_radius : int, optional
        Morgan fingerprint radius (default 2, i.e. ECFP4).
    fp_bits : int, optional
        Number of fingerprint bits (default 2048).
    max_mols : int or None, optional
        Maximum number of unique valid molecules used for the pairwise
        computation.  When the deduplicated set is larger, a random subsample
        of this size is drawn (with a fixed seed for reproducibility) and a
        warning is printed.  ``None`` disables subsampling.

    Returns
    -------
    dict
        Keys:

        * ``internal_diversity`` – 1 − mean pairwise Tanimoto similarity.
          Primary diversity score; higher is more diverse.
        * ``mean_internal_similarity`` – mean pairwise Tanimoto similarity
          (complement of ``internal_diversity``).
        * ``median_internal_similarity`` – median pairwise Tanimoto similarity;
          more robust to outlier pairs.
        * ``max_internal_similarity`` – similarity of the most-similar pair;
          useful for detecting near-duplicate clusters.
        * ``sim_gt_0_4`` – fraction of pairs with similarity > 0.4.
        * ``sim_gt_0_6`` – fraction of pairs with similarity > 0.6.
        * ``sim_gt_0_8`` – fraction of pairs with similarity > 0.8.
        * ``sim_gt_0_9`` – fraction of pairs with similarity > 0.9.
        * ``n_duplicates_removed`` – number of duplicate canonical SMILES
          removed before fingerprinting.
        * ``n_invalid`` – number of SMILES that could not be parsed by RDKit.
        * ``n_molecules`` – number of unique valid molecules used (after
          deduplication and optional subsampling).
        * ``subsampled`` – ``True`` if the set was randomly subsampled due
          to *max_mols*.
        * ``pairwise_similarities`` – flat ``numpy`` array of all upper-triangle
          pairwise similarities (length ``n*(n-1)//2``).  Empty array when
          fewer than two molecules are available.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    _empty: dict = {
        "internal_diversity": 0.0,
        "mean_internal_similarity": 0.0,
        "median_internal_similarity": 0.0,
        "max_internal_similarity": 0.0,
        "sim_gt_0_4": 0.0,
        "sim_gt_0_6": 0.0,
        "sim_gt_0_8": 0.0,
        "sim_gt_0_9": 0.0,
        "n_duplicates_removed": 0,
        "n_invalid": 0,
        "n_molecules": 0,
        "subsampled": False,
        "pairwise_similarities": np.array([], dtype=np.float32),
    }

    # ------------------------------------------------------------------
    # Canonicalize, validate, and deduplicate
    # ------------------------------------------------------------------
    n_invalid = 0
    seen_canonical: dict = {}  # canonical_smiles → RDKit fingerprint

    for smi in generated_smiles:
        if smi is None:
            n_invalid += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical not in seen_canonical:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, fp_radius, nBits=fp_bits
                )
                seen_canonical[canonical] = fp
            except Exception:
                n_invalid += 1

    n_duplicates_removed = len(generated_smiles) - n_invalid - len(seen_canonical)
    fps = list(seen_canonical.values())

    if len(fps) < 2:
        if len(fps) == 1:
            warnings.warn(
                "compute_internal_diversity: only one unique valid molecule — "
                "internal_diversity is 0.0 by convention.",
                stacklevel=2,
            )
        result = dict(_empty)
        result["n_duplicates_removed"] = max(0, n_duplicates_removed)
        result["n_invalid"] = n_invalid
        result["n_molecules"] = len(fps)
        return result

    # ------------------------------------------------------------------
    # Optional subsampling for large sets
    # ------------------------------------------------------------------
    subsampled = False
    if max_mols is not None and len(fps) > max_mols:
        print(
            f"-- Warning: compute_internal_diversity received {len(fps)} unique "
            f"molecules; subsampling to {max_mols} for pairwise computation.",
            flush=True,
        )
        rng = random.Random(42)
        fps = rng.sample(fps, max_mols)
        subsampled = True

    n_mols = len(fps)

    # ------------------------------------------------------------------
    # Compute upper-triangle pairwise similarities
    # ------------------------------------------------------------------
    n_pairs = n_mols * (n_mols - 1) // 2
    pairwise = np.empty(n_pairs, dtype=np.float32)
    idx = 0
    for i in range(n_mols - 1):
        row = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        n_row = len(row)
        pairwise[idx : idx + n_row] = row
        idx += n_row

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    mean_sim = float(np.mean(pairwise))

    return {
        "internal_diversity": 1.0 - mean_sim,
        "mean_internal_similarity": mean_sim,
        "median_internal_similarity": float(np.median(pairwise)),
        "max_internal_similarity": float(np.max(pairwise)),
        "sim_gt_0_4": float(np.mean(pairwise > 0.4)),
        "sim_gt_0_6": float(np.mean(pairwise > 0.6)),
        "sim_gt_0_8": float(np.mean(pairwise > 0.8)),
        "sim_gt_0_9": float(np.mean(pairwise > 0.9)),
        "n_duplicates_removed": max(0, n_duplicates_removed),
        "n_invalid": n_invalid,
        "n_molecules": n_mols,
        "subsampled": subsampled,
        "pairwise_similarities": pairwise,
    }
