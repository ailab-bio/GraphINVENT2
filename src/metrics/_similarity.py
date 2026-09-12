"""
Test-set similarity metrics for molecular generation evaluation.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def compute_test_set_similarity(
    generated_smiles: list[str],
    test_smiles: list[str],
    *,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    top_k: int = 10,
    condition_filter: Optional[dict[str, dict[str, float]]] = None,
    test_conditions: Optional[list[dict[str, float]]] = None,
    max_refs: Optional[int] = None,
) -> dict:
    """
    Compute nearest-neighbour Tanimoto similarity between generated molecules
    and a hold-out test set using Morgan fingerprints.

    For each valid generated molecule the maximum Tanimoto similarity to any
    molecule in the (optionally filtered) test set is recorded.  Aggregate
    statistics are computed over these per-molecule scores.

    Parameters
    ----------
    generated_smiles : list[str]
        SMILES strings of generated molecules.  Invalid entries are skipped.
    test_smiles : list[str]
        SMILES strings of the reference test set.  Invalid entries are skipped.
    fp_radius : int, optional
        Morgan fingerprint radius (default 2, i.e. ECFP4).
    fp_bits : int, optional
        Number of fingerprint bits (default 2048).
    top_k : int, optional
        Number of top-scoring generated molecules used for ``top_k_similarity``
        (default 10).
    condition_filter : dict[str, dict[str, float]] or None, optional
        Restricts the reference test set to molecules whose conditions match
        the filter.  Keys are property names; values are dicts with ``"value"``
        and ``"tolerance"`` keys, e.g.
        ``{"pLogS": {"value": -1.5, "tolerance": 0.3}}``.
        Requires *test_conditions* to be provided.
    test_conditions : list[dict[str, float]] or None, optional
        Per-molecule condition dicts parallel to *test_smiles*, e.g.
        ``[{"pLogS": -0.51, "MW": 60.05}, ...]``.  Required when
        *condition_filter* is set.
    max_refs : int or None, optional
        Cap on the number of test-set molecules used as references.  If the
        filtered test set is larger, a deterministic subsample is taken.
        ``None`` (default) uses all references.

    Returns
    -------
    dict
        Keys:

        * ``mean_similarity`` – mean nearest-neighbour similarity across all
          valid generated molecules.
        * ``median_similarity`` – median nearest-neighbour similarity.
        * ``top_k_similarity`` – mean similarity of the *top_k* most similar
          generated molecules.
        * ``sim_gt_0_4`` – fraction of valid generated molecules with
          nearest-neighbour similarity > 0.4.
        * ``sim_gt_0_6`` – fraction > 0.6.
        * ``sim_gt_0_8`` – fraction > 0.8.
        * ``sim_gt_0_9`` – fraction > 0.9.
        * ``exact_rediscovery_count`` – number of *distinct* test molecules
          exactly rediscovered, confirmed by canonical-SMILES identity rather
          than by a fingerprint similarity of 1.0.  Falls back to the
          fingerprint criterion when a ``condition_filter`` is active.
        * ``n_invalid_generated`` – number of generated SMILES that could not
          be parsed.
        * ``n_invalid_test`` – number of test SMILES that could not be parsed
          (before condition filtering).
        * ``n_test_after_filter`` – number of test molecules used as references
          after condition filtering and subsampling.
        * ``per_mol_similarity`` – list of per-molecule nearest-neighbour
          similarities (one per *valid* generated molecule), for downstream
          analysis.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    _empty = {
        "mean_similarity": 0.0,
        "median_similarity": 0.0,
        "top_k_similarity": 0.0,
        "sim_gt_0_4": 0.0,
        "sim_gt_0_6": 0.0,
        "sim_gt_0_8": 0.0,
        "sim_gt_0_9": 0.0,
        "exact_rediscovery_count": 0,
        "n_invalid_generated": 0,
        "n_invalid_test": 0,
        "n_test_after_filter": 0,
        "per_mol_similarity": [],
    }

    # ------------------------------------------------------------------
    # Parse generated fingerprints
    # ------------------------------------------------------------------
    n_invalid_gen = 0
    gen_fps: list = []
    gen_canonical: list[str] = []
    for smi in generated_smiles:
        if smi is None:
            n_invalid_gen += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid_gen += 1
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
            gen_fps.append(fp)
            gen_canonical.append(Chem.MolToSmiles(mol))
        except Exception:
            n_invalid_gen += 1

    if not gen_fps:
        result = dict(_empty)
        result["n_invalid_generated"] = n_invalid_gen
        return result

    # ------------------------------------------------------------------
    # Parse test fingerprints (with optional condition filtering)
    # ------------------------------------------------------------------
    n_invalid_test = 0
    test_fps: list = []
    test_indices: list[int] = []
    test_canonical: list[str] = []

    for idx, smi in enumerate(test_smiles):
        if smi is None:
            n_invalid_test += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid_test += 1
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
        except Exception:
            n_invalid_test += 1
            continue
        test_fps.append(fp)
        test_indices.append(idx)
        test_canonical.append(Chem.MolToSmiles(mol))

    # Apply condition filter if provided
    if condition_filter and test_conditions is not None:
        if len(test_conditions) != len(test_smiles):
            raise ValueError(
                f"test_conditions length ({len(test_conditions)}) must match "
                f"test_smiles length ({len(test_smiles)}); they are parallel "
                "arrays and are indexed together."
            )
        kept_fps: list = []
        for fp, idx in zip(test_fps, test_indices):
            cond = test_conditions[idx]
            passes = True
            for prop, spec in condition_filter.items():
                val = cond.get(prop)
                if val is None:
                    passes = False
                    break
                if abs(val - spec["value"]) > spec["tolerance"]:
                    passes = False
                    break
            if passes:
                kept_fps.append(fp)
        test_fps = kept_fps
        test_canonical = []  # no longer parallel to test_fps after filtering
    elif condition_filter and test_conditions is None:
        warnings.warn(
            "condition_filter was provided but test_conditions is None; "
            "condition filtering will be skipped.",
            stacklevel=2,
        )

    if not test_fps:
        result = dict(_empty)
        result["n_invalid_generated"] = n_invalid_gen
        result["n_invalid_test"] = n_invalid_test
        return result

    # Apply max_refs subsampling deterministically
    if max_refs is not None and max_refs <= 0:
        raise ValueError(
            f"max_refs must be a positive integer or None, got {max_refs}."
        )
    if max_refs is not None and len(test_fps) > max_refs:
        # Use a fixed stride for determinism (no random seed side-effects)
        step = len(test_fps) / max_refs
        idc = [int(i * step) for i in range(max_refs)]
        test_fps = [test_fps[i] for i in idc]
        if test_canonical:
            test_canonical = [test_canonical[i] for i in idc]

    n_test_after_filter = len(test_fps)

    # ------------------------------------------------------------------
    # Compute nearest-neighbour similarities (bulk, test fps precomputed)
    # ------------------------------------------------------------------
    nn_sims: list[float] = []
    for gen_fp in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(gen_fp, test_fps)
        nn_sims.append(max(sims))

    arr = np.array(nn_sims, dtype=float)

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    actual_k = min(top_k, len(arr))
    top_k_sim = float(np.mean(np.sort(arr)[-actual_k:])) if actual_k > 0 else 0.0

    # Exact rediscovery must be confirmed by canonical-SMILES identity: a
    # Tanimoto of 1.0 on folded Morgan bits is *not* proof of identity
    # (enantiomers and homologues such as decane/dodecane collide).  Count
    # distinct rediscovered molecules so a mode-collapsed run cannot inflate it.
    if test_canonical:
        n_exact_rediscovered = len(set(gen_canonical) & set(test_canonical))
    else:
        # test_canonical is dropped when a condition filter is applied; fall
        # back to the fingerprint criterion rather than silently reporting 0.
        n_exact_rediscovered = int(np.sum(arr >= 1.0 - 1e-6))

    return {
        "mean_similarity": float(np.mean(arr)),
        "median_similarity": float(np.median(arr)),
        "top_k_similarity": top_k_sim,
        "sim_gt_0_4": float(np.mean(arr > 0.4)),
        "sim_gt_0_6": float(np.mean(arr > 0.6)),
        "sim_gt_0_8": float(np.mean(arr > 0.8)),
        "sim_gt_0_9": float(np.mean(arr > 0.9)),
        "exact_rediscovery_count": n_exact_rediscovered,
        "n_invalid_generated": n_invalid_gen,
        "n_invalid_test": n_invalid_test,
        "n_test_after_filter": n_test_after_filter,
        "per_mol_similarity": nn_sims,
    }
