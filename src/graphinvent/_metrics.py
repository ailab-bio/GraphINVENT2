"""
Standalone metric functions for molecular generation evaluation.

These functions are independent of the rest of GraphINVENT2 and operate on
plain SMILES strings, RDKit Mol objects, and torch Tensors.
"""

import random
from typing import Optional

import numpy as np
import torch

# Module-level cache for sascorer to avoid repeated path manipulation
_sascorer = None
_fcd_warned = False


def _get_sascorer():
    """Lazily import sascorer from RDKit contrib."""
    global _sascorer
    if _sascorer is None:
        import os
        import sys

        from rdkit import RDConfig

        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer as _sa

        _sascorer = _sa
    return _sascorer


def compute_sa_scores(mols: list) -> tuple:
    """
    Compute SA (Synthetic Accessibility) scores for a list of RDKit Mol objects.

    Returns:
        (mean, median, std) — all NaN if no valid molecules.
    """

    sascorer = _get_sascorer()
    scores = []
    for mol in mols:
        if mol is None:
            continue
        try:
            scores.append(sascorer.calculateScore(mol))
        except Exception:
            pass
    if not scores:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(scores, dtype=float)
    return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))


def compute_novelty(generated_smiles: list, training_smiles: set) -> float:
    """
    Fraction of unique valid generated SMILES not present in the training set.

    Args:
        generated_smiles: list of SMILES strings (may include None for invalid).
        training_smiles:  set of training-set SMILES strings.

    Returns:
        novelty in [0.0, 1.0], or 0.0 if no valid SMILES.
    """
    unique_gen = {s for s in generated_smiles if s is not None}
    if not unique_gen:
        return 0.0
    novel = unique_gen - training_smiles
    return len(novel) / len(unique_gen)


def compute_diversity(smiles_list: list, subsample: int = 1000) -> float:
    """
    Average pairwise Tanimoto distance using Morgan fingerprints (ECFP4, 2048 bits).

    A random subsample is taken for tractability. Returns 0.0 if fewer than 2
    valid molecules are available.

    Args:
        smiles_list: list of SMILES strings.
        subsample:   maximum number of molecules to use (default 1000).

    Returns:
        diversity in [0.0, 1.0].
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mols = []
    for smi in smiles_list:
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)

    if len(mols) < 2:
        return 0.0

    if len(mols) > subsample:
        random.seed(42)
        mols = random.sample(mols, subsample)

    fps = []
    for mol in mols:
        try:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        except Exception:
            pass

    n = len(fps)
    if n < 2:
        return 0.0

    total_sim = 0.0
    count = 0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        total_sim += sum(sims)
        count += len(sims)

    if count == 0:
        return 0.0
    mean_similarity = total_sim / count
    return 1.0 - mean_similarity


def compute_fcd(
    generated_smiles: list,
    reference_smiles: list,
) -> Optional[float]:
    """
    Fréchet ChemNet Distance between generated and reference SMILES distributions.

    Requires the `fcd_torch` package. Returns None if not installed.

    Args:
        generated_smiles:  list of SMILES strings.
        reference_smiles:  list of reference SMILES strings.

    Returns:
        FCD value (lower is better), or None if fcd_torch is not available.
    """
    global _fcd_warned
    try:
        import fcd as fcd_lib

        gen = [s for s in generated_smiles if s is not None]
        ref = [s for s in reference_smiles if s is not None]
        if not gen or not ref:
            return None
        return float(fcd_lib.get_fcd(gen, ref))
    except ImportError:
        if not _fcd_warned:
            print(
                "-- Warning: fcd_torch not installed; FCD metric will be skipped. "
                "Install with: pip install fcd_torch",
                flush=True,
            )
            _fcd_warned = True
        return None
    except Exception as e:
        print(f"-- Warning: FCD computation failed: {e}", flush=True)
        return None


def compute_rediscovery_rate(
    generated_smiles: list,
    test_smiles: set,
) -> float:
    """
    Fraction of test set molecules rediscovered in the unique generated set.

    rediscovery_rate = |unique_gen ∩ test_smiles| / |test_smiles|

    Returns 0.0 if test_smiles is empty.
    """
    if not test_smiles:
        return 0.0
    unique_gen = {s for s in generated_smiles if s is not None}
    found = unique_gen & test_smiles
    return len(found) / len(test_smiles)


def compute_internal_diversity(
    generated_smiles: list,
    max_mols=10000,
) -> dict:
    """
    Pairwise Tanimoto similarity within the generated set (Morgan ECFP4, 2048 bits).

    Thin wrapper around ``metrics.compute_internal_diversity`` from
    ``src/metrics/_internal_diversity.py`` for use inside ``Analyzer``.

    Args:
        generated_smiles: List of generated SMILES strings.
        max_mols:         Subsample cap (default 10 000).  None = no limit.

    Returns:
        dict with keys internal_diversity, mean_internal_similarity,
        median_internal_similarity, max_internal_similarity,
        sim_gt_0_4/0_6/0_8/0_9, n_duplicates_removed, n_invalid,
        n_molecules, subsampled, pairwise_similarities.
    """
    import sys
    from pathlib import Path

    _src = str(Path(__file__).resolve().parent.parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from metrics._internal_diversity import compute_internal_diversity as _fn

    return _fn(generated_smiles, max_mols=max_mols)


def compute_test_set_similarity(
    generated_smiles: list,
    test_smiles: list,
    top_k: int = 10,
    condition_filter=None,
    test_conditions=None,
    max_refs=None,
) -> dict:
    """
    Nearest-neighbour Tanimoto similarity (Morgan ECFP4, 2048 bits) between
    generated molecules and a hold-out test set.

    Thin wrapper around ``metrics.compute_test_set_similarity`` from
    ``src/metrics/_similarity.py`` for use inside ``Analyzer``.

    Args:
        generated_smiles: List of generated SMILES strings.
        test_smiles:      List of test-set SMILES strings.
        top_k:            Number of top-scoring molecules for top-k statistic.
        condition_filter: Optional dict ``{prop: {"value": v, "tolerance": t}}``
                          to restrict the test set before comparison.
        test_conditions:  Per-molecule condition dicts parallel to test_smiles.
        max_refs:         Cap on reference molecules; None uses all.

    Returns:
        dict with keys mean_similarity, median_similarity, top_k_similarity,
        sim_gt_0_4/0_6/0_8/0_9, exact_rediscovery_count, n_invalid_generated,
        n_invalid_test, n_test_after_filter, per_mol_similarity.
    """
    import sys
    from pathlib import Path

    _src = str(Path(__file__).resolve().parent.parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from metrics._similarity import compute_test_set_similarity as _fn

    return _fn(
        generated_smiles,
        test_smiles,
        top_k=top_k,
        condition_filter=condition_filter,
        test_conditions=test_conditions,
        max_refs=max_refs,
    )


def compute_success_rate(scores: torch.Tensor, threshold: float) -> float:
    """
    Fraction of molecules with score > threshold.

    Args:
        scores:    1-D tensor of per-molecule scores.
        threshold: success threshold.

    Returns:
        success rate in [0.0, 1.0].
    """
    if scores.numel() == 0:
        return 0.0
    return float((scores > threshold).float().mean().item())
