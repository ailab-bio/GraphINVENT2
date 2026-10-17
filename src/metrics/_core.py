"""
Core metrics, reported for every setting.

The funnel -- validity, uniqueness, novelty -- composes sequentially on the
surviving subset, so each stage's denominator is the previous stage's numerator.
The denominators are in the variable names here on purpose: every one of these
numbers is a ratio, and the only way to read the code is to know what it is a
ratio of.
"""

from __future__ import annotations

import numpy as np

from ._internal_diversity import compute_internal_diversity
from ._properties import sa_score

# Morgan radius 2, 2048 bits: ECFP4, as CLAUDE.md's definitions specify.
ECFP4_RADIUS = 2
ECFP4_BITS = 2048

# CLAUDE.md: near_duplicate_rate counts a generated molecule as a near duplicate
# when its maximum ECFP4 Tanimoto to any training molecule exceeds this.
NEAR_DUPLICATE_THRESHOLD = 0.9


def ecfp4(mol):
    """ECFP4 fingerprint as a packed RDKit bit vector."""
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=ECFP4_RADIUS, fpSize=ECFP4_BITS
    )
    return generator.GetFingerprint(mol)


def canonical_smiles(smiles_or_mols: list) -> list:
    """
    Canonical SMILES of each input that RDKit parses and sanitizes, in order.

    Entries RDKit rejects become None, so the result stays aligned with the
    input and the caller can still count them.
    """
    from rdkit import Chem

    out: list = []
    for item in smiles_or_mols:
        if item is None:
            out.append(None)
            continue
        mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
        if mol is None:
            out.append(None)
            continue
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            out.append(None)
            continue
        out.append(smi or None)
    return out


def bemis_murcko(smiles: str) -> str | None:
    """Bemis-Murcko scaffold SMILES, or None when RDKit cannot derive one."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        return None


def max_similarity_to_reference(query_smiles: list, reference_smiles: list) -> list:
    """
    For each query molecule, its maximum ECFP4 Tanimoto to any reference molecule.

    `BulkTanimotoSimilarity` over packed bit vectors, not a Python loop over
    pairs: the reference set is the whole training split, so the pair count is
    the product of two large numbers and the only affordable version of this is
    the one that stays inside RDKit's C++ for the inner loop.
    """
    from rdkit import Chem, DataStructs

    reference_fps = []
    for smi in reference_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            reference_fps.append(ecfp4(mol))
    if not reference_fps:
        return [None] * len(query_smiles)

    out: list = []
    for smi in query_smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            out.append(None)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(ecfp4(mol), reference_fps)
        out.append(max(sims) if sims else 0.0)
    return out


def compute_core_metrics(
    generated: list,
    n_sampled: int,
    training_smiles: list,
    *,
    subsample: int = 10000,
) -> dict:
    """
    The core metric table from CLAUDE.md, for one sample of generated molecules.

    Parameters
    ----------
    generated
        The generated molecules, as SMILES or RDKit Mols. Invalid entries may
        be present or absent; `n_sampled` is what fixes the denominator.
    n_sampled
        N: how many molecules the model was asked for. Every "fraction of N"
        in the funnel divides by this, not by the length of `generated`.
    training_smiles
        The training split, the reference set for novelty and for
        near_duplicate_rate.
    subsample
        Cap for the pairwise internal-diversity computation.

    Returns
    -------
    dict of the CLAUDE.md core keys, plus `n_sampled` and the funnel counts so
    a reader can reconstruct every denominator.
    """
    smiles_or_none = canonical_smiles(generated)
    valid_smiles = [s for s in smiles_or_none if s is not None]
    n_valid = len(valid_smiles)

    # Funnel stage 1: valid / N.
    validity = n_valid / n_sampled if n_sampled else 0.0

    # Funnel stage 2: unique canonical SMILES among valid / valid.
    unique_valid_smiles = sorted(set(valid_smiles))
    n_unique_valid = len(unique_valid_smiles)
    uniqueness = n_unique_valid / n_valid if n_valid else 0.0

    # Funnel stage 3: unique valid absent from the training split / unique valid.
    # The training side is canonicalized too, or a molecule the model copied and
    # wrote differently would count as novel.
    training_canonical = {s for s in canonical_smiles(training_smiles) if s}
    novel_smiles = [s for s in unique_valid_smiles if s not in training_canonical]
    n_novel = len(novel_smiles)
    novelty = n_novel / n_unique_valid if n_unique_valid else 0.0

    # The funnel composite: unique, valid and novel over N. Equal to the product
    # of the three stages above, computed from the counts so it does not inherit
    # their rounding.
    vun = n_novel / n_sampled if n_sampled else 0.0

    results: dict = {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "vun": vun,
        "n_sampled": n_sampled,
        "n_valid": n_valid,
        "n_unique_valid": n_unique_valid,
        "n_unique_valid_novel": n_novel,
    }

    # near_duplicate_rate: over unique valid molecules, the fraction whose
    # nearest training neighbor is closer than the threshold. This is the metric
    # that separates a model with high novelty from a model that memorized and
    # then added a methyl.
    if training_canonical and unique_valid_smiles:
        max_sims = max_similarity_to_reference(
            unique_valid_smiles, sorted(training_canonical)
        )
        scored = [s for s in max_sims if s is not None]
        results["near_duplicate_rate"] = (
            sum(1 for s in scored if s > NEAR_DUPLICATE_THRESHOLD) / len(scored)
            if scored
            else 0.0
        )
    else:
        results["near_duplicate_rate"] = None

    # internal_diversity: mean pairwise Tanimoto distance over unique valid.
    if n_unique_valid >= 2:
        results["internal_diversity"] = compute_internal_diversity(
            unique_valid_smiles, max_mols=subsample
        )["internal_diversity"]
    else:
        results["internal_diversity"] = None

    # scaffold_diversity: unique Bemis-Murcko scaffolds / unique valid.
    if n_unique_valid:
        scaffolds = {
            s
            for s in (bemis_murcko(smi) for smi in unique_valid_smiles)
            if s is not None
        }
        results["scaffold_diversity"] = len(scaffolds) / n_unique_valid
        results["n_scaffolds"] = len(scaffolds)
    else:
        results["scaffold_diversity"] = None
        results["n_scaffolds"] = 0

    # filter_pass_rate, Ro5_pass_rate, bRo5_pass_rate -- same denominator.
    from ._filters import compute_filter_metrics

    filter_metrics, filter_unavailable = compute_filter_metrics(unique_valid_smiles)
    results.update(filter_metrics)
    results["_filter_unavailable"] = filter_unavailable

    # Synthetic accessibility over unique valid molecules.
    from rdkit import Chem

    sa_values = []
    for smi in unique_valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                sa_values.append(sa_score(mol))
            except Exception:
                pass
    if sa_values:
        results["sa_score_mean"] = float(np.mean(sa_values))
        results["sa_score_median"] = float(np.median(sa_values))
    else:
        results["sa_score_mean"] = None
        results["sa_score_median"] = None

    return results
