"""
Unconditional molecular generation metrics.
"""

from __future__ import annotations

import numpy as np

from ._internal_diversity import compute_internal_diversity
from ._properties import sa_score
from ._utils import to_mols, to_smiles


def _canonicalize(smiles: set[str]) -> set[str]:
    """Canonical-SMILES form of a set, skipping anything RDKit cannot parse."""
    from rdkit import Chem

    out: set[str] = set()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            out.add(Chem.MolToSmiles(mol))
    return out


def evaluate_unconditional(
    mols: list,
    reference_mols: list,
    *,
    training_smiles: set[str] | None = None,
    subsample: int = 1000,
    include_fcd: bool = False,
) -> dict[str, float | None]:
    """
    Compute unconditional molecular generation metrics.

    Parameters
    ----------
    mols : list
        Generated molecules (SMILES strings or RDKit Mol objects).
    reference_mols : list
        Reference molecules (SMILES strings or RDKit Mol objects) for FCD.
    training_smiles : set of str, optional
        Training-set canonical SMILES for novelty computation.  If None,
        novelty (and vun) are returned as None.
    subsample : int
        Max molecules used for diversity computation (default 1000).
    include_fcd : bool
        Whether to compute FCD (requires fcd_torch). Default False.

    Returns
    -------
    dict with keys:
        'validity'   : float in [0, 1]
        'uniqueness' : float in [0, 1]
        'novelty'    : float in [0, 1] or None
        'vun'        : float in [0, 1] (validity x uniqueness x novelty) or None
        'internal_diversity' : float in [0, 1]
        'sa_score_mean'      : float
        'sa_score_median'    : float
        'sa_score_std'       : float
        'fcd'        : float or None

    Examples
    --------
    >>> from metrics import evaluate_unconditional
    >>> results = evaluate_unconditional(generated_smiles, reference_smiles,
    ...                                  training_smiles=train_set)
    """
    # ------------------------------------------------------------------
    # Validity
    # ------------------------------------------------------------------
    n_total = len(mols)
    if n_total == 0:
        return {
            "validity": 0.0,
            "uniqueness": 0.0,
            "novelty": None if training_smiles is None else 0.0,
            "vun": None,
            "internal_diversity": 0.0,
            "sa_score_mean": float("nan"),
            "sa_score_median": float("nan"),
            "sa_score_std": float("nan"),
            "fcd": None,
        }

    rdkit_mols = to_mols(mols)
    valid_mols = [m for m in rdkit_mols if m is not None]
    validity = len(valid_mols) / n_total

    # ------------------------------------------------------------------
    # Canonical SMILES for valid molecules
    # ------------------------------------------------------------------
    from rdkit import Chem

    valid_smiles: list[str] = []
    for mol in valid_mols:
        try:
            smi = Chem.MolToSmiles(mol)
            if smi:
                valid_smiles.append(smi)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Uniqueness
    # ------------------------------------------------------------------
    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0.0

    # ------------------------------------------------------------------
    # Novelty
    # ------------------------------------------------------------------
    if training_smiles is None:
        novelty = None
    else:
        if not unique_smiles:
            novelty = 0.0
        else:
            # The generated side is canonical (built via MolToSmiles above), so
            # the training side must be too -- otherwise a model that reproduces
            # a training molecule written in a different-but-equivalent SMILES
            # form is scored as novel.
            training_canonical = _canonicalize(training_smiles)
            novel = unique_smiles - training_canonical
            novelty = len(novel) / len(unique_smiles)

    # ------------------------------------------------------------------
    # VUN
    # ------------------------------------------------------------------
    if novelty is None:
        vun = None
    else:
        vun = validity * uniqueness * novelty

    # ------------------------------------------------------------------
    # Diversity (average pairwise Tanimoto distance, Morgan ECFP4)
    # ------------------------------------------------------------------
    # sorted() -- unique_smiles is a set, whose iteration order varies with
    # PYTHONHASHSEED and would make the subsampled diversity irreproducible.
    diversity = compute_internal_diversity(sorted(unique_smiles), max_mols=subsample)[
        "internal_diversity"
    ]

    # ------------------------------------------------------------------
    # SA scores
    # ------------------------------------------------------------------
    sa_values: list[float] = []
    for mol in valid_mols:
        try:
            sa_values.append(sa_score(mol))
        except Exception:
            pass

    if sa_values:
        arr = np.array(sa_values, dtype=float)
        sa_mean = float(np.mean(arr))
        sa_median = float(np.median(arr))
        sa_std = float(np.std(arr))
    else:
        sa_mean = float("nan")
        sa_median = float("nan")
        sa_std = float("nan")

    # ------------------------------------------------------------------
    # FCD (optional)
    # ------------------------------------------------------------------
    fcd_value: float | None = None
    if include_fcd:
        ref_smiles_list = [s for s in to_smiles(reference_mols) if s is not None]
        fcd_value = _compute_fcd(list(valid_smiles), ref_smiles_list)

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "vun": vun,
        "internal_diversity": diversity,
        "sa_score_mean": sa_mean,
        "sa_score_median": sa_median,
        "sa_score_std": sa_std,
        "fcd": fcd_value,
    }


# ---------------------------------------------------------------------------
# Internal helpers (not part of the public API)
# ---------------------------------------------------------------------------


_fcd_warned = False


def _compute_fcd(
    generated_smiles: list[str],
    reference_smiles: list[str],
) -> float | None:
    """
    Frechet ChemNet Distance, or None when it cannot be computed.

    Composed from `get_predictions` and `calculate_frechet_distance` rather
    than calling `fcd.get_fcd`, for one reason: `get_fcd` does not expose
    `n_jobs`, so it inherits the default of 1, and `get_predictions` turns that
    into `DataLoader(..., num_workers=1)` -- one worker process. Called from a
    job that is itself a subprocess, that worker deadlocks: the generate smoke
    test hung indefinitely at 0% CPU, with and without pytest's output capture,
    and an orphaned process kept a core busy afterwards. `n_jobs=0` loads in
    this process, which removes the deadlock and is also about forty times
    faster on a small sample.

    None is returned when the `fcd` package is absent or either set is empty;
    the caller records the reason.
    """
    global _fcd_warned

    gen = [s for s in generated_smiles if s]
    ref = [s for s in reference_smiles if s]
    if not gen or not ref:
        return None

    try:
        import fcd as fcd_lib
    except ImportError:
        if not _fcd_warned:
            print(
                "-- Warning: the fcd package is not installed; the fcd metric "
                "will be skipped. Install with: uv pip install fcd",
                flush=True,
            )
            _fcd_warned = True
        return None

    try:
        model = fcd_lib.load_ref_model()
        act_gen = fcd_lib.get_predictions(model, gen, n_jobs=0)
        act_ref = fcd_lib.get_predictions(model, ref, n_jobs=0)
        return float(
            fcd_lib.calculate_frechet_distance(
                mu1=np.mean(act_gen, axis=0),
                sigma1=np.cov(act_gen.T),
                mu2=np.mean(act_ref, axis=0),
                sigma2=np.cov(act_ref.T),
            )
        )
    except Exception as error:
        print(f"-- Warning: FCD computation failed: {error}", flush=True)
        return None
