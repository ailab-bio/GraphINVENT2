"""
The Frechet ChemNet Distance helper.

This module used to carry a second implementation of the core metric table --
`evaluate_unconditional`, with its own validity, uniqueness, novelty, internal
diversity and SA score, plus `_conditional` and `_goal_directed` layered on
top. CLAUDE.md requires exactly one implementation of each metric, and
`src/metrics/_core.py` is it, so those three are gone. They were exported and
uncalled, and they had drifted: no `isomeric` parameter, so they still carried
the novelty-pinning bug that `_core` fixed, a diversity subsample of 1,000
against the specified 10,000, and SA computed over valid rather than unique
valid molecules.
"""

from __future__ import annotations

import numpy as np

# CLAUDE.md reports fcd as the headline unconditional number. These two bound
# when it is meaningful: see _compute_fcd for why.
FCD_MIN_MOLECULES = 1000
FCD_COMPARABLE_MOLECULES = 10000


def _compute_fcd(
    generated_smiles: list,
    reference_smiles: list,
) -> dict:
    """
    Frechet ChemNet Distance, with the counts it was computed from.

    Returns a dict: `value`, `reason` (None when a value was produced),
    `n_generated`, `n_reference`, and `biased` (True when the sample is above
    the floor but below the size published numbers use).

    **Both sides must be canonicalized the same way before they get here.** The
    caller passes canonical SMILES produced with one `isomeric` flag for the
    generated set and the reference set alike. They used not to be: the
    generated side was canonicalized and flattened to match `use_chirality`
    while the reference split went in raw from `read_smiles`, so one side
    carried `@` and `/` tokens and the other did not. ChemNet reads SMILES
    characters, so that was a systematic distribution shift inflating the
    distance by an unknown amount.

    The floor is 1,000 unique molecules a side, and it is a statistical floor
    rather than a mathematical one. ChemNet activations are 512-dimensional, so
    a sample covariance below 512 molecules is singular and is badly
    conditioned well above it. An FCD on a few hundred molecules is not a noisy
    estimate of the real number, it is a different and much larger quantity, so
    it is not computed at all rather than reported with a caveat. Published
    values use 30,000 (MOSES) or 10,000 (GuacaMol), so between the floor and
    10,000 the value is returned with `biased` set: usable for comparing our
    own runs to each other, not comparable to a published number.

    Two is still the mathematical minimum and the reason the old floor existed:
    `np.cov` of a single row divides by `n - 1 == 0` and returns a 512x512
    matrix of NaN, and `calculate_frechet_distance` hands that to
    `scipy.linalg.sqrtm` *before* its own finiteness check. `sqrtm` of a NaN
    matrix that size never finishes, so the process sits at 100% CPU with no
    error. The 1,000 floor subsumes it; the regression test for the
    one-molecule case stays because the 1,000 floor is a policy and that was a
    hang.
    """
    # Deduplicated, because the floor is stated in unique molecules and the
    # value has to be computed from the same set the counts describe. Sorted so
    # the subsample a caller might take downstream is reproducible.
    gen = sorted({s for s in generated_smiles if s})
    ref = sorted({s for s in reference_smiles if s})

    if len(gen) < FCD_MIN_MOLECULES or len(ref) < FCD_MIN_MOLECULES:
        return {
            "value": None,
            "reason": (
                f"insufficient_sample: the Frechet ChemNet Distance needs at "
                f"least {FCD_MIN_MOLECULES} unique valid molecules on each side "
                f"and got {len(gen)} generated against {len(ref)} reference. "
                "ChemNet activations are 512-dimensional, so a covariance from "
                "fewer is singular and the result would be a different quantity "
                "rather than a noisy one."
            ),
            "n_generated": len(gen),
            "n_reference": len(ref),
            "biased": False,
        }

    try:
        import fcd as fcd_lib
    except ImportError:
        return {
            "value": None,
            "reason": (
                "optional_tool_missing: the fcd package is not installed; "
                "install it with `uv sync --extra metrics`"
            ),
            "n_generated": len(gen),
            "n_reference": len(ref),
            "biased": False,
        }

    model = fcd_lib.load_ref_model()
    act_gen = fcd_lib.get_predictions(model, gen, n_jobs=0)
    act_ref = fcd_lib.get_predictions(model, ref, n_jobs=0)

    for label, act in (("generated", act_gen), ("reference", act_ref)):
        if act.shape[0] < FCD_MIN_MOLECULES:
            return {
                "value": None,
                "reason": (
                    f"insufficient_sample: ChemNet embedded only "
                    f"{act.shape[0]} of the {label} molecules, below the "
                    f"{FCD_MIN_MOLECULES} floor"
                ),
                "n_generated": int(act_gen.shape[0]),
                "n_reference": int(act_ref.shape[0]),
                "biased": False,
            }

    mu_gen, sigma_gen = np.mean(act_gen, axis=0), np.cov(act_gen.T)
    mu_ref, sigma_ref = np.mean(act_ref, axis=0), np.cov(act_ref.T)

    # Above the floor this cannot happen, so it means something upstream is
    # wrong. Raise rather than hand sqrtm a matrix it will never finish.
    for label, sigma in (("generated", sigma_gen), ("reference", sigma_ref)):
        if not np.isfinite(sigma).all():
            raise ValueError(
                f"the {label} ChemNet covariance contains non-finite values "
                "above the sample floor. Refusing to call sqrtm, which would "
                "not terminate."
            )

    value = float(
        fcd_lib.calculate_frechet_distance(
            mu1=mu_gen, sigma1=sigma_gen, mu2=mu_ref, sigma2=sigma_ref
        )
    )
    return {
        "value": value,
        "reason": None,
        "n_generated": int(act_gen.shape[0]),
        "n_reference": int(act_ref.shape[0]),
        "biased": min(act_gen.shape[0], act_ref.shape[0]) < FCD_COMPARABLE_MOLECULES,
    }
