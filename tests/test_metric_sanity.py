"""
Metric sanity over fixtures with hand-known answers.

Justification for a fifth test file: the other four exercise `src/graphinvent/`
and none of them imports `src/metrics/`, which is where every reported number
now comes from. Each assertion here guards a defect that shipped undetected and
whose failure mode is a plausible wrong number rather than a crash:

  - novelty was pinned at 1.0 by an isomeric-canonicalization mismatch, which
    would have made every generative result look better than it was
  - the funnel denominators could disagree, letting validity exceed 1.0
  - fcd was computed on a few hundred molecules, which is a different quantity
    from the published one rather than a noisy version of it, and on one
    molecule it hung the suite at 100% CPU with no error
  - fcd compared a flattened generated set against a raw stereo-bearing
    reference
  - the budget curve integrated one component of a multi-component objective
  - three implementations of the core metrics disagreed on success_rate
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "src" / "graphinvent")]

from metrics._core import canonical_smiles, compute_core_metrics  # noqa: E402
from metrics._unconditional import (  # noqa: E402
    FCD_MIN_MOLECULES,
    _compute_fcd,
)

# Two of these carry stereocenters. Flattening them collapses each onto its
# own flat form, which is what makes the novelty asymmetry measurable.
TRAINING = [
    "C[C@H](N)C(=O)O",
    "C[C@@H](N)C(=O)O",
    "CCO",
    "c1ccccc1",
]
# The three flat forms of the training set, plus one molecule that is genuinely
# absent from it.
GENERATED_FLAT = ["CC(N)C(=O)O", "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]


class TestNoveltyUnderBothChiralitySettings:
    """The bug: novelty read 1.0 whichever setting was used."""

    def test_flat_comparison_sees_the_copies(self):
        # With isomeric=False both sides are flat, so three of the four
        # generated molecules are training molecules and only aspirin is novel.
        core = compute_core_metrics(
            GENERATED_FLAT, n_sampled=4, training_smiles=TRAINING, isomeric=False
        )
        assert core["n_unique_valid"] == 4
        assert core["n_unique_valid_novel"] == 1, (
            "flattened, three of these four ARE training molecules; a novelty "
            "above 1/4 means the training side was not canonicalized the same way"
        )
        assert core["novelty"] == pytest.approx(0.25)

    def test_isomeric_comparison_counts_the_flattened_copies_as_novel(self):
        # With isomeric=True the flat alanine cannot equal either stereoisomer,
        # so it counts as novel: two of four, not one.
        core = compute_core_metrics(
            GENERATED_FLAT, n_sampled=4, training_smiles=TRAINING, isomeric=True
        )
        assert core["n_unique_valid_novel"] == 2
        assert core["novelty"] == pytest.approx(0.5)

    def test_the_two_settings_must_disagree(self):
        flat = compute_core_metrics(
            GENERATED_FLAT, n_sampled=4, training_smiles=TRAINING, isomeric=False
        )
        iso = compute_core_metrics(
            GENERATED_FLAT, n_sampled=4, training_smiles=TRAINING, isomeric=True
        )
        assert flat["novelty"] != iso["novelty"], (
            "identical novelty under both settings is the signature of the "
            "original bug: one side was canonicalized and the other was not"
        )

    def test_canonicalization_is_aligned_with_its_input(self):
        out = canonical_smiles(["CCO", "not_a_molecule", "c1ccccc1"])
        assert len(out) == 3
        assert out[1] is None, "entries RDKit rejects must stay in position"


class TestFunnelDenominators:
    """The bug: validity could exceed 1.0 when batches over-produced."""

    def test_the_funnel_composes(self):
        core = compute_core_metrics(
            GENERATED_FLAT + ["garbage", None],
            n_sampled=6,
            training_smiles=TRAINING,
            isomeric=False,
        )
        assert core["validity"] == pytest.approx(core["n_valid"] / core["n_sampled"])
        assert core["uniqueness"] == pytest.approx(
            core["n_unique_valid"] / core["n_valid"]
        )
        assert core["novelty"] == pytest.approx(
            core["n_unique_valid_novel"] / core["n_unique_valid"]
        )
        assert core["vun"] == pytest.approx(
            core["n_unique_valid_novel"] / core["n_sampled"]
        )
        # The composite is the product of the three stages, which is the
        # property that makes the funnel a funnel.
        assert core["vun"] == pytest.approx(
            core["validity"] * core["uniqueness"] * core["novelty"]
        )

    def test_no_rate_exceeds_one(self):
        core = compute_core_metrics(
            GENERATED_FLAT, n_sampled=4, training_smiles=TRAINING, isomeric=False
        )
        for key in ("validity", "uniqueness", "novelty", "vun"):
            assert 0.0 <= core[key] <= 1.0, f"{key} = {core[key]} is not a rate"


class TestFcdSampleFloor:
    """
    The bug: one molecule produced a NaN covariance and `scipy.linalg.sqrtm`
    never returned, hanging the suite at 100% CPU with no error. The timeout is
    the point of this test: without it a regression hangs rather than fails.
    """

    @pytest.mark.timeout(60)
    def test_one_molecule_returns_rather_than_hanging(self):
        result = _compute_fcd(["CCO"], ["CCO", "c1ccccc1"])
        assert result["value"] is None
        assert "insufficient_sample" in result["reason"]
        assert result["n_generated"] == 1

    @pytest.mark.timeout(60)
    def test_below_the_floor_is_not_computed(self):
        # Comfortably more than two molecules and far fewer than the floor: the
        # old two-molecule floor would have computed a number here.
        gen = [f"C{'C' * i}O" for i in range(1, 40)]
        result = _compute_fcd(gen, gen)
        assert result["value"] is None, (
            f"fcd must not be computed from {len(gen)} molecules; a 512-"
            "dimensional covariance from that sample is singular"
        )
        assert "insufficient_sample" in result["reason"]
        assert result["n_generated"] < FCD_MIN_MOLECULES

    def test_the_floor_is_a_statistical_one(self):
        assert FCD_MIN_MOLECULES >= 512, (
            "ChemNet activations are 512-dimensional, so a floor below 512 "
            "admits a singular covariance"
        )

    @pytest.mark.timeout(60)
    def test_counts_are_reported_whatever_happens(self):
        result = _compute_fcd(["CCO"], ["CCO"])
        assert "n_generated" in result and "n_reference" in result


class TestExactlyOneImplementation:
    """The bug: three implementations of the core metrics, two of them live."""

    def test_the_duplicate_module_is_gone(self):
        assert not (REPO_ROOT / "src" / "graphinvent" / "_metrics.py").exists(), (
            "src/graphinvent/_metrics.py is a second implementation of the core "
            "metrics and CLAUDE.md requires exactly one"
        )

    def test_the_dead_layer_is_not_exported(self):
        import metrics

        for dead in (
            "evaluate_unconditional",
            "evaluate_conditional",
            "evaluate_goal_directed",
        ):
            assert dead not in metrics.__all__
            assert not hasattr(metrics, dead)

    def test_success_rate_uses_at_least(self):
        # The two implementations disagreed: one used a strict `>`. The config
        # key reads as "a molecule is a success at or above this score", so a
        # molecule exactly on the threshold must count.
        from metrics._budget import compute_set_level_metrics

        result = compute_set_level_metrics([("CCO", 0.5)], success_threshold=0.5)
        assert result["success_rate"] == pytest.approx(
            1.0
        ), "a score exactly at the threshold must count as a success"


class TestTruncationInvariant:
    """
    The bug: batches over-produce, so more molecules were measured than were
    asked for and validity exceeded 1.0. Latent at low validity, which is why
    a smoke test on a briefly-trained model cannot catch it: the check has to
    be the invariant itself.
    """

    def test_more_valid_than_sampled_is_refused(self):
        with pytest.raises(ValueError, match="not truncated to n_samples"):
            compute_core_metrics(
                ["CCO", "c1ccccc1", "CCC"],
                n_sampled=2,
                training_smiles=TRAINING,
                isomeric=False,
            )

    def test_exactly_n_sampled_is_fine(self):
        core = compute_core_metrics(
            ["CCO", "c1ccccc1"], n_sampled=2, training_smiles=TRAINING, isomeric=False
        )
        assert core["validity"] == pytest.approx(1.0)
