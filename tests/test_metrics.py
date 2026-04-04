"""
Unit tests for graphinvent/metrics.py.

Covers all six public functions:
  compute_novelty, compute_sa_scores, compute_diversity,
  compute_fcd, compute_rediscovery_rate, compute_success_rate

No real dataset or trained model is required — tests use simple hand-crafted
SMILES strings and RDKit Mol objects.

Usage:
    pytest tests/test_metrics.py -v
"""

import math
import sys
from pathlib import Path

import pytest
import torch
from rdkit import Chem

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_GRAPHINVENT = Path(__file__).resolve().parent.parent / "src/graphinvent"
if str(_GRAPHINVENT) not in sys.path:
    sys.path.insert(0, str(_GRAPHINVENT))

import _metrics as metrics  # noqa: E402  (after sys.path setup)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# A handful of simple, valid SMILES
_ETHANOL = "CCO"
_BENZENE = "c1ccccc1"
_METHANE = "C"
_ACETONE = "CC(=O)C"
_ACETIC = "CC(=O)O"

_VALID_SMILES = [_ETHANOL, _BENZENE, _METHANE, _ACETONE, _ACETIC]
_VALID_MOLS = [Chem.MolFromSmiles(s) for s in _VALID_SMILES]


# ===========================================================================
# compute_novelty
# ===========================================================================


class TestComputeNovelty:

    def test_all_novel(self):
        """All generated SMILES absent from training → novelty = 1.0."""
        gen = [_ETHANOL, _BENZENE]
        train = {_METHANE, _ACETONE}
        assert metrics.compute_novelty(gen, train) == pytest.approx(1.0)

    def test_none_novel(self):
        """All generated SMILES already in training set → novelty = 0.0."""
        gen = [_ETHANOL, _BENZENE]
        train = {_ETHANOL, _BENZENE, _METHANE}
        assert metrics.compute_novelty(gen, train) == pytest.approx(0.0)

    def test_partial_novelty(self):
        """Half novel → novelty ≈ 0.5."""
        gen = [_ETHANOL, _BENZENE]
        train = {_ETHANOL}
        assert metrics.compute_novelty(gen, train) == pytest.approx(0.5)

    def test_none_entries_ignored(self):
        """None entries (invalid molecules) are excluded from both numerator and denominator."""
        gen = [_ETHANOL, None, None]
        train = set()
        # Only 1 valid unique SMILES, all novel → 1.0
        assert metrics.compute_novelty(gen, train) == pytest.approx(1.0)

    def test_all_none_returns_zero(self):
        """List of only None entries → 0.0 (no valid SMILES)."""
        assert metrics.compute_novelty([None, None], {"CCO"}) == pytest.approx(0.0)

    def test_empty_list_returns_zero(self):
        assert metrics.compute_novelty([], {"CCO"}) == pytest.approx(0.0)

    def test_empty_training_set(self):
        """Empty training set → all generated are novel → 1.0."""
        gen = [_ETHANOL, _BENZENE]
        assert metrics.compute_novelty(gen, set()) == pytest.approx(1.0)

    def test_duplicates_in_generated_count_once(self):
        """Duplicate SMILES in generated list count as one unique entry."""
        gen = [_ETHANOL, _ETHANOL, _ETHANOL]  # 1 unique
        train = {_BENZENE}  # _ETHANOL is novel
        assert metrics.compute_novelty(gen, train) == pytest.approx(1.0)

    def test_result_in_range(self):
        """Result must always be in [0.0, 1.0]."""
        gen = _VALID_SMILES + [None]
        train = {_ETHANOL, _BENZENE}
        result = metrics.compute_novelty(gen, train)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# compute_sa_scores
# ===========================================================================


class TestComputeSaScores:

    def test_returns_three_floats(self):
        mean, median, std = metrics.compute_sa_scores(_VALID_MOLS)
        assert isinstance(mean, float)
        assert isinstance(median, float)
        assert isinstance(std, float)

    def test_scores_in_valid_range(self):
        """SA scores are in [1, 10]."""
        mean, median, std = metrics.compute_sa_scores(_VALID_MOLS)
        assert 1.0 <= mean <= 10.0
        assert 1.0 <= median <= 10.0
        assert std >= 0.0

    def test_empty_list_returns_nan(self):
        mean, median, std = metrics.compute_sa_scores([])
        assert math.isnan(mean)
        assert math.isnan(median)
        assert math.isnan(std)

    def test_all_none_returns_nan(self):
        mean, median, std = metrics.compute_sa_scores([None, None])
        assert math.isnan(mean)

    def test_none_entries_skipped(self):
        """None entries don't crash; valid mols still scored."""
        mols_with_none = [None, _VALID_MOLS[0], None, _VALID_MOLS[1]]
        mean, median, std = metrics.compute_sa_scores(mols_with_none)
        assert not math.isnan(mean)
        assert 1.0 <= mean <= 10.0

    def test_single_molecule_std_zero(self):
        """Single molecule → std = 0.0."""
        _, _, std = metrics.compute_sa_scores([_VALID_MOLS[0]])
        assert std == pytest.approx(0.0)

    def test_aspirin_reasonable_sa_score(self):
        """Aspirin is a simple, well-known drug — SA score should be in [1, 4]."""
        aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        mean, _, _ = metrics.compute_sa_scores([aspirin])
        assert 1.0 <= mean <= 4.0, f"Unexpected SA score for aspirin: {mean}"


# ===========================================================================
# compute_diversity
# ===========================================================================


class TestComputeDiversity:

    def test_returns_float_in_range(self):
        result = metrics.compute_diversity(_VALID_SMILES)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_single_molecule_returns_zero(self):
        assert metrics.compute_diversity([_ETHANOL]) == pytest.approx(0.0)

    def test_empty_list_returns_zero(self):
        assert metrics.compute_diversity([]) == pytest.approx(0.0)

    def test_all_none_returns_zero(self):
        assert metrics.compute_diversity([None, None]) == pytest.approx(0.0)

    def test_identical_molecules_diversity_near_zero(self):
        """A set of identical molecules has Tanimoto similarity = 1 → diversity ≈ 0."""
        dupes = [_BENZENE] * 5
        result = metrics.compute_diversity(dupes)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_structurally_different_molecules_have_positive_diversity(self):
        """Structurally distinct molecules should have diversity > 0."""
        diverse = [_METHANE, _BENZENE, "c1ccc(cc1)N", "CC(=O)Nc1ccc(O)cc1"]
        result = metrics.compute_diversity(diverse)
        assert result > 0.0

    def test_subsample_limits_molecules_used(self):
        """Passing subsample=2 still returns a valid float and doesn't crash."""
        result = metrics.compute_diversity(_VALID_SMILES, subsample=2)
        assert 0.0 <= result <= 1.0

    def test_none_entries_filtered_out(self):
        """None entries in smiles_list are skipped gracefully."""
        smiles_with_none = [_ETHANOL, None, _BENZENE, None]
        result = metrics.compute_diversity(smiles_with_none)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# compute_fcd
# ===========================================================================


class TestComputeFcd:

    def test_returns_none_or_float(self):
        """Returns either None (fcd_torch missing) or a non-negative float."""
        result = metrics.compute_fcd(_VALID_SMILES, _VALID_SMILES)
        assert result is None or (isinstance(result, float) and result >= 0.0)

    def test_empty_generated_returns_none(self):
        result = metrics.compute_fcd([], _VALID_SMILES)
        assert result is None

    def test_empty_reference_returns_none(self):
        result = metrics.compute_fcd(_VALID_SMILES, [])
        assert result is None

    def test_all_none_generated_returns_none(self):
        result = metrics.compute_fcd([None, None], _VALID_SMILES)
        assert result is None

    def test_does_not_raise(self):
        """compute_fcd must never raise an exception regardless of input."""
        try:
            metrics.compute_fcd(_VALID_SMILES, _VALID_SMILES)
            metrics.compute_fcd([], [])
            metrics.compute_fcd([None], [None])
        except Exception as e:
            pytest.fail(f"compute_fcd raised an exception: {e}")


# ===========================================================================
# compute_rediscovery_rate
# ===========================================================================


class TestComputeRediscoveryRate:

    def test_full_rediscovery(self):
        """All test SMILES present in generated → rate = 1.0."""
        gen = [_ETHANOL, _BENZENE, _METHANE]
        test = {_ETHANOL, _BENZENE}
        assert metrics.compute_rediscovery_rate(gen, test) == pytest.approx(1.0)

    def test_no_rediscovery(self):
        """No test SMILES in generated → rate = 0.0."""
        gen = [_ACETONE, _ACETIC]
        test = {_ETHANOL, _BENZENE}
        assert metrics.compute_rediscovery_rate(gen, test) == pytest.approx(0.0)

    def test_partial_rediscovery(self):
        """Half of test set rediscovered → rate = 0.5."""
        gen = [_ETHANOL, _ACETONE]
        test = {_ETHANOL, _BENZENE}
        assert metrics.compute_rediscovery_rate(gen, test) == pytest.approx(0.5)

    def test_empty_test_set_returns_zero(self):
        assert metrics.compute_rediscovery_rate([_ETHANOL], set()) == pytest.approx(0.0)

    def test_empty_generated_returns_zero(self):
        assert metrics.compute_rediscovery_rate([], {_ETHANOL}) == pytest.approx(0.0)

    def test_none_entries_in_generated_ignored(self):
        gen = [_ETHANOL, None, None]
        test = {_ETHANOL}
        assert metrics.compute_rediscovery_rate(gen, test) == pytest.approx(1.0)

    def test_result_in_range(self):
        gen = _VALID_SMILES + [None]
        test = {_ETHANOL, _BENZENE, "NOTASMILE"}
        result = metrics.compute_rediscovery_rate(gen, test)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# compute_success_rate
# ===========================================================================


class TestComputeSuccessRate:

    def test_all_above_threshold(self):
        scores = torch.tensor([0.8, 0.9, 1.0])
        assert metrics.compute_success_rate(scores, threshold=0.5) == pytest.approx(1.0)

    def test_none_above_threshold(self):
        scores = torch.tensor([0.1, 0.2, 0.3])
        assert metrics.compute_success_rate(scores, threshold=0.5) == pytest.approx(0.0)

    def test_half_above_threshold(self):
        scores = torch.tensor([0.0, 0.0, 1.0, 1.0])
        assert metrics.compute_success_rate(scores, threshold=0.5) == pytest.approx(0.5)

    def test_empty_tensor_returns_zero(self):
        assert metrics.compute_success_rate(
            torch.tensor([]), threshold=0.5
        ) == pytest.approx(0.0)

    def test_threshold_is_strict_greater_than(self):
        """Score equal to threshold should NOT count as success (strict >)."""
        scores = torch.tensor([0.5, 0.5, 0.5])
        assert metrics.compute_success_rate(scores, threshold=0.5) == pytest.approx(0.0)

    def test_threshold_zero_all_positive_scores_succeed(self):
        scores = torch.tensor([0.1, 0.5, 0.9])
        assert metrics.compute_success_rate(scores, threshold=0.0) == pytest.approx(1.0)

    def test_result_is_float(self):
        scores = torch.tensor([0.6, 0.4])
        result = metrics.compute_success_rate(scores, threshold=0.5)
        assert isinstance(result, float)

    def test_result_in_range(self):
        scores = torch.rand(20)
        result = metrics.compute_success_rate(scores, threshold=0.5)
        assert 0.0 <= result <= 1.0
