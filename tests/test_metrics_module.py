"""
Unit tests for the src.metrics package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable regardless of install state
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from rdkit import Chem

# ---------------------------------------------------------------------------
# Shared fixture: load molecules from fixture_molecules.smi
# ---------------------------------------------------------------------------

_FIXTURE_SMI = Path(__file__).resolve().parent / "fixture_molecules.smi"


@pytest.fixture(scope="module")
def fixture_smiles() -> list[str]:
    lines = _FIXTURE_SMI.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


@pytest.fixture(scope="module")
def fixture_mols(fixture_smiles):
    return [Chem.MolFromSmiles(s) for s in fixture_smiles]


# ---------------------------------------------------------------------------
# TestUtils
# ---------------------------------------------------------------------------


class TestUtils:
    def test_to_mols_from_smiles(self, fixture_smiles):
        from metrics._utils import to_mols

        result = to_mols(fixture_smiles)
        assert isinstance(result, list)
        assert len(result) == len(fixture_smiles)
        # All fixture SMILES should be valid
        for mol in result:
            assert mol is not None

    def test_to_mols_with_none_entries(self):
        from metrics._utils import to_mols

        result = to_mols(["CCO", None, "c1ccccc1", None])
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is not None
        assert result[3] is None

    def test_to_mols_passthrough_mol_objects(self, fixture_mols):
        from metrics._utils import to_mols

        result = to_mols(fixture_mols)
        assert len(result) == len(fixture_mols)
        for original, returned in zip(fixture_mols, result):
            if original is None:
                assert returned is None
            else:
                assert returned is not None

    def test_to_smiles_from_mols(self, fixture_mols):
        from metrics._utils import to_smiles

        result = to_smiles(fixture_mols)
        assert len(result) == len(fixture_mols)
        for smi in result:
            # fixture_mols are all valid — all should return a string
            assert isinstance(smi, str) and len(smi) > 0

    def test_to_smiles_invalid_returns_none(self):
        from metrics._utils import to_smiles

        result = to_smiles(["CCO", "not_a_smiles_!!!", None])
        assert result[0] is not None  # valid
        assert result[1] is None  # invalid
        assert result[2] is None  # None input


# ---------------------------------------------------------------------------
# TestProperties
# ---------------------------------------------------------------------------


class TestProperties:
    @pytest.fixture(scope="class")
    def aspirin(self):
        return Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")

    @pytest.fixture(scope="class")
    def ethanol(self):
        return Chem.MolFromSmiles("CCO")

    def test_qed_returns_float_in_range(self, aspirin):
        from metrics._properties import qed

        val = qed(aspirin)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_sa_score_returns_float_in_range(self, aspirin):
        from metrics._properties import sa_score

        val = sa_score(aspirin)
        assert isinstance(val, float)
        assert 1.0 <= val <= 10.0

    def test_mol_weight_aspirin(self, aspirin):
        from metrics._properties import mol_weight

        val = mol_weight(aspirin)
        # Aspirin MW ~ 180.16
        assert abs(val - 180.0) < 5.0

    def test_logp_ethanol(self, ethanol):
        from metrics._properties import logp

        val = logp(ethanol)
        # Ethanol LogP ~ -0.31
        assert val < 1.0


# ---------------------------------------------------------------------------
# TestSuccessCriterion
# ---------------------------------------------------------------------------


class TestSuccessCriterion:
    @pytest.fixture(scope="class")
    def high_qed_mol(self):
        # Ibuprofen — QED ~ 0.72
        return Chem.MolFromSmiles("CC(C)Cc1ccc(CC(C)C(=O)O)cc1")

    @pytest.fixture(scope="class")
    def low_qed_mol(self):
        # Large aromatic — low QED
        return Chem.MolFromSmiles("c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34")

    def test_threshold_greater_passes(self, high_qed_mol):
        from metrics._criteria import SuccessCriterion

        sc = SuccessCriterion(
            property="qed", type="threshold", value=0.5, direction="greater"
        )
        assert sc.is_satisfied(high_qed_mol) is True

    def test_threshold_greater_fails(self, low_qed_mol):
        from metrics._criteria import SuccessCriterion

        sc = SuccessCriterion(
            property="qed", type="threshold", value=0.5, direction="greater"
        )
        assert sc.is_satisfied(low_qed_mol) is False

    def test_threshold_less_passes(self, low_qed_mol):
        from metrics._criteria import SuccessCriterion

        sc = SuccessCriterion(
            property="qed", type="threshold", value=0.9, direction="less"
        )
        assert sc.is_satisfied(low_qed_mol) is True

    def test_threshold_less_fails(self, high_qed_mol):
        from metrics._criteria import SuccessCriterion

        # QED of ibuprofen is above 0.1
        sc = SuccessCriterion(
            property="qed", type="threshold", value=0.1, direction="less"
        )
        assert sc.is_satisfied(high_qed_mol) is False

    def test_range_passes(self, high_qed_mol):
        from metrics._criteria import SuccessCriterion

        sc = SuccessCriterion(property="qed", type="range", min=0.0, max=1.0)
        assert sc.is_satisfied(high_qed_mol) is True

    def test_range_fails_below(self, high_qed_mol):
        from metrics._criteria import SuccessCriterion

        sc = SuccessCriterion(property="qed", type="range", min=0.99, max=1.0)
        assert sc.is_satisfied(high_qed_mol) is False

    def test_range_fails_above(self, high_qed_mol):
        from metrics._criteria import SuccessCriterion

        sc = SuccessCriterion(property="qed", type="range", min=0.0, max=0.01)
        assert sc.is_satisfied(high_qed_mol) is False

    def test_target_passes(self):
        from metrics._criteria import SuccessCriterion

        mol = Chem.MolFromSmiles("CCO")
        # mol_weight of ethanol ~ 46
        sc = SuccessCriterion(
            property="mol_weight", type="target", value=46.0, tolerance=5.0
        )
        assert sc.is_satisfied(mol) is True

    def test_target_fails(self):
        from metrics._criteria import SuccessCriterion

        mol = Chem.MolFromSmiles("CCO")
        sc = SuccessCriterion(
            property="mol_weight", type="target", value=200.0, tolerance=5.0
        )
        assert sc.is_satisfied(mol) is False

    def test_molecule_passes_all_criteria(self, high_qed_mol):
        from metrics._criteria import SuccessCriterion, molecule_passes

        criteria = [
            SuccessCriterion(
                property="qed", type="threshold", value=0.3, direction="greater"
            ),
            SuccessCriterion(property="qed", type="range", min=0.0, max=1.0),
        ]
        assert molecule_passes(high_qed_mol, criteria) is True

    def test_molecule_passes_fails_if_any_criterion_fails(self, high_qed_mol):
        from metrics._criteria import SuccessCriterion, molecule_passes

        criteria = [
            SuccessCriterion(
                property="qed", type="threshold", value=0.3, direction="greater"
            ),
            SuccessCriterion(
                property="qed", type="threshold", value=0.99, direction="greater"
            ),  # will fail
        ]
        assert molecule_passes(high_qed_mol, criteria) is False


# ---------------------------------------------------------------------------
# TestEvaluateUnconditional
# ---------------------------------------------------------------------------


class TestEvaluateUnconditional:
    @pytest.fixture(scope="class")
    def results_with_training(self, fixture_smiles):
        from metrics import evaluate_unconditional

        return evaluate_unconditional(
            fixture_smiles,
            fixture_smiles,
            training_smiles=set(),  # empty training set → novelty = 1.0
        )

    @pytest.fixture(scope="class")
    def results_no_training(self, fixture_smiles):
        from metrics import evaluate_unconditional

        return evaluate_unconditional(
            fixture_smiles,
            fixture_smiles,
            training_smiles=None,
        )

    def test_validity_in_range(self, results_with_training):
        v = results_with_training["validity"]
        assert 0.0 <= v <= 1.0

    def test_uniqueness_in_range(self):
        from metrics import evaluate_unconditional

        smiles_with_dups = ["CCO", "CCO", "c1ccccc1", "c1ccccc1", "CC(=O)O"]
        results = evaluate_unconditional(
            smiles_with_dups, smiles_with_dups, training_smiles=None
        )
        u = results["uniqueness"]
        assert 0.0 <= u <= 1.0
        # 3 unique out of 5 valid
        assert u < 1.0

    def test_diversity_in_range(self, results_with_training):
        d = results_with_training["diversity"]
        assert 0.0 <= d <= 1.0

    def test_sa_mean_in_range(self, results_with_training):
        sa = results_with_training["sa_mean"]
        assert 1.0 <= sa <= 10.0

    def test_novelty_one_when_training_empty(self, results_with_training):
        assert results_with_training["novelty"] == 1.0

    def test_novelty_none_when_training_none(self, results_no_training):
        assert results_no_training["novelty"] is None

    def test_vun_none_when_novelty_none(self, results_no_training):
        assert results_no_training["vun"] is None

    def test_fcd_none_when_not_requested(self, results_with_training):
        assert results_with_training["fcd"] is None

    def test_empty_list_returns_zeros(self):
        from metrics import evaluate_unconditional

        results = evaluate_unconditional([], [], training_smiles=set())
        assert results["validity"] == 0.0
        assert results["uniqueness"] == 0.0
        assert results["diversity"] == 0.0


# ---------------------------------------------------------------------------
# TestEvaluateConditional
# ---------------------------------------------------------------------------


class TestEvaluateConditional:
    def test_success_rate_all_pass(self, fixture_smiles):
        from metrics import SuccessCriterion, evaluate_conditional

        # QED > 0.0 — always true
        criteria = [
            SuccessCriterion(
                property="qed", type="threshold", value=0.0, direction="greater"
            )
        ]
        results = evaluate_conditional(
            fixture_smiles, fixture_smiles, criteria, training_smiles=None
        )
        assert results["success_rate"] == pytest.approx(1.0)

    def test_success_rate_none_pass(self, fixture_smiles):
        from metrics import SuccessCriterion, evaluate_conditional

        # QED > 1.0 — impossible
        criteria = [
            SuccessCriterion(
                property="qed", type="threshold", value=1.0, direction="greater"
            )
        ]
        results = evaluate_conditional(
            fixture_smiles, fixture_smiles, criteria, training_smiles=None
        )
        assert results["success_rate"] == pytest.approx(0.0)

    def test_rediscovery_rate_none_when_no_reference(self, fixture_smiles):
        from metrics import SuccessCriterion, evaluate_conditional

        criteria = [
            SuccessCriterion(
                property="qed", type="threshold", value=0.0, direction="greater"
            )
        ]
        results = evaluate_conditional(
            fixture_smiles, [], criteria, training_smiles=None
        )
        assert results["rediscovery_rate"] is None

    def test_all_unconditional_keys_present(self, fixture_smiles):
        from metrics import SuccessCriterion, evaluate_conditional

        criteria = [
            SuccessCriterion(
                property="qed", type="threshold", value=0.0, direction="greater"
            )
        ]
        results = evaluate_conditional(
            fixture_smiles, fixture_smiles, criteria, training_smiles=None
        )
        for key in (
            "validity",
            "uniqueness",
            "novelty",
            "vun",
            "diversity",
            "sa_mean",
            "sa_median",
            "sa_std",
            "fcd",
        ):
            assert key in results, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# TestEvaluateGoalDirected
# ---------------------------------------------------------------------------


class TestComputeInternalDiversity:
    """Tests for compute_internal_diversity in src/metrics/_internal_diversity.py."""

    def _fn(self, *args, **kwargs):
        from metrics._internal_diversity import compute_internal_diversity

        return compute_internal_diversity(*args, **kwargs)

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_identical_molecules_diversity_zero(self):
        """A set of identical molecules should have internal diversity 0.0."""
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        result = self._fn([aspirin] * 10)
        assert result["internal_diversity"] == pytest.approx(0.0, abs=1e-6)
        assert result["n_duplicates_removed"] == 9

    def test_diverse_fixture_set_high_diversity(self, fixture_smiles):
        """A diverse set of real molecules should have internal diversity > 0.5."""
        result = self._fn(fixture_smiles)
        assert result["internal_diversity"] > 0.5

    def test_result_keys_present(self, fixture_smiles):
        result = self._fn(fixture_smiles)
        for key in (
            "internal_diversity",
            "mean_internal_similarity",
            "median_internal_similarity",
            "max_internal_similarity",
            "sim_gt_0_4",
            "sim_gt_0_6",
            "sim_gt_0_8",
            "sim_gt_0_9",
            "n_duplicates_removed",
            "n_invalid",
            "n_molecules",
            "subsampled",
            "pairwise_similarities",
        ):
            assert key in result, f"Missing key: {key}"

    def test_diversity_plus_mean_sim_equals_one(self, fixture_smiles):
        """internal_diversity + mean_internal_similarity should sum to 1."""
        result = self._fn(fixture_smiles)
        total = result["internal_diversity"] + result["mean_internal_similarity"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_pairwise_length_correct(self, fixture_smiles):
        """Upper-triangle has n*(n-1)//2 entries for n unique valid molecules."""
        result = self._fn(fixture_smiles)
        n = result["n_molecules"]
        expected_pairs = n * (n - 1) // 2
        assert len(result["pairwise_similarities"]) == expected_pairs

    def test_similarities_in_range(self, fixture_smiles):
        result = self._fn(fixture_smiles)
        assert 0.0 <= result["internal_diversity"] <= 1.0
        assert 0.0 <= result["mean_internal_similarity"] <= 1.0
        assert 0.0 <= result["max_internal_similarity"] <= 1.0

    def test_max_gte_mean(self, fixture_smiles):
        result = self._fn(fixture_smiles)
        assert result["max_internal_similarity"] >= result["mean_internal_similarity"]

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_list_returns_gracefully(self):
        result = self._fn([])
        assert result["internal_diversity"] == 0.0
        assert result["n_molecules"] == 0
        assert len(result["pairwise_similarities"]) == 0

    def test_single_molecule_warns_and_returns_zero(self):
        with pytest.warns(UserWarning, match="one unique valid molecule"):
            result = self._fn(["CCO"])
        assert result["internal_diversity"] == 0.0
        assert result["n_molecules"] == 1

    def test_invalid_smiles_skipped(self):
        result = self._fn(["CCO", "NOT_SMILES", "c1ccccc1"])
        assert result["n_invalid"] == 1
        assert result["n_molecules"] == 2

    def test_deduplication_counts(self):
        """Duplicate SMILES should be deduplicated and counted."""
        smiles = ["CCO", "CCO", "CCO", "c1ccccc1"]
        result = self._fn(smiles)
        assert result["n_duplicates_removed"] == 2
        assert result["n_molecules"] == 2

    def test_max_mols_subsamples(self, fixture_smiles):
        """max_mols should limit the number of molecules used."""
        result = self._fn(fixture_smiles, max_mols=5)
        assert result["n_molecules"] == 5
        assert result["subsampled"] is True

    def test_no_subsampling_when_below_limit(self, fixture_smiles):
        result = self._fn(fixture_smiles, max_mols=10000)
        assert result["subsampled"] is False

    def test_max_mols_none_disables_subsampling(self, fixture_smiles):
        result = self._fn(fixture_smiles, max_mols=None)
        assert result["subsampled"] is False


class TestComputeTestSetSimilarity:
    """Tests for compute_test_set_similarity in src/metrics/_similarity.py."""

    def _fn(self, *args, **kwargs):
        from metrics._similarity import compute_test_set_similarity

        return compute_test_set_similarity(*args, **kwargs)

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_self_similarity_is_one(self):
        """A molecule should have similarity 1.0 to itself."""
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        result = self._fn([aspirin], [aspirin])
        assert result["mean_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_low_similarity_sanity_check(self):
        """Methane vs sulfasalazine should give a low similarity."""
        methane = "C"
        sulfasalazine = "Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1"
        result = self._fn([methane], [sulfasalazine])
        assert result["mean_similarity"] < 0.3

    def test_result_keys_present(self, fixture_smiles):
        """All expected keys must be present in the result."""
        result = self._fn(fixture_smiles, fixture_smiles)
        for key in (
            "mean_similarity",
            "median_similarity",
            "top_k_similarity",
            "sim_gt_0_4",
            "sim_gt_0_6",
            "sim_gt_0_8",
            "sim_gt_0_9",
            "exact_rediscovery_count",
            "n_invalid_generated",
            "n_invalid_test",
            "n_test_after_filter",
            "per_mol_similarity",
        ):
            assert key in result, f"Missing key: {key}"

    def test_similarities_in_range(self, fixture_smiles):
        """Similarity values must lie in [0, 1]."""
        result = self._fn(fixture_smiles, fixture_smiles)
        assert 0.0 <= result["mean_similarity"] <= 1.0
        assert 0.0 <= result["median_similarity"] <= 1.0
        assert 0.0 <= result["top_k_similarity"] <= 1.0

    def test_per_mol_length_matches_valid_generated(self):
        """per_mol_similarity should have one entry per valid generated molecule."""
        smiles = ["CCO", "INVALID_SMILES_XYZ", "c1ccccc1"]
        test = ["CC(=O)O"]
        result = self._fn(smiles, test)
        assert len(result["per_mol_similarity"]) == 2  # two valid SMILES
        assert result["n_invalid_generated"] == 1

    def test_exact_rediscovery_when_identical(self, fixture_smiles):
        """Generating exact test molecules should yield non-zero exact_rediscovery_count."""
        result = self._fn(fixture_smiles, fixture_smiles)
        assert result["exact_rediscovery_count"] == len(fixture_smiles)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_generated_returns_gracefully(self, fixture_smiles):
        result = self._fn([], fixture_smiles)
        assert result["mean_similarity"] == 0.0
        assert result["per_mol_similarity"] == []

    def test_empty_test_returns_gracefully(self, fixture_smiles):
        result = self._fn(fixture_smiles, [])
        assert result["mean_similarity"] == 0.0
        assert result["n_test_after_filter"] == 0

    def test_all_invalid_generated(self, fixture_smiles):
        result = self._fn(["NOT_A_SMILES", "ALSO_BAD"], fixture_smiles)
        assert result["mean_similarity"] == 0.0
        assert result["n_invalid_generated"] == 2

    def test_invalid_test_smiles_skipped(self):
        """Invalid SMILES in the test set are counted but not used."""
        valid = "CCO"
        result = self._fn([valid], ["NOT_SMILES", "CCO"])
        assert result["n_invalid_test"] == 1
        assert result["mean_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_top_k_lte_mean(self, fixture_smiles):
        """top_k_similarity should be >= mean_similarity (it's the top-k mean)."""
        result = self._fn(fixture_smiles, fixture_smiles, top_k=5)
        assert result["top_k_similarity"] >= result["mean_similarity"] - 1e-9

    def test_max_refs_subsamples(self, fixture_smiles):
        """max_refs limits the number of reference molecules used."""
        result = self._fn(fixture_smiles, fixture_smiles, max_refs=3)
        assert result["n_test_after_filter"] == 3

    # ------------------------------------------------------------------
    # Condition filtering
    # ------------------------------------------------------------------

    def test_condition_filter_subsets_test_set(self):
        """condition_filter should reduce n_test_after_filter."""
        smiles = ["CCO", "c1ccccc1"]
        conditions = [{"logp": 0.1}, {"logp": 2.5}]
        cond_filter = {"logp": {"value": 0.1, "tolerance": 0.5}}
        result_filtered = self._fn(
            smiles,
            smiles,
            condition_filter=cond_filter,
            test_conditions=conditions,
        )
        result_unfiltered = self._fn(smiles, smiles)
        assert (
            result_filtered["n_test_after_filter"]
            < result_unfiltered["n_test_after_filter"]
        )

    def test_condition_filter_without_test_conditions_warns(self, fixture_smiles):
        """Providing condition_filter without test_conditions should warn, not crash."""
        cond_filter = {"logp": {"value": 1.0, "tolerance": 0.5}}
        with pytest.warns(UserWarning, match="test_conditions is None"):
            result = self._fn(
                fixture_smiles, fixture_smiles, condition_filter=cond_filter
            )
        # Should still return results (unfiltered)
        assert result["mean_similarity"] >= 0.0


class TestEvaluateGoalDirected:
    @pytest.fixture(scope="class")
    def gd_results(self, fixture_smiles):
        from metrics import SuccessCriterion, evaluate_goal_directed

        criteria = [
            SuccessCriterion(
                property="qed", type="threshold", value=0.0, direction="greater"
            )
        ]
        return evaluate_goal_directed(
            fixture_smiles,
            fixture_smiles,
            criteria,
            oracle_calls=5000,
            training_smiles=None,
        )

    def test_oracle_calls_matches_input(self, gd_results):
        assert gd_results["oracle_calls"] == 5000

    def test_sample_efficiency_is_none(self, gd_results):
        assert gd_results["sample_efficiency"] is None

    def test_all_conditional_keys_present(self, gd_results):
        for key in (
            "validity",
            "uniqueness",
            "novelty",
            "vun",
            "diversity",
            "sa_mean",
            "sa_median",
            "sa_std",
            "fcd",
            "success_rate",
            "conditional_vun",
            "rediscovery_rate",
        ):
            assert key in gd_results, f"Missing key: {key}"
