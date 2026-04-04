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
