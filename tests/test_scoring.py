"""
Unit tests for graphinvent/ScoringFunction.py.

Focuses on the refactored compute_score / compute_score_with_components methods:
  - compute_score returns the same final tensor as compute_score_with_components
  - component_scores dict has the expected keys and value shapes
  - invalid / duplicate / improperly-terminated molecules are zeroed out
  - QED score is in [0, 1]
  - continuous vs binary score_type produce different but sensible results

No real QSAR model is needed — only the QED and target_size components are
tested (both are computed from the molecule alone without external files).

Usage:
    pytest tests/test_scoring.py -v
"""

import sys
from collections import namedtuple
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

from ScoringFunction import ScoringFunction  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal fake graph class
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Minimal stand-in for a GenerationGraph used by ScoringFunction."""

    def __init__(self, smiles: str, n_nodes: int = 5):
        self.molecule = Chem.MolFromSmiles(smiles) if smiles else None
        self.n_nodes = n_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMILES = ["CCO", "c1ccccc1", "CC(=O)C", "CC(=O)O", "C"]


def _make_constants(
    score_components=("QED",),
    score_thresholds=(0.0,),
    score_type="continuous",
    max_n_nodes=20,
    oracles=None,
    uncertainty_modulation=None,
) -> namedtuple:
    fields = [
        "score_components",
        "score_thresholds",
        "score_type",
        "qsar_models",
        "device",
        "max_n_nodes",
        "oracles",
        "uncertainty_modulation",
    ]
    C = namedtuple("C", fields)
    return C(
        score_components=list(score_components),
        score_thresholds=list(score_thresholds),
        score_type=score_type,
        qsar_models={},
        device="cpu",
        max_n_nodes=max_n_nodes,
        oracles=dict(oracles or {}),
        uncertainty_modulation=dict(uncertainty_modulation or {}),
    )


def _all_valid_unique(n):
    """All-ones validity, uniqueness, and termination tensors."""
    return (
        torch.ones(n),
        torch.ones(n),
        torch.ones(n),
    )


# ===========================================================================
# compute_score and compute_score_with_components consistency
# ===========================================================================


class TestScoreConsistency:

    def test_compute_score_matches_with_components(self):
        """compute_score must return the same tensor as compute_score_with_components."""
        graphs = [_FakeGraph(s) for s in _SMILES]
        n = len(graphs)
        termination, validity, uniqueness = _all_valid_unique(n)
        c = _make_constants()
        sf = ScoringFunction(c)

        score_simple = sf.compute_score(graphs, termination, validity, uniqueness)
        score_full, _ = sf.compute_score_with_components(
            graphs, termination, validity, uniqueness
        )

        assert torch.allclose(score_simple, score_full, atol=1e-6)

    def test_component_keys_match_score_components(self):
        """component_scores dict must have exactly the keys in score_components."""
        graphs = [_FakeGraph(s) for s in _SMILES]
        n = len(graphs)
        termination, validity, uniqueness = _all_valid_unique(n)
        components = ["QED", "target_size=5"]
        c = _make_constants(
            score_components=components,
            score_thresholds=[0.0, 0.0],
            score_type="continuous",
        )
        sf = ScoringFunction(c)
        _, comp_dict = sf.compute_score_with_components(
            graphs, termination, validity, uniqueness
        )

        assert set(comp_dict.keys()) == set(components)

    def test_component_tensors_have_correct_length(self):
        """Each component tensor must have length == number of graphs."""
        graphs = [_FakeGraph(s) for s in _SMILES]
        n = len(graphs)
        termination, validity, uniqueness = _all_valid_unique(n)
        c = _make_constants()
        sf = ScoringFunction(c)
        _, comp_dict = sf.compute_score_with_components(
            graphs, termination, validity, uniqueness
        )

        for name, tensor in comp_dict.items():
            assert tensor.shape[0] == n, f"Component '{name}' has wrong length"


# ===========================================================================
# Masking: invalid / duplicate / improperly-terminated molecules
# ===========================================================================


class TestMasking:

    def test_invalid_molecules_zeroed(self):
        """Molecules flagged as invalid (validity=0) must receive score 0."""
        graphs = [_FakeGraph(s) for s in _SMILES]
        n = len(graphs)
        validity = torch.zeros(n)  # all invalid
        uniqueness = torch.ones(n)
        termination = torch.ones(n)
        c = _make_constants()
        sf = ScoringFunction(c)
        score = sf.compute_score(graphs, termination, validity, uniqueness)
        assert (score == 0).all(), "Invalid molecules should have score 0"

    def test_duplicate_molecules_zeroed(self):
        """Molecules flagged as duplicates (uniqueness=0) must receive score 0."""
        graphs = [_FakeGraph(s) for s in _SMILES]
        n = len(graphs)
        validity = torch.ones(n)
        uniqueness = torch.zeros(n)  # all duplicates
        termination = torch.ones(n)
        c = _make_constants()
        sf = ScoringFunction(c)
        score = sf.compute_score(graphs, termination, validity, uniqueness)
        assert (score == 0).all(), "Duplicate molecules should have score 0"

    def test_improperly_terminated_zeroed(self):
        """Molecules with termination=0 must receive score 0."""
        graphs = [_FakeGraph(s) for s in _SMILES]
        n = len(graphs)
        validity = torch.ones(n)
        uniqueness = torch.ones(n)
        termination = torch.zeros(n)  # all improperly terminated
        c = _make_constants()
        sf = ScoringFunction(c)
        score = sf.compute_score(graphs, termination, validity, uniqueness)
        assert (score == 0).all()

    def test_partial_mask(self):
        """Only properly valid unique molecules should have non-zero score."""
        graphs = [_FakeGraph(s) for s in _SMILES]
        n = len(graphs)
        validity = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])
        uniqueness = torch.ones(n)
        termination = torch.ones(n)
        c = _make_constants()
        sf = ScoringFunction(c)
        score = sf.compute_score(graphs, termination, validity, uniqueness)
        assert score[1] == pytest.approx(0.0), "Invalid molecule should have score 0"
        assert score[0] > 0.0, "Valid molecule should have positive score"


# ===========================================================================
# QED component
# ===========================================================================


class TestQEDComponent:

    def test_qed_scores_in_range(self):
        """QED scores must be in [0, 1]."""
        graphs = [_FakeGraph(s) for s in _SMILES]
        n = len(graphs)
        termination, validity, uniqueness = _all_valid_unique(n)
        c = _make_constants(score_components=["QED"], score_thresholds=[0.0])
        sf = ScoringFunction(c)
        _, comp = sf.compute_score_with_components(
            graphs, termination, validity, uniqueness
        )
        qed_scores = comp["QED"]
        assert (qed_scores >= 0.0).all() and (qed_scores <= 1.0).all()

    def test_none_molecule_gets_zero_qed(self):
        """A graph with molecule=None should get QED score 0."""
        graphs = [_FakeGraph(None)]
        termination, validity, uniqueness = _all_valid_unique(1)
        c = _make_constants()
        sf = ScoringFunction(c)
        score = sf.compute_score(graphs, termination, validity, uniqueness)
        assert score[0] == pytest.approx(0.0)


# ===========================================================================
# target_size component
# ===========================================================================


class TestTargetSizeComponent:

    def test_exact_target_size_gives_score_one(self):
        """A molecule with exactly the target number of nodes should get score 1."""
        target = 5
        graphs = [_FakeGraph("CCCCC", n_nodes=target)]  # 5 carbons
        termination, validity, uniqueness = _all_valid_unique(1)
        c = _make_constants(
            score_components=[f"target_size={target}"],
            score_thresholds=[0.0],
            max_n_nodes=10,
        )
        sf = ScoringFunction(c)
        _, comp = sf.compute_score_with_components(
            graphs, termination, validity, uniqueness
        )
        assert comp[f"target_size={target}"][0] == pytest.approx(1.0)

    def test_far_from_target_gives_low_score(self):
        """A molecule far from the target size should get a low score."""
        target = 5
        graphs = [_FakeGraph("C", n_nodes=1)]  # 1 node, target is 5
        termination, validity, uniqueness = _all_valid_unique(1)
        c = _make_constants(
            score_components=[f"target_size={target}"],
            score_thresholds=[0.0],
            max_n_nodes=10,
        )
        sf = ScoringFunction(c)
        _, comp = sf.compute_score_with_components(
            graphs, termination, validity, uniqueness
        )
        assert comp[f"target_size={target}"][0] < 1.0


# ===========================================================================
# Binary vs continuous score_type
# ===========================================================================


class TestScoreType:

    def test_binary_below_threshold_gives_zero(self):
        """Binary scoring: molecules failing ALL thresholds → score 0."""
        graphs = [_FakeGraph("CCCCC", n_nodes=5)]
        termination, validity, uniqueness = _all_valid_unique(1)
        c = _make_constants(
            score_components=["QED", "target_size=5"],
            score_thresholds=[0.999, 0.999],  # both impossible to exceed
            score_type="binary",
            max_n_nodes=10,
        )
        sf = ScoringFunction(c)
        score = sf.compute_score(graphs, termination, validity, uniqueness)
        assert (score == 0).all()

    def test_binary_applies_threshold_with_one_component(self):
        """score_type='binary' must be honoured for a single component too.

        The shipped goal_directed template uses exactly one component with a
        threshold; a short-circuit used to return the raw continuous score, so
        that job silently optimised continuous QED instead of the binary reward
        the config asked for.
        """
        graphs = [_FakeGraph("CCO", n_nodes=3)]
        termination, validity, uniqueness = _all_valid_unique(1)
        c = _make_constants(
            score_components=["QED"],
            score_thresholds=[0.99],  # QED of ethanol is well below this
            score_type="binary",
            max_n_nodes=10,
        )
        score = ScoringFunction(c).compute_score(
            graphs, termination, validity, uniqueness
        )
        assert (score == 0).all()

    def test_target_size_score_is_never_negative(self):
        """A molecule far from the target size scores 0, not a negative number.

        The unclamped form was unbounded below, and two negative components
        multiplied to a positive reward.
        """
        graphs = [_FakeGraph("C", n_nodes=1)]
        termination, validity, uniqueness = _all_valid_unique(1)
        c = _make_constants(
            score_components=["target_size=10"],
            score_thresholds=[0.0],
            score_type="continuous",
            max_n_nodes=13,
        )
        score = ScoringFunction(c).compute_score(
            graphs, termination, validity, uniqueness
        )
        assert (score >= 0).all()

    def test_binary_above_threshold_gives_nonzero(self):
        """Binary scoring: molecule above the QED threshold → score > 0."""
        graphs = [_FakeGraph(s) for s in _SMILES]
        n = len(graphs)
        termination, validity, uniqueness = _all_valid_unique(n)
        c = _make_constants(
            score_components=["QED"],
            score_thresholds=[0.0],  # always passes
            score_type="binary",
        )
        sf = ScoringFunction(c)
        score = sf.compute_score(graphs, termination, validity, uniqueness)
        assert (score >= 0).all()
        assert score.sum() > 0

    def test_continuous_score_is_product_of_components(self):
        """Continuous scoring with two components gives their product."""
        graphs = [_FakeGraph("CCCCC", n_nodes=5)]
        termination, validity, uniqueness = _all_valid_unique(1)
        c = _make_constants(
            score_components=["QED", "target_size=5"],
            score_thresholds=[0.0, 0.0],
            score_type="continuous",
            max_n_nodes=10,
        )
        sf = ScoringFunction(c)
        final, comp = sf.compute_score_with_components(
            graphs, termination, validity, uniqueness
        )
        expected = comp["QED"][0] * comp["target_size=5"][0]
        assert final[0] == pytest.approx(expected.item(), abs=1e-5)

    def test_component_scores_unaffected_by_masks(self):
        """Component tensors in the dict must be RAW scores, not masked by validity etc."""
        graphs = [_FakeGraph(s) for s in _SMILES]
        n = len(graphs)
        validity = torch.zeros(n)  # all invalid → final score = 0
        uniqueness = torch.ones(n)
        termination = torch.ones(n)
        c = _make_constants()
        sf = ScoringFunction(c)
        final, comp = sf.compute_score_with_components(
            graphs, termination, validity, uniqueness
        )

        # Final score is all zeros due to validity mask
        assert (final == 0).all()
        # But component QED scores should be > 0 for valid molecules
        assert (
            comp["QED"].sum() > 0
        ), "Component scores should not be masked by validity"


# ===========================================================================
# Oracle-backed components
# ===========================================================================


def _constant_half(smiles):
    """Module-level so a PythonOracle can import it by path."""
    return [0.5] * len(smiles)


class TestOracleComponents:
    def test_oracle_component_is_scored(self):
        c = _make_constants(
            score_components=["my_target"],
            score_thresholds=[0.0],
            oracles={
                "my_target": {
                    "type": "python",
                    "target": f"{__name__}:_constant_half",
                }
            },
        )
        graphs = [_FakeGraph("CCO")]
        termination, validity, uniqueness = _all_valid_unique(1)
        score = ScoringFunction(c).compute_score(
            graphs, termination, validity, uniqueness
        )
        assert float(score[0]) == pytest.approx(0.5)

    def test_oracle_may_be_named_with_an_activity_suffix(self):
        """
        A component containing "activity" used to be routed to the qsar_models
        branch on the substring alone, so an oracle called "EGFR_activity"
        passed validation and then raised KeyError on the first scored batch.
        """
        c = _make_constants(
            score_components=["EGFR_activity"],
            score_thresholds=[0.0],
            oracles={
                "EGFR_activity": {
                    "type": "python",
                    "target": f"{__name__}:_constant_half",
                }
            },
        )
        graphs = [_FakeGraph("CCO")]
        termination, validity, uniqueness = _all_valid_unique(1)
        score = ScoringFunction(c).compute_score(
            graphs, termination, validity, uniqueness
        )
        assert float(score[0]) == pytest.approx(0.5)

    def test_undeclared_component_fails_at_construction(self):
        """Not three batches into a training run."""
        c = _make_constants(score_components=["nowhere"], score_thresholds=[0.0])
        with pytest.raises(ValueError, match="neither built-in nor"):
            ScoringFunction(c)
