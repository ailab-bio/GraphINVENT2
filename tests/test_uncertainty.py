"""
Tests for uncertainty-aware reward and loss shaping.

The implementation follows Medina and Janet (arXiv:2606.24990), so the
assertions here are against hand-computed values from their equations rather
than against the implementation's own output.  Two properties matter most and
are easy to get subtly wrong: loss weights must be normalised to mean 1 so the
modulation does not act as a covert learning-rate change, and each component
must be modulated on its own terms, since different oracles report uncertainty
in incomparable units.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oracles._uncertainty import (  # noqa: E402
    UncertaintyModulation,
    combine_loss_weights,
    combine_score_weights,
    modulate_loss_weights,
    reliability_weight,
    reliability_weights,
)

# ===========================================================================
# Uncertainty -> reliability
# ===========================================================================


class TestReliabilityWeight:
    def test_none_is_the_identity(self):
        assert reliability_weight(999.0, "none") == 1.0

    def test_linear_matches_the_paper(self):
        """w = 1 - u/max, their Eq. 14 with an explicit scale."""
        assert reliability_weight(0.0, "linear", max_uncertainty=2.0) == 1.0
        assert reliability_weight(0.5, "linear", max_uncertainty=2.0) == 0.75
        assert reliability_weight(2.0, "linear", max_uncertainty=2.0) == 0.0

    def test_linear_clips_beyond_the_scale(self):
        assert reliability_weight(10.0, "linear", max_uncertainty=1.0) == 0.0

    def test_sigmoid_is_half_at_beta(self):
        """Their Eq. 13, read as reliability rather than distance."""
        assert reliability_weight(0.4, "sigmoid", beta=0.4) == pytest.approx(0.5)

    def test_sigmoid_hand_computed(self):
        got = reliability_weight(0.5, "sigmoid", beta=0.4, alpha=10.0)
        expected = 1.0 - 1.0 / (1.0 + math.exp(-10.0 * (0.5 - 0.4)))
        assert got == pytest.approx(expected)

    def test_inverse_is_bounded_at_zero_uncertainty(self):
        """The paper's raw 1/u diverges; the bounded form must not."""
        assert reliability_weight(0.0, "inverse", scale=0.5) == 1.0
        assert reliability_weight(0.5, "inverse", scale=0.5) == pytest.approx(0.5)

    def test_exponential_decays(self):
        assert reliability_weight(0.0, "exponential", beta=2.0) == 1.0
        assert reliability_weight(1.0, "exponential", beta=2.0) == pytest.approx(
            math.exp(-2.0)
        )

    @pytest.mark.parametrize(
        "method,params",
        [
            ("linear", {"max_uncertainty": 1.0}),
            ("sigmoid", {"beta": 0.5}),
            ("inverse", {"scale": 1.0}),
            ("exponential", {"beta": 1.0}),
        ],
    )
    def test_every_method_is_monotonically_decreasing_and_bounded(self, method, params):
        values = [reliability_weight(u / 10, method, **params) for u in range(0, 40)]
        assert all(0.0 <= v <= 1.0 for v in values)
        assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))

    def test_non_finite_uncertainty_is_fully_distrusted(self):
        """A NaN means the estimator failed; it must not propagate."""
        for bad in (float("nan"), float("inf"), -1.0):
            assert reliability_weight(bad, "linear", max_uncertainty=1.0) == 0.0

    def test_huge_uncertainty_does_not_overflow(self):
        assert reliability_weight(1e300, "exponential", beta=1.0) == 0.0
        assert reliability_weight(1e300, "sigmoid", beta=1.0) == 0.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown uncertainty method"):
            reliability_weight(0.1, "bayesian_magic")

    def test_missing_required_parameter_raises(self):
        """Falling back to no modulation would leave the run unprotected."""
        with pytest.raises(ValueError, match="max_uncertainty"):
            reliability_weight(0.1, "linear")
        with pytest.raises(ValueError, match="beta"):
            reliability_weight(0.1, "sigmoid")

    def test_vectorised_form_matches_scalar(self):
        got = reliability_weights([0.0, 0.5, 1.0], "linear", max_uncertainty=1.0)
        assert got == [1.0, 0.5, 0.0]


# ===========================================================================
# Per-component configuration
# ===========================================================================


class TestPerComponentModulation:
    def _config(self, mode="loss"):
        return {
            "mode": mode,
            "components": {
                "EGFR": {"method": "linear", "max_uncertainty": 1.0},
                "hERG": {"method": "exponential", "beta": 10.0},
                "QED": {"method": "none"},
            },
        }

    def test_each_component_uses_its_own_method(self):
        """
        The requirement that makes multi-objective modulation meaningful: a
        docking spread in kcal/mol and an ensemble std in probability units
        cannot share a threshold.
        """
        modulation = UncertaintyModulation(self._config())
        assert modulation.weights_for("EGFR", [0.5]) == [pytest.approx(0.5)]
        assert modulation.weights_for("hERG", [0.5]) == [pytest.approx(math.exp(-5.0))]

    def test_the_same_uncertainty_gives_different_weights_per_component(self):
        modulation = UncertaintyModulation(self._config())
        egfr = modulation.weights_for("EGFR", [0.2])[0]
        herg = modulation.weights_for("hERG", [0.2])[0]
        assert egfr != pytest.approx(herg)

    def test_unconfigured_component_is_untouched(self):
        modulation = UncertaintyModulation(self._config())
        assert modulation.weights_for("not_mentioned", [0.9, 0.1]) == [1.0, 1.0]

    def test_method_none_disables_that_component_only(self):
        modulation = UncertaintyModulation(self._config())
        assert not modulation.is_configured("QED")
        assert modulation.is_configured("EGFR")
        assert modulation.weights_for("QED", [0.9]) == [1.0]

    @pytest.mark.parametrize(
        "mode,score,loss",
        [
            ("none", False, False),
            ("score", True, False),
            ("loss", False, True),
            ("both", True, True),
        ],
    )
    def test_mode_selects_where_modulation_acts(self, mode, score, loss):
        modulation = UncertaintyModulation(self._config(mode=mode))
        assert modulation.modulates_score() is score
        assert modulation.modulates_loss() is loss

    def test_empty_config_is_disabled(self):
        assert not UncertaintyModulation(None).enabled
        assert not UncertaintyModulation({}).enabled

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown uncertainty_modulation mode"):
            UncertaintyModulation({"mode": "sometimes"})

    def test_bad_component_spec_fails_at_construction(self):
        """Not on the first batch that happens to carry an uncertainty."""
        with pytest.raises(ValueError, match="component 'EGFR'"):
            UncertaintyModulation(
                {"mode": "loss", "components": {"EGFR": {"method": "linear"}}}
            )

    def test_comment_keys_are_ignored(self):
        modulation = UncertaintyModulation(
            {
                "_comment": "params.json has no comment syntax",
                "mode": "loss",
                "components": {"EGFR": {"_why": "noisy assay", "method": "none"}},
            }
        )
        assert modulation.mode == "loss"


# ===========================================================================
# Aggregation across components
# ===========================================================================


class TestAggregation:
    def test_loss_weights_use_the_arithmetic_mean(self):
        """
        The paper's choice, made "to prevent extreme values from dominating";
        a product would let one distrusted component delete the molecule.
        """
        combined = combine_loss_weights({"a": [1.0, 0.0], "b": [0.0, 1.0]}, 2)
        assert combined.tolist() == [0.5, 0.5]

    def test_score_weights_use_the_geometric_mean(self):
        """
        The paper's Eq. 2 with equal weights.  A raw product would shrink the
        aggregate just because more components were added.
        """
        combined = combine_score_weights({"a": [1.0, 0.5], "b": [0.5, 0.5]}, 2)
        assert combined.tolist() == [pytest.approx(0.5**0.5), pytest.approx(0.5)]

    def test_adding_a_fully_trusted_component_does_not_shrink_the_score(self):
        two = combine_score_weights({"a": [0.4], "b": [1.0]}, 1)[0]
        three = combine_score_weights({"a": [0.4], "b": [1.0], "c": [1.0]}, 1)[0]
        assert three > two

    def test_a_single_distrusted_component_cannot_delete_a_molecule_from_the_loss(self):
        loss_combined = combine_loss_weights({"a": [0.0], "b": [1.0], "c": [1.0]}, 1)
        score_combined = combine_score_weights({"a": [0.0], "b": [1.0], "c": [1.0]}, 1)
        assert loss_combined[0] == pytest.approx(2.0 / 3.0)
        assert score_combined[0] == 0.0

    def test_no_components_means_no_modulation(self):
        assert combine_loss_weights({}, 3).tolist() == [1.0, 1.0, 1.0]
        assert combine_score_weights({}, 3).tolist() == [1.0, 1.0, 1.0]


# ===========================================================================
# Loss-weight normalisation
# ===========================================================================


class TestLossWeightNormalisation:
    def test_weights_are_normalised_to_mean_one(self):
        """
        Eq. 8's ``w_j / mean(w)``.  Without it a uniformly-distrusted batch
        would shrink the loss, which is a learning-rate change in disguise.
        """
        normalised = modulate_loss_weights([0.2, 0.4, 0.6])
        assert float(np.mean(normalised)) == pytest.approx(1.0)

    def test_uniform_distrust_does_not_shrink_the_update(self):
        assert (
            modulate_loss_weights([0.1, 0.1, 0.1]).tolist() == [pytest.approx(1.0)] * 3
        )

    def test_relative_ordering_is_preserved(self):
        normalised = modulate_loss_weights([0.1, 0.5, 0.9])
        assert normalised[0] < normalised[1] < normalised[2]

    def test_hand_computed_values(self):
        # mean([0.5, 1.0]) = 0.75 -> [0.6667, 1.3333]
        normalised = modulate_loss_weights([0.5, 1.0])
        assert normalised.tolist() == [
            pytest.approx(2.0 / 3.0),
            pytest.approx(4.0 / 3.0),
        ]

    def test_all_zero_weights_fall_back_to_ones(self):
        """Dividing by zero, or returning zeros, would silently halt learning."""
        assert modulate_loss_weights([0.0, 0.0]).tolist() == [1.0, 1.0]

    def test_empty_batch_is_handled(self):
        assert modulate_loss_weights([]).size == 0


# ===========================================================================
# The property the whole feature exists for
# ===========================================================================


class TestUncertainMoleculesAreRewardedLess:
    def test_equal_scores_but_unequal_confidence_are_ranked(self):
        """
        Two molecules the surrogate scores identically, one supported by the
        training data and one not, must not contribute equally.
        """
        modulation = UncertaintyModulation(
            {
                "mode": "both",
                "components": {"EGFR": {"method": "linear", "max_uncertainty": 1.0}},
            }
        )
        confident, uncertain = modulation.weights_for("EGFR", [0.05, 0.8])
        assert confident > uncertain

        raw_scores = np.array([0.9, 0.9])
        modulated = raw_scores * combine_score_weights(
            {"EGFR": [confident, uncertain]}, 2
        )
        assert modulated[0] > modulated[1]

        loss_weights = modulate_loss_weights(
            combine_loss_weights({"EGFR": [confident, uncertain]}, 2)
        )
        assert loss_weights[0] > loss_weights[1]

    def test_modulation_also_damps_a_minimised_component(self):
        """
        An anti-target prediction that cannot be trusted must not earn full
        credit either.  Reliability multiplies the contribution, so it damps
        the reward regardless of which way the component is optimised.
        """
        modulation = UncertaintyModulation(
            {
                "mode": "score",
                "components": {"hERG": {"method": "linear", "max_uncertainty": 1.0}},
            }
        )
        confident, uncertain = modulation.weights_for("hERG", [0.0, 0.9])
        # After `direction: minimize` the component score is already inverted,
        # so a high value here means "confidently avoids hERG".
        avoided = np.array([0.95, 0.95])
        modulated = avoided * combine_score_weights({"hERG": [confident, uncertain]}, 2)
        assert modulated[0] > modulated[1]
        assert modulated[1] < 0.95
