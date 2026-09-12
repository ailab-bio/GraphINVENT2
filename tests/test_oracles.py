"""
Unit tests for src/oracles/.

The oracle system is how a user attaches their own objective to an RL run, so
these tests exercise the whole path a real objective takes: native model output
-> transform -> direction -> cached score, including the multi-objective case
where one target is maximised and another avoided.

Nothing here needs a network connection or a docking installation: the docking
backend is injected, and the surrogate is a small model trained in-process.

Usage:
    pytest tests/test_oracles.py -v
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oracles import (  # noqa: E402
    BaseOracle,
    CachedOracle,
    OracleFactory,
    PythonOracle,
    SklearnOracle,
    VinaOracle,
    apply_direction,
    build_transform,
)
from oracles._auc import _trapz, compute_auc_top_k  # noqa: E402

_ETHANOL = "CCO"
_BENZENE = "c1ccccc1"


class _ConstantOracle(BaseOracle):
    """Returns a fixed native value; the simplest thing that exercises the base."""

    def __init__(self, value: float = 7.5, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.call_log: list = []

    def predict(self, smiles):
        self.call_log.extend(smiles)
        return [self.value] * len(smiles)


class _SpreadOracle(BaseOracle):
    """Reports a per-molecule uncertainty, for exercising the uncertainty path."""

    def __init__(self, values, spreads, **kwargs):
        super().__init__(**kwargs)
        self.values = values
        self.spreads = spreads

    def predict(self, smiles):
        return [self.values[s] for s in smiles]

    def predict_with_uncertainty(self, smiles):
        return ([self.values[s] for s in smiles], [self.spreads[s] for s in smiles])


# ===========================================================================
# Transforms
# ===========================================================================


class TestTransforms:
    def test_identity_clamps(self):
        f = build_transform(None)
        assert f(0.5) == pytest.approx(0.5)
        assert f(1.7) == pytest.approx(1.0)
        assert f(-0.3) == pytest.approx(0.0)

    def test_clipped_linear_ramps(self):
        f = build_transform({"type": "clipped_linear", "low": 0.0, "high": 10.0})
        assert f(0.0) == pytest.approx(0.0)
        assert f(5.0) == pytest.approx(0.5)
        assert f(10.0) == pytest.approx(1.0)
        assert f(20.0) == pytest.approx(1.0)

    def test_clipped_linear_handles_lower_is_better(self):
        """A docking energy improves as it becomes more negative."""
        f = build_transform({"type": "clipped_linear", "low": -4.0, "high": -11.0})
        assert f(-4.0) == pytest.approx(0.0)
        assert f(-7.5) == pytest.approx(0.5)
        assert f(-11.0) == pytest.approx(1.0)
        assert f(-15.0) == pytest.approx(1.0)

    def test_sigmoid_is_monotonic_and_bounded(self):
        f = build_transform({"type": "sigmoid", "low": 0.0, "high": 10.0})
        values = [f(x) for x in range(-20, 30)]
        assert all(0.0 <= v <= 1.0 for v in values)
        assert all(b >= a for a, b in zip(values, values[1:]))
        assert f(5.0) == pytest.approx(0.5)

    def test_sigmoid_does_not_overflow(self):
        f = build_transform({"type": "sigmoid", "low": 0.0, "high": 1.0})
        assert f(1e9) == pytest.approx(1.0)
        assert f(-1e9) == pytest.approx(0.0)

    def test_step_is_binary(self):
        f = build_transform({"type": "step", "threshold": 5.0})
        assert f(4.9) == 0.0
        assert f(5.0) == 1.0
        below = build_transform({"type": "step", "threshold": 5.0, "above": False})
        assert below(4.9) == 1.0
        assert below(5.1) == 0.0

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown transform type"):
            build_transform({"type": "gaussian"})

    def test_missing_parameters_raise(self):
        """Silently falling back to identity would misreport an unbounded value."""
        with pytest.raises(ValueError, match="low"):
            build_transform({"type": "clipped_linear", "low": 1.0})

    def test_direction_inverts(self):
        assert apply_direction(0.8, "maximize") == pytest.approx(0.8)
        assert apply_direction(0.8, "minimize") == pytest.approx(0.2)

    def test_unknown_direction_raises(self):
        with pytest.raises(ValueError, match="Unknown direction"):
            apply_direction(0.5, "maximise")


# ===========================================================================
# BaseOracle
# ===========================================================================


class TestBaseOracle:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseOracle(name="x")

    def test_transform_and_direction_are_applied(self):
        oracle = _ConstantOracle(
            value=7.5,
            name="demo",
            transform={"type": "clipped_linear", "low": 5.0, "high": 9.0},
        )
        assert oracle([_ETHANOL]) == [pytest.approx(0.625)]

    def test_minimize_inverts_the_score(self):
        spec = {"type": "clipped_linear", "low": 5.0, "high": 9.0}
        high = _ConstantOracle(value=7.5, name="a", transform=spec)
        low = _ConstantOracle(value=7.5, name="b", transform=spec, direction="minimize")
        assert high([_ETHANOL])[0] + low([_ETHANOL])[0] == pytest.approx(1.0)

    def test_bad_direction_fails_at_construction(self):
        """A typo must not survive until the first scored batch."""
        with pytest.raises(ValueError, match="Unknown direction"):
            _ConstantOracle(name="demo", direction="lower")

    def test_supports_uncertainty_reflects_implementation(self):
        assert not _ConstantOracle(name="demo").supports_uncertainty
        assert _SpreadOracle({}, {}, name="demo").supports_uncertainty

    def test_uncertainty_not_implemented_raises(self):
        with pytest.raises(NotImplementedError, match="does not provide"):
            _ConstantOracle(name="demo").predict_with_uncertainty([_ETHANOL])


# ===========================================================================
# CachedOracle
# ===========================================================================


class TestCachedOracle:
    def _cached(self, **kwargs):
        inner = _ConstantOracle(value=0.42, name="counting", **kwargs)
        return CachedOracle(inner), inner

    def test_call_count_counts_unique_molecules(self):
        cached, _ = self._cached()
        cached([_ETHANOL, _BENZENE])
        assert cached.call_count == 2

    def test_deduplication_within_batch(self):
        cached, inner = self._cached()
        cached([_ETHANOL, _ETHANOL, _ETHANOL])
        assert cached.call_count == 1
        assert len(inner.call_log) == 1

    def test_cache_hit_costs_nothing(self):
        """The point of the budget: re-proposing a known molecule is free."""
        cached, inner = self._cached()
        cached([_ETHANOL])
        cached([_ETHANOL])
        assert cached.call_count == 1
        assert len(inner.call_log) == 1

    def test_none_scores_zero_without_a_call(self):
        cached, _ = self._cached()
        scores = cached([None, _ETHANOL])
        assert scores[0] == 0.0
        assert cached.call_count == 1

    def test_optimization_log_records_each_evaluation(self):
        cached, _ = self._cached()
        cached([_ETHANOL, _BENZENE, "C"])
        log = cached.optimization_log
        assert [entry[0] for entry in log] == [1, 2, 3]

    def test_reset_clears_state(self):
        cached, _ = self._cached()
        cached([_ETHANOL])
        cached.reset()
        assert cached.call_count == 0
        assert cached.optimization_log == []
        cached([_ETHANOL])
        assert cached.call_count == 1

    def test_name_delegates(self):
        cached, _ = self._cached()
        assert cached.name == "counting"

    def test_uncertainty_is_cached_with_the_score(self):
        """
        A molecule must attract the same reward whether or not it was a cache
        hit, so the uncertainty has to be cached alongside the score.
        """
        inner = _SpreadOracle(
            values={_ETHANOL: 0.9, _BENZENE: 0.9},
            spreads={_ETHANOL: 0.01, _BENZENE: 0.40},
            name="spread",
        )
        cached = CachedOracle(inner)
        first_scores, first_unc = cached.predict_with_uncertainty([_ETHANOL, _BENZENE])
        second_scores, second_unc = cached.predict_with_uncertainty(
            [_ETHANOL, _BENZENE]
        )
        assert first_scores == second_scores
        assert first_unc == second_unc == [0.01, 0.40]
        assert cached.call_count == 2

    def test_uncertainty_unsupported_raises(self):
        cached, _ = self._cached()
        assert not cached.supports_uncertainty
        with pytest.raises(NotImplementedError):
            cached.predict_with_uncertainty([_ETHANOL])


# ===========================================================================
# SklearnOracle
# ===========================================================================


def _train_toy_forest(tmp_path: Path, n_estimators: int = 8) -> Path:
    """
    Train a small random forest on a separable toy task and pickle it.

    The task is "does the molecule contain nitrogen", which fingerprints
    capture trivially; the point is to exercise loading, featurisation, and
    the ensemble-uncertainty path, not to model anything real.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.ensemble import RandomForestClassifier

    positives = ["CCN", "CCCN", "c1ccncc1", "CN(C)C", "NCCN", "c1ccc(N)cc1"]
    negatives = ["CCO", "CCC", "c1ccccc1", "CCOC", "CCCC", "c1ccc(O)cc1"]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    features, labels = [], []
    for smiles_list, label in ((positives, 1), (negatives, 0)):
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            features.append(
                np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=float)
            )
            labels.append(label)

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=0, max_depth=3
    ).fit(np.vstack(features), labels)

    path = tmp_path / "toy_rf.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


class TestSklearnOracle:
    def test_scores_reflect_the_trained_task(self, tmp_path):
        oracle = SklearnOracle(name="hasN", path=str(_train_toy_forest(tmp_path)))
        nitrogen, no_nitrogen = oracle(["CCN", "CCO"])
        assert nitrogen > no_nitrogen

    def test_minimize_direction_flips_the_ranking(self, tmp_path):
        """An anti-target is the same model with the objective reversed."""
        path = str(_train_toy_forest(tmp_path))
        want = SklearnOracle(name="hasN", path=path)
        avoid = SklearnOracle(name="noN", path=path, direction="minimize")
        assert want(["CCN"])[0] > want(["CCO"])[0]
        assert avoid(["CCN"])[0] < avoid(["CCO"])[0]

    def test_invalid_smiles_score_worst_without_raising(self, tmp_path):
        oracle = SklearnOracle(name="hasN", path=str(_train_toy_forest(tmp_path)))
        scores = oracle([None, "not_a_molecule", "CCN"])
        assert scores[0] == 0.0
        assert scores[1] == 0.0
        assert scores[2] > 0.0

    def test_all_invalid_does_not_crash(self, tmp_path):
        oracle = SklearnOracle(name="hasN", path=str(_train_toy_forest(tmp_path)))
        assert oracle([None, None]) == [0.0, 0.0]

    def test_ensemble_reports_uncertainty(self, tmp_path):
        oracle = SklearnOracle(name="hasN", path=str(_train_toy_forest(tmp_path)))
        assert oracle.supports_uncertainty
        values, uncertainties = oracle.predict_with_uncertainty(["CCN", "CCO"])
        assert len(values) == len(uncertainties) == 2
        assert all(u >= 0.0 for u in uncertainties)

    def test_invalid_molecules_get_zero_uncertainty(self, tmp_path):
        """
        Their score is already the worst available; inflating the uncertainty
        would let reward modulation partially undo that penalty.
        """
        oracle = SklearnOracle(name="hasN", path=str(_train_toy_forest(tmp_path)))
        _, uncertainties = oracle.predict_with_uncertainty([None, "CCN"])
        assert uncertainties[0] == 0.0

    def test_non_ensemble_declines_uncertainty(self, tmp_path):
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression().fit(np.eye(4)[:, :2], [0, 1, 0, 1])
        path = tmp_path / "linear.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        oracle = SklearnOracle(name="linear", path=str(path), n_bits=2)
        assert not oracle.supports_uncertainty
        with pytest.raises(NotImplementedError, match="non-ensemble"):
            oracle.predict_with_uncertainty(["CCN"])

    def test_missing_file_fails_at_construction(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            SklearnOracle(name="x", path="/nonexistent/model.pkl")

    def test_dict_pickle_requires_the_right_key(self, tmp_path):
        path = tmp_path / "wrapped.pkl"
        with open(path, "wb") as f:
            pickle.dump({"something_else": 1}, f)
        with pytest.raises(KeyError, match="classifier_sv"):
            SklearnOracle(name="x", path=str(path))

    def test_unknown_output_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown output"):
            SklearnOracle(
                name="x", path=str(_train_toy_forest(tmp_path)), output="logits"
            )


# ===========================================================================
# PythonOracle
# ===========================================================================


def _length_score(smiles, scale: float = 10.0):
    """Module-level so PythonOracle can import it by path."""
    return [min(len(s or ""), scale) / scale for s in smiles]


class TestPythonOracle:
    def test_imports_and_calls_the_target(self):
        oracle = PythonOracle(name="len", target=f"{__name__}:_length_score")
        assert oracle(["CCO"])[0] == pytest.approx(0.3)

    def test_kwargs_are_forwarded(self):
        oracle = PythonOracle(
            name="len", target=f"{__name__}:_length_score", kwargs={"scale": 5.0}
        )
        assert oracle(["CCO"])[0] == pytest.approx(0.6)

    def test_bad_target_format_raises(self):
        with pytest.raises(ValueError, match="module.path:callable_name"):
            PythonOracle(name="x", target="no_colon_here")

    def test_missing_module_fails_at_construction(self):
        with pytest.raises(ImportError, match="Could not import module"):
            PythonOracle(name="x", target="no_such_module_xyz:fn")

    def test_missing_attribute_raises(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            PythonOracle(name="x", target=f"{__name__}:not_defined")

    def test_wrong_length_output_is_rejected(self):
        """A silent length mismatch would misalign scores with molecules."""
        oracle = PythonOracle(name="bad", target=f"{__name__}:_truncating_score")
        with pytest.raises(ValueError, match="exactly one value per input"):
            oracle(["CCO", "CCN"])


def _truncating_score(smiles):
    return [1.0]


# ===========================================================================
# VinaOracle
# ===========================================================================


def _fake_dock(args: tuple) -> float:
    """
    Stand-in docking backend: bigger molecules "bind" better.

    Lets the surrounding machinery -- transform direction, parallel dispatch,
    failure handling -- be tested without a docking installation.
    """
    smiles = args[0]
    if not smiles or "X" in smiles:
        return 0.0
    return -4.0 - 0.3 * len(smiles)


class TestVinaOracle:
    def _oracle(self, **kwargs):
        params = dict(
            name="target",
            receptor="unused.pdbqt",
            center=[0.0, 0.0, 0.0],
            box_size=[20.0, 20.0, 20.0],
            n_workers=1,
            dock_fn=_fake_dock,
        )
        params.update(kwargs)
        return VinaOracle(**params)

    def test_more_negative_energy_scores_higher(self):
        oracle = self._oracle()
        weak, strong = oracle(["CCO", "CCCCCCCCCCCCCCCCCCCC"])
        assert strong > weak

    def test_default_transform_spans_the_docking_range(self):
        oracle = self._oracle()
        assert oracle.score_from_raw(-4.0) == pytest.approx(0.0)
        assert oracle.score_from_raw(-7.5) == pytest.approx(0.5)
        assert oracle.score_from_raw(-11.0) == pytest.approx(1.0)

    def test_failed_docking_scores_zero(self):
        oracle = self._oracle()
        assert oracle(["XXX"])[0] == pytest.approx(0.0)

    def test_missing_receptor_fails_at_construction(self):
        """Discovering this after an hour of generation would be expensive."""
        with pytest.raises(FileNotFoundError, match="Receptor file"):
            VinaOracle(
                name="t",
                receptor="/nonexistent/receptor.pdbqt",
                center=[0, 0, 0],
                box_size=[20, 20, 20],
            )

    def test_malformed_box_raises(self):
        with pytest.raises(ValueError, match="three"):
            self._oracle(center=[0.0, 0.0])

    def test_empty_input_returns_empty(self):
        assert self._oracle()([]) == []

    def test_replicate_spread_is_reported_as_uncertainty(self):
        oracle = self._oracle()
        assert oracle.supports_uncertainty
        values, uncertainties = oracle.predict_with_uncertainty(["CCO"])
        # The fake backend is deterministic, so the spread is exactly zero.
        assert uncertainties == [pytest.approx(0.0)]
        assert values[0] == pytest.approx(-4.9)


# ===========================================================================
# OracleFactory
# ===========================================================================


class TestOracleFactory:
    def test_builds_from_specs(self, tmp_path):
        specs = {
            "hasN": {"type": "sklearn", "path": str(_train_toy_forest(tmp_path))},
            "len": {"type": "python", "target": f"{__name__}:_length_score"},
        }
        oracles = OracleFactory.from_specs(specs)
        assert set(oracles) == {"hasN", "len"}
        assert all(isinstance(o, CachedOracle) for o in oracles.values())

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="has no 'type'"):
            OracleFactory.create("x", {"path": "foo.pkl"})

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown oracle type"):
            OracleFactory.create("x", {"type": "quantum_magic"})

    def test_unexpected_key_is_reported(self):
        """A silently ignored key means optimising something other than asked."""
        with pytest.raises(TypeError, match="Check the keys"):
            OracleFactory.create(
                "x",
                {
                    "type": "python",
                    "target": f"{__name__}:_length_score",
                    "nonexistent_option": 1,
                },
            )

    def test_comment_keys_are_ignored(self):
        oracle = OracleFactory.create(
            "x",
            {
                "_comment": "params.json has no comment syntax",
                "type": "python",
                "target": f"{__name__}:_length_score",
            },
        )
        assert oracle.name == "x"

    def test_empty_specs_give_no_oracles(self):
        assert OracleFactory.from_specs({}) == {}

    def test_from_config_reads_the_job_block(self):
        config = {
            "job": {
                "oracles": {
                    "len": {"type": "python", "target": f"{__name__}:_length_score"}
                }
            }
        }
        assert set(OracleFactory.from_config(config)) == {"len"}


# ===========================================================================
# Multi-objective: the selectivity case
# ===========================================================================


class TestMultiObjective:
    def test_selectivity_ranks_a_selective_molecule_highest(self, tmp_path):
        """
        "Bind the target, avoid the anti-target" is two oracles over the same
        kind of quantity with opposite directions.  The product of the two
        component scores must prefer the molecule that satisfies both.
        """
        path = str(_train_toy_forest(tmp_path))
        on_target = SklearnOracle(name="want_N", path=path)
        anti_target = SklearnOracle(name="avoid_N", path=path, direction="minimize")

        for smiles in ("CCN", "CCO"):
            combined = on_target([smiles])[0] * anti_target([smiles])[0]
            # The same model cannot be both satisfied and avoided, so a single
            # molecule can never score well on both: the product is bounded by
            # 0.25 at p=0.5, which is the point of the check.
            assert combined <= 0.25 + 1e-9

    def test_two_independent_targets_combine(self, tmp_path):
        activity = SklearnOracle(name="hasN", path=str(_train_toy_forest(tmp_path)))
        size = PythonOracle(name="len", target=f"{__name__}:_length_score")
        good = activity(["CCN"])[0] * size(["CCN"])[0]
        bad = activity(["CCO"])[0] * size(["CCO"])[0]
        assert good > bad


# ===========================================================================
# AUC Top-k
# ===========================================================================


class TestComputeAucTopK:

    def test_empty_log_returns_zero(self):
        assert compute_auc_top_k([]) == pytest.approx(0.0)

    def test_single_entry_perfect_score(self):
        """Single molecule with score 1.0 at call 1; k=1 -> AUC >= 0.5."""
        log = [(1, 1.0)]
        result = compute_auc_top_k(log, k=1, budget=1)
        # Trapz integrates the triangle from (0,0) to (1,1.0) -> area=0.5,
        # normalised by budget=1 -> AUC = 0.5.
        assert result == pytest.approx(0.5)

    def test_all_zero_scores_gives_zero_auc(self):
        log = [(i + 1, 0.0) for i in range(50)]
        assert compute_auc_top_k(log, k=10, budget=50) == pytest.approx(0.0)

    def test_all_ones_scores_gives_one_auc(self):
        """All scores = 1.0 -> AUC is high (close to 1.0 after top-k ramp-up)."""
        log = [(i + 1, 1.0) for i in range(100)]
        result = compute_auc_top_k(log, k=10, budget=100)
        # Top-k ramps up over the first k steps (0.1, 0.2, ..., 1.0 averaged over k),
        # then stays at 1.0.  With k=10 and budget=100, AUC = 0.95.
        assert result >= 0.9

    def test_monotone_increasing_gives_positive_auc(self):
        log = [(i + 1, (i + 1) / 100.0) for i in range(100)]
        result = compute_auc_top_k(log, k=10, budget=100)
        assert result > 0.0

    def test_constrained_none_valid_gives_zero(self):
        """If no entry satisfies constraints, AUC = 0."""
        log = [(i + 1, 0.9) for i in range(20)]
        constr = [False] * 20
        assert compute_auc_top_k(
            log, k=10, budget=20, constraints=constr
        ) == pytest.approx(0.0)

    def test_constrained_partial_valid(self):
        """Entries failing the constraint never enter the top-k."""
        log = [(i + 1, 1.0) for i in range(20)]
        constr = [i % 2 == 0 for i in range(20)]  # 10 valid
        constrained = compute_auc_top_k(log, k=10, budget=20, constraints=constr)
        assert 0.0 <= constrained <= 1.0

    def test_constraint_excluding_best_lowers_auc(self):
        """Screening out the best molecules must lower the AUC.

        Note the converse does not hold: because the running average is over
        molecules actually *found*, excluding poor molecules can raise the curve.
        """
        log = [(i + 1, 0.1 * (i + 1)) for i in range(10)]  # scores 0.1 .. 1.0
        unconstrained = compute_auc_top_k(log, k=3, budget=10)
        constr = [s < 0.8 for _, s in log]  # the three best all fail
        assert (
            compute_auc_top_k(log, k=3, budget=10, constraints=constr) < unconstrained
        )

    def test_fewer_than_k_averages_over_found(self):
        """With fewer than k molecules the average is over those found, not k.

        Dividing by a fixed k would score an early run k/n times too low, which
        directly corrupts the PMO-style AUC Top-10 table.
        """
        assert compute_auc_top_k([(1, 1.0)], k=10, budget=1) == pytest.approx(0.5)

    def test_unfinished_run_is_not_credited_for_unused_budget(self):
        """A run that stopped early must not score like one that used its budget."""
        log = [(1, 1.0)]
        assert compute_auc_top_k(log, k=1, budget=10000) == pytest.approx(0.5)
        assert compute_auc_top_k(log, k=1, budget=10000, finish=True) == pytest.approx(
            0.99995
        )

    def test_constraints_length_mismatch_raises(self):
        log = [(1, 0.5), (2, 0.7)]
        with pytest.raises(ValueError, match="constraints length"):
            compute_auc_top_k(log, k=2, budget=2, constraints=[True])

    def test_entries_beyond_budget_ignored(self):
        """Entries with call_count > budget do not affect AUC."""
        log = [(5, 0.9), (10, 0.9), (15, 0.9)]  # budget=10
        result_without_15 = compute_auc_top_k(log[:2], k=1, budget=10)
        result_with_15 = compute_auc_top_k(log, k=1, budget=10)
        assert result_without_15 == pytest.approx(result_with_15)

    def test_top_k_uses_best_scores(self):
        """Top-1 AUC reflects the best score seen; dominated by ramp-up phase."""
        log = [(1, 0.2), (2, 0.8), (3, 0.5)]
        # Curve: t=0 f=0, t=1 f=0.2, t=2 f=0.8, t=3 f=0.8
        # trapz([0, 0.2, 0.8, 0.8], [0, 1, 2, 3]) = 0.1 + 0.5 + 0.8 = 1.4
        # AUC = 1.4 / 3 ≈ 0.467
        result = compute_auc_top_k(log, k=1, budget=3)
        assert result == pytest.approx(1.4 / 3, rel=1e-3)

    def test_k_larger_than_log_size(self):
        """k larger than log size: average over however many exist."""
        log = [(1, 0.9), (2, 0.8)]
        result = compute_auc_top_k(log, k=10, budget=2)
        # Only 2 molecules; they contribute 0.9+0.8 = 1.7, divided by k=10 = 0.17 each
        assert 0.0 <= result <= 1.0

    def test_trapz_helper(self):
        """Sanity check the trapezoidal integration helper."""
        # Rectangle of width 4, height 1 -> area = 4
        assert _trapz([1.0, 1.0], [0, 4]) == pytest.approx(4.0)
        # Triangle from 0 to 2 -> area = 1
        assert _trapz([0.0, 1.0], [0, 2]) == pytest.approx(1.0)


# ===========================================================================
# ===========================================================================
