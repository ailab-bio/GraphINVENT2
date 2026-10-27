"""
Justification for this file, per the rule in CLAUDE.md's Tests section.

Nothing in `tests/` imported `src/oracles/` or `scripts/train-surrogate.py`
before this file, and the three failures it guards are all silent:

1. **A group on both sides of a fold split still produces a plausible R².**
   Grouped cross-validation exists only to stop that, and a leaking grouping
   reports a better number, not an error, so there is no symptom to notice. The
   POI case is the one with real logic in it: a composite target such as
   `KRAS/CRAF` must land in the same fold as plain `KRAS`, or the fold "holds
   out" a protein the model saw.
2. **Fold models the oracle ignores still produce a plausible uncertainty.**
   The detector CLAUDE.md specifies is disagreement between models fitted on
   different data. If the oracle silently falls back to the spread across one
   forest's trees, every number still arrives, in the same units, and reward
   modulation runs on a signal that cannot detect extrapolation.
3. **A feature-width mismatch in the uncertainty path only fires when
   uncertainty modulation is switched on.** It shipped broken: the method
   featurized without the descriptors and one-hots that `predict` uses, so the
   first RL run to enable modulation would have died mid-run.

Each case below was mutation-tested: the thing it guards was broken
deliberately, the test was confirmed to fail, and the break reverted.
"""

import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from oracles._surrogate import SklearnOracle  # noqa: E402

# train-surrogate.py is not importable by name, because of the hyphen.
_spec = importlib.util.spec_from_file_location(
    "train_surrogate", REPO / "scripts" / "train-surrogate.py"
)
train_surrogate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_surrogate)


# Four real, distinct molecules; enough to featurize and fit on.
MOLECULES = [
    "c1ccccc1",
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2ccccc2c1",
    "CCN(CC)CC",
    "OCC1OC(O)C(O)C(O)C1O",
]


def test_poi_groups_put_a_composite_target_with_both_of_its_members():
    """
    `KRAS/CRAF` names two proteins. Grouping on the raw string would put it in
    a different fold from plain `KRAS` and from plain `CRAF`, so a fold that
    "holds out KRAS" would train on KRAS chemistry. The connected component
    over shared symbols is what makes the hold-out strict.
    """
    poi = ["KRAS", "RAF1", "KRAS;RAF1", "SMARCA2"]
    keys = train_surrogate.group_keys(MOLECULES[:4], "poi", poi=poi)
    assert keys[0] == keys[1] == keys[2], (
        "KRAS, RAF1 and the KRAS;RAF1 composite must share one group; "
        f"got {keys[:3]}"
    )
    assert keys[3] != keys[0], "SMARCA2 shares no symbol and must stay separate"


def test_poi_groups_are_order_independent():
    """`14-3-3s/ERa` and `ERa/14-3-3s` are one complex, so one group."""
    keys = train_surrogate.group_keys(
        MOLECULES[:2], "poi", poi=["SFN;ESR1", "ESR1;SFN"]
    )
    assert keys[0] == keys[1]


def test_poi_grouping_without_a_target_column_refuses_rather_than_inventing_one():
    with pytest.raises(SystemExit, match="needs the target column"):
        train_surrogate.group_keys(MOLECULES[:2], "poi", poi=None)


def test_an_unimplemented_grouping_raises_and_names_the_implemented_set():
    with pytest.raises(SystemExit, match="implemented kinds are"):
        train_surrogate.group_keys(MOLECULES[:2], "by_vibes")


def test_compound_grouping_collapses_the_replicate_rows_of_one_compound():
    """
    The same compound written two ways is one group, because its rows are one
    piece of evidence. This is also the clustering unit of the bootstrap.
    """
    keys = train_surrogate.group_keys(["c1ccccc1", "C1=CC=CC=C1", "CCO"], "compound")
    assert keys[0] == keys[1]
    assert keys[2] != keys[0]


def test_the_noise_floor_does_not_treat_two_targets_as_replicates():
    """
    A compound measured against SMARCA2 and SMARCA4 has two different true
    potencies, not two replicates. Keying the replicate group on the compound
    alone counted that difference as assay noise and dropped the measured R²
    ceiling of the pooled table from 0.755 to 0.410, which would have made a
    usable model look impossible.
    """
    labels = [6.0, 9.0, 6.1, 9.1]
    by_compound = ["A", "A", "B", "B"]
    by_compound_and_target = ["A|SMARCA2", "A|SMARCA4", "B|SMARCA2", "B|SMARCA4"]

    wrong = train_surrogate.label_noise_floor(labels, by_compound)
    right = train_surrogate.label_noise_floor(labels, by_compound_and_target)
    assert wrong[0] > 1.0, "keying on the compound alone inflates the noise"
    # Keyed correctly, no (compound, target) pair repeats, so there is no
    # replicate evidence at all and the function says so rather than guessing.
    assert right[0] is None and right[2] is None


def test_the_noise_floor_caps_r2_below_one():
    labels = [5.0, 5.4, 7.0, 7.4, 9.0, 9.4]
    keys = ["A", "A", "B", "B", "C", "C"]
    noise, total, ceiling, n = train_surrogate.label_noise_floor(labels, keys)
    assert n == 3
    assert 0.0 < ceiling < 1.0
    assert np.isclose(ceiling, 1.0 - (noise / total) ** 2)


def test_the_clustered_bootstrap_is_wider_than_an_unclustered_one():
    """
    Three rows of one compound are one piece of evidence. Resampling rows
    treats them as three, which narrows the interval in the direction the kill
    criterion depends on: on the real SMARCA2 fit the lower bound moved from
    0.017 to 0.003 when the resample was clustered.
    """
    rng = np.random.default_rng(0)
    n_compounds = 30
    truth, predictions, clusters = [], [], []
    for compound in range(n_compounds):
        level = rng.normal(7.0, 0.8)
        error = rng.normal(0.0, 0.3)
        for _ in range(4):
            truth.append(level + rng.normal(0.0, 0.02))
            # One correlated error per compound, which is what clustering is for.
            predictions.append(level + error)
            clusters.append(f"compound{compound}")
    truth = np.asarray(truth)
    predictions = np.asarray(predictions)
    baseline = np.full_like(truth, truth.mean())

    lo_c, hi_c, n_blocks = train_surrogate.clustered_bootstrap_margin(
        truth, predictions, baseline, clusters, seed=1, n=400
    )
    lo_i, hi_i, n_rows = train_surrogate.clustered_bootstrap_margin(
        truth, predictions, baseline, list(range(len(truth))), seed=1, n=400
    )
    assert n_blocks == n_compounds
    assert n_rows == len(truth)
    assert (hi_c - lo_c) > (hi_i - lo_i), (
        "clustering on the compound must widen the interval when the error is "
        f"correlated within compound; clustered {hi_c - lo_c:.4f} vs "
        f"unclustered {hi_i - lo_i:.4f}"
    )


# --- the oracle half -------------------------------------------------------


def _fit_table(tmp_path, with_cv: bool, extra_args=()):
    """Fit a real surrogate on a tiny table and return the pickle path."""
    table = tmp_path / "labels.tsv"
    rows = ["smiles\tpdc50\tcell_line\tligase\tpoi"]
    for i, smi in enumerate(MOLECULES):
        rows.append(f"{smi}\t{6.0 + 0.5 * i:.2f}\thek293\tvhl\tSMARCA2")
    table.write_text("\n".join(rows) + "\n")
    out = tmp_path / ("cv.pickle" if with_cv else "plain.pickle")
    command = [
        sys.executable,
        str(REPO / "scripts" / "train-surrogate.py"),
        "--input",
        str(table),
        "--smiles-column",
        "smiles",
        "--label-column",
        "pdc50",
        "--split",
        "random",
        "--test-fraction",
        "0.34",
        "--seed",
        "1",
        "--n-estimators",
        "12",
        "--count-fingerprint",
        "--categorical",
        "cell_line",
        "--output",
        str(out),
        *extra_args,
    ]
    if with_cv:
        command += ["--cv", "grouped", "--group-by", "compound", "--folds", "3"]
    done = subprocess.run(command, capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stdout + done.stderr
    return out, done.stdout


def _oracle(path):
    return SklearnOracle(
        name="t",
        path=str(path),
        output="predict",
        transform={"type": "clipped_linear", "low": 5.0, "high": 10.0},
    )


def test_the_uncertainty_comes_from_the_fold_models_when_they_are_present(tmp_path):
    """
    The detector is disagreement between models fitted on DIFFERENT data. A
    silent fall-back to one forest's trees still returns numbers in the same
    units, so the source is asserted, not the magnitude.
    """
    cv_path, _ = _fit_table(tmp_path, with_cv=True)
    oracle = _oracle(cv_path)
    assert oracle.uncertainty_source == "cv_fold_models"
    assert len(oracle._sub_estimators) == 3, "three folds were requested"
    assert oracle.supports_uncertainty


def test_without_cross_validation_the_source_is_the_forest_and_says_so(tmp_path):
    plain_path, _ = _fit_table(tmp_path, with_cv=False)
    oracle = _oracle(plain_path)
    assert oracle.uncertainty_source == "forest_trees"
    assert len(oracle._sub_estimators) == 12, "the forest's own trees"


def test_no_store_folds_keeps_the_numbers_and_drops_the_models(tmp_path):
    path, stdout = _fit_table(tmp_path, with_cv=True, extra_args=("--no-store-folds",))
    assert "fold 1:" in stdout, "the CV still runs and still reports"
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    assert payload["models"] == []
    assert payload["cv"] is not None, "the summary is kept even without models"
    assert _oracle(path).uncertainty_source == "forest_trees"


def test_predict_with_uncertainty_featurizes_exactly_as_predict_does(tmp_path):
    """
    Regression test for a shipped crash. This path omitted the descriptors and
    the one-hot block, so it offered 2048 columns to a model fitted on more and
    raised an sklearn feature-count error the first time uncertainty modulation
    was enabled. The guarantee is stronger than "it does not raise": the value
    must be the same number `predict` returns, or the uncertainty is not the
    uncertainty of the score.
    """
    cv_path, _ = _fit_table(tmp_path, with_cv=True)
    oracle = _oracle(cv_path)
    probe = ["c1ccccc1", "CCO", "not_a_molecule", "CC(=O)Oc1ccccc1C(=O)O"]
    values, uncertainties = oracle.predict_with_uncertainty(probe)
    # Parallel tree accumulation makes these agree to float noise, not bitwise.
    assert np.allclose(values, oracle.predict(probe), atol=1e-9)
    assert uncertainties[2] == 0.0, (
        "an unparseable molecule already scores the worst available value; "
        "giving it uncertainty would let modulation undo the penalty"
    )
    assert all(u >= 0.0 for u in uncertainties)


def test_a_models_entry_that_is_not_estimators_is_refused(tmp_path):
    """
    A pickle whose `models` list holds something unfittable must fail at load,
    naming the problem, rather than at the first scoring call deep inside an
    RL run.
    """
    cv_path, _ = _fit_table(tmp_path, with_cv=True)
    with open(cv_path, "rb") as handle:
        payload = pickle.load(handle)
    payload["models"] = ["not an estimator"]
    broken = tmp_path / "broken.pickle"
    with open(broken, "wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match="no predict method"):
        _oracle(broken)


def test_every_fold_holds_out_whole_groups(tmp_path):
    """
    The guarantee the scheme rests on. The trainer asserts it internally and
    exits if a group straddles a split; this checks the assertion is reached
    and that a clean run reports groups held out on every fold.
    """
    _, stdout = _fit_table(tmp_path, with_cv=True)
    folds = [line for line in stdout.splitlines() if line.strip().startswith("fold ")]
    assert len(folds) == 3
    for line in folds:
        assert "groups held out" in line
        held = int(line.split("(")[-1].split(" groups")[0])
        assert held >= 1, f"a fold that holds out nothing is not a fold: {line}"


# --- Added 2026-10-26, after a code review found four further silent failures.
# Each is mutation-tested the same way: break the thing, confirm the failure,
# revert.


def test_a_column_in_both_extra_and_optional_is_read_once(tmp_path):
    """
    The duplication bug this guards was introduced and caught in one session,
    and it is the worst kind: `read_table` appended a column named in both
    `extra` and `optional` twice per row, so every value after the first
    belonged to a different molecule than its index said. Nothing raised. The
    POI group count fell from 18 to 12 and the label-noise ceiling moved from
    0.755 to 0.418 -- numbers that look like a modeling result.
    """
    table = tmp_path / "t.tsv"
    table.write_text(
        "smiles\tpdc50\tpoi\n" "CCO\t7.0\tSMARCA2\n" "CCC\t8.0\tAR\n" "CCN\t9.0\tKRAS\n"
    )
    smiles, labels, extras = train_surrogate.read_table(
        table, "smiles", "pdc50", extra=("poi",), optional=("poi",)
    )
    assert len(smiles) == 3
    assert len(extras["poi"]) == 3, "the column was appended once per request"
    assert extras["poi"] == ["SMARCA2", "AR", "KRAS"], "row alignment survived"


def test_an_absent_optional_column_is_omitted_rather_than_fatal(tmp_path):
    """
    The noise floor wants a target column from the tables that have one. Making
    it mandatory would break every fit whose table has no target at all, which
    is how it came to be requested only for POI grouping -- and that silently
    degraded the replicate key to compound alone everywhere else.
    """
    table = tmp_path / "t.tsv"
    table.write_text("smiles\tpdc50\nCCO\t7.0\nCCC\t8.0\n")
    _, _, extras = train_surrogate.read_table(
        table, "smiles", "pdc50", optional=("poi",)
    )
    assert "poi" not in extras
    with pytest.raises(ValueError, match="poi"):
        train_surrogate.read_table(table, "smiles", "pdc50", extra=("poi",))


def test_a_pooled_poi_pickle_is_refused_as_an_oracle(tmp_path):
    """
    A pooled fit carries one one-hot column per target and an inference default
    of the training mode. Nothing in SklearnOracle can set it per campaign, so
    loading one as a reward would score every molecule of every campaign
    against that one target -- `ar`, in the committed pooled PROTAC fit, over
    an 18-target vocabulary -- with nothing raising and every number arriving
    in the right units.
    """
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=2, random_state=0)
    model.fit(np.zeros((4, 3)), [1.0, 2.0, 3.0, 4.0])
    path = tmp_path / "pooled.pickle"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "featurization": {
                    "radius": 2,
                    "n_bits": 3,
                    "categorical_vocab": {"poi": ["ar", "smarca2", "kras"]},
                    "categorical_defaults": {"poi": "ar"},
                },
            },
            f,
        )
    with pytest.raises(ValueError, match="pooled fit"):
        SklearnOracle(name="x", path=str(path), output="predict")


def test_a_single_target_pickle_is_still_accepted(tmp_path):
    """
    The guard above must not reject the per-target fits, whose `poi` vocabulary
    has exactly one entry and so cannot be pinned to the wrong target.
    """
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=2, random_state=0)
    model.fit(np.zeros((4, 3)), [1.0, 2.0, 3.0, 4.0])
    path = tmp_path / "one.pickle"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "featurization": {
                    "radius": 2,
                    "n_bits": 3,
                    "categorical_vocab": {"poi": ["smarca2"]},
                    "categorical_defaults": {"poi": "smarca2"},
                },
            },
            f,
        )
    oracle = SklearnOracle(
        name="x",
        path=str(path),
        output="predict",
        transform={"type": "clipped_linear", "low": 5.0, "high": 10.0},
    )
    assert oracle.uncertainty_source == "forest_trees"


def test_the_target_map_refuses_two_rows_for_one_string(tmp_path):
    """
    This file is the leakage guard. Two rows for one raw string with different
    symbol sets would change fold membership depending on which came last,
    with nothing printed -- against the fail-loudly rule the rest of the loader
    already follows.
    """
    labels = REPO / "scripts" / "tpddb-activity-labels.py"
    spec = importlib.util.spec_from_file_location("tpddb_labels", labels)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    good = tmp_path / "map.tsv"
    good.write_text("raw\tsymbols\tkind\tnote\nAR\tAR\tpoi\t\nAR-V7\tAR\tpoi\t\n")
    table = module.load_target_map(good)
    assert table["AR-V7"][0] == frozenset({"AR"})

    clashing = tmp_path / "clash.tsv"
    clashing.write_text("raw\tsymbols\tkind\tnote\nAR\tAR\tpoi\t\nAR\tESR1\tpoi\t\n")
    with pytest.raises(SystemExit, match="two canonical symbol sets"):
        module.load_target_map(clashing)
