"""
Every behavior-selecting guard, triggered on the path real jobs take.

Justification for a sixth test file: the guards these cover are correct code
that review passes and that does not run. Two shipped dead because the check
sat where a value is read from a config layer, and almost every parameter is
inherited rather than set, so the check ran on the rare path and skipped the
common one. A seventh guard raised only after the model was built, and an
eighth only after a preprocess job had scanned every molecule. None of those
is visible in the file; the only way to know a guard works is to construct the
input it rejects and watch it fire.

Every case below INHERITS the bad value from the experiment layer rather than
setting it in the job's own config, because that is the path that was dead.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO / "src/graphinvent", _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from parameters import config  # noqa: E402

# A minimal preprocessing.yaml: a dataset's recorded feature vocabulary.
VOCABULARY = {
    "atom_types": ["C", "N", "O"],
    "formal_charge": [0],
    "imp_H": [0, 1, 2, 3],
    "chirality": ["None", "R", "S"],
    "max_n_nodes": 10,
    "use_aromatic_bonds": True,
    "use_canon": True,
    "use_chirality": False,
    "use_explicit_H": False,
    "ignore_H": False,
}


@pytest.fixture
def workspace():
    root = Path(tempfile.mkdtemp())
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _dataset(root, name="ds", split_type="scaffold", **overrides):
    """A dataset directory: the source corpus at its root, the splits and the
    recorded vocabulary in the per-strategy subdirectory every job reads."""
    directory = root / "data" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "corpus.smi").write_text("SMILES\nCCO\n")
    split_dir = directory / f"split_{split_type}"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "preprocessing.yaml").write_text(
        yaml.safe_dump({**VOCABULARY, **overrides})
    )
    for split in ("train.smi", "valid.smi", "test.smi"):
        (split_dir / split).write_text("SMILES\nCCO\n")
    return directory


def _job(root, job_config, experiment_config=None):
    job = root / "exp" / "jobs" / "j"
    job.mkdir(parents=True, exist_ok=True)
    (job / "config.yaml").write_text(yaml.safe_dump(job_config))
    if experiment_config:
        (root / "exp" / "experiment.yaml").write_text(yaml.safe_dump(experiment_config))
    return job


def _resolve(job):
    return config.derive(config.resolve(job))


@pytest.mark.parametrize(
    "key,bad,expected",
    [
        ("split_type", "butina", "split strategy"),
        ("sample_mode", "sampl", "sampling mode"),
        ("device", "tpu", "device"),
        ("condition_type", "film", "condition_type"),
        ("n_workers", 4, "n_workers"),
    ],
)
def test_a_behavior_selecting_key_is_rejected_when_inherited(
    workspace, key, bad, expected
):
    dataset = _dataset(workspace)
    job = _job(
        workspace,
        {
            "job_type": "preprocess",
            "dataset_dir": str(dataset),
            "smiles_file": str(dataset / "corpus.smi"),
        },
        {key: bad},
    )
    with pytest.raises((ValueError, NotImplementedError)) as caught:
        _resolve(job)
    message = " ".join(str(caught.value).split())
    assert expected.lower() in message.lower()
    assert str(bad) in message, "the message must name the value it rejected"


def test_the_custom_split_raises_before_any_molecule_is_read(workspace):
    """CLAUDE.md: a hook with no implementation, and selecting it raises.

    It used to raise inside `split_smiles_file`, after a preprocess job had
    scanned the whole corpus to build the feature vocabulary.
    """
    dataset = _dataset(workspace)
    job = _job(
        workspace,
        {
            "job_type": "preprocess",
            "dataset_dir": str(dataset),
            "smiles_file": str(dataset / "corpus.smi"),
        },
        {"split_type": "custom"},
    )
    with pytest.raises(NotImplementedError) as caught:
        config.resolve(job)
    assert "custom" in str(caught.value)


def _checkpoint(root, **trained_with):
    checkpoints = root / "prior" / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "m.pth").write_text("not a real checkpoint")
    results = root / "prior" / "results"
    results.mkdir(parents=True, exist_ok=True)
    results.joinpath("config.resolved.yaml").write_text(
        yaml.safe_dump({**VOCABULARY, **trained_with})
    )
    return checkpoints / "m.pth"


def _generate_job(root, dataset, checkpoint, conditions=None):
    return _job(
        root,
        {
            "job_type": "generate",
            "sample_mode": "generate",
            "dataset_dir": str(dataset),
            "pretrained_model_path": str(checkpoint),
            **({"sample_conditions": conditions} if conditions else {}),
        },
    )


def test_a_transposed_property_order_is_refused(workspace):
    """The hazard that writes a byte-identical file.

    The order is positional in two places, so a transposed declaration
    conditions on one property and reports it as another. Nothing about the
    stored data differs.
    """
    conditioning = {"source": "smiles_file"}
    dataset = _dataset(
        workspace,
        condition_dim=2,
        conditioning={"properties": ["pLogS", "MW"], **conditioning},
    )
    checkpoint = _checkpoint(
        workspace,
        condition_dim=2,
        conditioning={"properties": ["MW", "pLogS"], **conditioning},
    )
    job = _generate_job(workspace, dataset, checkpoint, {"pLogS": 1.0, "MW": 300.0})
    with pytest.raises(ValueError) as caught:
        _resolve(job)
    message = " ".join(str(caught.value).split())
    assert "different order" in message
    assert "pLogS" in message and "MW" in message


def test_the_same_order_is_accepted(workspace):
    """The control. A guard that rejects everything is not a guard."""
    conditioning = {"properties": ["pLogS", "MW"], "source": "smiles_file"}
    dataset = _dataset(workspace, condition_dim=2, conditioning=conditioning)
    checkpoint = _checkpoint(workspace, condition_dim=2, conditioning=conditioning)
    job = _generate_job(workspace, dataset, checkpoint, {"pLogS": 1.0, "MW": 300.0})
    _resolve(job)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"atom_types": ["O", "N", "C"]}, "SAME VALUES, DIFFERENT ORDER"),
        ({"atom_types": ["C", "N", "O", "S"]}, "atom_types"),
        ({"max_n_nodes": 54}, "max_n_nodes"),
        ({"use_chirality": True}, "use_chirality"),
    ],
)
def test_a_feature_vocabulary_mismatch_is_refused(workspace, overrides, expected):
    """These set the model's tensor widths and come from the dataset.

    No config layer sets `atom_types`, so the checkpoint comparison that keys
    on explicitly-set values never saw them. A reordered vocabulary of the same
    size does not crash on load: the model reads every atom as a different
    element and the job completes.
    """
    dataset = _dataset(workspace)
    checkpoint = _checkpoint(workspace, **overrides)
    job = _generate_job(workspace, dataset, checkpoint)
    with pytest.raises(ValueError) as caught:
        _resolve(job)
    assert expected in " ".join(str(caught.value).split())


def test_a_requested_class_outside_the_panel_is_refused(workspace):
    """It was dropped silently, leaving that slot at zero."""
    conditioning = {"properties": ["is_a", "is_b"], "source": "smiles_file"}
    dataset = _dataset(workspace, condition_dim=2, conditioning=conditioning)
    checkpoint = _checkpoint(workspace, condition_dim=2, conditioning=conditioning)
    job = _generate_job(workspace, dataset, checkpoint, {"is_a": 1.0, "is_typo": 0.0})
    with pytest.raises(ValueError) as caught:
        _resolve(job)
    message = " ".join(str(caught.value).split())
    assert "is_typo" in message and "is_a" in message


def _preprocess_job(workspace, dataset, **overrides):
    return _job(
        workspace,
        {
            "job_type": "preprocess",
            "dataset_dir": str(dataset),
            "smiles_file": str(dataset / "corpus.smi"),
            "split_type": "scaffold",
            "train_frac": 0.8,
            "valid_frac": 0.1,
            "seed": 1,
            **overrides,
        },
    )


def _record_split_as_built(constants, split_dir):
    """Write the preprocessing.yaml a completed run would have written.

    Built from the resolved constants rather than a literal, so the recorded
    vocabulary is the one this corpus actually detects and the only thing a
    later job can differ on is what that job changed.
    """
    from Workflow import Workflow

    recorded = {key: getattr(constants, key) for key in Workflow.VOCABULARY_KEYS}
    recorded["max_n_nodes"] = constants.max_n_nodes
    recorded["use_chirality"] = constants.use_chirality
    recorded["seed"] = constants.seed
    recorded["split"] = {
        "strategy": constants.split_type,
        "train_frac": constants.train_frac,
        "valid_frac": constants.valid_frac,
    }
    (split_dir / "preprocessing.yaml").write_text(yaml.safe_dump(recorded))


def _guard(constants):
    from Workflow import Workflow

    workflow = Workflow.__new__(Workflow)
    workflow.constants = constants
    return workflow._guard_existing_split(constants.split_dir)


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"seed": 99}, "seed"),
        ({"train_frac": 0.5}, "train_frac"),
        ({"valid_frac": 0.05}, "valid_frac"),
        ({"max_n_nodes": 2}, "max_n_nodes"),
        ({"use_chirality": True}, "use_chirality"),
    ],
)
def test_rebuilding_a_split_set_under_different_parameters_is_refused(
    workspace, override, expected
):
    """A split set is identified by split_type, seed, train_frac, valid_frac,
    max_n_nodes and use_chirality.

    Preprocessing a dataset a second time under different parameters used to
    move the existing splits into `_previous_run_<timestamp>/` and write new
    ones, so a model trained against one split was then evaluated against
    another's test split with nothing failing. The guard runs however the job
    was launched, not only under restart, because the harm is overwriting a
    split other jobs have already been evaluated against. `max_n_nodes` and
    `use_chirality` are in the fatal set because both act before the split:
    one drops oversized molecules, the other is the deduplication key.
    """
    dataset = _dataset(workspace)
    as_built = _resolve(_preprocess_job(workspace, dataset))
    _record_split_as_built(as_built, Path(as_built.split_dir))

    changed = _resolve(_preprocess_job(workspace, dataset, **override))
    with pytest.raises(ValueError) as caught:
        _guard(changed)
    message = " ".join(str(caught.value).split())
    assert expected in message and "new dataset_dir" in message


def test_rebuilding_a_split_set_under_the_same_parameters_is_allowed(workspace):
    """The legitimate re-run, which must not be caught by the guard above."""
    dataset = _dataset(workspace)
    as_built = _resolve(_preprocess_job(workspace, dataset))
    _record_split_as_built(as_built, Path(as_built.split_dir))

    assert _guard(_resolve(_preprocess_job(workspace, dataset))) is True


def test_a_changed_vocabulary_rebuilds_rather_than_raising(workspace):
    """A vocabulary change makes a half-built HDF5 unresumable but moves no
    molecule between splits, so it must not be fatal."""
    dataset = _dataset(workspace)
    as_built = _resolve(_preprocess_job(workspace, dataset))
    split_dir = Path(as_built.split_dir)
    _record_split_as_built(as_built, split_dir)

    recorded = yaml.safe_load((split_dir / "preprocessing.yaml").read_text())
    recorded["decoding_route"] = "dfs"
    (split_dir / "preprocessing.yaml").write_text(yaml.safe_dump(recorded))

    assert _guard(_resolve(_preprocess_job(workspace, dataset))) is False
