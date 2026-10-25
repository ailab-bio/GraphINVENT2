"""
One smoke test per job type.

A few steps on a tiny dataset, asserting the job finishes and writes what it
says it writes. These catch interface breakage: a renamed config key, a moved
output directory, a dispatch branch that no longer exists.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEBUG_SMILES = REPO_ROOT / "data/datasets/debug/debug.smi"
CONDITIONAL_SMILES = (
    REPO_ROOT / "data/datasets/tpddb_protac10_logp/tpddb_protac10_logp.smi"
)

TRAINING = {
    "epochs": 2,
    "batch_size": 50,
    "accumulation_steps": 1,
    "sample_every": 2,
    "n_samples": 20,
    "device": "cpu",
    "seed": 1,
}


def run_job(job_dir: Path, config: dict) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "config.yaml").write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, "submit.py", str(job_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    """An unconditional dataset, preprocessed once for the jobs that follow."""
    root = tmp_path_factory.mktemp("smoke")
    dataset_dir = root / "debug"
    dataset_dir.mkdir()
    shutil.copy(DEBUG_SMILES, dataset_dir / "molecules.smi")

    job_dir = root / "exp" / "jobs" / "preprocess"
    run_job(
        job_dir,
        {
            "job_type": "preprocess",
            "dataset_dir": str(dataset_dir),
            "smiles_file": str(dataset_dir / "molecules.smi"),
            "batch_size": 100,
            "seed": 1,
        },
    )
    return dataset_dir


@pytest.fixture(scope="module")
def checkpoint(dataset, tmp_path_factory) -> Path:
    """A trained model for the jobs that need one."""
    job_dir = dataset.parent / "exp" / "jobs" / "train"
    run_job(
        job_dir, {"job_type": "unconditional", "dataset_dir": str(dataset), **TRAINING}
    )
    return job_dir / "checkpoints" / "model_restart_2.pth"


def test_preprocess_writes_hdf5_and_splits(dataset):
    # The splits and the recorded vocabulary live in the per-strategy
    # subdirectory, not at the dataset root, so two split sets of one dataset
    # cannot overwrite each other.
    split_dir = dataset / "split_scaffold"
    for name in ("train", "valid", "test"):
        assert (split_dir / f"{name}.smi").exists()
        assert (split_dir / f"{name}.h5").exists()
    assert (split_dir / "preprocessing.yaml").exists()


def test_unconditional_writes_a_checkpoint(checkpoint):
    assert checkpoint.exists()
    job_dir = checkpoint.parent.parent
    assert (job_dir / "results" / "config.resolved.yaml").exists()
    assert (job_dir / "logs" / "convergence.log").exists()


def test_transfer_learning_resumes_from_a_checkpoint(dataset, checkpoint):
    job_dir = dataset.parent / "exp" / "jobs" / "transfer"
    run_job(
        job_dir,
        {
            "job_type": "unconditional",
            "dataset_dir": str(dataset),
            "resume_from": str(checkpoint),
            **TRAINING,
        },
    )
    assert (job_dir / "checkpoints" / "model_restart_2.pth").exists()


def test_a_killed_job_resumes_from_its_last_checkpoint(dataset):
    """
    The bug: `restart: true` resumed, started the next epoch, and raised
    "Tried to step N times. The specified number of total steps is M" from
    OneCycleLR, so no job could be resumed after a walltime kill and a
    48-hour prior was an all-or-nothing submission.

    The cause was reading `epochs` as a count of ADDITIONAL epochs on a
    restart, against a one-cycle schedule sized for the whole run. The
    learning rate on the resumed epochs is the assertion that matters: if the
    cycle had restarted rather than continued, it would come back to its
    opening value.
    """
    job_dir = dataset.parent / "exp" / "jobs" / "killable"
    config = {"job_type": "unconditional", "dataset_dir": str(dataset), **TRAINING}
    config["epochs"] = 4
    run_job(job_dir, config)

    log = (job_dir / "logs" / "convergence.log").read_text().splitlines()
    full = {
        int(r.split(",")[0].split()[1]): r.split(",")[1].strip()
        for r in log
        if r.startswith("Epoch")
    }
    assert set(full) == {1, 2, 3, 4}

    # Simulate the kill: keep only the epoch-2 checkpoint.
    for epoch in (3, 4):
        stale = job_dir / "checkpoints" / f"model_restart_{epoch}.pth"
        if stale.exists():
            stale.unlink()
    (job_dir / "logs" / "convergence.log").unlink()

    config["restart"] = True
    run_job(job_dir, config)

    assert (
        job_dir / "checkpoints" / "model_restart_4.pth"
    ).exists(), "the resumed run must reach the configured total of 4 epochs"
    resumed = {
        int(r.split(",")[0].split()[1]): r.split(",")[1].strip()
        for r in (job_dir / "logs" / "convergence.log").read_text().splitlines()
        if r.startswith("Epoch")
    }
    assert set(resumed) == {3, 4}, (
        f"epochs is the total for the run, so a restart at 2 of 4 runs 3 and 4; "
        f"got {sorted(resumed)}"
    )
    for epoch in (3, 4):
        assert resumed[epoch] == full[epoch], (
            f"epoch {epoch} learning rate is {resumed[epoch]} on the resumed run "
            f"against {full[epoch]} on the uninterrupted one, so the one-cycle "
            "schedule restarted instead of continuing"
        )


def test_conditional_trains_on_a_dataset_with_properties(tmp_path):
    dataset_dir = tmp_path / "logp"
    dataset_dir.mkdir()
    shutil.copy(CONDITIONAL_SMILES, dataset_dir / "molecules.smi")

    run_job(
        tmp_path / "exp" / "jobs" / "preprocess",
        {
            "job_type": "preprocess",
            "dataset_dir": str(dataset_dir),
            "smiles_file": str(dataset_dir / "molecules.smi"),
            "batch_size": 100,
            "condition_dim": 1,
            "conditioning": {"properties": ["logP"], "source": "smiles_file"},
            "seed": 1,
        },
    )

    job_dir = tmp_path / "exp" / "jobs" / "train"
    run_job(
        job_dir,
        {
            "job_type": "conditional",
            "dataset_dir": str(dataset_dir),
            "sample_conditions": {"logP": 1.0},
            **TRAINING,
        },
    )
    assert (job_dir / "checkpoints" / "model_restart_2.pth").exists()


def test_goal_directed_optimizes_against_an_oracle(dataset, checkpoint):
    job_dir = dataset.parent / "exp" / "jobs" / "rl"
    run_job(
        job_dir,
        {
            "job_type": "goal_directed",
            "dataset_dir": str(dataset),
            "pretrained_model_path": str(checkpoint),
            "score_components": ["QED"],
            "score_thresholds": [0.0],
            "score_type": "continuous",
            # The RL schedule has its own keys; a goal_directed job that sets
            # `epochs` or `sample_every` is rejected, so TRAINING's supervised
            # names are translated rather than reused.
            **{
                k: v for k, v in TRAINING.items() if k not in ("epochs", "sample_every")
            },
            "rl_steps": TRAINING["epochs"],
            "rl_eval_every": TRAINING["sample_every"],
            "batch_size": 20,
        },
    )
    score_log = job_dir / "logs" / "score.log"
    assert score_log.exists()
    # Not just existence: the initial pre-fine-tuning evaluation writes a row
    # before any learning step, so a file with one row means the training loop
    # recorded nothing.
    steps = [
        line for line in score_log.read_text().splitlines() if line.startswith("Step ")
    ]
    assert len(steps) > 1, f"score.log recorded no learning step: {steps}"


def test_generate_writes_molecules(dataset, checkpoint):
    job_dir = dataset.parent / "exp" / "jobs" / "generate"
    run_job(
        job_dir,
        {
            "job_type": "generate",
            "dataset_dir": str(dataset),
            "pretrained_model_path": str(checkpoint),
            "sample_mode": "generate",
            "n_samples": 40,
            "batch_size": 20,
            "device": "cpu",
            "seed": 1,
        },
    )
    # CLAUDE.md's Job types table names this file molecules.smi.
    assert (job_dir / "results" / "molecules.smi").exists()

    # A batch can come back larger than batch_size, so the generated list is
    # truncated to n_samples before anything measures it. Without that,
    # validity divides by n_samples while counting more molecules than that and
    # exceeds 1.0 on a model with high validity. This is the only place that
    # check can live: it needs a real generation run.
    import json

    payload = json.loads((job_dir / "results" / "metrics.json").read_text())
    reported = payload["metrics"]
    assert reported["n_sampled"] == 40
    assert reported["validity"] <= 1.0, (
        f"validity {reported['validity']} exceeds 1.0, so the generated list "
        "was not truncated to n_samples before being measured"
    )
    assert reported["n_valid"] <= reported["n_sampled"]
    assert "validity_n_generated_raw" in reported, (
        "the pre-truncation count must be recorded, or the over-production is "
        "hidden rather than visible"
    )
    assert reported["validity_n_generated_raw"] >= reported["n_sampled"]

    # An open shell is not a molecule. RDKit sanitizes a radical without
    # complaint, so before the writer gated on it every molecule counted valid
    # in every run was a polyradical: 253 of 253 in the first toy runs,
    # methylidyne `[CH]` among them, and validity, novelty, diversity and the
    # synthesizability scores were all computed over them.
    from rdkit import Chem

    written = (job_dir / "results" / "molecules.smi").read_text().split()
    for smiles in written:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"{smiles} was written but does not parse"
        assert not any(
            a.GetNumRadicalElectrons() for a in mol.GetAtoms()
        ), f"{smiles} was written to molecules.smi and is an open shell"


def test_generate_in_evaluate_mode_scores_the_model(dataset, checkpoint):
    job_dir = dataset.parent / "exp" / "jobs" / "evaluate"
    run_job(
        job_dir,
        {
            "job_type": "generate",
            "dataset_dir": str(dataset),
            "pretrained_model_path": str(checkpoint),
            "sample_mode": "evaluate",
            "n_samples": 40,
            "batch_size": 20,
            "device": "cpu",
            "seed": 1,
        },
    )
    # `sample_mode: evaluate` exists to compute NLL and UC-JSD, and
    # metrics.json is where CLAUDE.md puts them. It used to assert that
    # logs/generation.log existed, which the generate path no longer writes:
    # that file recorded the first generation batch only, so every rate in it
    # described `batch_size` molecules while the sample was `n_samples`.
    import json

    payload = json.loads((job_dir / "results" / "metrics.json").read_text())
    reported = payload["metrics"]
    for key in ("nll_train", "nll_validation", "nll_test", "uc_jsd"):
        assert key in reported, f"{key} missing from metrics.json"
    assert reported["nll_train"] is not None, (
        "evaluate mode computed no training-set NLL, which is the one thing "
        "this job type is for"
    )
