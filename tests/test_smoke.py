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
    for name in ("train", "valid", "test"):
        assert (dataset / f"{name}.smi").exists()
        assert (dataset / f"{name}.h5").exists()
    assert (dataset / "preprocessing.yaml").exists()


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
            **{**TRAINING, "batch_size": 20},
        },
    )
    assert (job_dir / "logs" / "score.log").exists()


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
    assert (job_dir / "results" / "40_samples.smi").exists()


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
    assert (job_dir / "logs" / "generation.log").exists()
