"""
Split disjointness.

Preprocessing decomposes each molecule into many subgraphs. A split applied
after decomposition would scatter subgraphs of one molecule across train,
validation and test, and every reported number would be leakage, with nothing
visible in the training curves. These tests run a real preprocessing job and
check the splits at both the molecule and the subgraph level.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import pytest
import yaml
from rdkit import Chem, RDLogger

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SMILES = REPO_ROOT / "data/datasets/debug/debug.smi"
SPLITS = ("train", "valid", "test")

RDLogger.DisableLog("rdApp.*")


@pytest.fixture(scope="module")
def preprocessed(tmp_path_factory) -> Path:
    """Run one preprocessing job and return its dataset directory."""
    root = tmp_path_factory.mktemp("split")
    dataset_dir = root / "dataset"
    dataset_dir.mkdir()
    shutil.copy(SOURCE_SMILES, dataset_dir / "molecules.smi")

    job_dir = root / "exp" / "jobs" / "preprocess"
    job_dir.mkdir(parents=True)
    (job_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "job_type": "preprocess",
                "dataset_dir": str(dataset_dir),
                "smiles_file": str(dataset_dir / "molecules.smi"),
                "split_type": "scaffold",
                "train_frac": 0.6,
                "valid_frac": 0.2,
                "batch_size": 100,
                "seed": 3,
            }
        )
    )
    subprocess.run(
        [sys.executable, "submit.py", str(job_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return dataset_dir


def _canonical(path: Path) -> list:
    lines = [line.split("\t")[0].strip() for line in path.read_text().splitlines()]
    return [Chem.CanonSmiles(s) for s in lines if s and not s.startswith("SMILES")]


def test_no_molecule_appears_in_two_splits(preprocessed):
    molecules = {s: set(_canonical(preprocessed / f"{s}.smi")) for s in SPLITS}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        assert (
            not molecules[a] & molecules[b]
        ), f"{len(molecules[a] & molecules[b])} molecules are in both {a} and {b}"


def test_splits_cover_every_input_molecule(preprocessed):
    molecules = set()
    for split in SPLITS:
        molecules |= set(_canonical(preprocessed / f"{split}.smi"))
    assert molecules == set(_canonical(preprocessed / "molecules.smi"))


def test_no_scaffold_is_shared_between_splits(preprocessed):
    """What a scaffold split buys over a random one: without this, close
    analogues sit on both sides and generalization is overstated."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = {
        split: {
            MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
            for s in _canonical(preprocessed / f"{split}.smi")
        }
        for split in SPLITS
    }
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        assert not scaffolds[a] & scaffolds[b]


def test_no_subgraph_derives_from_another_splits_molecule(preprocessed):
    """
    Decompose each split's own SMILES and check that every row of that split's
    HDF5 is one of the resulting subgraphs. A split applied after decomposition
    would put rows in the file that its own molecules cannot account for.

    Containment rather than equality, because identical subgraphs are merged
    within a group, which removes rows but never adds any.
    """
    sys.path.insert(0, str(REPO_ROOT / "src/graphinvent"))
    import numpy as np
    from parameters import config

    recorded = yaml.safe_load((preprocessed / "preprocessing.yaml").read_text())
    constants = config.derive(
        {
            **yaml.safe_load((REPO_ROOT / "config/defaults.yaml").read_text()),
            **{k: v for k, v in recorded.items() if k not in ("split", "run_info")},
            "job_type": "generate",
            "dataset_dir": str(preprocessed),
            "job_dir": str(preprocessed),
        }
    )
    from MolecularGraph import PreprocessingGraph

    def fingerprint(nodes, edges) -> bytes:
        return (
            np.asarray(nodes, dtype=np.float32).tobytes()
            + np.asarray(edges, dtype=np.float32).tobytes()
        )

    for split in SPLITS:
        own_subgraphs = set()
        for smiles in _canonical(preprocessed / f"{split}.smi"):
            graph = PreprocessingGraph(
                constants=constants, molecule=Chem.MolFromSmiles(smiles)
            )
            for i in range(graph.get_decoding_route_length()):
                (nodes, edges), _ = graph.get_decoding_route_state(subgraph_idx=i)
                own_subgraphs.add(fingerprint(nodes, edges))

        with h5py.File(preprocessed / f"{split}.h5", "r") as f:
            stored = [fingerprint(n, e) for n, e in zip(f["nodes"][:], f["edges"][:])]

        foreign = [row for row in stored if row not in own_subgraphs]
        assert not foreign, (
            f"{len(foreign)} of {len(stored)} subgraphs in {split}.h5 do not "
            f"come from any molecule in {split}.smi"
        )
