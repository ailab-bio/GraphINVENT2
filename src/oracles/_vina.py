"""
AutoDock Vina docking oracle.

Docking gives a structure-based objective for a specific target, which is what
makes "optimise against this protein" possible without first assembling enough
assay data to train a surrogate.  The trade-off is cost: a single pose search
takes on the order of a second per molecule per CPU core, so an RL run of a few
hundred steps at batch 64 is tens of thousands of dockings.  Deduplication in
:class:`~._cache.CachedOracle` helps, since a converging agent re-proposes
molecules often, but a docking-driven run is hours-to-days rather than minutes,
and that should be planned for rather than discovered.

The scoring function itself is the deeper limitation.  Vina's empirical score
correlates only loosely with measured affinity, and an RL agent is an efficient
adversary against exactly that kind of imperfect objective: it will find
molecules that score well and do not bind. Docking scores are therefore best
combined with property constraints (QED, size, an off-target surrogate), and
treated as a filter rather than as ground truth.

Requirements, none installed by default:

  * ``vina`` -- either the Python package (``pip install vina``) or the
    ``vina`` executable on PATH.
  * A receptor prepared as PDBQT, with the search box specified.
  * Ligand preparation: ``meeko`` (``pip install meeko``) is preferred; the
    ``obabel`` executable is used as a fallback.

The backend is injectable so the surrounding logic can be tested without a
docking installation; see ``dock_fn`` in :class:`VinaOracle`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from ._base import BaseOracle

#: Returned for molecules that cannot be prepared or docked.  Chosen as a
#: plausible "no binding" energy rather than 0.0, which on the kcal/mol scale
#: would sit between a good and a bad binder and reward failure.
FAILED_DOCKING_SCORE = 0.0


def _prepare_ligand_pdbqt(smiles: str, out_path: Path, seed: int = 42) -> bool:
    """
    Generate a 3D conformer for *smiles* and write it as PDBQT.

    Returns False rather than raising when embedding or conversion fails: for
    a generated molecule that is a routine outcome, not an exceptional one.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return False
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except (ValueError, RuntimeError):
        # An unoptimised conformer still docks; a failed force field is not
        # a reason to discard the molecule.
        pass

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy

        preparation = MoleculePreparation()
        setups = preparation.prepare(mol)
        pdbqt_string, is_ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        if not is_ok:
            return False
        out_path.write_text(pdbqt_string)
        return True
    except ImportError:
        pass

    if shutil.which("obabel") is None:
        raise RuntimeError(
            "Ligand preparation needs either the 'meeko' package "
            "(pip install meeko) or the 'obabel' executable on PATH."
        )
    sdf_path = out_path.with_suffix(".sdf")
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    result = subprocess.run(
        ["obabel", str(sdf_path), "-O", str(out_path), "--partialcharge", "gasteiger"],
        capture_output=True,
        timeout=120,
        check=False,
    )
    sdf_path.unlink(missing_ok=True)
    return result.returncode == 0 and out_path.exists()


def _dock_one(args: tuple) -> float:
    """
    Dock one molecule and return its best binding energy in kcal/mol.

    A module-level function taking a single tuple, because
    :class:`ProcessPoolExecutor` must pickle whatever it dispatches and cannot
    pickle a bound method or a closure.
    """
    (
        smiles,
        receptor,
        center,
        box_size,
        exhaustiveness,
        n_poses,
        seed,
        cpu_per_dock,
    ) = args

    if not smiles:
        return FAILED_DOCKING_SCORE

    with tempfile.TemporaryDirectory() as workdir:
        ligand_path = Path(workdir) / "ligand.pdbqt"
        try:
            if not _prepare_ligand_pdbqt(smiles, ligand_path, seed=seed):
                return FAILED_DOCKING_SCORE
        except RuntimeError:
            raise
        except Exception:
            return FAILED_DOCKING_SCORE

        try:
            from vina import Vina

            docking = Vina(sf_name="vina", cpu=cpu_per_dock, seed=seed, verbosity=0)
            docking.set_receptor(str(receptor))
            docking.set_ligand_from_file(str(ligand_path))
            docking.compute_vina_maps(center=list(center), box_size=list(box_size))
            docking.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            energies = docking.energies(n_poses=n_poses)
            return float(energies[0][0])
        except ImportError:
            pass
        except Exception:
            return FAILED_DOCKING_SCORE

        vina_exe = shutil.which("vina")
        if vina_exe is None:
            raise RuntimeError(
                "AutoDock Vina not found: install the Python package "
                "(pip install vina) or put the 'vina' executable on PATH."
            )
        out_path = Path(workdir) / "out.pdbqt"
        command = [
            vina_exe,
            "--receptor",
            str(receptor),
            "--ligand",
            str(ligand_path),
            "--center_x",
            str(center[0]),
            "--center_y",
            str(center[1]),
            "--center_z",
            str(center[2]),
            "--size_x",
            str(box_size[0]),
            "--size_y",
            str(box_size[1]),
            "--size_z",
            str(box_size[2]),
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            str(n_poses),
            "--seed",
            str(seed),
            "--cpu",
            str(cpu_per_dock),
            "--out",
            str(out_path),
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=600, check=False
            )
        except subprocess.TimeoutExpired:
            return FAILED_DOCKING_SCORE
        if result.returncode != 0 or not out_path.exists():
            return FAILED_DOCKING_SCORE
        for line in out_path.read_text().splitlines():
            # "REMARK VINA RESULT:    -8.4      0.000      0.000"
            if line.startswith("REMARK VINA RESULT:"):
                try:
                    return float(line.split()[3])
                except (IndexError, ValueError):
                    return FAILED_DOCKING_SCORE
        return FAILED_DOCKING_SCORE


class VinaOracle(BaseOracle):
    """
    Scores molecules by AutoDock Vina binding energy against a fixed receptor.

    The native value is the best pose's energy in kcal/mol, where more negative
    is better, so the transform must be written with ``high`` below ``low``.
    The default maps -4 to 0 and -11 to 1, which spans roughly "no meaningful
    binding" to "as good as docking usefully resolves"; those numbers are a
    convention rather than a calibration and should be revisited per target.

    Parameters
    ----------
    receptor
        Path to the prepared receptor PDBQT.
    center, box_size
        Search box centre and dimensions in Angstrom.  Both are required:
        docking into the wrong pocket produces confident, meaningless numbers,
        so there is no default.
    exhaustiveness
        Vina's search effort.  8 is Vina's default; 32 is common for final
        scoring.  Cost scales roughly linearly.
    n_poses
        Poses to generate; only the best-scoring one is used for the score.
    n_workers
        Molecules docked in parallel as separate processes.  Defaults to one
        less than the CPU count, leaving a core for the training process.
    dock_fn
        Injectable docking backend, used for testing.  Takes the same argument
        tuple as :func:`_dock_one` and returns a float.
    """

    def __init__(
        self,
        name: str,
        receptor: str,
        center: Sequence[float],
        box_size: Sequence[float],
        transform: Optional[dict] = None,
        direction: str = "maximize",
        exhaustiveness: int = 8,
        n_poses: int = 1,
        seed: int = 42,
        n_workers: Optional[int] = None,
        cpu_per_dock: int = 1,
        dock_fn: Optional[Callable[[tuple], float]] = None,
    ) -> None:
        if transform is None:
            transform = {"type": "clipped_linear", "low": -4.0, "high": -11.0}
        super().__init__(name=name, transform=transform, direction=direction)

        self.receptor = Path(receptor)
        if dock_fn is None and not self.receptor.exists():
            raise FileNotFoundError(
                f"Receptor file for oracle '{name}' not found: '{receptor}'. "
                "It must be a prepared PDBQT."
            )
        if len(center) != 3 or len(box_size) != 3:
            raise ValueError(
                f"Oracle '{name}': 'center' and 'box_size' must each have three "
                f"elements, got {len(center)} and {len(box_size)}."
            )

        self.center = tuple(float(c) for c in center)
        self.box_size = tuple(float(b) for b in box_size)
        self.exhaustiveness = int(exhaustiveness)
        self.n_poses = int(n_poses)
        self.seed = int(seed)
        self.cpu_per_dock = int(cpu_per_dock)
        self.n_workers = (
            int(n_workers) if n_workers else max(1, (os.cpu_count() or 2) - 1)
        )
        self._dock_fn = dock_fn or _dock_one

    def _job(self, smiles: Optional[str]) -> tuple:
        return (
            smiles,
            str(self.receptor),
            self.center,
            self.box_size,
            self.exhaustiveness,
            self.n_poses,
            self.seed,
            self.cpu_per_dock,
        )

    def predict(self, smiles: Sequence[Optional[str]]) -> List[float]:
        jobs = [self._job(s) for s in smiles]
        if not jobs:
            return []

        if self.n_workers == 1 or len(jobs) == 1:
            return [float(self._dock_fn(job)) for job in jobs]

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            return [float(v) for v in pool.map(self._dock_fn, jobs)]

    def predict_with_uncertainty(
        self, smiles: Sequence[Optional[str]]
    ) -> Tuple[List[float], List[float]]:
        """
        Dock each molecule under several random seeds and report the spread.

        Vina's search is stochastic, so re-docking the same ligand gives
        slightly different energies; a large spread means the pose search did
        not converge and the reported energy should be trusted less.  Note this
        captures only *search* variance, not the error of the scoring function
        itself, which is by far the larger source of uncertainty in docking.
        Cost multiplies by the number of replicates.
        """
        n_replicates = 3
        base_seed = self.seed
        per_replicate: List[List[float]] = []
        for replicate in range(n_replicates):
            self.seed = base_seed + replicate
            per_replicate.append(self.predict(smiles))
        self.seed = base_seed

        values: List[float] = []
        uncertainties: List[float] = []
        for i in range(len(smiles)):
            runs = [rep[i] for rep in per_replicate]
            if all(v == FAILED_DOCKING_SCORE for v in runs):
                values.append(FAILED_DOCKING_SCORE)
                uncertainties.append(0.0)
                continue
            succeeded = [v for v in runs if v != FAILED_DOCKING_SCORE]
            mean = sum(succeeded) / len(succeeded)
            variance = sum((v - mean) ** 2 for v in succeeded) / len(succeeded)
            values.append(mean)
            uncertainties.append(variance**0.5)
        return values, uncertainties
