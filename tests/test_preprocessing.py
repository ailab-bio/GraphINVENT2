"""
Unit tests for GraphINVENT2 preprocessing output.

Checks:
  1. The three split .smi files together contain the same number of molecules
     as the original SMILES file (only when SMILES_FILE is set in config.py).
  2. Each HDF5 file contains the same number of complete molecules as its
     corresponding .smi file (determined via the termination flag in the APD).
  3. Every complete molecule in each HDF5 can be decoded back to a valid,
     non-empty SMILES string.

Configuration:
  Edit tests/config.py to point at your dataset.  All paths are relative to
  the repository root (the working directory when you run pytest).

Usage:
  pytest tests/test_preprocessing.py -v
"""
import json
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pytest
import rdkit
import rdkit.Chem
from rdkit.Chem import MolToSmiles
from rdkit.Chem.rdchem import BondType

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
try:
    from config import DATASET_DIR, SMILES_FILE  # type: ignore
except ImportError:
    from tests.config import DATASET_DIR, SMILES_FILE  # type: ignore

SPLITS = ["train", "valid", "test"]


# ---------------------------------------------------------------------------
# Helpers: reading .smi files
# ---------------------------------------------------------------------------

def read_smiles_file(path: Path) -> List[str]:
    """
    Return a list of SMILES strings from a .smi file.
    Skips blank lines, '#' comment lines, and header lines containing 'SMILES'.
    The first whitespace-delimited token on each line is taken as the SMILES.
    """
    smiles = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "SMILES" in line:
                continue
            smiles.append(line.split()[0])
    return smiles


# ---------------------------------------------------------------------------
# Helpers: reading HDF5 files
# ---------------------------------------------------------------------------

def load_preprocessing_params(dataset_dir: Path) -> dict:
    params_path = dataset_dir / "preprocessing_params.json"
    if not params_path.exists():
        pytest.skip(f"preprocessing_params.json not found in {dataset_dir}")
    with open(params_path) as fh:
        return json.load(fh)


def count_molecules_in_hdf(h5_path: Path) -> int:
    """
    Count complete molecules in an HDF5 file.

    A molecule is 'complete' when the last element of its APD vector
    (the termination flag f_term) equals 1.
    """
    with h5py.File(h5_path, "r") as fh:
        apds = fh["APDs"][:]
    return int((apds[:, -1] == 1).sum())


def get_complete_graphs(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (nodes, edges) arrays for all complete-molecule rows in an HDF5.

    Shape:
        nodes : (n_molecules, max_n_nodes, n_node_features)  int8
        edges : (n_molecules, max_n_nodes, max_n_nodes, n_edge_features)  int8
    """
    with h5py.File(h5_path, "r") as fh:
        apds  = fh["APDs"][:]
        nodes = fh["nodes"][:]
        edges = fh["edges"][:]
    mask = apds[:, -1] == 1
    return nodes[mask], edges[mask]


# ---------------------------------------------------------------------------
# Helpers: reconstructing molecules from graph arrays
# ---------------------------------------------------------------------------

def _build_constants(params: dict):
    """
    Build a minimal namedtuple from preprocessing_params sufficient for
    graph-to-molecule reconstruction.
    """
    atom_types     = params["atom_types"]
    formal_charge  = params["formal_charge"]
    imp_H          = params.get("imp_H", [])
    chirality      = params.get("chirality", [])
    use_explicit_H = params.get("use_explicit_H", False)
    ignore_H       = params.get("ignore_H", False)
    use_chirality  = params.get("use_chirality", False)
    use_aromatic   = params.get("use_aromatic_bonds", False)

    bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}
    if use_aromatic:
        bondtype_to_int[BondType.AROMATIC] = 3
    int_to_bondtype = {v: k for k, v in bondtype_to_int.items()}

    n_atom_types    = len(atom_types)
    n_formal_charge = len(formal_charge)
    n_imp_H         = 0 if (use_explicit_H or ignore_H) else len(imp_H)
    n_edge_features = len(bondtype_to_int)

    fields = [
        "atom_types", "formal_charge", "imp_H", "chirality",
        "use_explicit_H", "ignore_H", "use_chirality",
        "n_atom_types", "n_formal_charge", "n_imp_H",
        "n_edge_features", "int_to_bondtype",
    ]
    C = namedtuple("C", fields)
    return C(
        atom_types=atom_types,
        formal_charge=formal_charge,
        imp_H=imp_H,
        chirality=chirality,
        use_explicit_H=use_explicit_H,
        ignore_H=ignore_H,
        use_chirality=use_chirality,
        n_atom_types=n_atom_types,
        n_formal_charge=n_formal_charge,
        n_imp_H=n_imp_H,
        n_edge_features=n_edge_features,
        int_to_bondtype=int_to_bondtype,
    )


def _node_row_to_atom(row: np.ndarray, c) -> Optional[rdkit.Chem.Atom]:
    """
    Convert a single node feature vector to an RDKit Atom, or None if the
    row is all-zero (padding).

    The feature vector layout (one-hot segments concatenated) is:
        [ atom_types | formal_charge | imp_H (optional) | chirality (optional) ]
    """
    nonzero = np.nonzero(row)[0]
    # A valid atom has at least atom_type and formal_charge encoded.
    if len(nonzero) < 2:
        return None

    atom = rdkit.Chem.Atom(c.atom_types[int(nonzero[0])])

    fc_idx = int(nonzero[1]) - c.n_atom_types
    atom.SetFormalCharge(c.formal_charge[fc_idx])

    if not c.use_explicit_H and not c.ignore_H and c.n_imp_H > 0:
        h_idx = int(nonzero[2]) - c.n_atom_types - c.n_formal_charge
        atom.SetUnsignedProp("_TotalNumHs", c.imp_H[h_idx])

    if c.use_chirality:
        # The chirality one-hot is always the last segment; use the last
        # nonzero index, mirroring MolecularGraph.features_to_atom().
        cip_idx = (
            int(nonzero[-1])
            - c.n_atom_types
            - c.n_formal_charge
            - (0 if (c.use_explicit_H or c.ignore_H) else c.n_imp_H)
        )
        cip_code = c.chirality[cip_idx]
        if cip_code != "None":
            atom.SetProp("_CIPCode", cip_code)

    return atom


def graph_to_smiles(
    nodes: np.ndarray, edges: np.ndarray, c
) -> Optional[str]:
    """
    Reconstruct a canonical SMILES string from node/edge feature arrays.

    Args:
        nodes : (max_n_nodes, n_node_features) — node feature matrix.
        edges : (max_n_nodes, max_n_nodes, n_edge_features) — edge tensor.
        c     : constants namedtuple from _build_constants().

    Returns:
        Canonical SMILES string, or None if reconstruction fails.
    """
    # Atoms are present where at least one node feature is non-zero.
    atom_mask = np.any(nodes != 0, axis=1)
    n_nodes   = int(atom_mask.sum())
    if n_nodes == 0:
        return None

    mol = rdkit.Chem.RWMol()
    idx_map = {}
    for v in range(n_nodes):
        atom = _node_row_to_atom(nodes[v], c)
        if atom is None:
            return None
        idx_map[v] = mol.AddAtom(atom)

    for bond_type in range(c.n_edge_features):
        for i in range(n_nodes):
            for j in range(i):
                if edges[i, j, bond_type]:
                    mol.AddBond(
                        idx_map[i], idx_map[j],
                        c.int_to_bondtype[bond_type],
                    )

    try:
        mol = mol.GetMol()
        # Always sanitize so RDKit re-perceives aromaticity.
        # This normalises Kekulé forms (produced when use_aromatic_bonds=False)
        # back to aromatic SMILES, matching the reference .smi canonicalization.
        rdkit.Chem.SanitizeMol(mol)
        return MolToSmiles(mol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dataset_dir() -> Path:
    d = DATASET_DIR if isinstance(DATASET_DIR, Path) else Path(DATASET_DIR)
    if not d.exists():
        pytest.skip(f"Dataset directory not found: {d}")
    return d


@pytest.fixture(scope="session")
def preprocessing_params(dataset_dir) -> dict:
    return load_preprocessing_params(dataset_dir)


@pytest.fixture(scope="session")
def constants(preprocessing_params):
    return _build_constants(preprocessing_params)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitFileCounts:
    """Checks on the .smi split files."""

    def test_split_files_exist(self, dataset_dir):
        """train.smi, valid.smi, and test.smi must all be present."""
        missing = [
            name for name in ("train.smi", "valid.smi", "test.smi")
            if not (dataset_dir / name).exists()
        ]
        assert not missing, (
            f"Missing split file(s) in {dataset_dir}: {missing}"
        )

    def test_split_counts_sum_to_original(self, dataset_dir):
        """
        The total number of SMILES across the three split files must equal
        the number of molecules in the original SMILES file.
        Skipped if SMILES_FILE is None.
        """
        if SMILES_FILE is None:
            pytest.skip("SMILES_FILE not configured — skipping total-count check.")

        original_path = SMILES_FILE if isinstance(SMILES_FILE, Path) else Path(SMILES_FILE)
        if not original_path.exists():
            pytest.skip(f"Original SMILES file not found: {original_path}")

        n_original = len(read_smiles_file(original_path))
        n_splits   = sum(
            len(read_smiles_file(dataset_dir / f"{s}.smi")) for s in SPLITS
        )
        assert n_splits == n_original, (
            f"Split files contain {n_splits} molecules total, "
            f"but original file has {n_original}."
        )

    def test_no_overlap_between_splits(self, dataset_dir):
        """
        No SMILES string should appear in more than one split.
        (Canonicalise before comparing to be robust to equivalent representations.)
        """
        split_smiles = {}
        for split in SPLITS:
            path = dataset_dir / f"{split}.smi"
            if not path.exists():
                pytest.skip(f"{split}.smi not found — skipping overlap check.")
            raw = read_smiles_file(path)
            canon = set()
            for s in raw:
                mol = rdkit.Chem.MolFromSmiles(s)
                if mol is not None:
                    canon.add(MolToSmiles(mol))
            split_smiles[split] = canon

        for i, s1 in enumerate(SPLITS):
            for s2 in SPLITS[i + 1:]:
                overlap = split_smiles[s1] & split_smiles[s2]
                assert not overlap, (
                    f"Overlap between {s1} and {s2} splits "
                    f"({len(overlap)} molecule(s)): {list(overlap)[:5]}"
                )


class TestHDFFileCounts:
    """Checks that HDF5 molecule counts match their .smi counterparts."""

    @pytest.mark.parametrize("split", SPLITS)
    def test_hdf_exists(self, dataset_dir, split):
        h5_path = dataset_dir / f"{split}.h5"
        assert h5_path.exists(), f"{split}.h5 not found in {dataset_dir}"

    @pytest.mark.parametrize("split", SPLITS)
    def test_hdf_has_required_datasets(self, dataset_dir, split):
        """HDF5 file must contain 'nodes', 'edges', and 'APDs' datasets."""
        h5_path = dataset_dir / f"{split}.h5"
        if not h5_path.exists():
            pytest.skip(f"{split}.h5 not found")
        with h5py.File(h5_path, "r") as fh:
            missing = [k for k in ("nodes", "edges", "APDs") if k not in fh]
        assert not missing, (
            f"{split}.h5 is missing dataset(s): {missing}"
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_hdf_molecule_count_matches_smi(self, dataset_dir, split):
        """
        The number of complete molecules in <split>.h5 (rows where APD[-1]==1)
        must equal the number of SMILES in <split>.smi.
        """
        smi_path = dataset_dir / f"{split}.smi"
        h5_path  = dataset_dir / f"{split}.h5"
        if not smi_path.exists():
            pytest.skip(f"{split}.smi not found")
        if not h5_path.exists():
            pytest.skip(f"{split}.h5 not found")

        n_smi = len(read_smiles_file(smi_path))
        n_h5  = count_molecules_in_hdf(h5_path)

        assert n_h5 == n_smi, (
            f"{split}: HDF5 contains {n_h5} complete molecules, "
            f"but {split}.smi has {n_smi}."
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_hdf_subgraph_count_geq_molecule_count(self, dataset_dir, split):
        """
        Total subgraph rows must be >= molecule count (each molecule
        decomposes into ≥1 subgraph).
        """
        h5_path = dataset_dir / f"{split}.h5"
        if not h5_path.exists():
            pytest.skip(f"{split}.h5 not found")

        with h5py.File(h5_path, "r") as fh:
            n_subgraphs = fh["APDs"].shape[0]
        n_molecules = count_molecules_in_hdf(h5_path)

        assert n_subgraphs >= n_molecules, (
            f"{split}: {n_subgraphs} subgraph rows < {n_molecules} molecules."
        )


class TestSMILESReconstruction:
    """Checks that complete molecules in HDF5 can be decoded to valid SMILES."""

    @pytest.mark.parametrize("split", SPLITS)
    def test_all_molecules_reconstruct_to_valid_smiles(
        self, dataset_dir, constants, split
    ):
        """
        Every complete-molecule graph in <split>.h5 must decode to a valid,
        non-empty SMILES string.
        """
        h5_path = dataset_dir / f"{split}.h5"
        if not h5_path.exists():
            pytest.skip(f"{split}.h5 not found")

        nodes_arr, edges_arr = get_complete_graphs(h5_path)
        failures = []
        for i, (nodes, edges) in enumerate(zip(nodes_arr, edges_arr)):
            smi = graph_to_smiles(nodes, edges, constants)
            if not smi:
                failures.append(i)

        assert not failures, (
            f"{split}: {len(failures)} molecule(s) failed SMILES reconstruction "
            f"(indices: {failures[:10]}{'...' if len(failures) > 10 else ''})"
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_reconstructed_smiles_match_smi_file(
        self, dataset_dir, constants, split
    ):
        """
        The set of canonical SMILES reconstructed from <split>.h5 must equal
        the set of canonical SMILES in <split>.smi.

        This is a strong sanity check: it verifies the round-trip
        SMILES → graph → HDF5 → graph → SMILES is lossless.
        """
        smi_path = dataset_dir / f"{split}.smi"
        h5_path  = dataset_dir / f"{split}.h5"
        if not smi_path.exists():
            pytest.skip(f"{split}.smi not found")
        if not h5_path.exists():
            pytest.skip(f"{split}.h5 not found")

        # Canonical SMILES from the .smi file
        ref_smiles = set()
        for s in read_smiles_file(smi_path):
            mol = rdkit.Chem.MolFromSmiles(s)
            if mol is not None:
                ref_smiles.add(MolToSmiles(mol))

        # Canonical SMILES reconstructed from .h5
        nodes_arr, edges_arr = get_complete_graphs(h5_path)
        rec_smiles = set()
        for nodes, edges in zip(nodes_arr, edges_arr):
            smi = graph_to_smiles(nodes, edges, constants)
            if smi:
                rec_smiles.add(smi)

        only_in_ref = ref_smiles - rec_smiles
        only_in_h5  = rec_smiles - ref_smiles

        assert not only_in_ref and not only_in_h5, (
            f"{split} SMILES mismatch.\n"
            f"  In .smi but not in .h5 ({len(only_in_ref)}): "
            f"{list(only_in_ref)[:5]}\n"
            f"  In .h5 but not in .smi ({len(only_in_h5)}): "
            f"{list(only_in_h5)[:5]}"
        )
