"""
The `DataProcessor` class contains functions for pre-processing training data,
including dataset splitting (random, Butina, or custom) and HDF5 conversion.
"""

# load general packages and functions
import os
import random
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import parameters.load as load
import rdkit
import util

# load GraphINVENT-specific functions
from Analyzer import Analyzer
from MolecularGraph import PreprocessingGraph
from parameters.config import constants
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset splitting — module-level functions
# ---------------------------------------------------------------------------


def _read_smiles(path: Path) -> List[str]:
    """Read one SMILES per line; skip blank lines and comment lines."""
    smiles = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smiles.append(line.split()[0])  # first token is the SMILES
    return smiles


def _write_smiles(smiles: List[str], path: Path) -> None:
    """Write a list of SMILES strings, one per line, to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for smi in smiles:
            f.write(smi + "\n")


def _read_smiles_with_conditions(
    path: Path,
) -> Tuple[List[str], List[np.ndarray], List[str]]:
    """
    Read a tab-separated file with a SMILES column and optional property columns.

    Expected format (header required when properties are present)::

        SMILES\\tprop1\\tprop2
        CC(=O)O\\t-0.5\\t4.75

    If the file contains only SMILES (no tabs), returns empty condition lists.

    Returns:
        smiles_list:      SMILES strings, one per molecule.
        condition_vectors: Float32 arrays of shape (n_properties,), one per
                           molecule.  Empty list when no properties are present.
        condition_names:  Names of the property columns.  Empty list when none.

    Raises:
        ValueError: If any property value is non-numeric or a row has a wrong
                    number of columns.
    """
    smiles_list: List[str] = []
    condition_vectors: List[np.ndarray] = []
    condition_names: List[str] = []

    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]

    if not lines:
        return smiles_list, condition_vectors, condition_names

    # Detect TSV: first line has tabs
    has_tabs = "\t" in lines[0]
    if not has_tabs:
        # Plain SMILES file — no conditions.  Skip a header line, matching the
        # detection `parameters.load.molecules` uses; otherwise the literal
        # string "SMILES" is split into the dataset as if it were a molecule.
        if "SMILES" in lines[0].upper().split()[0]:
            lines = lines[1:]
        for ln in lines:
            smiles_list.append(ln.split()[0])
        return smiles_list, condition_vectors, condition_names

    # Parse header
    header = lines[0].split("\t")
    if header[0].upper() != "SMILES":
        raise ValueError(
            f"Expected first column header to be 'SMILES', got '{header[0]}'. "
            "Conditioning TSV files must start with a 'SMILES' column."
        )
    condition_names = header[1:]
    n_props = len(condition_names)

    for line_no, ln in enumerate(lines[1:], start=2):
        parts = ln.split("\t")
        if len(parts) != len(header):
            raise ValueError(
                f"Line {line_no}: expected {len(header)} tab-separated fields, "
                f"got {len(parts)}."
            )
        smiles_list.append(parts[0])
        try:
            cond = np.array([float(v) for v in parts[1:]], dtype=np.float32)
        except ValueError as exc:
            raise ValueError(
                f"Line {line_no}: non-numeric property value — {exc}"
            ) from exc
        condition_vectors.append(cond)

    if n_props == 0:
        condition_vectors = []

    return smiles_list, condition_vectors, condition_names


def _write_smiles_with_conditions(
    smiles: List[str],
    condition_vectors: List[np.ndarray],
    condition_names: List[str],
    path: Path,
) -> None:
    """
    Write SMILES (and optional properties) to a tab-separated file.

    If ``condition_vectors`` is empty, writes plain SMILES (one per line).
    Otherwise writes a TSV with a header row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if condition_vectors:
            f.write("SMILES\t" + "\t".join(condition_names) + "\n")
            for smi, cond in zip(smiles, condition_vectors):
                vals = "\t".join(str(v) for v in cond.tolist())
                f.write(f"{smi}\t{vals}\n")
        else:
            for smi in smiles:
                f.write(smi + "\n")


def _random_split(
    smiles: List[str],
    train_frac: float,
    valid_frac: float,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle with a fixed seed and split into train / valid / test."""
    train_idx, valid_idx, test_idx = _random_split_indices(
        len(smiles), train_frac, valid_frac, seed
    )
    return (
        [smiles[i] for i in train_idx],
        [smiles[i] for i in valid_idx],
        [smiles[i] for i in test_idx],
    )


def _random_split_indices(
    n: int,
    train_frac: float,
    valid_frac: float,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Return shuffled integer indices split into train / valid / test."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_train = int(round(n * train_frac))
    n_valid = int(round(n * valid_frac))

    return (
        indices[:n_train],
        indices[n_train : n_train + n_valid],
        indices[n_train + n_valid :],
    )


def _butina_split_indices(
    smiles: List[str],
    train_frac: float,
    valid_frac: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Butina clustering split returning integer indices."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from rdkit.ML.Cluster import Butina
    except ImportError as exc:
        raise ImportError(
            "Butina splitting requires RDKit. "
            "Install it with: conda install -c conda-forge rdkit"
        ) from exc

    fps, valid_indices, invalid_count = [], [], 0
    for orig_idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_count += 1
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        valid_indices.append(orig_idx)

    if invalid_count:
        print(
            f"  Warning: {invalid_count} SMILES failed RDKit parsing "
            "and will be excluded from the Butina split.",
            flush=True,
        )

    n = len(valid_indices)
    if n == 0:
        raise ValueError("No valid SMILES found for Butina splitting.")

    print(f"  Computing Tanimoto distance matrix for {n} molecules …", flush=True)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, cutoff=0.4, isDistData=True)

    n_train_target = int(round(n * train_frac))
    n_valid_target = int(round(n * valid_frac))

    train_idx, valid_idx, test_idx = [], [], []
    n_assigned_train, n_assigned_valid = 0, 0

    for cluster in clusters:
        orig_cluster = [valid_indices[i] for i in cluster]
        if n_assigned_train < n_train_target:
            train_idx.extend(orig_cluster)
            n_assigned_train += len(orig_cluster)
        elif n_assigned_valid < n_valid_target:
            valid_idx.extend(orig_cluster)
            n_assigned_valid += len(orig_cluster)
        else:
            test_idx.extend(orig_cluster)

    return train_idx, valid_idx, test_idx


def _butina_split(
    smiles: List[str],
    train_frac: float,
    valid_frac: float,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Cluster SMILES with the Butina algorithm (ECFP4 / Tanimoto distance ≤ 0.4)
    and assign clusters to splits so the test set is chemically dissimilar to
    the training set.
    """
    train_idx, valid_idx, test_idx = _butina_split_indices(
        smiles, train_frac, valid_frac
    )
    return (
        [smiles[i] for i in train_idx],
        [smiles[i] for i in valid_idx],
        [smiles[i] for i in test_idx],
    )


def _custom_split(
    smiles: List[str],
    train_frac: float,
    valid_frac: float,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Placeholder for a user-defined splitting strategy.

    Replace the body of this function with your own logic.  The function must
    return three lists of SMILES strings: (train, valid, test).

    Example skeleton:

        # 1. Compute a custom score or property for each molecule.
        scores = [my_score_fn(smi) for smi in smiles]

        # 2. Sort or partition by those scores.
        sorted_smiles = [smi for _, smi in sorted(zip(scores, smiles))]

        # 3. Slice into splits.
        n = len(sorted_smiles)
        n_train = int(round(n * train_frac))
        n_valid = int(round(n * valid_frac))
        train = sorted_smiles[:n_train]
        valid = sorted_smiles[n_train:n_train + n_valid]
        test  = sorted_smiles[n_train + n_valid:]
        return train, valid, test
    """
    raise NotImplementedError(
        "Custom split selected but _custom_split() has not been implemented.  "
        "Edit _custom_split() in graphinvent/DataProcessor.py and replace "
        "this placeholder with your splitting logic."
    )


def split_smiles_file(
    smiles_file: str,
    dataset_dir: str,
    split_type: str,
    train_frac: float,
    valid_frac: float,
) -> None:
    """
    Read `smiles_file`, split it, and write train.smi / valid.smi / test.smi
    into `dataset_dir`.

    When the input file is a tab-separated TSV with property columns, the split
    files preserve the TSV format (including the header).

    Args:
        smiles_file : Path to the input SMILES file (plain or TSV).
        dataset_dir : Directory where the split .smi files will be written.
        split_type  : One of "random", "butina", or "custom".
        train_frac  : Fraction of molecules for the training set.
        valid_frac  : Fraction for the validation set.
    """
    smiles_file = Path(smiles_file)
    dataset_dir = Path(dataset_dir)
    test_frac = 1.0 - train_frac - valid_frac

    if test_frac < 0:
        raise ValueError(f"train_frac ({train_frac}) + valid_frac ({valid_frac}) > 1.0")
    if not smiles_file.exists():
        raise FileNotFoundError(f"smiles_file not found: {smiles_file}")

    print(f"* Reading SMILES from {smiles_file} …", flush=True)
    smiles, condition_vectors, condition_names = _read_smiles_with_conditions(
        smiles_file
    )
    has_conditions = bool(condition_vectors)
    print(f"  {len(smiles)} molecules read.", flush=True)

    # Deduplicate on canonical SMILES before splitting.  A molecule present
    # twice in the input would otherwise land in two different splits, leaking
    # test molecules into training and inflating every novelty number.
    from rdkit import Chem

    seen: dict = {}
    keep: list = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        key = Chem.MolToSmiles(mol) if mol is not None else smi
        if key in seen:
            continue
        seen[key] = i
        keep.append(i)
    n_dupes = len(smiles) - len(keep)
    if n_dupes:
        print(
            f"  Removed {n_dupes} duplicate molecule(s) before splitting.",
            flush=True,
        )
        smiles = [smiles[i] for i in keep]
        if has_conditions:
            condition_vectors = [condition_vectors[i] for i in keep]
    if has_conditions:
        print(f"  Condition properties: {condition_names}", flush=True)
    print(
        f"* Splitting ({split_type}): "
        f"{train_frac:.0%} train / {valid_frac:.0%} valid / {test_frac:.0%} test",
        flush=True,
    )

    if split_type == "random":
        train_idx, valid_idx, test_idx = _random_split_indices(
            len(smiles), train_frac, valid_frac
        )
    elif split_type == "butina":
        train_idx, valid_idx, test_idx = _butina_split_indices(
            smiles, train_frac, valid_frac
        )
    elif split_type == "custom":
        raise NotImplementedError(
            "Custom split selected but _custom_split() has not been implemented.  "
            "Edit split_smiles_file() in graphinvent/DataProcessor.py and add "
            "your splitting logic."
        )
    else:
        raise ValueError(
            f"Unknown split_type '{split_type}'. "
            "Choose from: 'random', 'butina', 'custom'."
        )

    train_smi = [smiles[i] for i in train_idx]
    valid_smi = [smiles[i] for i in valid_idx]
    test_smi = [smiles[i] for i in test_idx]

    dataset_dir.mkdir(parents=True, exist_ok=True)

    if has_conditions:
        train_cond = [condition_vectors[i] for i in train_idx]
        valid_cond = [condition_vectors[i] for i in valid_idx]
        test_cond = [condition_vectors[i] for i in test_idx]
        _write_smiles_with_conditions(
            train_smi, train_cond, condition_names, dataset_dir / "train.smi"
        )
        _write_smiles_with_conditions(
            valid_smi, valid_cond, condition_names, dataset_dir / "valid.smi"
        )
        _write_smiles_with_conditions(
            test_smi, test_cond, condition_names, dataset_dir / "test.smi"
        )
    else:
        _write_smiles(train_smi, dataset_dir / "train.smi")
        _write_smiles(valid_smi, dataset_dir / "valid.smi")
        _write_smiles(test_smi, dataset_dir / "test.smi")

    print(
        f"  Wrote {len(train_smi)} train / {len(valid_smi)} valid / {len(test_smi)} test "
        f"molecules to {dataset_dir}/",
        flush=True,
    )


def _dataset_dtype(name: str) -> np.dtype:
    """
    Storage dtype for an HDF5 dataset.

    `nodes`/`edges` are one-hot, so int8 suffices.  `action_probs` accumulates a
    COUNT each time an identical subgraph is deduplicated into an existing row,
    which routinely exceeds int8's 127 for the small subgraphs many molecules
    share; those counts were silently clipped, corrupting the relative action
    weights the row encodes.
    """
    if name == "condition_vector":
        return np.dtype("float32")
    if name == "action_probs":
        return np.dtype("int16")
    return np.dtype("int8")


class DataProcessor:
    """
    A class for preprocessing molecular sets and writing them to HDF files.
    """

    def __init__(self, path: str, is_training_set: bool = False) -> None:
        """
        Args:
        ----
            path (string)          : Full path/filename to SMILES file containing
                                     molecules.
            is_training_set (bool) : Indicates if this is the training set, as we
                                     calculate a few additional things for the training
                                     set.
        """
        # define some variables for later use
        self.path = path
        self.is_training_set = is_training_set
        self.condition_dim = getattr(constants, "condition_dim", 0)

        # When conditioning is enabled, store condition_vector in HDF5.
        if self.condition_dim > 0:
            self.dataset_names = ["nodes", "edges", "action_probs", "condition_vector"]
        else:
            self.dataset_names = ["nodes", "edges", "action_probs"]

        self.get_dataset_dims()  # creates `self.dims`

        # load the molecules
        self.molecule_set = load.molecules(self.path)

        # Load condition vectors when conditioning is enabled.
        # condition_vectors[i] is a float32 array of shape (condition_dim,) for
        # molecule i (matching the order of molecule_set).
        self.condition_vectors: List[np.ndarray] = []
        if self.condition_dim > 0:
            _, raw_cond, cond_names = _read_smiles_with_conditions(Path(self.path))
            if not raw_cond:
                raise ValueError(
                    f"condition_dim={self.condition_dim} but no property columns "
                    f"found in {self.path}. Provide a tab-separated file with a "
                    "'SMILES' header and property columns."
                )
            if len(raw_cond[0]) != self.condition_dim:
                raise ValueError(
                    f"condition_dim={self.condition_dim} but the file has "
                    f"{len(raw_cond[0])} property columns: {cond_names}."
                )
            self.condition_vectors = raw_cond

        # placeholders
        self.molecule_subset = None
        self.condition_subset: List[np.ndarray] = []
        self.dataset = None
        self.skip_collection = None
        self.resume_idx = None
        self.training_set_properties = None
        self.restart_index_file = None
        self.hdf_file = None
        self.dataset_size = None

        # get total number of molecules, and total number of subgraphs in their
        # decoding routes
        self.n_molecules = len(self.molecule_set)
        self.total_n_subgraphs = self.get_n_subgraphs()
        print(f"-- {self.n_molecules} molecules in set.", flush=True)
        print(f"-- {self.total_n_subgraphs} total subgraphs in set.", flush=True)

    def preprocess(self) -> None:
        """
        Prepares an HDF file to save three different datasets to it (`nodes`,
        `edges`, `action probabilities`), and slowly fills it in by looping over all the
        molecules in the data in groups (or "mini-batches").
        """
        chunked_path = f"{self.path[:-3]}h5.chunked"
        restart_index_file = constants.dataset_dir + "index.restart"
        resuming = constants.restart and os.path.exists(restart_index_file)

        # "a" would reopen a half-written file from a crashed run and then fail
        # in `create_datasets` with "name already exists"; a fresh run must
        # start from an empty file.
        with h5py.File(chunked_path, "a" if resuming else "w") as self.hdf_file:

            self.restart_index_file = restart_index_file

            if resuming:
                self.restart_preprocessing_job()
            else:
                self.start_new_preprocessing_job()

                # keep track of the dataset size (to resize later)
                self.dataset_size = 0

            self.training_set_properties = None

            # this is where we fill the datasets with actual data by looping
            # over subgraphs in blocks of size `constants.batch_size`
            for idx in range(0, self.total_n_subgraphs, constants.batch_size):

                if not self.skip_collection:

                    self.get_molecule_subset()

                    # add `constants.batch_size` subgraphs from
                    # `self.molecule_subset` to the dataset (and if training
                    # set, calculate their properties and add these to
                    # `self.training_set_properties`)
                    self.get_subgraphs(init_idx=idx)

                    util.write_last_molecule_idx(
                        last_molecule_idx=self.resume_idx,
                        dataset_size=self.dataset_size,
                        restart_file_path=constants.dataset_dir,
                    )

                if self.resume_idx >= self.n_molecules:
                    # all molecules have been processed

                    self.resize_datasets()  # remove padding from initialization
                    print("Datasets resized.", flush=True)

                    if self.is_training_set:
                        # No `not constants.restart` guard: that made the write
                        # impossible in exactly the case restart exists for, so
                        # a resumed job finished without train.csv and every
                        # later training job then failed to load it.
                        print("Writing training set properties.", flush=True)
                        util.save_training_set_properties(
                            training_set_properties=self.training_set_properties
                        )

                    break

        print("* Resaving datasets in unchunked format.")
        self.resave_datasets_unchunked()

    def restart_preprocessing_job(self) -> None:
        """
        Restarts a preprocessing job. Uses an index specified in the dataset
        directory to know where to resume preprocessing.
        """
        try:
            self.resume_idx, self.dataset_size = util.read_last_molecule_idx(
                restart_file_path=constants.dataset_dir
            )
        except (OSError, ValueError):
            self.resume_idx, self.dataset_size = 0, 0
        self.skip_collection = bool(
            self.resume_idx == self.n_molecules and self.is_training_set
        )

        # load dictionary of previously created datasets (`self.dataset`)
        self.load_datasets(hdf_file=self.hdf_file)

    def start_new_preprocessing_job(self) -> None:
        """
        Starts a fresh preprocessing job.
        """
        self.resume_idx = 0
        self.skip_collection = False

        # create a dictionary of empty HDF datasets (`self.dataset`)
        self.create_datasets(hdf_file=self.hdf_file)

    def resave_datasets_unchunked(self) -> None:
        """
        Resaves the HDF datasets in an unchunked format to remove initial
        padding.
        """
        with h5py.File(f"{self.path[:-3]}h5.chunked", "r", swmr=True) as chunked_file:
            keys = list(chunked_file.keys())
            data = [chunked_file.get(key)[:] for key in keys]
            data_zipped = tuple(zip(data, keys))

            with h5py.File(f"{self.path[:-3]}h5", "w") as unchunked_file:
                for d, k in tqdm(data_zipped):
                    # Must match `create_datasets`; re-casting to int8 here
                    # silently undid the wider dtype used while writing.
                    unchunked_file.create_dataset(
                        k, chunks=None, data=d, dtype=_dataset_dtype(k)
                    )

        # remove the restart file and chunked file (don't need them anymore)
        os.remove(self.restart_index_file)
        os.remove(f"{self.path[:-3]}h5.chunked")

    def get_subgraphs(self, init_idx: int) -> None:
        """
        Adds `constants.batch_size` subgraphs from `self.molecule_subset` to the
        HDF dataset (and if currently processing the training set, also
        calculates the full graphs' properties and adds these to
        `self.training_set_properties`).

        Args:
        ----
            init_idx (int) : As analysis is done in blocks/slices, `init_idx` is
                             the start index for the next block/slice to be taken
                             from `self.molecule_subset`.
        """
        data_subgraphs, data_action_probs, molecular_graph_list = [], [], []
        data_condition_vectors: List[np.ndarray] = []

        molecules_processed = 0  # keep track of the number of molecules processed

        # loop over all the `PreprocessingGraph`s, enumerated so we can look up
        # the corresponding condition vector by position in the subset.
        for mol_pos, graph in enumerate(map(self.get_graph, self.molecule_subset)):
            if graph is None:
                # unparseable SMILES: skipped here and by `get_n_subgraphs`
                continue
            molecular_graph_list.append(graph)

            # Condition vector for this molecule (all-zero when unconditional).
            if self.condition_dim > 0:
                mol_condition = self.condition_subset[mol_pos]
            else:
                mol_condition = None

            # get the number of decoding graphs
            n_subgraphs = graph.get_decoding_route_length()

            for new_subgraph_idx in range(n_subgraphs):

                # `get_decoding_route_state() returns a list of [`subgraph`, `action_probs`],
                subgraph, action_probs = graph.get_decoding_route_state(
                    subgraph_idx=new_subgraph_idx
                )

                if self.condition_dim > 0:
                    # When conditioning is enabled, store every subgraph
                    # separately (no deduplication) so each carries its
                    # molecule's condition vector.
                    data_subgraphs.append(subgraph)
                    data_action_probs.append(action_probs)
                    data_condition_vectors.append(mol_condition)
                else:
                    # Deduplicate identical subgraphs, merging their action
                    # counts.  The append must be guarded by whether a match was
                    # found: the old `count == len(data_subgraphs)` test was also
                    # true when the match was the LAST entry, so such subgraphs
                    # were merged and appended, double-counting their actions.
                    matched = False
                    for idx, existing_subgraph in enumerate(data_subgraphs):
                        try:
                            nodes_equal = (subgraph[0] == existing_subgraph[0]).all()
                        except AttributeError:
                            nodes_equal = False
                        try:
                            edges_equal = (subgraph[1] == existing_subgraph[1]).all()
                        except AttributeError:
                            edges_equal = False
                        if nodes_equal and edges_equal:
                            data_action_probs[idx] += action_probs
                            matched = True
                            break
                    if not matched:
                        data_subgraphs.append(subgraph)
                        data_action_probs.append(action_probs)

            # This molecule's whole decoding route is now in the buffer.
            molecules_processed += 1

            # Flush only on a molecule boundary.  Flushing mid-route and then
            # counting the molecule as processed silently discarded the rest of
            # its route -- ~9% of all training states on the shipped debug set.
            # The group may therefore end slightly above `batch_size`, which is
            # fine: the HDF datasets are allocated from `get_n_subgraphs`, an
            # upper bound that deduplication only ever reduces.
            len_data_subgraphs = len(data_subgraphs)
            if len_data_subgraphs >= constants.batch_size:
                self.save_group(
                    data_subgraphs=data_subgraphs,
                    data_action_probs=data_action_probs,
                    group_size=len_data_subgraphs,
                    init_idx=init_idx,
                    data_condition_vectors=data_condition_vectors,
                )

                # get molecular properties for group iff it's the training set
                self.compute_training_set_properties(
                    molecular_graphs=molecular_graph_list,
                    group_size=len_data_subgraphs,
                )

                # keep track of the last molecule to be processed in
                # `self.resume_idx`
                self.resume_idx += molecules_processed
                self.dataset_size += len_data_subgraphs

                return None

        n_processed_subgraphs = len(data_subgraphs)
        if n_processed_subgraphs == 0:
            # Nothing left to write (e.g. an empty trailing subset); writing a
            # zero-length group would raise a broadcast error in `save_group`.
            return None

        # save group with < `constants.batch_size` subgraphs (e.g. last block)
        self.save_group(
            data_subgraphs=data_subgraphs,
            data_action_probs=data_action_probs,
            group_size=n_processed_subgraphs,
            init_idx=init_idx,
            data_condition_vectors=data_condition_vectors,
        )

        # get molecular properties for this group iff it's the training set
        self.compute_training_set_properties(
            molecular_graphs=molecular_graph_list, group_size=n_processed_subgraphs
        )

        # keep track of the last molecule to be processed in `self.resume_idx`
        self.resume_idx += molecules_processed  # number of molecules processed
        self.dataset_size += n_processed_subgraphs  # subgraphs processed

        return None

    def create_datasets(self, hdf_file: h5py._hl.files.File) -> None:
        """
        Creates a dictionary of HDF5 datasets (`self.dataset`).

        Args:
        ----
            hdf_file (h5py._hl.files.File) : HDF5 file which will contain the datasets.
        """
        self.dataset = {}  # initialize

        for ds_name in self.dataset_names:
            dtype = _dataset_dtype(ds_name)
            self.dataset[ds_name] = hdf_file.create_dataset(
                ds_name,
                (self.total_n_subgraphs, *self.dims[ds_name]),
                chunks=True,  # must be True for resizing later
                dtype=dtype,
            )

    def resize_datasets(self) -> None:
        """
        Resizes the HDF datasets, since much longer datasets are initialized
        when first creating the HDF datasets (it it is impossible to predict
        how many graphs will be equivalent beforehand).
        """
        for dataset_name in self.dataset_names:
            try:
                self.dataset[dataset_name].resize(
                    (self.dataset_size, *self.dims[dataset_name])
                )
            except KeyError:  # `f_term` has no extra dims
                self.dataset[dataset_name].resize((self.dataset_size,))

    def get_dataset_dims(self) -> None:
        """
        Calculates the dimensions of the node features, edge features, and action probabilities,
        and stores them as lists in a dict (`self.dims`), where keys are the
        dataset name.

        Shapes:
        ------
            dims["nodes"] : [max N nodes, N atom types + N formal charges]
            dims["edges"] : [max N nodes, max N nodes, N bond types]
            dims["action_probs"]  : [action probabilities length = f_add length + f_conn length + f_term length]
        """
        self.dims = {}
        self.dims["nodes"] = constants.dim_nodes
        self.dims["edges"] = constants.dim_edges
        self.dims["action_probs"] = constants.dim_action_probs
        cond_dim = getattr(constants, "condition_dim", 0)
        if cond_dim > 0:
            self.dims["condition_vector"] = (cond_dim,)

    def get_graph(self, mol: rdkit.Chem.Mol) -> PreprocessingGraph:
        """
        Converts an `rdkit.Chem.Mol` object to `PreprocessingGraph`.

        Args:
        ----
            mol (rdkit.Chem.Mol) : Molecule to convert.

        Returns:
        -------
            molecular_graph (PreprocessingGraph) : Molecule, now as a graph.
        """
        if mol is None:
            return None
        if not constants.use_aromatic_bonds:
            rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
        return PreprocessingGraph(molecule=mol, constants=constants)

    def get_molecule_subset(self) -> None:
        """
        Slices `self.molecule_set` into a subset of molecules of size
        `constants.batch_size`, starting from `self.resume_idx`.
        `self.n_molecules` is the number of molecules in the full
        `self.molecule_set`.
        """
        init_idx = self.resume_idx
        subset_size = constants.batch_size
        self.molecule_subset = []
        self.condition_subset = []
        max_idx = min(init_idx + subset_size, self.n_molecules)

        count = -1
        for mol_idx, mol in enumerate(self.molecule_set):
            if mol is not None:
                count += 1
                if count < init_idx:
                    continue
                elif count >= max_idx:
                    return self.molecule_subset
                else:
                    self.molecule_subset.append(mol)
                    if self.condition_dim > 0:
                        self.condition_subset.append(self.condition_vectors[mol_idx])

    def get_n_subgraphs(self) -> int:
        """
        Calculates the total number of subgraphs in the decoding route of all
        molecules in `self.molecule_set`. Loads training, testing, or validation
        set. First, the `PreprocessingGraph` for each molecule is obtained, and
        then the length of the decoding route is trivially calculated for each.

        Returns:
        -------
            n_subgraphs (int) : Sum of number of subgraphs in decoding routes for
                                all molecules in `self.molecule_set`.
        """
        n_subgraphs = 0  # start the count

        # convert molecules in `self.molecule_set` to `PreprocessingGraph`s
        molecular_graph_generator = map(self.get_graph, self.molecule_set)

        # loop over all the `PreprocessingGraph`s
        n_valid = 0
        for molecular_graph in molecular_graph_generator:
            if molecular_graph is None:
                # `get_graph` returns None for SMILES RDKit cannot parse; they
                # are skipped in `get_subgraphs` too, so they must not be
                # counted here either or `resume_idx` can never reach
                # `n_molecules` and the run stalls on empty subsets.
                continue
            n_valid += 1

            # get the number of decoding graphs (i.e. the decoding route length)
            # and add them to the running count
            n_subgraphs += molecular_graph.get_decoding_route_length()

        self.n_molecules = n_valid
        return int(n_subgraphs)

    def compute_training_set_properties(
        self, molecular_graphs: list, group_size: int
    ) -> None:
        """
        Gets molecular properties for group of molecular graphs, only for the
        training set.

        Args:
        ----
            molecular_graphs (list) : Contains `PreprocessingGraph`s.
            group_size (int)        : Size of "group" (i.e. slice of graphs).
        """
        if self.is_training_set:

            analyzer = Analyzer()
            batch_properties = analyzer.evaluate_training_set(
                preprocessing_graphs=molecular_graphs
            )

            # merge properties of current group with the accumulated running total
            if self.training_set_properties:
                self.training_set_properties = analyzer.merge_training_set_properties(
                    prev_properties=self.training_set_properties,
                    next_properties=batch_properties,
                    weight_next=group_size,
                )
            else:
                self.training_set_properties = batch_properties
        else:
            self.training_set_properties = None

    def load_datasets(self, hdf_file: h5py._hl.files.File) -> None:
        """
        Creates a dictionary of HDF datasets (`self.dataset`) which have been
        previously created (for restart jobs only).

        Args:
        ----
            hdf_file (h5py._hl.files.File) : HDF file containing all the datasets.
        """
        self.dataset = {}  # initialize dictionary of datasets

        # use the names of the datasets as the keys in `self.dataset`
        for ds_name in self.dataset_names:
            self.dataset[ds_name] = hdf_file.get(ds_name)

    def save_group(
        self,
        data_subgraphs: list,
        data_action_probs: list,
        group_size: int,
        init_idx: int,
        data_condition_vectors: List[np.ndarray] = None,
    ) -> None:
        """
        Saves a group of padded subgraphs and their corresponding action probabilities to the HDF
        datasets as `numpy.ndarray`s.

        Args:
        ----
            data_subgraphs (list) : Contains molecular subgraphs.
            data_action_probs (list)      : Contains action probabilities.
            group_size (int)      : Size of HDF "slice".
            init_idx (int)        : Index to begin slicing.
            data_condition_vectors (list) : Condition vectors, one per subgraph.
                                           Only used when condition_dim > 0.
        """
        # convert to `np.ndarray`s
        nodes = np.array([graph_tuple[0] for graph_tuple in data_subgraphs])
        edges = np.array([graph_tuple[1] for graph_tuple in data_subgraphs])
        action_probs = np.array(data_action_probs)

        end_idx = init_idx + group_size  # idx to end slicing

        # once data is padded, save it to dataset slice
        self.dataset["nodes"][init_idx:end_idx] = nodes
        self.dataset["edges"][init_idx:end_idx] = edges
        self.dataset["action_probs"][init_idx:end_idx] = action_probs

        if self.condition_dim > 0 and data_condition_vectors:
            condition_array = np.array(data_condition_vectors, dtype=np.float32)
            self.dataset["condition_vector"][init_idx:end_idx] = condition_array
