"""
The `DataProcessor` class contains functions for pre-processing training data,
including dataset splitting (random, Butina, or custom) and HDF5 conversion.
"""
# load general packages and functions
import os
import random
import numpy as np
import rdkit
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

# load GraphINVENT-specific functions
from Analyzer import Analyzer
from parameters.constants import constants
import parameters.load as load
from MolecularGraph import PreprocessingGraph
import util


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


def _random_split(
    smiles: List[str],
    train_frac: float,
    valid_frac: float,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle with a fixed seed and split into train / valid / test."""
    rng = random.Random(seed)
    shuffled = list(smiles)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_valid = int(round(n * valid_frac))

    train = shuffled[:n_train]
    valid = shuffled[n_train:n_train + n_valid]
    test  = shuffled[n_train + n_valid:]
    return train, valid, test


def _butina_split(
    smiles: List[str],
    train_frac: float,
    valid_frac: float,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Cluster SMILES with the Butina algorithm (ECFP4 / Tanimoto distance ≤ 0.4)
    and assign clusters to splits so the test set is chemically dissimilar to
    the training set.

    Strategy:
      1. Compute ECFP4 fingerprints and all-pairs Tanimoto distances.
      2. Run Butina clustering (distance threshold = 0.4).
      3. Sort clusters by descending size.
      4. Greedily assign clusters to train until train_frac is reached,
         then to valid until valid_frac is reached, then the rest to test.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from rdkit.ML.Cluster import Butina
    except ImportError as exc:
        raise ImportError(
            "Butina splitting requires RDKit. "
            "Install it with: conda install -c conda-forge rdkit"
        ) from exc

    fps, valid_smiles, invalid_smiles = [], [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_smiles.append(smi)
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        valid_smiles.append(smi)

    if invalid_smiles:
        print(
            f"  Warning: {len(invalid_smiles)} SMILES failed RDKit parsing "
            "and will be excluded from the Butina split.",
            flush=True,
        )

    n = len(valid_smiles)
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
        if n_assigned_train < n_train_target:
            train_idx.extend(cluster)
            n_assigned_train += len(cluster)
        elif n_assigned_valid < n_valid_target:
            valid_idx.extend(cluster)
            n_assigned_valid += len(cluster)
        else:
            test_idx.extend(cluster)

    return (
        [valid_smiles[i] for i in train_idx],
        [valid_smiles[i] for i in valid_idx],
        [valid_smiles[i] for i in test_idx],
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

    Args:
        smiles_file : Path to the input SMILES file.
        dataset_dir : Directory where the split .smi files will be written.
        split_type  : One of "random", "butina", or "custom".
        train_frac  : Fraction of molecules for the training set.
        valid_frac  : Fraction for the validation set.
    """
    smiles_file = Path(smiles_file)
    dataset_dir = Path(dataset_dir)
    test_frac   = 1.0 - train_frac - valid_frac

    if test_frac < 0:
        raise ValueError(
            f"train_frac ({train_frac}) + valid_frac ({valid_frac}) > 1.0"
        )
    if not smiles_file.exists():
        raise FileNotFoundError(f"smiles_file not found: {smiles_file}")

    print(f"* Reading SMILES from {smiles_file} …", flush=True)
    smiles = _read_smiles(smiles_file)
    print(f"  {len(smiles)} molecules read.", flush=True)
    print(
        f"* Splitting ({split_type}): "
        f"{train_frac:.0%} train / {valid_frac:.0%} valid / {test_frac:.0%} test",
        flush=True,
    )

    if split_type == "random":
        train, valid, test = _random_split(smiles, train_frac, valid_frac)
    elif split_type == "butina":
        train, valid, test = _butina_split(smiles, train_frac, valid_frac)
    elif split_type == "custom":
        train, valid, test = _custom_split(smiles, train_frac, valid_frac)
    else:
        raise ValueError(
            f"Unknown split_type '{split_type}'. "
            "Choose from: 'random', 'butina', 'custom'."
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_smiles(train, dataset_dir / "train.smi")
    _write_smiles(valid, dataset_dir / "valid.smi")
    _write_smiles(test,  dataset_dir / "test.smi")

    print(
        f"  Wrote {len(train)} train / {len(valid)} valid / {len(test)} test "
        f"molecules to {dataset_dir}/",
        flush=True,
    )


class DataProcessor:
    """
    A class for preprocessing molecular sets and writing them to HDF files.
    """
    def __init__(self, path : str, is_training_set : bool=False) -> None:
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
        self.path            = path
        self.is_training_set = is_training_set
        self.dataset_names   = ["nodes", "edges", "APDs"]
        self.get_dataset_dims()  # creates `self.dims`

        # load the molecules
        self.molecule_set = load.molecules(self.path)

        # placeholders
        self.molecule_subset    = None
        self.dataset            = None
        self.skip_collection    = None
        self.resume_idx         = None
        self.training_set_properties      = None
        self.restart_index_file = None
        self.hdf_file           = None
        self.dataset_size       = None

        # get total number of molecules, and total number of subgraphs in their
        # decoding routes
        self.n_molecules       = len(self.molecule_set)
        self.total_n_subgraphs = self.get_n_subgraphs()
        print(f"-- {self.n_molecules} molecules in set.", flush=True)
        print(f"-- {self.total_n_subgraphs} total subgraphs in set.",
              flush=True)

    def preprocess(self) -> None:
        """
        Prepares an HDF file to save three different datasets to it (`nodes`,
        `edges`, `APDs`), and slowly fills it in by looping over all the
        molecules in the data in groups (or "mini-batches").
        """
        with h5py.File(f"{self.path[:-3]}h5.chunked", "a") as self.hdf_file:

            self.restart_index_file = constants.dataset_dir + "index.restart"

            if constants.restart and os.path.exists(self.restart_index_file):
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
                        restart_file_path=constants.dataset_dir
                    )


                if self.resume_idx == self.n_molecules:
                    # all molecules have been processed

                    self.resize_datasets()  # remove padding from initialization
                    print("Datasets resized.", flush=True)

                    if self.is_training_set and not constants.restart:

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
        self.resume_idx      = 0
        self.skip_collection = False

        # create a dictionary of empty HDF datasets (`self.dataset`)
        self.create_datasets(hdf_file=self.hdf_file)

    def resave_datasets_unchunked(self) -> None:
        """
        Resaves the HDF datasets in an unchunked format to remove initial
        padding.
        """
        with h5py.File(f"{self.path[:-3]}h5.chunked", "r", swmr=True) as chunked_file:
            keys        = list(chunked_file.keys())
            data        = [chunked_file.get(key)[:] for key in keys]
            data_zipped = tuple(zip(data, keys))

            with h5py.File(f"{self.path[:-3]}h5", "w") as unchunked_file:
                for d, k in tqdm(data_zipped):
                    unchunked_file.create_dataset(
                        k, chunks=None, data=d, dtype=np.dtype("int8")
                    )

        # remove the restart file and chunked file (don't need them anymore)
        os.remove(self.restart_index_file)
        os.remove(f"{self.path[:-3]}h5.chunked")

    def get_subgraphs(self, init_idx : int) -> None:
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
        data_subgraphs, data_apds, molecular_graph_list = [], [], []  # initialize

        # convert all molecules in `self.molecules_subset` to `PreprocessingGraphs`
        molecular_graph_generator = map(self.get_graph, self.molecule_subset)

        molecules_processed       = 0  # keep track of the number of molecules processed

        # loop over all the `PreprocessingGraph`s
        for graph in molecular_graph_generator:
            molecules_processed += 1

            # store `PreprocessingGraph` object
            molecular_graph_list.append(graph)

            # get the number of decoding graphs
            n_subgraphs = graph.get_decoding_route_length()

            for new_subgraph_idx in range(n_subgraphs):

                # `get_decoding_route_state() returns a list of [`subgraph`, `apd`],
                subgraph, apd = graph.get_decoding_route_state(
                    subgraph_idx=new_subgraph_idx
                )

                # "collect" all APDs corresponding to pre-existing subgraphs,
                # otherwise append both new subgraph and new APD
                count = 0
                for idx, existing_subgraph in enumerate(data_subgraphs):

                    count += 1
                    # check if subgraph `subgraph` is "already" in
                    # `data_subgraphs` as `existing_subgraph`, and if so, add
                    # the "new" APD to the "old"
                    try:  # first compare the node feature matrices
                        nodes_equal = (subgraph[0] == existing_subgraph[0]).all()
                    except AttributeError:
                        nodes_equal = False
                    try:  # then compare the edge feature tensors
                        edges_equal = (subgraph[1] == existing_subgraph[1]).all()
                    except AttributeError:
                        edges_equal = False

                    # if both matrices have a match, then subgraphs are the same
                    if nodes_equal and edges_equal:
                        existing_apd = data_apds[idx]
                        existing_apd += apd
                        break

                # if subgraph is not already in `data_subgraphs`, append it
                if count == len(data_subgraphs) or count == 0:
                    data_subgraphs.append(subgraph)
                    data_apds.append(apd)

                # if `constants.batch_size` unique subgraphs have been
                # processed, save group to the HDF dataset
                len_data_subgraphs = len(data_subgraphs)
                if len_data_subgraphs == constants.batch_size:
                    self.save_group(data_subgraphs=data_subgraphs,
                                    data_apds=data_apds,
                                    group_size=len_data_subgraphs,
                                    init_idx=init_idx)

                    # get molecular properties for group iff it's the training set
                    self.compute_training_set_properties(molecular_graphs=molecular_graph_list,
                                           group_size=constants.batch_size)

                    # keep track of the last molecule to be processed in
                    # `self.resume_idx`
                    # number of molecules processed:
                    self.resume_idx   += molecules_processed
                    # subgraphs processed:
                    self.dataset_size += constants.batch_size

                    return None

        n_processed_subgraphs = len(data_subgraphs)

        # save group with < `constants.batch_size` subgraphs (e.g. last block)
        self.save_group(data_subgraphs=data_subgraphs,
                        data_apds=data_apds,
                        group_size=n_processed_subgraphs,
                        init_idx=init_idx)

        # get molecular properties for this group iff it's the training set
        self.compute_training_set_properties(molecular_graphs=molecular_graph_list,
                               group_size=n_processed_subgraphs)

        # keep track of the last molecule to be processed in `self.resume_idx`
        self.resume_idx   += molecules_processed    # number of molecules processed
        self.dataset_size += n_processed_subgraphs  # subgraphs processed

        return None

    def create_datasets(self, hdf_file : h5py._hl.files.File) -> None:
        """
        Creates a dictionary of HDF5 datasets (`self.dataset`).

        Args:
        ----
            hdf_file (h5py._hl.files.File) : HDF5 file which will contain the datasets.
        """
        self.dataset = {}  # initialize

        for ds_name in self.dataset_names:
            self.dataset[ds_name] = hdf_file.create_dataset(
                ds_name,
                (self.total_n_subgraphs, *self.dims[ds_name]),
                chunks=True,  # must be True for resizing later
                dtype=np.dtype("int8")
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
                    (self.dataset_size, *self.dims[dataset_name]))
            except KeyError:  # `f_term` has no extra dims
                self.dataset[dataset_name].resize((self.dataset_size,))

    def get_dataset_dims(self) -> None:
        """
        Calculates the dimensions of the node features, edge features, and APDs,
        and stores them as lists in a dict (`self.dims`), where keys are the
        dataset name.

        Shapes:
        ------
            dims["nodes"] : [max N nodes, N atom types + N formal charges]
            dims["edges"] : [max N nodes, max N nodes, N bond types]
            dims["APDs"]  : [APD length = f_add length + f_conn length + f_term length]
        """
        self.dims = {}
        self.dims["nodes"] = constants.dim_nodes
        self.dims["edges"] = constants.dim_edges
        self.dims["APDs"]  = constants.dim_apd

    def get_graph(self, mol : rdkit.Chem.Mol) -> PreprocessingGraph:
        """
        Converts an `rdkit.Chem.Mol` object to `PreprocessingGraph`.

        Args:
        ----
            mol (rdkit.Chem.Mol) : Molecule to convert.

        Returns:
        -------
            molecular_graph (PreprocessingGraph) : Molecule, now as a graph.
        """
        if mol is not None:
            if not constants.use_aromatic_bonds:
                rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
            molecular_graph = PreprocessingGraph(molecule=mol,
                                                 constants=constants)
        return molecular_graph

    def get_molecule_subset(self) -> None:
        """
        Slices `self.molecule_set` into a subset of molecules of size
        `constants.batch_size`, starting from `self.resume_idx`.
        `self.n_molecules` is the number of molecules in the full
        `self.molecule_set`.
        """
        init_idx             = self.resume_idx
        subset_size          = constants.batch_size
        self.molecule_subset = []
        max_idx              = min(init_idx + subset_size, self.n_molecules)

        count = -1
        for mol in self.molecule_set:
            if mol is not None:
                count += 1
                if count < init_idx:
                    continue
                elif count >= max_idx:
                    return self.molecule_subset
                else:
                    self.molecule_subset.append(mol)

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
        for molecular_graph in molecular_graph_generator:

            # get the number of decoding graphs (i.e. the decoding route length)
            # and add them to the running count
            n_subgraphs += molecular_graph.get_decoding_route_length()

        return int(n_subgraphs)

    def compute_training_set_properties(self, molecular_graphs : list, group_size : int) -> \
        None:
        """
        Gets molecular properties for group of molecular graphs, only for the
        training set.

        Args:
        ----
            molecular_graphs (list) : Contains `PreprocessingGraph`s.
            group_size (int)        : Size of "group" (i.e. slice of graphs).
        """
        if self.is_training_set:

            analyzer         = Analyzer()
            batch_properties = analyzer.evaluate_training_set(
                preprocessing_graphs=molecular_graphs
            )

            # merge properties of current group with the accumulated running total
            if self.training_set_properties:
                self.training_set_properties = analyzer.merge_training_set_properties(
                    prev_properties=self.training_set_properties,
                    next_properties=batch_properties,
                    weight_next=group_size
                )
            else:
                self.training_set_properties = batch_properties
        else:
            self.training_set_properties = None

    def load_datasets(self, hdf_file : h5py._hl.files.File) -> None:
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

    def save_group(self, data_subgraphs : list, data_apds : list,
                   group_size : int, init_idx : int) -> None:
        """
        Saves a group of padded subgraphs and their corresponding APDs to the HDF
        datasets as `numpy.ndarray`s.

        Args:
        ----
            data_subgraphs (list) : Contains molecular subgraphs.
            data_apds (list)      : Contains APDs.
            group_size (int)      : Size of HDF "slice".
            init_idx (int)        : Index to begin slicing.
        """
        # convert to `np.ndarray`s
        nodes = np.array([graph_tuple[0] for graph_tuple in data_subgraphs])
        edges = np.array([graph_tuple[1] for graph_tuple in data_subgraphs])
        apds  = np.array(data_apds)

        end_idx = init_idx + group_size  # idx to end slicing

        # once data is padded, save it to dataset slice
        self.dataset["nodes"][init_idx:end_idx] = nodes
        self.dataset["edges"][init_idx:end_idx] = edges
        self.dataset["APDs"][init_idx:end_idx]  = apds
