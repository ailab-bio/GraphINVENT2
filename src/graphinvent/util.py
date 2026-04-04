"""
Contains various miscellaneous useful functions.
"""

import ast
import csv
import datetime
import json
import re
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from typing import Iterator, Tuple, Union
from warnings import filterwarnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import rdkit
import torch
from matplotlib import pyplot as plt
from parameters.config import constants
from rdkit import RDLogger
from rdkit.Chem import MolToSmiles
from torch.utils.tensorboard import SummaryWriter


def get_feature_vector_indices() -> list:
    """
    Gets the indices of the different segments of the feature vector. The
    indices are analogous to the lengths of the various segments.

    Returns:
    -------
        idc (list) : Contains the indices of the different one-hot encoded
          segments used in the feature vector representations of nodes in
          `MolecularGraph`s. These segments are, in order, atom type, formal
          charge, number of implicit Hs, and chirality.
    """
    idc = [constants.n_atom_types, constants.n_formal_charge]

    # indices corresponding to implicit H's and chirality are optional (below)
    if not constants.use_explicit_H and not constants.ignore_H:
        idc.append(constants.n_imp_H)

    if constants.use_chirality:
        idc.append(constants.n_chirality)

    return np.cumsum(idc).tolist()


def get_last_epoch() -> str:
    """
    Gets previous training epoch by reading it from the "convergence.log" file.

    Returns:
    -------
        epoch_key (str) : A string that indicates the final epoch written to
          "convergence.log".
    """

    convergence_path = constants.job_dir + "convergence.log"

    if constants.job_type in ("rl", "constrained_rl", "goal_directed"):
        # RL logs use "Step N" labels; extract just "Step N"
        try:
            epoch_key_tmp, _, _ = read_row(path=convergence_path, row=-1, col=(0, 1, 2))
            epoch_key = " ".join(epoch_key_tmp.split()[:2])
        except (ValueError, IndexError):
            epoch_key = "Step init"
    else:
        # All supervised jobs (pretrain, transfer, generate, test) use "Epoch N"
        try:
            epoch_key, _, _ = read_row(path=convergence_path, row=-1, col=(0, 1, 2))
        except (ValueError, FileNotFoundError):
            epoch_key = "Epoch 1"

        if constants.job_type == "generate" or (
            constants.job_type == "sample"
            and getattr(constants, "sample_mode", "generate") == "generate"
        ):
            if constants.pretrained_model_path:
                import re as _re

                _m = _re.search(
                    r"model_restart_(\d+)\.pth", constants.pretrained_model_path
                )
                gen_epoch = int(_m.group(1)) if _m else 0
            else:
                gen_epoch = constants.generation_epoch
            epoch_key = f"Epoch GEN{gen_epoch}"
        elif constants.job_type == "test" or (
            constants.job_type == "sample"
            and getattr(constants, "sample_mode", "generate") == "evaluate"
        ):
            epoch_key = f"Epoch EVAL{constants.generation_epoch}"

    return epoch_key


def normalize_evaluation_metrics(
    property_histograms: dict, epoch_key: str
) -> Tuple[list, ...]:
    """
    Normalizes histograms in `props_dict`, converts them to `list`s (from
    `torch.Tensor`s) and rounds the elements. This is done for clarity when
    saving the histograms to CSV.

    Args:
    ----
        property_histograms (dict) : Contains histograms of evaluation metrics
          of interest.
        epoch_key (str) : Indicates the training epoch.

    Returns:
    -------
        norm_n_nodes_hist (torch.Tensor) : Normalized histogram of the number of
          nodes per molecule.
        norm_atom_type_hist (torch.Tensor) : Normalized histogram of the atom
          types present in the molecules.
        norm_charge_hist (torch.Tensor) : Normalized histogram of the formal
          charges present in the molecules.
        norm_numh_hist (torch.Tensor) : Normalized histogram of the number of
          implicit hydrogens present in the molecules.
        norm_n_edges_hist (torch.Tensor) : Normalized histogram of the number of
          edges per node in the molecules.
        norm_edge_feature_hist (torch.Tensor) : Normalized histogram of the
          edge features (types of bonds) present in the molecules.
        norm_chirality_hist (torch.Tensor) : Normalized histogram of the chiral
          centers present in the molecules.
    """
    # compute histograms for non-optional features
    norm_n_nodes_hist = [
        round(i, 2)
        for i in normalize(property_histograms[(epoch_key, "n_nodes_hist")]).tolist()
    ]
    norm_atom_type_hist = [
        round(i, 2)
        for i in normalize(property_histograms[(epoch_key, "atom_type_hist")]).tolist()
    ]
    norm_charge_hist = [
        round(i, 2)
        for i in normalize(
            property_histograms[(epoch_key, "formal_charge_hist")]
        ).tolist()
    ]
    norm_n_edges_hist = [
        round(i, 2)
        for i in normalize(property_histograms[(epoch_key, "n_edges_hist")]).tolist()
    ]
    norm_edge_feature_hist = [
        round(i, 2)
        for i in normalize(
            property_histograms[(epoch_key, "edge_feature_hist")]
        ).tolist()
    ]

    # compute histograms for optional features
    if not constants.use_explicit_H and not constants.ignore_H:
        norm_numh_hist = [
            round(i, 2)
            for i in normalize(property_histograms[(epoch_key, "numh_hist")]).tolist()
        ]
    else:
        norm_numh_hist = [0] * len(constants.imp_H)

    if constants.use_chirality:
        norm_chirality_hist = [
            round(i, 2)
            for i in normalize(
                property_histograms[(epoch_key, "chirality_hist")]
            ).tolist()
        ]
    else:
        norm_chirality_hist = [1, 0, 0]

    return (
        norm_n_nodes_hist,
        norm_atom_type_hist,
        norm_charge_hist,
        norm_numh_hist,
        norm_n_edges_hist,
        norm_edge_feature_hist,
        norm_chirality_hist,
    )


def get_restart_epoch() -> Union[int, str]:
    """
    Gets the restart epoch e.g. epoch for the last saved model state
    (`model_restart.pth`). Will simply return zero if called outside of a
    restart job.

    Returns:
    -------
        epoch (int or str) :
    """
    if (
        constants.job_type in ("rl", "constrained_rl", "goal_directed")
        and constants.restart
    ):
        # RL restart: find the last saved step from score.log.
        ft_log_path = constants.job_dir + "score.log"
        if not Path(ft_log_path).exists():
            raise FileNotFoundError(
                f"RL restart requires '{ft_log_path}' but the file was not found. "
                "Cannot determine the last completed RL step."
            )
        with open(ft_log_path, "r") as _f:
            _rows = [r for r in csv.reader(_f) if r]
        epoch = None
        for _row in reversed(_rows[1:]):  # skip header row
            try:
                epoch = int(_row[0].strip()[5:])  # "Step 10" → 10
                break
            except (ValueError, IndexError):
                continue
        if epoch is None:
            raise RuntimeError(
                f"Could not determine the last RL step from '{ft_log_path}'. "
                "The file may be empty or corrupted."
            )
    elif constants.job_type in ("rl", "constrained_rl", "goal_directed"):
        # Fresh RL start: if a direct checkpoint path was given, the epoch
        # number is meaningless for step counting — start from 0.
        # Otherwise use generation_epoch (only set when using pretrained_model_dir).
        if constants.pretrained_model_path:
            epoch = 0
        else:
            epoch = constants.generation_epoch
    elif (
        constants.restart
        or constants.job_type == "test"
        or (
            constants.job_type == "sample"
            and getattr(constants, "sample_mode", "generate") == "evaluate"
        )
    ):
        # Supervised restart or test: find the last saved epoch from generation.log.
        generation_path = constants.job_dir + "generation.log"
        if not Path(generation_path).exists():
            raise FileNotFoundError(
                f"Restart requires '{generation_path}' but the file was not found. "
                "Cannot determine the last completed epoch. "
                "Set restart=false to start a fresh run."
            )
        with open(generation_path, "r") as _f:
            _rows = [r for r in csv.reader(_f) if r]
        epoch = None
        for _row in reversed(_rows[1:]):  # skip header row
            try:
                epoch = int(_row[0].strip()[6:])  # "Epoch 10" → 10
                break
            except (ValueError, IndexError):
                continue
        if epoch is None:
            raise RuntimeError(
                f"Could not determine the last epoch from '{generation_path}'. "
                "The file may be empty or corrupted. "
                "Set restart=false to start a fresh run."
            )
    else:
        epoch = 0

    return epoch


def load_training_set_properties(csv_path: str) -> dict:
    """
    Loads training set properties from CSV and returns them as a dictionary.

    Args:
        csv_path: Path to the semicolon-delimited CSV file.

    Returns:
        properties: Training set properties, with tuple keys and
                    `torch.Tensor` values where applicable.
    """
    print("* Loading training set properties.", flush=True)

    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        csv_dict = dict(reader)

    properties: dict = {}
    for key, value in csv_dict.items():
        parsed_key = ast.literal_eval(key)
        if isinstance(parsed_key, (list, tuple)) and len(parsed_key) > 1:
            tuple_key = (str(parsed_key[0]), str(parsed_key[1]))
        else:
            tuple_key = parsed_key

        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Fallback for NumPy 2.0 repr like "[np.float64(0.0), np.float64(1.0)]"
            # Strip np.typeN(...) wrappers so literal_eval can handle the value.
            cleaned = re.sub(r"np\.\w+\(([^)]*)\)", r"\1", value)
            try:
                parsed_value = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                parsed_value = value

        if isinstance(parsed_value, list):
            parsed_value = torch.Tensor(parsed_value)

        properties[tuple_key] = parsed_value

    return properties


def normalize(list_of_nums: list) -> list:
    """
    Normalizes a list of numbers. Returns the list unchanged if the sum is zero.

    Args:
        list_of_nums: Numeric list or array to normalize.

    Returns:
        Normalized version of `list_of_nums`, or the original if the sum is zero.
    """
    total = sum(list_of_nums)
    if total == 0:
        return list_of_nums
    return list_of_nums / total


def one_hot_encode(x: Union[str, int], allowable_set: list) -> Iterator[int]:
    """
    Returns a one-hot encoding of `x` over `allowable_set`.

    Yields a sequence of ints (0 or 1) of length ``len(allowable_set)``, where
    the position corresponding to `x` is 1 and all others are 0.

    Args:
        x:              The value to encode.  Must be an element of `allowable_set`.
        allowable_set:  The ordered list of all possible values.

    Returns:
        Generator of ints representing the one-hot vector.

    Raises:
        Exception: If `x` is not in `allowable_set`.
    """
    if x not in set(allowable_set):  # use set for speedup over list
        raise Exception(
            f"Input {x} not in allowable set {allowable_set}. Add {x} to "
            f"allowable set in either a) `features.py` or b) your submission "
            f"script (`submit.py`) and run again."
        )
    one_hot_generator = (int(x == s) for s in allowable_set)
    return one_hot_generator


def properties_to_csv(
    prop_dict: dict,
    csv_filename: str,
    epoch_key: str,
    tb_writer: Union[SummaryWriter, None],
    append: bool = True,
    extra_cols: list = None,
) -> None:
    """
    Writes a CSV summarizing how training is going by comparing the properties
    of the generated structures during evaluation to the training set. Also
    writes the properties to an active tensorboard.

    Args:
    ----
        prop_dict (dict)   : Contains molecular properties.
        csv_filename (str) : Full path/filename to CSV file.
        epoch_key (str)    : For example, "Training set" or "Epoch {n}".
        tb_writer (Union[SummaryWriter, None]) : Tensorboard SummaryWriter (if one
                             has been created).
        append (bool)      : Indicates whether to append to the output file (if
                             the file exists) or start a new one. Default `True`.
    """
    # get all the relevant properties from the dictionary
    frac_valid = prop_dict[(epoch_key, "fraction_valid")]
    avg_n_nodes = prop_dict[(epoch_key, "avg_n_nodes")]
    avg_n_edges = prop_dict[(epoch_key, "avg_n_edges")]
    frac_unique = prop_dict[(epoch_key, "fraction_unique")]

    # use the following properties if they exist
    try:
        run_time = prop_dict[(epoch_key, "run_time")]
        frac_valid_pt = round(
            float(prop_dict[(epoch_key, "fraction_valid_properly_terminated")]), 5
        )
        frac_pt = round(
            float(prop_dict[(epoch_key, "fraction_properly_terminated")]), 5
        )
    except KeyError:
        run_time = "NA"
        frac_valid_pt = "NA"
        frac_pt = "NA"

    # helper: format "mean±std" when a std key is present in prop_dict
    def _fmt(mean_val, std_key: str) -> str:
        std = prop_dict.get((epoch_key, std_key))
        if std is not None:
            return f"{float(mean_val):.3f}\u00b1{float(std):.3f}"
        return f"{float(mean_val):.3f}"

    def _fmt_or_na(mean_val, std_key: str) -> str:
        if mean_val == "NA":
            return "NA"
        return _fmt(mean_val, std_key)

    (
        norm_n_nodes_hist,
        norm_atom_type_hist,
        norm_formal_charge_hist,
        norm_numh_hist,
        norm_n_edges_hist,
        norm_edge_feature_hist,
        norm_chirality_hist,
    ) = normalize_evaluation_metrics(prop_dict, epoch_key)

    import os as _os

    write_header = not append or not _os.path.exists(csv_filename)
    _extra_header = (", " + ", ".join(extra_cols)) if extra_cols else ""
    if write_header:
        with open(csv_filename, "w") as output_file:
            output_file.write(
                "set, fraction_valid, fraction_valid_pt, fraction_pt, run_time, "
                "avg_n_nodes, avg_n_edges, fraction_unique, atom_type_hist, "
                "formal_charge_hist, numh_hist, chirality_hist, "
                "n_nodes_hist, n_edges_hist, edge_feature_hist" + _extra_header + "\n"
            )

    _extra_values = ""
    if extra_cols:
        _parts = []
        for _col in extra_cols:
            _val = prop_dict.get((epoch_key, _col))
            if _val is None or _val != _val:  # None or NaN check
                _parts.append("NA")
            elif isinstance(_val, float):
                _parts.append(f"{_val:.5f}")
            else:
                _parts.append(str(_val))
        _extra_values = ", " + ", ".join(_parts)

    # append the properties of interest to the CSV file
    with open(csv_filename, "a") as output_file:
        output_file.write(
            f"{epoch_key}, "
            f"{_fmt(frac_valid, 'fraction_valid_std')}, "
            f"{_fmt_or_na(frac_valid_pt, 'fraction_valid_pt_std')}, "
            f"{_fmt_or_na(frac_pt, 'fraction_pt_std')}, "
            f"{run_time}, "
            f"{_fmt(avg_n_nodes, 'avg_n_nodes_std')}, "
            f"{_fmt(avg_n_edges, 'avg_n_edges_std')}, "
            f"{_fmt(frac_unique, 'fraction_unique_std')}, "
            f"{norm_atom_type_hist}, {norm_formal_charge_hist}, "
            f"{norm_numh_hist}, {norm_chirality_hist}, {norm_n_nodes_hist}, "
            f"{norm_n_edges_hist}, {norm_edge_feature_hist}" + _extra_values + "\n"
        )

    try:
        epoch = int(epoch_key.split()[1])
    except (IndexError, ValueError):
        pass
    else:
        if tb_writer is not None:
            tb_writer.add_scalar("Evaluation/fraction_valid", frac_valid, epoch)
            tb_writer.add_scalar(
                "Evaluation/fraction_valid_and_properly_term", frac_valid_pt, epoch
            )
            tb_writer.add_scalar(
                "Evaluation/fraction_properly_terminated", frac_pt, epoch
            )
            tb_writer.add_scalar("Evaluation/avg_n_nodes", avg_n_nodes, epoch)
            tb_writer.add_scalar("Evaluation/fraction_unique", frac_unique, epoch)


def read_last_molecule_idx(restart_file_path: str) -> Tuple[int, int]:
    """
    Reads the index of the last preprocessed molecule from a file called
    "index.restart" located in the same directory as the data. Also returns the
    dataset size thus far.

    Args:
    ----
        restart_file_path (str) : Path to the index restart file.

    Returns:
    -------
        Tuple[int, int] : The first integer is the index of the last preprocessed
                          molecule and the second integer is the dataset size.
    """
    with open(restart_file_path + "index.restart", "r") as txt_file:
        last_molecule_idx = np.genfromtxt(txt_file, delimiter=",")
    return int(last_molecule_idx[0]), int(last_molecule_idx[1])


def read_row(path: str, row: int, col: int) -> np.ndarray:
    """
    Reads a row from CSV file. Returns it as a `numpy.ndarray`. Removes "NA"
    missing values from the column before returning.

    Args:
    ----
        path (str) : Path to CSV file.
        row (int)  : Row to read.
        col (int)  : Column to read.

    Returns:
        np.ndarray : Desired row from the file.
    """
    with open(path, "r") as csv_file:
        data = np.genfromtxt(
            csv_file, dtype=str, delimiter=",", skip_header=1, usecols=col
        )
    data = np.array(data)
    return data[:][row]


def suppress_warnings() -> None:
    """
    Suppresses unimportant warnings for a cleaner readout.
    """
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    filterwarnings(action="ignore", category=UserWarning)
    filterwarnings(action="ignore", category=FutureWarning)
    # could instead suppress ALL warnings with:
    # `filterwarnings(action="ignore")`
    # but choosing not to do this


def turn_off_empty_axes(n_plots_y: int, n_plots_x: int, ax: plt.axes) -> plt.axes:
    """
    Turns off empty axes in a `n_plots_y` by `n_plots_x` grid of plots.

    Args:
    ----
        n_plots_y (int) : Number of plots along the y-axis.
        n_plots_x (int) : Number of plots along the x-axis.
        ax (plt.axes)   : Matplotlib object containing grid of plots.

    Returns:
    -------
        plt.axes : The updated Matplotlib object.
    """
    for vi in range(n_plots_y):
        for vj in range(n_plots_x):
            # if nothing plotted on ax, it will contain `inf`
            # in axes lims, so clean up (turn off)
            if "inf" in str(ax[vi, vj].dataLim):
                ax[vi, vj].axis("off")
    return ax


def write_last_molecule_idx(
    last_molecule_idx: int, dataset_size: int, restart_file_path: str
) -> None:
    """
    Writes the index of the last preprocessed molecule and the current dataset
    size to a file.

    Args:
    ----
        last_molecules_idx (int) : Index of last preprocessed molecule.
        dataset_size (int) : The dataset size.
        restart_file_path (str) : Path indicating where to save indices (should
          be same directory as the dataset).
    """
    with open(restart_file_path + "index.restart", "w") as txt_file:
        txt_file.write(str(last_molecule_idx) + ", " + str(dataset_size))


class _ConstantsEncoder(json.JSONEncoder):  # this might be dead code?
    """JSON encoder that handles non-serializable types gracefully."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        # RDKit BondType and other opaque objects → string representation
        return str(obj)


def _build_run_info(seed: int = 0) -> dict:
    """
    Collect reproducibility metadata: timestamp, library versions, git commit,
    device details, and the random seed used for this run.

    Returns a dict suitable for embedding in any params_all.json.
    """
    # --- timestamp ---
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    # --- library versions ---
    try:
        import rdkit as _rdkit

        rdkit_version = _rdkit.__version__
    except Exception:
        rdkit_version = "unknown"
    try:
        cuda_version = torch.version.cuda or "n/a"
    except Exception:
        cuda_version = "unknown"

    # --- device details ---
    device_name = "cpu"
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "cuda (unknown model)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"

    # --- git commit hash ---
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    return {
        "timestamp": timestamp,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "rdkit_version": rdkit_version,
        "cuda_version": cuda_version,
        "device_name": device_name,
        "git_hash": git_hash,
        "seed": seed,
    }


def write_job_parameters(params: namedtuple) -> None:
    """
    Writes all resolved job parameters/hyperparameters to ``params_all.json``,
    together with a ``run_info`` block containing reproducibility metadata
    (timestamp, library versions, git hash, device, seed).

    Args:
        params: Resolved constants namedtuple.
    """
    dict_path = Path(params.job_dir) / "params_all.json"
    params_dict = {field: getattr(params, field) for field in params._fields}
    params_dict["run_info"] = _build_run_info(seed=getattr(params, "seed", 0))

    with open(dict_path, "w") as f:
        json.dump(params_dict, f, indent=2, cls=_ConstantsEncoder)


def write_preprocessing_parameters(params: namedtuple) -> None:
    """
    Writes the feature-vocabulary parameters needed to verify preprocessing
    consistency to ``preprocessing_params.json`` in the dataset directory,
    together with split configuration and a ``run_info`` block.

    The actual split counts (n_train, n_valid, n_test) are appended later by
    :func:`update_preprocessing_stats` once the split files exist on disk.

    Args:
        params: Resolved constants namedtuple.
    """
    dict_path = Path(params.dataset_dir) / "preprocessing_params.json"
    vocab_keys = {
        "atom_types",
        "formal_charge",
        "imp_H",
        "chirality",
        "max_n_nodes",
        "use_aromatic_bonds",
        "use_canon",
        "use_chirality",
        "use_explicit_H",
        "ignore_H",
    }
    split_keys = {"split_type", "train_frac", "valid_frac", "smiles_file"}
    keys_to_write = vocab_keys | split_keys
    preproc_dict = {
        key: getattr(params, key) for key in keys_to_write if hasattr(params, key)
    }
    preproc_dict["run_info"] = _build_run_info(seed=getattr(params, "seed", 0))
    with open(dict_path, "w") as f:
        json.dump(preproc_dict, f, indent=2)


def update_preprocessing_stats(dataset_dir: str) -> None:
    """
    Append split counts (n_train, n_valid, n_test) to the existing
    ``preprocessing_params.json`` in *dataset_dir* by counting lines in the
    ``train.smi``, ``valid.smi``, and ``test.smi`` files.

    Safe to call even if a split file is absent (count is recorded as 0).

    Args:
        dataset_dir: Path to the dataset directory.
    """
    base = Path(dataset_dir)
    json_path = base / "preprocessing_params.json"
    if not json_path.exists():
        return

    with open(json_path) as f:
        data = json.load(f)

    for split, fname in [
        ("n_train", "train.smi"),
        ("n_valid", "valid.smi"),
        ("n_test", "test.smi"),
    ]:
        p = base / fname
        data[split] = sum(1 for _ in open(p)) if p.exists() else 0

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def write_graphs_to_smi(
    smi_filename: str, molecular_graphs_list: list, write: bool = False
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Calculates the validity and uniqueness of input molecular graphs. Then,
    (optional) writes the input molecular graphs a SMILES file.

    Args:
    ----
        smi_filename (str)           : Full path/filename to output SMILES file.
        molecular_graphs_list (list) : Contains molecular graphs.
        write (bool, optional)       : If True, writes the input molecular graphs
                                       to SMILES. If False, skips writing the SMILES,
                                       but calculates the rest of the metrics
                                       (validity, uniqueness) anyways.

    Returns:
    -------
        fraction_valid (float)           : The fraction of molecular graphs which
                                           are valid molecules.
        validity_tensor (torch.Tensor)   : A binary vector indicating the chemical
                                           validity of the input graphs using either
                                           a 1 (valid) or 0 (invalid).
        uniqueness_tensor (torch.Tensor) : A binary vector indicating the uniqueness
                                           of the input graphs using either a 1
                                           (unique or first duplicate instance)
                                           or 0 (duplicate).
    """
    validity_tensor = torch.zeros(len(molecular_graphs_list), device=constants.device)
    uniqueness_tensor = torch.ones(len(molecular_graphs_list), device=constants.device)
    smiles = []

    with open(smi_filename, "w") as smi_file:

        if write:
            smi_writer = rdkit.Chem.rdmolfiles.SmilesWriter(smi_file)

        for idx, molecular_graph in enumerate(molecular_graphs_list):

            mol = molecular_graph.get_molecule()
            try:
                mol.UpdatePropertyCache(strict=False)
                rdkit.Chem.SanitizeMol(mol)
                current_smiles = MolToSmiles(mol)
                if (
                    len(current_smiles) == 0
                ):  # TODO would we rather leave it blank for an empty SMILES?
                    raise ValueError
                if write:
                    smi_writer.write(mol)
                validity_tensor[idx] = 1
                if current_smiles in smiles:
                    uniqueness_tensor[idx] = 0
                smiles.append(current_smiles)
            except (ValueError, RuntimeError, AttributeError):
                # molecule cannot be written to file (likely contains unphysical
                # aromatic bond(s) or an unphysical valence), so put placeholder
                if write:
                    # `validity_tensor` remains 0
                    smi_writer.write(rdkit.Chem.MolFromSmiles("[Xe]"))

        if write:
            smi_writer.close()

    fraction_valid = torch.sum(validity_tensor, dim=0) / len(validity_tensor)

    return fraction_valid, validity_tensor, uniqueness_tensor


def write_training_status(
    tb_writer: Union[SummaryWriter, None],
    epoch: Union[int, None] = None,
    lr: Union[float, None] = None,
    training_loss: Union[float, None] = None,
    validation_loss: Union[float, None] = None,
    score: Union[float, None] = None,
    append: bool = True,
) -> None:
    """
    Writes the current epoch, loss, learning rate, and model score to CSV.

    Args:
    ----
        tb_writer (Union[SummaryWriter, None]) : Tensorboard SummaryWriter (if one
                                                 has been created).
        epoch (Union[int, None])             : Current epoch.
        lr (Union[float, None])              : Learning rate.
        training_loss (Union[float, None])   : Training loss.
        validation_loss (Union[float, None]) : Validation loss.
        score (Union[float, None])           : Model score (the UC-JSD).
        append (bool)                        : If True, appends to existing file.
                                               If False, creates a new file.
    """
    convergence_path = constants.job_dir + "convergence.log"
    # RL progress is measured in steps; all supervised jobs use epochs
    epoch_label = (
        "Step"
        if constants.job_type in ("rl", "constrained_rl", "goal_directed")
        else "Epoch"
    )

    is_rl = constants.job_type in ("rl", "constrained_rl", "goal_directed")

    if not append:  # create the file
        with open(convergence_path, "w") as output_file:
            if is_rl:
                output_file.write(
                    f"{epoch_label.lower()}, lr, avg_train_loss, model_score\n"
                )
            else:
                output_file.write(
                    f"{epoch_label.lower()}, lr, avg_train_loss, "
                    f"avg_valid_loss, model_score\n"
                )
    else:  # append to existing file
        if constants.job_type in [
            "pretrain",
            "transfer",
            "rl",
            "constrained_rl",
            "unconditional",
            "goal_directed",
        ]:
            if score is None:
                with open(convergence_path, "a") as output_file:
                    if is_rl:
                        output_file.write(
                            f"{epoch_label} {epoch}, {lr:.8f}, "
                            f"{training_loss:.8f}, "
                        )
                    else:
                        output_file.write(
                            f"{epoch_label} {epoch}, {lr:.8f}, "
                            f"{training_loss:.8f}, "
                            f"{validation_loss:.8f}, "
                        )
                # write to tensorboard
                if tb_writer is not None:
                    tb_writer.add_scalar("Training/training_loss", training_loss, epoch)
                    if not is_rl:
                        tb_writer.add_scalar(
                            "Training/validation_loss", validation_loss, epoch
                        )
                    tb_writer.add_scalar("Training/lr", lr, epoch)

            elif score == "NA":
                with open(convergence_path, "a") as output_file:
                    output_file.write(f"{score}\n")

            elif score is not None and training_loss is not None:
                with open(convergence_path, "a") as output_file:
                    if is_rl:
                        output_file.write(
                            f"{epoch_label} {epoch}, {lr:.8f}, "
                            f"{training_loss:.8f}, {score:.6f}\n"
                        )
                    else:
                        output_file.write(
                            f"{epoch_label} {epoch}, {lr:.8f}, "
                            f"{training_loss:.8f}, "
                            f"{validation_loss:.8f}, {score:.6f}\n"
                        )

            else:
                with open(convergence_path, "a") as output_file:
                    output_file.write(f"{score:.6f}\n")


def write_molecules(
    molecules: list,
    final_likelihoods: torch.Tensor,
    epoch: str,
    write: bool = False,
    label: str = "test",
) -> Tuple[list, list, list]:
    """
    Writes generated molecular graphs and their NLLs. In writing the structures
    to a SMILES file, determines if structures are valid and returns this
    information (to avoid recomputing later).

    Args:
    ----
        molecules (list)                 : Contains generated `MolecularGraph`s.
        final_likelihoods (torch.Tensor) : Contains final NLLs for the graphs.
        epoch (str)                      : Number corresponding to the current
                                           training epoch.
        write (bool)                     : Whether or not to write the molecules
                                           to SMILES.
        label (str)                      : Label to use when saving the molecules.

    Returns:
    -------
        fraction_valid (float)           : The fraction of molecular graphs which are
                                           valid molecules.
        validity_tensor (torch.Tensor)   : A binary vector indicating the chemical
                                           validity of the input graphs using either
                                           a 1 (valid) or 0 (invalid).
        uniqueness_tensor (torch.Tensor) : A binary vector indicating the uniqueness
                                           of the input graphs using either a 1
                                           (unique or first duplicate instance) or
                                           0 (duplicate).
    """
    # RL outputs are labelled by step number; supervised jobs by epoch/label
    if constants.job_type in ("rl", "constrained_rl", "goal_directed"):
        step = epoch.split(" ")[1]
        smi_filename = constants.job_dir + f"generation/step{step}_{label}.smi"
    else:
        smi_filename = constants.job_dir + f"generation/{label}.smi"

    fraction_valid, validity_tensor, uniqueness_tensor = write_graphs_to_smi(
        smi_filename=smi_filename, molecular_graphs_list=molecules, write=write
    )
    # save the NLLs and validity status
    write_likelihoods(
        likelihood_filename=f"{smi_filename[:-3]}likelihood",
        likelihoods=final_likelihoods,
    )
    write_validity(
        validity_file_path=f"{smi_filename[:-3]}valid", validity_tensor=validity_tensor
    )

    return fraction_valid, validity_tensor, uniqueness_tensor


def write_likelihoods(likelihood_filename: str, likelihoods: torch.Tensor) -> None:
    """
    Writes the final likelihoods of each molecule to a file in the same order as
    the molecules are written in the corresponding SMILES file.

    Args:
    ----
        likelihood_filename (str)  : Path to the likelihood-containing file.
        likelihoods (torch.Tensor) : Likelihoods.
    """
    with open(likelihood_filename, "w") as likelihood_file:
        for likelihood in likelihoods:
            likelihood_file.write(f"{likelihood}\n")


def save_training_set_properties(training_set_properties: dict) -> None:
    """
    Writes the training set properties to CSV.

    Args:
    ----
        training_set_properties (dict) : The properties of the training set.
    """
    training_set = constants.training_set  # path to "train.smi"
    dict_path = f"{training_set[:-4]}.csv"

    with open(dict_path, "w") as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=";")
        for key, value in training_set_properties.items():
            if "validity_tensor" in key:
                # skip writing the validity tensor here because it is really
                # long, instead it gets its own file elsewhere
                continue
            if isinstance(value, np.ndarray):
                csv_writer.writerow([key, [float(x) for x in value]])
            elif isinstance(value, torch.Tensor):
                try:
                    csv_writer.writerow([key, float(value)])
                except ValueError:
                    csv_writer.writerow([key, [float(i) for i in value]])
            else:
                csv_writer.writerow([key, value])


def write_validation_scores(
    output_dir: str,
    epoch_key: str,
    model_scores: dict,
    tb_writer: Union[SummaryWriter, None],
    append: bool = True,
) -> None:
    """
    Writes a CSV with the model validation scores as a function of the epoch.

    Args:
    ----
        output_dir (str)    : Full path/filename to CSV file.
        epoch_key (str)     : For example, "Training set" or "Epoch {n}".
        model_scores (dict) : Contains the average NLL per molecule of
                              {validation/train/generated} structures, and the
                              average model score (weighted mean of above two scores).
        tb_writer (Union[SummaryWriter, None]) : Tensorboard SummaryWriter
                              object (if it was created).
        append (bool)       : Indicates whether to append to the output file or
                              start a new one.
    """
    validation_file_path = output_dir + "validation.log"
    avg_likelihood_val = model_scores["avg_likelihood_val"]
    avg_likelihood_train = model_scores["avg_likelihood_train"]
    avg_likelihood_gen = model_scores["avg_likelihood_gen"]
    uc_jsd = model_scores["UC-JSD"]

    if not append:  # create file
        with open(validation_file_path, "w") as output_file:
            # write headeres
            output_file.write(
                "set, avg_likelihood_per_molecule_val, "
                "avg_likelihood_per_molecule_train, "
                "avg_likelihood_per_molecule_gen, uc_jsd\n"
            )

    # append the properties of interest to the CSV file
    with open(validation_file_path, "a") as output_file:
        output_file.write(
            f"{epoch_key:}, {avg_likelihood_val:.5f}, "
            f"{avg_likelihood_train:.5f}, "
            f"{avg_likelihood_gen:.5f}, {uc_jsd:.7f}\n"
        )

    try:
        epoch = int(epoch_key.split()[1])
    except (IndexError, ValueError):
        pass
    else:
        if tb_writer is not None:
            tb_writer.add_scalar(
                "Evaluation/avg_validation_likelihood", avg_likelihood_val, epoch
            )
            tb_writer.add_scalar(
                "Evaluation/avg_training_likelihood", avg_likelihood_train, epoch
            )
            tb_writer.add_scalar(
                "Evaluation/avg_generation_likelihood", avg_likelihood_gen, epoch
            )
            tb_writer.add_scalar("Evaluation/uc_jsd", uc_jsd, epoch)


def write_validity(validity_file_path: str, validity_tensor: torch.Tensor) -> None:
    """
    Writes the validity (0 or 1) of each molecule to a file in the same
    order as the molecules are written in the corresponding SMILES file.

    Args:
    ----
        validity_file_path (str)       : Path to validity file.
        validity_tensor (torch.Tensor) : A binary vector indicating the chemical
                                         validity of the input graphs using either
                                         a 1 (valid) or 0 (invalid).
    """
    with open(validity_file_path, "w") as valid_file:
        for valid in validity_tensor:
            valid_file.write(f"{valid}\n")


def log_likelihoods_to_tensorboard(
    tb_writer: Union[SummaryWriter, None],
    step: Union[int, None] = None,
    agent_loglikelihoods: Union[torch.Tensor, None] = None,
    prior_loglikelihoods: Union[torch.Tensor, None] = None,
) -> None:
    """
    Writes the current epoch and log-likelihoods to the tensorboard during
    fine-tuning jobs.

    Args:
    ----
        tb_writer (Union[SummaryWriter, None])           : Tensorboard SummaryWriter
                                                           object (if it was created).
        step (Union[int, None])                          : Current fine-tuning step.
        agent_loglikelihoods (Union[torch.Tensor, None]) : Vector containing the
                                                           agent log-likelihoods
                                                           for sampled molecules.
        prior_loglikelihoods (Union[torch.Tensor, None]) : Vector containing the
                                                           prior log-likelihoods
                                                           for sampled molecules.
    """
    # TODO this function may only be called when the tb_writer is not None, and thus this if statement may be redundant
    if tb_writer is not None:
        avg_agent_loglikelihood = torch.mean(agent_loglikelihoods)
        avg_prior_loglikelihood = torch.mean(prior_loglikelihoods)
        tb_writer.add_scalar("Train/agent_loglikelihood", avg_agent_loglikelihood, step)
        tb_writer.add_scalar("Train/prior_loglikelihood", avg_prior_loglikelihood, step)


def load_saved_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Loads a pre-saved neural net model.

    Args:
    ----
        model (torch.nn.Module) : Existing but bare-bones model variable.
        path (str)              : Path to the saved model.

    Returns:
    -------
        model (torch.nn.Module) : Loaded model.
    """
    try:
        # first try to load model as if it was created using GraphINVENT
        # v1.0 (will raise an exception if it was actually created with
        # GraphINVENT v2.0)
        model.state_dict = torch.load(path, weights_only=False).state_dict()
    except AttributeError:
        # try to load the model as if created using GraphINVENT v2.0 or
        # later
        model.load_state_dict(torch.load(path, weights_only=False))
    return model
