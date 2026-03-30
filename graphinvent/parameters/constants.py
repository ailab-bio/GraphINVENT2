"""
Loads input parameters from `defaults.py` and defines global constants that
depend on the input features, creating a `namedtuple` from them.

If a `params.json` file exists in the job directory it overrides the defaults
(falling back to the legacy `input.csv` format for backwards compatibility).
"""
import ast
import csv
import json
import os
import pickle
import sys
from collections import namedtuple
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import rdkit
from rdkit.Chem import AddHs
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

sys.path.insert(1, "./parameters/")
import parameters.args as args
import parameters.defaults as defaults


def scan_smiles_features(
    smi_paths: list,
    use_explicit_H: bool,
    ignore_H: bool,
) -> tuple:
    """
    Scans one or more SMILES files to auto-detect the molecular feature
    vocabulary needed for preprocessing.

    Args:
        smi_paths     : List of paths to .smi files (train / valid / test).
                        Missing paths are silently skipped.
        use_explicit_H: Whether explicit Hs are added before scanning atoms.
        ignore_H      : Whether H atoms are excluded from the feature vector.

    Returns:
        atom_types    : Sorted list of element symbols present in the data.
        formal_charge : Sorted list of unique formal charges present.
        imp_H         : Sorted list of unique implicit H counts present
                        (empty list when use_explicit_H or ignore_H is True).
        max_n_nodes   : Maximum heavy-atom count across all scanned molecules.
    """
    atom_types_set    = set()
    formal_charge_set = set()
    imp_H_set         = set()
    max_n_nodes       = 0

    for path in smi_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            first_line = f.readline()
        has_header = "SMILES" in first_line
        supplier = SmilesMolSupplier(
            path, sanitize=True, nameColumn=-1, titleLine=has_header
        )
        for mol in supplier:
            if mol is None:
                continue
            if use_explicit_H and not ignore_H:
                mol = AddHs(mol)
            n = mol.GetNumAtoms()
            if n > max_n_nodes:
                max_n_nodes = n
            for atom in mol.GetAtoms():
                atom_types_set.add(atom.GetSymbol())
                formal_charge_set.add(atom.GetFormalCharge())
                if not use_explicit_H and not ignore_H:
                    imp_H_set.add(atom.GetTotalNumHs())

    atom_types    = sorted(atom_types_set)
    formal_charge = sorted(formal_charge_set)
    imp_H         = sorted(imp_H_set)

    return atom_types, formal_charge, imp_H, max_n_nodes


def get_feature_dimensions(parameters: dict) -> Tuple[int, int, int, int]:
    """Returns dimensions for all node feature segments."""
    n_atom_types    = len(parameters["atom_types"])
    n_formal_charge = len(parameters["formal_charge"])
    n_numh          = (
        int(not parameters["use_explicit_H"] and not parameters["ignore_H"])
        * len(parameters["imp_H"])
    )
    n_chirality = int(parameters["use_chirality"]) * len(parameters["chirality"])
    return n_atom_types, n_formal_charge, n_numh, n_chirality


def get_tensor_dimensions(
    n_atom_types: int,
    n_formal_charge: int,
    n_num_h: int,
    n_chirality: int,
    n_node_features: int,
    n_edge_features: int,
    parameters: dict,
) -> Tuple[list, list, list, list, int]:
    """
    Returns tensor shapes for molecular graph representations.
    Each return value is a list of dimension sizes, except `dim_f_term` (int).
    """
    max_nodes = parameters["max_n_nodes"]

    dim_nodes = [max_nodes, n_node_features]
    dim_edges = [max_nodes, max_nodes, n_edge_features]

    use_chirality  = parameters["use_chirality"]
    use_explicit_H = parameters["use_explicit_H"]
    ignore_H       = parameters["ignore_H"]

    if use_chirality:
        if use_explicit_H or ignore_H:
            dim_f_add = [max_nodes, n_atom_types, n_formal_charge, n_chirality, n_edge_features]
        else:
            dim_f_add = [max_nodes, n_atom_types, n_formal_charge, n_num_h, n_chirality, n_edge_features]
    else:
        if use_explicit_H or ignore_H:
            dim_f_add = [max_nodes, n_atom_types, n_formal_charge, n_edge_features]
        else:
            dim_f_add = [max_nodes, n_atom_types, n_formal_charge, n_num_h, n_edge_features]

    dim_f_conn = [max_nodes, n_edge_features]
    dim_f_term = 1

    return dim_nodes, dim_edges, dim_f_add, dim_f_conn, dim_f_term


def load_params(params_path: str) -> dict:
    """
    Loads job parameters from a JSON file or legacy CSV file.

    JSON is the preferred format. CSV (semicolon-delimited, one key-value pair
    per row) is supported for backwards compatibility; values are parsed with
    `ast.literal_eval` instead of the unsafe `eval`.
    """
    path = Path(params_path)
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)

    # Legacy CSV path
    params: dict = {}
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        for key, value in reader:
            try:
                params[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                params[key] = value
    return params


def override_params(all_params: dict) -> dict:
    """
    Overrides defaults with values from the job directory.

    Looks for `params.json` first; falls back to legacy `input.csv`.
    """
    job_dir  = Path(all_params["job_dir"])
    json_path = job_dir / "params.json"
    csv_path  = job_dir / "input.csv"

    if json_path.exists():
        overrides = load_params(str(json_path))
    elif csv_path.exists():
        overrides = load_params(str(csv_path))
    else:
        return all_params

    all_params.update(overrides)
    return all_params


def collect_global_constants(parameters: dict, job_dir: str) -> namedtuple:
    """
    Merges defaults with any job-directory overrides and derived constants,
    returning an immutable `namedtuple`.

    Args:
        parameters: Default parameter dictionary from `defaults.py`.
        job_dir:    Path to the current job directory (from the CLI).

    Returns:
        constants: Named tuple of all configuration and derived constants.
    """
    parameters["job_dir"] = job_dir
    parameters = override_params(all_params=parameters)

    # Normalize directory paths so they always end with exactly one '/'
    # regardless of whether the user included a trailing slash.
    for _path_key in ("dataset_dir", "job_dir", "tensorboard_dir", "pretrained_model_dir"):
        if parameters.get(_path_key):
            parameters[_path_key] = str(Path(parameters[_path_key])) + "/"

    # Auto-detect device if the user left the default "cuda" but CUDA is unavailable.
    # Priority: CUDA > MPS (Apple Silicon) > CPU.
    if parameters.get("device") == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            parameters["device"] = "mps"
        else:
            parameters["device"] = "cpu"
        print(
            f"* CUDA not available — using device '{parameters['device']}' instead.",
            flush=True,
        )

    if parameters["use_explicit_H"] and parameters["ignore_H"]:
        raise ValueError(
            "Cannot use explicit Hs and ignore Hs simultaneously. "
            "Please fix the flags in your params.json."
        )

    # Auto-detect molecular feature vocabulary from SMILES files when preprocessing
    if parameters.get("job_type") == "preprocess" and parameters.get("atom_types") is None:
        dataset_dir = parameters.get("dataset_dir", "")
        smiles_file = parameters.get("smiles_file") or None

        # In Mode A (single SMILES file + automatic splitting), the split files
        # don't exist yet — splitting happens later in Workflow.preprocess_phase().
        # Scan the original file directly instead.
        split_files = [
            dataset_dir + "train.smi",
            dataset_dir + "valid.smi",
            dataset_dir + "test.smi",
        ]
        if smiles_file and os.path.exists(smiles_file):
            smi_paths = [smiles_file]
        else:
            smi_paths = split_files

        print("* Auto-detecting molecular features from SMILES files...", flush=True)
        atom_types, formal_charge, imp_H, max_n_nodes = scan_smiles_features(
            smi_paths=smi_paths,
            use_explicit_H=parameters.get("use_explicit_H", False),
            ignore_H=parameters.get("ignore_H", False),
        )
        parameters["atom_types"]    = atom_types
        parameters["formal_charge"] = formal_charge
        parameters["max_n_nodes"]   = max_n_nodes
        if not parameters.get("use_explicit_H", False) and not parameters.get("ignore_H", False):
            parameters["imp_H"] = imp_H
        if parameters.get("use_chirality", False):
            parameters["chirality"] = ["None", "R", "S"]
        print(f"  atom_types    : {atom_types}", flush=True)
        print(f"  formal_charge : {formal_charge}", flush=True)
        if not parameters.get("use_explicit_H", False) and not parameters.get("ignore_H", False):
            print(f"  imp_H         : {imp_H}", flush=True)
        print(f"  max_n_nodes   : {max_n_nodes}", flush=True)

    # Bond type <-> integer mappings
    bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}
    if parameters["use_aromatic_bonds"]:
        bondtype_to_int[BondType.AROMATIC] = 3
    int_to_bondtype = {v: k for k, v in bondtype_to_int.items()}
    n_edge_features = len(bondtype_to_int)

    # Node feature dimensions
    n_atom_types, n_formal_charge, n_imp_H, n_chirality = get_feature_dimensions(parameters)
    n_node_features = n_atom_types + n_formal_charge + n_imp_H + n_chirality

    # Tensor dimensions
    (dim_nodes, dim_edges, dim_f_add, dim_f_conn, dim_f_term) = get_tensor_dimensions(
        n_atom_types, n_formal_charge, n_imp_H, n_chirality,
        n_node_features, n_edge_features, parameters,
    )

    len_f_add           = int(np.prod(dim_f_add))
    len_f_add_per_node  = int(np.prod(dim_f_add[1:]))
    len_f_conn          = int(np.prod(dim_f_conn))
    len_f_conn_per_node = int(np.prod(dim_f_conn[1:]))

    constants_dict = {
        "big_negative"       : -1e6,
        "big_positive"       : 1e6,
        "bondtype_to_int"    : bondtype_to_int,
        "int_to_bondtype"    : int_to_bondtype,
        "n_edge_features"    : n_edge_features,
        "n_atom_types"       : n_atom_types,
        "n_formal_charge"    : n_formal_charge,
        "n_imp_H"            : n_imp_H,
        "n_chirality"        : n_chirality,
        "n_node_features"    : n_node_features,
        "dim_nodes"          : dim_nodes,
        "dim_edges"          : dim_edges,
        "dim_f_add"          : dim_f_add,
        "dim_f_conn"         : dim_f_conn,
        "dim_f_term"         : dim_f_term,
        "dim_apd"            : [int(np.prod(dim_f_add)) + int(np.prod(dim_f_conn)) + 1],
        "len_f_add"          : len_f_add,
        "len_f_add_per_node" : len_f_add_per_node,
        "len_f_conn"         : len_f_conn,
        "len_f_conn_per_node": len_f_conn_per_node,
    }

    constants_dict.update(parameters)

    constants_dict["test_set"]       = parameters["dataset_dir"] + "test.smi"
    constants_dict["training_set"]   = parameters["dataset_dir"] + "train.smi"
    constants_dict["validation_set"] = parameters["dataset_dir"] + "valid.smi"

    if constants_dict["job_type"] != "preprocess":
        print(
            "* Running job using HDF datasets located at " + parameters["dataset_dir"],
            flush=True,
        )
        print(
            "* Checking that the relevant parameters match those used in preprocessing.",
            flush=True,
        )

        dataset_dir   = parameters["dataset_dir"]
        json_preproc  = Path(dataset_dir) / "preprocessing_params.json"
        csv_preproc   = Path(dataset_dir) / "preprocessing_params.csv"

        if json_preproc.exists():
            params_to_check = load_params(str(json_preproc))
        elif csv_preproc.exists():
            params_to_check = load_params(str(csv_preproc))
        else:
            print(
                "-- No preprocessing_params file found; skipping parameter check.",
                flush=True,
            )
            params_to_check = {}

        for key, value in params_to_check.items():
            if key in constants_dict and value != constants_dict[key]:
                raise ValueError(
                    f"Parameter mismatch between current job and preprocessing: "
                    f"'{key}' differs. Ensure all relevant parameters match."
                )

        print("-- Job parameters match preprocessing parameters.", flush=True)

    if constants_dict["job_type"] == "rl":
        print("-- Loading pre-trained scikit-learn activity model.", flush=True)
        for qsar_model_name, qsar_model_path in constants_dict["qsar_models"].items():
            with open(qsar_model_path, "rb") as f:
                model_dict = pickle.load(f)
                constants_dict["qsar_models"][qsar_model_name] = model_dict["classifier_sv"]

    Constants = namedtuple("CONSTANTS", sorted(constants_dict))
    return Constants(**constants_dict)


constants = collect_global_constants(
    parameters=defaults.parameters,
    job_dir=args.job_dir,
)
