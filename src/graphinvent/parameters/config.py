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
    atom_types_set = set()
    formal_charge_set = set()
    imp_H_set = set()
    max_n_nodes = 0

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

    atom_types = sorted(atom_types_set)
    formal_charge = sorted(formal_charge_set)
    imp_H = sorted(imp_H_set)

    return atom_types, formal_charge, imp_H, max_n_nodes


def get_feature_dimensions(parameters: dict) -> Tuple[int, int, int, int]:
    """Returns dimensions for all node feature segments."""
    n_atom_types = len(parameters["atom_types"])
    n_formal_charge = len(parameters["formal_charge"])
    n_numh = int(not parameters["use_explicit_H"] and not parameters["ignore_H"]) * len(
        parameters["imp_H"]
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

    use_chirality = parameters["use_chirality"]
    use_explicit_H = parameters["use_explicit_H"]
    ignore_H = parameters["ignore_H"]

    if use_chirality:
        if use_explicit_H or ignore_H:
            dim_f_add = [
                max_nodes,
                n_atom_types,
                n_formal_charge,
                n_chirality,
                n_edge_features,
            ]
        else:
            dim_f_add = [
                max_nodes,
                n_atom_types,
                n_formal_charge,
                n_num_h,
                n_chirality,
                n_edge_features,
            ]
    else:
        if use_explicit_H or ignore_H:
            dim_f_add = [max_nodes, n_atom_types, n_formal_charge, n_edge_features]
        else:
            dim_f_add = [
                max_nodes,
                n_atom_types,
                n_formal_charge,
                n_num_h,
                n_edge_features,
            ]

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
    job_dir = Path(all_params["job_dir"])
    json_path = job_dir / "params.json"
    csv_path = job_dir / "input.csv"

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
    for _path_key in (
        "dataset_dir",
        "job_dir",
        "tensorboard_dir",
        "pretrained_model_dir",
    ):
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

    # For non-preprocess jobs, load molecular feature parameters from
    # preprocessing_params.json (authoritative source written by the preprocessing
    # step).  These never need to be re-specified in the job's params.json.
    _FEATURE_KEYS = (
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
    )
    # List/int keys that can be explicitly overridden in non-preprocess job JSONs.
    # An empty list [] or 0 means "inherit from preprocessing_params.json".
    _OVERRIDABLE_LIST_KEYS = ("atom_types", "formal_charge", "imp_H", "chirality")
    _OVERRIDABLE_INT_KEYS = ("max_n_nodes",)

    # For transfer/RL jobs using pretrained_model_path, load GGNN architecture
    # parameters from the pretrained model's params_all.json so they don't need
    # to be re-specified in the job config.
    _ARCH_KEYS = (
        "enn_depth",
        "enn_hidden_dim",
        "enn_dropout_p",
        "mlp1_depth",
        "mlp1_hidden_dim",
        "mlp1_dropout_p",
        "mlp2_depth",
        "mlp2_hidden_dim",
        "mlp2_dropout_p",
        "gather_att_depth",
        "gather_att_hidden_dim",
        "gather_att_dropout_p",
        "gather_emb_depth",
        "gather_emb_hidden_dim",
        "gather_emb_dropout_p",
        "gather_width",
        "hidden_node_features",
        "message_passes",
        "message_size",
    )
    if parameters.get("job_type") in (
        "transfer",
        "rl",
        "constrained_rl",
        "generate",
        "unconditional",
        "conditional",
        "goal_directed",
        "sample",
    ):
        _pth_path = parameters.get("pretrained_model_path", "")
        if _pth_path:
            _pretrain_params_path = Path(_pth_path).parent / "params_all.json"
            if _pretrain_params_path.exists():
                print(
                    f"* Loading model architecture from pretrained model params: "
                    f"{_pretrain_params_path}",
                    flush=True,
                )
                _pretrain_params = load_params(str(_pretrain_params_path))
                # Read raw job params to detect any explicit arch overrides.
                _job_params_path = Path(parameters["job_dir"]) / "params.json"
                _raw_job = (
                    load_params(str(_job_params_path))
                    if _job_params_path.exists()
                    else {}
                )
                for _k in _ARCH_KEYS:
                    if _k not in _raw_job and _k in _pretrain_params:
                        parameters[_k] = _pretrain_params[_k]
            else:
                print(
                    f"-- Warning: pretrained model params_all.json not found at "
                    f"{_pretrain_params_path}. Architecture params must be specified manually.",
                    flush=True,
                )

    if parameters.get("job_type") != "preprocess":
        # Read the raw job params.json before it was merged with defaults, so we
        # can distinguish "user explicitly set a non-empty value" from "default []".
        _job_params_path = Path(parameters["job_dir"]) / "params.json"
        _raw_job = (
            load_params(str(_job_params_path)) if _job_params_path.exists() else {}
        )
        job_feature_overrides: dict = {}
        for _k in _OVERRIDABLE_LIST_KEYS:
            _v = _raw_job.get(_k, [])
            if _v:  # non-empty list → explicit override
                job_feature_overrides[_k] = _v
        for _k in _OVERRIDABLE_INT_KEYS:
            _v = _raw_job.get(_k, 0)
            if _v:  # non-zero → explicit override
                job_feature_overrides[_k] = _v

        dataset_dir = parameters.get("dataset_dir", "")
        json_preproc = Path(dataset_dir) / "preprocessing_params.json"
        csv_preproc = Path(dataset_dir) / "preprocessing_params.csv"

        if json_preproc.exists():
            preproc_params = load_params(str(json_preproc))
        elif csv_preproc.exists():
            preproc_params = load_params(str(csv_preproc))
        else:
            preproc_params = {}
            print(
                f"-- Warning: no preprocessing_params file found in "
                f"'{dataset_dir}'. Feature parameters will use defaults.",
                flush=True,
            )

        if preproc_params:
            print(
                "* Loading molecular feature parameters from "
                "preprocessing_params.json.",
                flush=True,
            )
            for key in _FEATURE_KEYS:
                if key in preproc_params:
                    parameters[key] = preproc_params[key]

        # Re-apply any non-empty job-level overrides on top of preprocessing_params.
        if job_feature_overrides:
            print("* Applying job-level feature overrides:", flush=True)
            for _k, _v in job_feature_overrides.items():
                parameters[_k] = _v
                print(f"  {_k} : {_v}", flush=True)

    if parameters["use_explicit_H"] and parameters["ignore_H"]:
        raise ValueError(
            "Cannot use explicit Hs and ignore Hs simultaneously. "
            "Please fix the flags in your params.json."
        )

    # Resolve molecular feature vocabulary for preprocessing jobs.
    if parameters.get("job_type") == "preprocess":
        dataset_dir = parameters.get("dataset_dir", "")
        smiles_file = parameters.get("smiles_file") or None

        # In Mode A the split files don't exist yet; scan the original file.
        split_files = [
            dataset_dir + "train.smi",
            dataset_dir + "valid.smi",
            dataset_dir + "test.smi",
        ]
        _primary_paths = (
            [smiles_file]
            if smiles_file and os.path.exists(smiles_file)
            else split_files
        )

        if parameters.get("auto_detect_features", True):
            smi_paths = list(_primary_paths)

            # If extra_dataset is provided, also scan it so the vocabulary covers
            # both datasets (useful for transfer learning compatibility).
            # Accepts a path to a .smi file OR a directory with train/valid/test.smi.
            extra_dataset = parameters.get("extra_dataset") or None
            if extra_dataset:
                extra_path = Path(extra_dataset)
                if extra_path.is_file():
                    smi_paths.append(str(extra_path))
                    print(
                        f"* Also scanning extra dataset for features: {extra_dataset}",
                        flush=True,
                    )
                elif extra_path.is_dir():
                    for _name in ("train.smi", "valid.smi", "test.smi"):
                        _p = extra_path / _name
                        if _p.exists():
                            smi_paths.append(str(_p))
                    print(
                        f"* Also scanning extra dataset for features: {extra_dataset}",
                        flush=True,
                    )
                else:
                    print(
                        f"-- Warning: extra_dataset path not found: {extra_dataset}",
                        flush=True,
                    )

            print(
                "* Auto-detecting molecular features from SMILES files...", flush=True
            )
            atom_types, formal_charge, imp_H, max_n_nodes = scan_smiles_features(
                smi_paths=smi_paths,
                use_explicit_H=parameters.get("use_explicit_H", False),
                ignore_H=parameters.get("ignore_H", False),
            )
            parameters["atom_types"] = atom_types
            parameters["formal_charge"] = formal_charge
            parameters["max_n_nodes"] = max_n_nodes
            if not parameters.get("use_explicit_H", False) and not parameters.get(
                "ignore_H", False
            ):
                parameters["imp_H"] = imp_H
            if parameters.get("use_chirality", False):
                parameters["chirality"] = ["None", "R", "S"]
            print(f"  atom_types    : {atom_types}", flush=True)
            print(f"  formal_charge : {formal_charge}", flush=True)
            if not parameters.get("use_explicit_H", False) and not parameters.get(
                "ignore_H", False
            ):
                print(f"  imp_H         : {imp_H}", flush=True)
            print(f"  max_n_nodes   : {max_n_nodes}", flush=True)
        else:
            # Manual mode: values come from params.json.
            # Any field left as [] or 0 is auto-detected from the primary dataset.
            needs_auto = []
            for _k in ("atom_types", "formal_charge"):
                if not parameters.get(_k):
                    needs_auto.append(_k)
            if not parameters.get("use_explicit_H", False) and not parameters.get(
                "ignore_H", False
            ):
                if not parameters.get("imp_H"):
                    needs_auto.append("imp_H")
            if not parameters.get("max_n_nodes"):
                needs_auto.append("max_n_nodes")

            if needs_auto:
                print(
                    f"* auto_detect_features=false — auto-detecting {needs_auto} from SMILES...",
                    flush=True,
                )
                _auto_at, _auto_fc, _auto_imph, _auto_nn = scan_smiles_features(
                    smi_paths=_primary_paths,
                    use_explicit_H=parameters.get("use_explicit_H", False),
                    ignore_H=parameters.get("ignore_H", False),
                )
                if "atom_types" in needs_auto:
                    parameters["atom_types"] = _auto_at
                if "formal_charge" in needs_auto:
                    parameters["formal_charge"] = _auto_fc
                if "imp_H" in needs_auto:
                    parameters["imp_H"] = _auto_imph
                if "max_n_nodes" in needs_auto:
                    parameters["max_n_nodes"] = _auto_nn
            else:
                print(
                    "* auto_detect_features=false — using all feature parameters from params.json.",
                    flush=True,
                )

            if parameters.get("use_chirality", False) and not parameters.get(
                "chirality"
            ):
                parameters["chirality"] = ["None", "R", "S"]

            print(f"  atom_types    : {parameters.get('atom_types')}", flush=True)
            print(f"  formal_charge : {parameters.get('formal_charge')}", flush=True)
            if not parameters.get("use_explicit_H", False) and not parameters.get(
                "ignore_H", False
            ):
                print(f"  imp_H         : {parameters.get('imp_H')}", flush=True)
            print(f"  max_n_nodes   : {parameters.get('max_n_nodes')}", flush=True)

    # Bond type <-> integer mappings
    bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}
    if parameters["use_aromatic_bonds"]:
        bondtype_to_int[BondType.AROMATIC] = 3
    int_to_bondtype = {v: k for k, v in bondtype_to_int.items()}
    n_edge_features = len(bondtype_to_int)

    # Node feature dimensions
    n_atom_types, n_formal_charge, n_imp_H, n_chirality = get_feature_dimensions(
        parameters
    )
    n_node_features = n_atom_types + n_formal_charge + n_imp_H + n_chirality

    # Tensor dimensions
    dim_nodes, dim_edges, dim_f_add, dim_f_conn, dim_f_term = get_tensor_dimensions(
        n_atom_types,
        n_formal_charge,
        n_imp_H,
        n_chirality,
        n_node_features,
        n_edge_features,
        parameters,
    )

    len_f_add = int(np.prod(dim_f_add))
    len_f_add_per_node = int(np.prod(dim_f_add[1:]))
    len_f_conn = int(np.prod(dim_f_conn))
    len_f_conn_per_node = int(np.prod(dim_f_conn[1:]))

    constants_dict = {
        "big_negative": -1e6,
        "big_positive": 1e6,
        "bondtype_to_int": bondtype_to_int,
        "int_to_bondtype": int_to_bondtype,
        "n_edge_features": n_edge_features,
        "n_atom_types": n_atom_types,
        "n_formal_charge": n_formal_charge,
        "n_imp_H": n_imp_H,
        "n_chirality": n_chirality,
        "n_node_features": n_node_features,
        "dim_nodes": dim_nodes,
        "dim_edges": dim_edges,
        "dim_f_add": dim_f_add,
        "dim_f_conn": dim_f_conn,
        "dim_f_term": dim_f_term,
        "dim_action_probs": [int(np.prod(dim_f_add)) + int(np.prod(dim_f_conn)) + 1],
        "len_f_add": len_f_add,
        "len_f_add_per_node": len_f_add_per_node,
        "len_f_conn": len_f_conn,
        "len_f_conn_per_node": len_f_conn_per_node,
    }

    # Strip underscore-prefixed keys (used as inline comments in params.json).
    constants_dict.update(
        {k: v for k, v in parameters.items() if not k.startswith("_")}
    )

    constants_dict["test_set"] = parameters["dataset_dir"] + "test.smi"
    constants_dict["training_set"] = parameters["dataset_dir"] + "train.smi"
    constants_dict["validation_set"] = parameters["dataset_dir"] + "valid.smi"

    if constants_dict["job_type"] != "preprocess":
        print(
            "* Running job using HDF datasets located at " + parameters["dataset_dir"],
            flush=True,
        )

    if constants_dict["job_type"] in ("rl", "constrained_rl", "goal_directed"):
        active_components = set(constants_dict["score_components"])
        for qsar_model_name, qsar_model_path in list(
            constants_dict["qsar_models"].items()
        ):
            # Only load models that are actually referenced in score_components;
            # stale entries (e.g. from the default config) are silently skipped.
            if qsar_model_name not in active_components:
                continue
            print(
                f"-- Loading pre-trained scikit-learn activity model: "
                f"'{qsar_model_name}'.",
                flush=True,
            )
            try:
                with open(qsar_model_path, "rb") as f:
                    model_dict = pickle.load(f)
                constants_dict["qsar_models"][qsar_model_name] = model_dict[
                    "classifier_sv"
                ]
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"QSAR model file not found: '{qsar_model_path}' "
                    f"(referenced by score component '{qsar_model_name}'). "
                    "Ensure the file exists or remove it from 'qsar_models'."
                ) from None
            except EOFError:
                raise RuntimeError(
                    f"QSAR model file is empty or corrupt: '{qsar_model_path}'. "
                    "Replace it with a valid trained model."
                ) from None
            except KeyError:
                raise KeyError(
                    f"QSAR model pickle at '{qsar_model_path}' does not contain "
                    "the expected key 'classifier_sv'. Check how the model was saved."
                ) from None

    # Validate conditional generation settings
    if constants_dict["condition_dim"] > 0:
        if constants_dict["job_type"] == "unconditional":
            raise ValueError(
                "condition_dim > 0 requires job_type 'conditional', not 'unconditional'."
            )
        if (
            constants_dict.get("conditioning") is None
            and constants_dict["job_type"] == "preprocess"
        ):
            pass  # allowed to preprocess without conditioning

    # For sample job: validate sample_conditions when sampling from conditional model
    if (
        constants_dict["job_type"] == "sample"
        and constants_dict["sample_mode"] == "generate"
    ):
        # If condition_dim > 0 (loaded from preprocessing_params), sample_conditions must be set
        if (
            constants_dict["condition_dim"] > 0
            and constants_dict["sample_conditions"] is None
        ):
            raise ValueError(
                "Sampling from a conditional model requires 'sample_conditions' in the config "
                '(e.g. {"pLogS": -1.5}). condition_dim from preprocessing is '
                f"{constants_dict['condition_dim']}."
            )

    # Default condition_embedding_dim to hidden_node_features if not explicitly set
    if constants_dict.get("condition_embedding_dim", 0) == 0:
        constants_dict["condition_embedding_dim"] = constants_dict.get(
            "hidden_node_features", 100
        )

    Constants = namedtuple("CONSTANTS", sorted(constants_dict))
    return Constants(**constants_dict)


constants = collect_global_constants(
    parameters=defaults.parameters,
    job_dir=args.job_dir,
)
