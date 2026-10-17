"""
Resolve a job's configuration and derive the constants the rest of the code reads.

Three layers are merged key by key, later wins: ``config/defaults.yaml``, the
optional ``experiments/<exp>/experiment.yaml``, and the job's ``config.yaml``.
A key left unset is inherited. ``resolve`` does the merge and all validation;
``derive`` adds the feature vocabulary and tensor shapes that depend on it.

``constants`` is the resolved namedtuple every module imports. It is set once,
by ``main.run``, before the modules that read it are imported.
"""

import math
import os
from collections import namedtuple
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from rdkit.Chem import AddHs
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULTS_PATH = REPO_ROOT / "config" / "defaults.yaml"

JOB_TYPES = (
    "preprocess",
    "unconditional",
    "conditional",
    "goal_directed",
    "generate",
)

SPLIT_TYPES = ("scaffold", "random", "custom")

# Keys that describe the model's shape. When a checkpoint is loaded they come
# from the job that produced it; a conflicting value in the config is an error.
ARCH_KEYS = (
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
    "condition_dim",
    "condition_embedding_dim",
    "condition_type",
    # The property order defines what each slot of the condition vector means,
    # so a sampling job must inherit it rather than fall back to the key order
    # of whatever `sample_conditions` mapping was written.
    "conditioning",
)

# Written by preprocessing into the dataset directory; every later job on that
# dataset reads them from there rather than restating them.
FEATURE_KEYS = (
    "conditioning",
    "condition_dim",
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


class _Constants:
    """
    Stand-in for the resolved constants, which do not exist at import time.

    Modules do ``from parameters.config import constants``, which binds a name
    once. Looking the value up on each attribute access is what lets ``derive``
    run after those imports.
    """

    _resolved = None

    def __getattr__(self, name: str):
        if _Constants._resolved is None:
            raise RuntimeError(
                "The configuration has not been resolved yet. "
                "Call parameters.config.derive() before using constants."
            )
        return getattr(_Constants._resolved, name)


constants = _Constants()


# ---------------------------------------------------------------------------
# Layer merge
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping, not {type(loaded).__name__}.")
    return loaded


def _merge(base: dict, override: dict) -> dict:
    """Merge key by key; nested mappings merge recursively."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve(job_dir: Path) -> dict:
    """
    Merge the three configuration layers for *job_dir* and validate the result.

    Returns the merged parameters, with ``job_dir`` and the checkpoint's
    architecture keys filled in.
    """
    job_dir = Path(job_dir)

    if not job_dir.is_dir():
        raise ValueError(
            f"{job_dir} is not a directory.\n"
            "Expected a job directory, for example "
            "experiments/my_experiment/jobs/my_job"
        )

    config_path = job_dir / "config.yaml"
    if not config_path.exists():
        if (job_dir / "jobs").is_dir():
            available = sorted(
                p.name for p in (job_dir / "jobs").iterdir() if p.is_dir()
            )
            raise ValueError(
                f"{job_dir} is an experiment directory, not a job directory.\n"
                f"Run one of its jobs instead, for example:\n"
                f"    python submit.py {job_dir}/jobs/{available[0] if available else '<job_name>'}"
            )
        raise ValueError(
            f"{config_path} does not exist.\n"
            "A job directory must contain config.yaml. Copy one from a "
            "tutorial experiment under experiments/ and edit it."
        )

    job_layer = _read_yaml(config_path)

    experiment_path = job_dir.parent.parent / "experiment.yaml"
    experiment_layer = _read_yaml(experiment_path) if experiment_path.exists() else {}

    params = _merge(_merge(_read_yaml(DEFAULTS_PATH), experiment_layer), job_layer)
    params["job_dir"] = str(job_dir)

    # Keys the job itself set, as opposed to inherited: these are what a
    # checkpoint's architecture is allowed to conflict with.
    explicit = _merge(experiment_layer, job_layer)

    _validate(params, config_path)
    _apply_checkpoint_architecture(params, explicit)
    _resolve_device(params)

    return params


def _validate(params: dict, config_path: Path) -> None:
    job_type = params.get("job_type")
    if job_type is None:
        raise ValueError(
            f"{config_path} does not set job_type.\nAdd one of: {', '.join(JOB_TYPES)}"
        )
    if job_type not in JOB_TYPES:
        raise ValueError(
            f"{config_path} sets job_type: {job_type}, which is not a job type.\n"
            f"Valid values: {', '.join(JOB_TYPES)}"
        )

    if not params.get("dataset_dir"):
        raise ValueError(f"{config_path} does not set dataset_dir.")
    # A relative dataset_dir hangs off data_root, so moving the data to a
    # project filesystem is one setting rather than an edit to every job.
    dataset_dir = Path(params["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = Path(params["data_root"]) / dataset_dir
    params["dataset_dir"] = str(dataset_dir)
    if job_type != "preprocess" and not dataset_dir.is_dir():
        raise ValueError(
            f"dataset_dir {dataset_dir} does not exist.\n"
            f"data_root is {params['data_root']}; a relative dataset_dir is "
            "resolved against it."
        )

    if params["use_explicit_H"] and params["ignore_H"]:
        raise ValueError(
            "use_explicit_H and ignore_H cannot both be true: a hydrogen is "
            "either represented explicitly or not represented at all."
        )

    if job_type == "preprocess":
        if params["split_type"] not in SPLIT_TYPES:
            raise ValueError(
                f"split_type: {params['split_type']} is not a split strategy.\n"
                f"Valid values: {', '.join(SPLIT_TYPES)}"
            )
        if params["train_frac"] + params["valid_frac"] >= 1.0:
            raise ValueError(
                f"train_frac ({params['train_frac']}) + valid_frac "
                f"({params['valid_frac']}) leaves nothing for the test split."
            )
        if params["smiles_file"] and not Path(params["smiles_file"]).exists():
            raise ValueError(f"smiles_file {params['smiles_file']} does not exist.")

    if job_type in ("unconditional", "conditional") and params["resume_from"]:
        if not Path(params["resume_from"]).exists():
            raise ValueError(
                f"resume_from {params['resume_from']} does not exist. Remove "
                "the key to train from scratch."
            )

    if job_type in ("goal_directed", "generate"):
        checkpoint = params.get("pretrained_model_path")
        if not checkpoint:
            raise ValueError(f"job_type: {job_type} needs pretrained_model_path.")
        if not Path(checkpoint).exists():
            raise ValueError(f"pretrained_model_path {checkpoint} does not exist.")

    if job_type == "goal_directed":
        if len(params["score_components"]) != len(params["score_thresholds"]):
            raise ValueError(
                f"score_components has {len(params['score_components'])} entries "
                f"but score_thresholds has {len(params['score_thresholds'])}."
            )


def checkpoint_config_path(checkpoint: str) -> Path:
    """Path to the resolved config of the job that wrote *checkpoint*."""
    return Path(checkpoint).resolve().parent.parent / "results" / "config.resolved.yaml"


def _apply_checkpoint_architecture(params: dict, explicit: dict) -> None:
    """
    Take the architecture keys from the checkpoint being loaded.

    Loading a model into the wrong architecture either crashes inside
    ``load_state_dict`` or, where the shapes happen to line up, produces a
    silently wrong model, so a conflict is reported here instead.
    """
    checkpoint = params.get("pretrained_model_path") or params.get("resume_from")
    if not checkpoint:
        return

    source = checkpoint_config_path(checkpoint)
    if not source.exists():
        raise ValueError(
            f"Cannot read the architecture of {checkpoint}: {source} is missing.\n"
            "A checkpoint is loaded together with the resolved config of the "
            "job that wrote it."
        )

    produced_by = _read_yaml(source)
    for key in ARCH_KEYS:
        if key not in produced_by:
            continue
        if key in explicit and explicit[key] != produced_by[key]:
            raise ValueError(
                f"config.yaml sets {key}: {explicit[key]}, but the checkpoint "
                f"{checkpoint} was trained with {key}: {produced_by[key]}.\n"
                "Architecture keys come from the checkpoint. Remove the key "
                "from config.yaml."
            )
        params[key] = produced_by[key]


def _resolve_device(params: dict) -> None:
    """
    Resolve `device` against what this machine actually has.

    One pattern for all three supported devices: an accelerator that is asked
    for but absent falls back, so the same job directory runs on a laptop and
    on a GPU node unchanged. A device name that is not one of the three is a
    typo, and that fails rather than silently running somewhere unintended.
    """
    requested = params["device"]
    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError(
            f"device: {requested} is not a device. Use one of: cpu, cuda, mps."
        )

    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if available[requested]:
        return

    fallback = "mps" if available["mps"] else "cpu"
    params["device"] = fallback
    print(f"* {requested} not available; using device '{fallback}'.", flush=True)


# ---------------------------------------------------------------------------
# Feature vocabulary
# ---------------------------------------------------------------------------


def scan_smiles_features(
    smi_paths: list,
    use_explicit_H: bool,
    ignore_H: bool,
) -> tuple:
    """
    Scan SMILES files for the feature vocabulary preprocessing needs.

    Returns the sorted element symbols, formal charges and implicit hydrogen
    counts present, and the 99th-percentile heavy-atom count. ``imp_H`` is
    empty when hydrogens are explicit or ignored. Missing paths are skipped.

    The 99th percentile rather than the maximum, because every stored tensor is
    padded to this width: the `edges` dataset grows with its square, so one
    outlier molecule used to set the size of every row in the dataset.
    Molecules above the cap are dropped at splitting time and counted in
    `preprocessing.yaml`.
    """
    atom_types_set = set()
    formal_charge_set = set()
    imp_H_set = set()
    n_nodes: list = []

    for path in smi_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            first_line = f.readline()
        has_header = "SMILES" in first_line
        # Property columns are tab-separated, so RDKit must be told the
        # delimiter; with the default it hands the whole "SMILES\tvalue"
        # line to the parser and every molecule comes back None.
        supplier = SmilesMolSupplier(
            path,
            sanitize=True,
            nameColumn=-1,
            titleLine=has_header,
            delimiter="\t" if "\t" in first_line else " \t",
        )
        for mol in supplier:
            if mol is None:
                continue
            if use_explicit_H and not ignore_H:
                mol = AddHs(mol)
            n_nodes.append(mol.GetNumAtoms())
            for atom in mol.GetAtoms():
                atom_types_set.add(atom.GetSymbol())
                formal_charge_set.add(atom.GetFormalCharge())
                if not use_explicit_H and not ignore_H:
                    imp_H_set.add(atom.GetTotalNumHs())

    # Rounded up, so the cap includes the percentile molecule itself.
    percentile_99 = int(math.ceil(float(np.percentile(n_nodes, 99)))) if n_nodes else 0
    return (
        sorted(atom_types_set),
        sorted(formal_charge_set),
        sorted(imp_H_set),
        percentile_99,
    )


def get_feature_dimensions(parameters: dict) -> Tuple[int, int, int, int]:
    """Return the size of each node feature segment."""
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
    Return the shapes of the molecular graph tensors.

    Each return value is a list of dimension sizes, except ``dim_f_term``.
    """
    max_nodes = parameters["max_n_nodes"]

    dim_f_add = [max_nodes, n_atom_types, n_formal_charge]
    if not parameters["use_explicit_H"] and not parameters["ignore_H"]:
        dim_f_add.append(n_num_h)
    if parameters["use_chirality"]:
        dim_f_add.append(n_chirality)
    dim_f_add.append(n_edge_features)

    return (
        [max_nodes, n_node_features],
        [max_nodes, max_nodes, n_edge_features],
        dim_f_add,
        [max_nodes, n_edge_features],
        1,
    )


# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------


def _load_dataset_features(params: dict) -> None:
    """Read the feature vocabulary written by the dataset's preprocessing job."""
    source = Path(params["dataset_dir"]) / "preprocessing.yaml"
    if not source.exists():
        raise ValueError(
            f"{source} is missing. Run a preprocess job on "
            f"{params['dataset_dir']} before training or sampling from it."
        )
    recorded = _read_yaml(source)
    for key in FEATURE_KEYS:
        if key in recorded:
            params[key] = recorded[key]


def _detect_features(params: dict) -> None:
    """Scan the SMILES a preprocessing job is about to decompose."""
    dataset_dir = Path(params["dataset_dir"])
    smiles_file = params["smiles_file"]

    # In the splitting case the split files do not exist yet.
    if smiles_file and Path(smiles_file).exists():
        smi_paths = [str(smiles_file)]
    else:
        smi_paths = [
            str(dataset_dir / n) for n in ("train.smi", "valid.smi", "test.smi")
        ]

    # Scanning a second dataset as well gives a transfer-learning pair one
    # shared vocabulary, so the checkpoint fits both.
    extra = params["extra_dataset"]
    if extra:
        extra_path = Path(extra)
        if extra_path.is_file():
            smi_paths.append(str(extra_path))
        elif extra_path.is_dir():
            smi_paths += [
                str(extra_path / n)
                for n in ("train.smi", "valid.smi", "test.smi")
                if (extra_path / n).exists()
            ]
        else:
            raise ValueError(f"extra_dataset {extra} does not exist.")

    detected = dict(
        zip(
            ("atom_types", "formal_charge", "imp_H", "max_n_nodes"),
            scan_smiles_features(
                smi_paths, params["use_explicit_H"], params["ignore_H"]
            ),
        )
    )
    if params["use_explicit_H"] or params["ignore_H"]:
        del detected["imp_H"]

    if params["auto_detect_features"]:
        # max_n_nodes is the exception to auto-detection winning: it is a
        # recorded decision about the dataset, not an observation of it, and it
        # sets the width of every stored tensor. A config that states it keeps
        # it, so a run is reproducible when the detected percentile shifts.
        stated_max_n_nodes = params.get("max_n_nodes") or 0
        params.update(detected)
        if stated_max_n_nodes:
            params["max_n_nodes"] = stated_max_n_nodes
    else:
        # Any field left empty is still detected; the rest come from the config.
        for key, value in detected.items():
            if not params.get(key):
                params[key] = value

    if params["use_chirality"] and not params["chirality"]:
        params["chirality"] = ["None", "R", "S"]

    for key in ("atom_types", "formal_charge", "imp_H", "max_n_nodes"):
        if key in detected:
            print(f"  {key:<14}: {params[key]}", flush=True)


def derive(params: dict) -> namedtuple:
    """Add the feature vocabulary and tensor shapes, and freeze the result."""
    params = dict(params)

    if params["job_type"] == "preprocess":
        _detect_features(params)
    else:
        # condition_dim is a property of the dataset, so whether a job can be
        # conditional is only knowable once the dataset has been read.
        _load_dataset_features(params)
        if params["job_type"] == "conditional" and params["condition_dim"] <= 0:
            raise ValueError(
                f"job_type: conditional, but {params['dataset_dir']} was "
                "preprocessed with condition_dim: 0, so its HDF5 carries no "
                "condition_vector. Preprocess a tab-separated file with "
                "property columns instead."
            )
        if params["job_type"] == "unconditional" and params["condition_dim"] > 0:
            raise ValueError(
                f"job_type: unconditional, but {params['dataset_dir']} was "
                f"preprocessed with condition_dim: {params['condition_dim']}. "
                "Use job_type: conditional, or an unconditional dataset."
            )

    bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}
    if params["use_aromatic_bonds"]:
        bondtype_to_int[BondType.AROMATIC] = 3
    n_edge_features = len(bondtype_to_int)

    n_atom_types, n_formal_charge, n_imp_H, n_chirality = get_feature_dimensions(params)
    n_node_features = n_atom_types + n_formal_charge + n_imp_H + n_chirality
    dim_nodes, dim_edges, dim_f_add, dim_f_conn, dim_f_term = get_tensor_dimensions(
        n_atom_types,
        n_formal_charge,
        n_imp_H,
        n_chirality,
        n_node_features,
        n_edge_features,
        params,
    )

    if params["condition_embedding_dim"] == 0:
        params["condition_embedding_dim"] = params["hidden_node_features"]

    job_dir = Path(params["job_dir"])
    derived = {
        "big_negative": -1e6,
        "big_positive": 1e6,
        "bondtype_to_int": bondtype_to_int,
        "int_to_bondtype": {v: k for k, v in bondtype_to_int.items()},
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
        "len_f_add": int(np.prod(dim_f_add)),
        "len_f_add_per_node": int(np.prod(dim_f_add[1:])),
        "len_f_conn": int(np.prod(dim_f_conn)),
        "len_f_conn_per_node": int(np.prod(dim_f_conn[1:])),
        "results_dir": str(job_dir / "results") + "/",
        "checkpoints_dir": str(job_dir / "checkpoints") + "/",
        "logs_dir": str(job_dir / "logs") + "/",
        "tensorboard_dir": str(job_dir / "logs" / "tensorboard") + "/",
        "training_set": str(Path(params["dataset_dir"]) / "train.smi"),
        "validation_set": str(Path(params["dataset_dir"]) / "valid.smi"),
        "test_set": str(Path(params["dataset_dir"]) / "test.smi"),
    }
    derived.update(params)
    derived["job_dir"] = str(job_dir) + "/"
    derived["dataset_dir"] = str(Path(params["dataset_dir"])) + "/"

    Constants = namedtuple("CONSTANTS", sorted(derived))
    _Constants._resolved = Constants(**derived)
    return _Constants._resolved
