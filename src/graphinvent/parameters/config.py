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

SPLIT_TYPES = ("scaffold", "warhead_scaffold", "random", "custom")
SAMPLE_MODES = ("generate", "evaluate")
OPTIMIZERS = ("graphinvent", "reinvent4")

# The only injection mechanism the GGNN implements. The key existed with no
# dispatch and no validation, so any value behaved as "virtual_node" and a
# config naming something else was accepted in silence.
CONDITION_TYPES = ("virtual_node",)

# Where a condition vector comes from. Only the property columns of the SMILES
# file are implemented.
CONDITION_SOURCES = ("smiles_file",)

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
    if (
        "condition_dropout" in explicit
        and float(explicit["condition_dropout"] or 0.0) > 0.0
        and params.get("job_type") != "conditional"
    ):
        raise ValueError(
            f"condition_dropout: {explicit['condition_dropout']} is set on a "
            f"{params.get('job_type')} job, which has no conditions to drop.\n"
            "It applies during conditional training only. Remove the key; "
            "inheriting the default is harmless because nothing reads it here."
        )
    _apply_checkpoint_architecture(params, explicit)
    _resolve_device(params)

    return params


def _job_keys(config_path: Path) -> set:
    """The keys this job's own config.yaml sets, ignoring what it inherits."""
    own = _read_yaml(Path(config_path))
    return set(own) if isinstance(own, dict) else set()


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

    # The HDF5 dataset holds an open file handle, which cannot be pickled for a
    # spawned worker and is not fork-safe for a forked one, where it corrupts
    # the loaded graphs rather than failing. A default of 0 is not enforcement.
    if params.get("n_workers", 0) != 0:
        raise ValueError(
            f"n_workers: {params.get('n_workers')} is not supported; it must be "
            "0.\nThe HDF5 dataset holds an open file handle that is neither "
            "picklable nor fork-safe, and a worker process corrupts the loaded "
            "graphs silently rather than failing.\nControl loading cost with "
            "block_size instead."
        )

    # Checked for every job type, not just generate: the default is inherited
    # everywhere, so a typo in an experiment.yaml would otherwise sit unnoticed
    # until a generate job in that experiment ran.
    if params.get("sample_mode") not in SAMPLE_MODES:
        raise ValueError(
            f"{config_path} sets sample_mode: {params.get('sample_mode')!r}, "
            "which is not a sampling mode.\n"
            f"Valid values: {', '.join(SAMPLE_MODES)}.\n"
            "generate writes molecules and computes the metric tables; "
            "evaluate computes NLL and UC-JSD."
        )

    optimizer = params.get("optimizer", "graphinvent")
    if optimizer not in OPTIMIZERS:
        raise ValueError(
            f"optimizer: {optimizer!r} is not an optimizer.\n"
            f"Valid values: {', '.join(OPTIMIZERS)}"
        )
    if optimizer != "graphinvent":
        raise NotImplementedError(
            f"This job declares optimizer: {optimizer}, which this repository "
            "does not run.\n"
            "It is a baseline arm for an external program, and submitting it "
            "here would optimize a GraphINVENT checkpoint and write a "
            "metrics.json that reads as that baseline's result.\n"
            "What it needs before it can run is in the experiment's "
            "instructions.md: the pinned REINVENT4 install, its prior from "
            "Zenodo specialized onto this task corpus, and a wrapper that "
            "exposes our scoring function as its ExternalProcess component."
        )

    if params.get("condition_type") not in CONDITION_TYPES:
        raise ValueError(
            f"condition_type: {params.get('condition_type')} is not an "
            "injection mechanism.\n"
            f"Valid values: {', '.join(CONDITION_TYPES)}"
        )

    conditioning = params.get("conditioning") or {}
    if conditioning:
        if not isinstance(conditioning, dict):
            raise ValueError(
                f"conditioning must be a block with 'properties' and 'source', "
                f"got {type(conditioning).__name__}."
            )
        source = conditioning.get("source")
        if source is not None and source not in CONDITION_SOURCES:
            raise ValueError(
                f"conditioning.source: {source} is not a condition source.\n"
                f"Valid values: {', '.join(CONDITION_SOURCES)}"
            )
        properties = conditioning.get("properties") or []
        if properties and params.get("condition_dim", 0) != len(properties):
            raise ValueError(
                f"conditioning.properties names {len(properties)} properties "
                f"{list(properties)} but condition_dim is "
                f"{params.get('condition_dim', 0)}. The condition vector has "
                "one slot per property."
            )

    dropout = params.get("condition_dropout")
    if dropout is not None and (
        not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0
    ):
        raise ValueError(
            f"condition_dropout: {dropout!r} is not a fraction in [0, 1).\n"
            "It is the fraction of training examples whose condition is "
            "replaced with zeros; 1.0 would train an unconditional model with "
            "a condition encoder attached."
        )

    # A categorical panel declares its classes, because nothing else can tell
    # a one-hot class column from a property that happens to be 0 or 1, and the
    # two have disjoint metric sets. Wrong here means the job reports
    # condition_success_rate on a one-hot, which measures whether the model
    # echoed its own input.
    classes = params.get("condition_classes")
    if classes is not None:
        if not isinstance(classes, list) or not all(
            isinstance(name, str) for name in classes
        ):
            raise ValueError(
                f"condition_classes must be a list of column names, got "
                f"{classes!r}.\nSet it to the one-hot panel, e.g. "
                '["is_ar", "is_btk", "is_egfr"], or leave it null for a '
                "continuous condition."
            )
        if len(set(classes)) != len(classes):
            raise ValueError(f"condition_classes repeats a class: {classes}.")
        properties = list((params.get("conditioning") or {}).get("properties") or [])
        if properties and list(classes) != properties:
            raise ValueError(
                f"condition_classes {list(classes)} is not "
                f"conditioning.properties {properties}.\nThe panel is the "
                "condition vector, in the same order: a different order means "
                "every per-class metric is attributed to the wrong class while "
                "still producing plausible numbers."
            )
        if params.get("condition_dim", 0) != len(classes):
            raise ValueError(
                f"condition_classes names {len(classes)} classes but "
                f"condition_dim is {params.get('condition_dim', 0)}. A one-hot "
                "panel has one slot per class."
            )

    if job_type == "preprocess":
        if params["split_type"] not in SPLIT_TYPES:
            raise ValueError(
                f"split_type: {params['split_type']} is not a split strategy.\n"
                f"Valid values: {', '.join(SPLIT_TYPES)}"
            )
        if params["split_type"] == "custom":
            raise NotImplementedError(
                "split_type: custom is a hook with no implementation, and "
                "selecting it raises by specification.\n"
                "Add your splitting logic to split_smiles_file() in "
                "src/graphinvent/DataProcessor.py, or choose one of: "
                + ", ".join(t for t in SPLIT_TYPES if t != "custom")
                + "."
            )
        if params["train_frac"] + params["valid_frac"] >= 1.0:
            raise ValueError(
                f"train_frac ({params['train_frac']}) + valid_frac "
                f"({params['valid_frac']}) leaves nothing for the test split."
            )
        if params["smiles_file"] and not Path(params["smiles_file"]).exists():
            raise ValueError(f"smiles_file {params['smiles_file']} does not exist.")

    if job_type in ("unconditional", "conditional"):
        if params["resume_from"] and not Path(params["resume_from"]).exists():
            raise ValueError(
                f"resume_from {params['resume_from']} does not exist. Remove "
                "the key to train from scratch."
            )
        # A training job loads weights from `resume_from` and never from
        # `pretrained_model_path`, which belongs to generate and goal_directed.
        # Setting the wrong one used to pass validation and be ignored, so a
        # transfer-learning run silently trained from scratch: measured on the
        # development set, epoch-1 validation loss was 2.870 either way, against
        # 2.217 for a real resume.
        if params.get("pretrained_model_path"):
            raise ValueError(
                f"job_type: {job_type} does not read pretrained_model_path, so "
                "the weights would be ignored and this would train from "
                "scratch.\n"
                "Transfer learning is an unconditional job with resume_from "
                "set. Rename the key:\n"
                f"    resume_from: {params['pretrained_model_path']}"
            )

    # A learning step is not an epoch. The RL path has its own keys, so a
    # goal_directed job that sets the supervised ones is naming the wrong
    # thing: silently honoring `epochs` there is how the two schedules came to
    # share a value in the first place.
    if job_type == "goal_directed":
        for supervised, rl_key in (
            ("epochs", "rl_steps"),
            ("sample_every", "rl_eval_every"),
        ):
            if supervised in _job_keys(config_path):
                raise ValueError(
                    f"job_type: goal_directed does not read '{supervised}'.\n"
                    f"A learning step is not an epoch; use '{rl_key}' instead.\n"
                    f"    {rl_key}: {params.get(supervised)}"
                )

    # A baseline that was declared and does not resolve is worse than none:
    # the metrics then report every baseline_ key as null with the reason "no
    # unconditional baseline sample is declared", which is false, and the
    # conditional numbers beside them look readable when they are not.
    baseline = params.get("baseline_results_dir")
    if baseline and not Path(baseline).exists():
        raise ValueError(
            f"baseline_results_dir {baseline} does not exist.\n"
            "It is the .smi of the same model sampled with no condition set, "
            "and every conditional metric is reported against it. Remove the "
            "key to run without a baseline, or point it at a file that exists."
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


def novelty_reference_sets(params: dict) -> list:
    """
    Every training split the model has seen, newest corpus first.

    `novelty` and `near_duplicate_rate` ask whether a generated molecule is new
    to the model, so the reference set has to be every corpus in the
    checkpoint's lineage rather than only the task corpus. A transfer-learned
    model that reproduced a base-corpus molecule verbatim scored novelty 1.0,
    which flattered the transfer arm of exactly the comparison the
    transfer-learning claim rests on.

    The walk follows `resume_from` and `pretrained_model_path` through each
    producing job's resolved config. A checkpoint whose config cannot be read
    stops the walk, because guessing at a lineage is worse than a short one;
    the sets that were found are returned and the caller reports how many.
    """
    sets, seen_ckpts = [], set()

    def add(dataset_dir) -> None:
        if not dataset_dir:
            return
        train = str(Path(dataset_dir) / "train.smi")
        if train not in sets:
            sets.append(train)

    add(params.get("dataset_dir"))
    checkpoint = params.get("resume_from") or params.get("pretrained_model_path")
    while checkpoint and checkpoint not in seen_ckpts:
        seen_ckpts.add(checkpoint)
        try:
            produced_by = _read_yaml(checkpoint_config_path(checkpoint))
        except Exception as error:
            # CLAUDE.md defines novelty as absent from EVERY training split in
            # the checkpoint's lineage. Breaking here measured it against a
            # shorter lineage than the model actually saw, which can only move
            # novelty upward, and nothing recorded that it had happened.
            raise ValueError(
                f"Cannot read the lineage of {checkpoint}: "
                f"{checkpoint_config_path(checkpoint)} could not be read "
                f"({type(error).__name__}: {error}).\n"
                "novelty is measured against every training split in the "
                "checkpoint's lineage, so a lineage that cannot be read would "
                "silently report novelty against a shorter one."
            ) from error
        if not produced_by:
            break
        add(produced_by.get("dataset_dir"))
        checkpoint = produced_by.get("resume_from") or produced_by.get(
            "pretrained_model_path"
        )
    return sets


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
            # condition_dim and conditioning are properties of the DATASET,
            # recorded in its preprocessing.yaml, so "remove the key from
            # config.yaml" is not advice anyone can follow: the value is not in
            # config.yaml. A conditional dataset resumed from an unconditional
            # checkpoint is the case that reaches this, and it needs a
            # different answer, not a shorter config.
            if key in ("condition_dim", "conditioning"):
                raise ValueError(
                    f"This job's dataset was preprocessed with {key}: "
                    f"{explicit[key]}, but the checkpoint {checkpoint} was "
                    f"trained with {key}: {produced_by[key]}.\n"
                    "A conditional model cannot be initialized from an "
                    "unconditional checkpoint: the condition encoder and the "
                    "virtual-edge message network do not exist in it, and "
                    "condition_dim is fixed by the dataset rather than by "
                    "this config.\n"
                    "Either drop resume_from and train this model from "
                    "scratch, or point resume_from at a checkpoint trained on "
                    "a dataset with the same conditioning."
                )
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

    One pattern for all three supported devices. A device name that is not one
    of the three is a typo, and that fails rather than silently running
    somewhere unintended.

    An accelerator that is asked for but absent falls back when run
    interactively, so the same job directory runs on a laptop and on a GPU node
    unchanged. Inside a SLURM allocation it does not fall back: see below.
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

    # Inside a batch allocation the fallback is fatal. Interactively it is a
    # convenience -- a laptop with no CUDA still runs the job. Under SLURM it
    # is the opposite: the walltime was requested for a GPU, nothing is
    # watching stdout, and on aarch64 a CPU-only torch wheel reports
    # `cuda.is_available() == False` with no error of its own. Falling back
    # there spends the whole allocation on CPU and produces metrics that look
    # right, stamped with a device nobody reads. So crash instead.
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            f"device: {requested} was requested but is not available on this "
            f"node (SLURM_JOB_ID={os.environ['SLURM_JOB_ID']}).\n"
            "Refusing to fall back, because that would spend the allocation on "
            "the wrong device and still write plausible metrics.\n"
            "Check that the CUDA module is loaded and that torch was built for "
            'this architecture: `python -c "import torch; '
            'print(torch.__version__, torch.cuda.is_available())"`.\n'
            "To run on CPU deliberately, set device: cpu in the job config."
        )

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


# The dataset properties that set the model's tensor widths. A checkpoint
# trained with a different value for any of them cannot be loaded into a model
# built from this dataset.
SHAPE_KEYS = (
    "atom_types",
    "formal_charge",
    "imp_H",
    "chirality",
    "use_chirality",
    "use_aromatic_bonds",
    "use_explicit_H",
    "ignore_H",
    "max_n_nodes",
)


def _check_dataset_features_match_checkpoint(params: dict) -> None:
    """
    The dataset's feature vocabulary must be the checkpoint's.

    These come from the dataset's `preprocessing.yaml` rather than from any
    config layer, so `_apply_checkpoint_architecture` never sees them: it
    compares only keys a config layer set explicitly, and nobody sets
    `atom_types` by hand. The result was a check that reads correctly and is
    dead on the path every real job takes.

    Both failure modes are bad and one of them is silent. Different widths
    crash inside `load_state_dict` with a shape mismatch several frames from
    the cause, which the tutorials describe as something to debug after the
    fact. Equal widths with a different ORDER do not crash at all: the model
    loads, every atom is read as a different element than it is, and the job
    produces molecules and metrics that look entirely normal.

    `max_n_nodes` is included because every stored tensor is padded to it, so
    a checkpoint trained at 108 fed graphs padded to 54 is reading a different
    tensor layout.
    """
    checkpoint = params.get("pretrained_model_path") or params.get("resume_from")
    if not checkpoint:
        return
    source = checkpoint_config_path(checkpoint)
    if not source.exists():
        return
    trained_with = _read_yaml(source)

    differences = []
    for key in SHAPE_KEYS:
        if key not in trained_with or key not in params:
            continue
        theirs, ours = trained_with[key], params[key]
        if theirs == ours:
            continue
        same_size = (
            isinstance(theirs, list)
            and isinstance(ours, list)
            and sorted(map(str, theirs)) == sorted(map(str, ours))
        )
        note = " (SAME VALUES, DIFFERENT ORDER)" if same_size else ""
        differences.append(f"  {key}: dataset {ours!r} vs checkpoint {theirs!r}{note}")
    if not differences:
        return

    raise ValueError(
        f"The dataset at {params['dataset_dir']} was preprocessed with a "
        f"different feature vocabulary than the checkpoint {checkpoint} was "
        "trained with:\n"
        + "\n".join(differences)
        + "\n\nThese set the model's tensor widths, so loading proceeds only "
        "by accident. Where the widths differ it crashes inside "
        "load_state_dict; where they match but the order differs it does not "
        "crash at all and every atom is read as a different element.\n"
        "Preprocess this dataset with `extra_dataset` pointing at the "
        "checkpoint's dataset, which gives the pair one shared vocabulary, or "
        "load a checkpoint trained on this dataset."
    )


def _check_dataset_mapping_matches_checkpoint(params: dict) -> None:
    """
    The dataset's class-to-index mapping must be the checkpoint's.

    `conditioning.properties` IS the mapping: the condition vector is built
    positionally from those columns at preprocessing time and each slot is
    labelled from the same list at generation time. A permuted list therefore
    conditions on one class, reports it as another, and writes a byte-identical
    HDF5 -- there is no shape, no count and no checksum that differs.

    `resolve` already compares a `conditioning` set in a config layer against
    the checkpoint's. This is the other half: the mapping a job inherits from
    its dataset's `preprocessing.yaml` never passed through that comparison,
    and inheriting it is the normal case.
    """
    checkpoint = params.get("pretrained_model_path") or params.get("resume_from")
    if not checkpoint or not params.get("condition_dim"):
        return
    source = checkpoint_config_path(checkpoint)
    if not source.exists():
        return
    trained_with = _read_yaml(source)

    dataset_mapping = list((params.get("conditioning") or {}).get("properties") or [])
    checkpoint_mapping = list(
        (trained_with.get("conditioning") or {}).get("properties") or []
    )
    if not checkpoint_mapping or dataset_mapping == checkpoint_mapping:
        return
    # The same list is the class panel for a categorical condition and the
    # property order for a continuous one, so the message says which.
    kind = "class" if params.get("condition_classes") else "property"
    reordering = (
        f" They are the same {kind}es in a different order, which is the "
        f"dangerous case: every per-{kind} number would be attributed to the "
        f"wrong {kind} and nothing would look wrong."
        if sorted(dataset_mapping) == sorted(checkpoint_mapping)
        else ""
    )
    raise ValueError(
        f"The dataset at {params['dataset_dir']} was preprocessed with the "
        f"{kind} order {dataset_mapping}, and the checkpoint {checkpoint} was "
        f"trained with {checkpoint_mapping}.{reordering}\n"
        f"The mapping is positional: slot i of the condition vector is {kind} "
        "i of this list, in both places, so a transposed declaration "
        "conditions on one and reports it as another while writing a "
        "byte-identical HDF5. Preprocess the dataset with the checkpoint's "
        "order, or load a checkpoint trained on this one."
    )


def _check_requested_classes_exist(params: dict) -> None:
    """
    Every class named in `sample_conditions` must be in the panel.

    A name that is not in the panel was silently dropped, because the condition
    vector is built by walking the panel and looking each name up: a typo, or a
    class from a different dataset, produced a vector of zeros in that slot and
    a job that reported results for a class it never asked for.
    """
    requested = params.get("sample_conditions")
    if not isinstance(requested, dict):
        # `all_classes` is a sentinel resolved against the panel itself.
        return
    panel = list((params.get("conditioning") or {}).get("properties") or [])
    if not panel:
        return
    unknown = [name for name in requested if name not in panel]
    if unknown:
        raise ValueError(
            f"sample_conditions names {unknown}, which the model was not "
            f"conditioned on.\nThis dataset's classes are: {panel}.\n"
            "A name outside the panel was dropped silently, leaving that slot "
            "at zero and the run reporting a class it never requested."
        )


def derive(params: dict) -> namedtuple:
    """Add the feature vocabulary and tensor shapes, and freeze the result."""
    params = dict(params)

    # The RL schedule has its own config keys, and `_validate` rejects the
    # supervised ones on a goal_directed job. They are mapped onto the internal
    # schedule here, once, rather than at each of the ten places the training
    # loop reads them: the config surface is `rl_*` and the loop is unchanged.
    if params["job_type"] == "goal_directed":
        params["epochs"] = params["rl_steps"]
        params["sample_every"] = params["rl_eval_every"]
        if params.get("rl_batch_size"):
            params["batch_size"] = params["rl_batch_size"]

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
        _check_dataset_features_match_checkpoint(params)
        _check_dataset_mapping_matches_checkpoint(params)
        _check_requested_classes_exist(params)

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
        # Every corpus the model has seen, for novelty and near_duplicate_rate.
        "novelty_reference_sets": novelty_reference_sets(params),
        "validation_set": str(Path(params["dataset_dir"]) / "valid.smi"),
        "test_set": str(Path(params["dataset_dir"]) / "test.smi"),
    }
    derived.update(params)
    derived["job_dir"] = str(job_dir) + "/"
    derived["dataset_dir"] = str(Path(params["dataset_dir"])) + "/"

    Constants = namedtuple("CONSTANTS", sorted(derived))
    _Constants._resolved = Constants(**derived)
    return _Constants._resolved
