"""
GraphINVENT2 job submission script.

Each job type has a dedicated config directory under jobs/:
    jobs/preprocess/params.json     -- data preprocessing
    jobs/unconditional/params.json  -- pre-training or transfer learning (unconditional model)
    jobs/conditional/params.json    -- pre-training or transfer learning (conditional model)
    jobs/goal_directed/params.json  -- RL or constrained-RL fine-tuning
    jobs/generate/params.json       -- molecule sampling / evaluation

Edit the relevant params.json, then run:
    python submit.py --config jobs/preprocess/params.json

Dataset input modes (preprocessing only)
-----------------------------------------
Mode A — single SMILES file:
    Set "smiles_file": "<path/to/molecules.smi>" in the submission block.
    The preprocessing workflow will split it into train/valid/test according
    to "split_type", "train_frac", and "valid_frac" in the job block.
    Only valid for a single dataset.

Mode B — pre-split directory (default):
    Leave "smiles_file" absent (or null).  The dataset directory must already
    contain train.smi, valid.smi, and test.smi; an error is raised otherwise.

Multi-dataset preprocessing
----------------------------
Set "dataset" to a list of dataset names, e.g.:
    "dataset": ["gdb13", "chembl"]

The feature vocabulary (atom_types, formal_charge, imp_H, max_n_nodes) is
computed as the union across all datasets so that the resulting HDF files share
a compatible vocabulary.  Each dataset is then preprocessed independently.

"data_path" may be a single string (applied to all datasets) or a list of
the same length as "dataset".

"smiles_file" may be null/absent (Mode B for all datasets), a list of the same
length as "dataset" (null entries use Mode B for that dataset, path entries use
Mode A), or a single path (applies to the first dataset only; remaining datasets
use Mode B).  Mixed Mode A and Mode B datasets are fully supported.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a GraphINVENT2 job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON config file, e.g. jobs/preprocess/params.json",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset normalization helpers
# ---------------------------------------------------------------------------


def _normalize_datasets(submission: dict) -> tuple:
    """
    Returns (datasets, data_paths, smiles_files) as equal-length lists.

    ``dataset`` and ``data_path`` may each be a plain string or a list of
    strings.  A single ``data_path`` is broadcast over all datasets.

    ``smiles_file`` may be:
    - absent / null  → [None, None, ...]   (Mode B for every dataset)
    - a single path  → [path, None, ...]   (Mode A for first dataset only)
    - a list         → used as-is (None entries = Mode B, path entries = Mode A)
    """
    raw_ds = submission.get("dataset") or None
    raw_dp = submission.get("data_path") or ""

    if raw_ds is None:
        # dataset will be resolved later (e.g. from pretrained model params)
        return [None], [raw_dp if isinstance(raw_dp, list) else raw_dp], [None]
    raw_sf = submission.get("smiles_file")

    datasets = raw_ds if isinstance(raw_ds, list) else [raw_ds]
    data_paths = raw_dp if isinstance(raw_dp, list) else [raw_dp]

    if len(data_paths) == 1 and len(datasets) > 1:
        data_paths = data_paths * len(datasets)

    if len(data_paths) != len(datasets):
        raise ValueError(
            f'"dataset" has {len(datasets)} entries but "data_path" has '
            f"{len(data_paths)}.  They must be the same length, or "
            '"data_path" can be a single string applied to all datasets.'
        )

    # Normalise smiles_file to a per-dataset list
    if isinstance(raw_sf, list):
        smiles_files = [sf or None for sf in raw_sf]
    elif raw_sf is None:
        smiles_files = [None] * len(datasets)
    else:
        # Single path: Mode A for the first dataset, Mode B for the rest
        smiles_files = [raw_sf] + [None] * (len(datasets) - 1)

    if len(smiles_files) != len(datasets):
        raise ValueError(
            f'"smiles_file" has {len(smiles_files)} entries but "dataset" has '
            f"{len(datasets)}.  They must be the same length."
        )

    return datasets, data_paths, smiles_files


def _resolve_dataset_from_pretrained(submission: dict, job_params: dict) -> None:
    """
    For RL/transfer jobs with ``pretrained_model_path`` and no explicit
    ``dataset``, infer ``dataset`` and ``data_path`` from the pretrained
    model's ``params_all.json``.  Mutates ``submission`` in-place.
    """
    if submission.get("dataset"):
        return
    pth = job_params.get("pretrained_model_path", "")
    if not pth:
        return
    params_all = Path(pth).parent / "params_all.json"
    if not params_all.exists():
        return
    with open(params_all) as f:
        pretrain_params = json.load(f)
    dataset_dir = pretrain_params.get("dataset_dir", "")
    if not dataset_dir:
        return
    # dataset_dir is like "data/datasets/debug/" → dataset="debug", data_path="data/datasets"
    dataset_path = Path(dataset_dir)
    submission["dataset"] = dataset_path.name
    submission["data_path"] = str(dataset_path.parent)
    print(
        f"* No dataset specified — using pretrained model's dataset: "
        f"'{submission['dataset']}' ({dataset_dir})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

_VALID_JOB_TYPES = {
    "preprocess",
    "unconditional",
    "conditional",
    "goal_directed",
    "generate",
}
_VALID_SPLIT_TYPES = {"random", "butina", "custom"}

_REQUIRED_SUBMISSION = set()

_REQUIRED_JOB: dict = {
    "preprocess": {"job_type"},
    "unconditional": {"job_type"},
    "conditional": {"job_type", "condition_dim"},
    "goal_directed": {"job_type", "score_components", "score_thresholds"},
    "generate": {"job_type"},
}


def validate_config(
    config_path: str,
    submission: dict,
    job_params: dict,
    datasets: list,
    data_paths: list,
    smiles_files: list,
) -> None:
    """
    Validate the config before any directories are created or jobs launched.
    Raises ``ValueError`` with an actionable message describing exactly what
    needs to be fixed.
    """
    errors = []

    # --- top-level structure ---
    missing_submission = _REQUIRED_SUBMISSION - set(submission.keys())
    if missing_submission:
        errors.append(
            'Missing required field(s) in "submission": '
            + ", ".join(f'"{k}"' for k in sorted(missing_submission))
        )
    job_type_for_dataset_check = job_params.get("job_type")
    _pretrained_optional_types = ("goal_directed", "generate")
    if (
        not submission.get("data_path")
        and job_type_for_dataset_check not in _pretrained_optional_types
    ):
        errors.append('"data_path" is required in "submission".')
    if (
        not submission.get("data_path")
        and job_type_for_dataset_check in _pretrained_optional_types
    ):
        if not job_params.get("pretrained_model_path"):
            errors.append(
                '"data_path" is required in "submission" when '
                '"pretrained_model_path" is not set.'
            )
    if (
        not submission.get("dataset")
        and job_type_for_dataset_check not in _pretrained_optional_types
    ):
        errors.append(
            '"dataset" is required in "submission" for '
            f'job_type="{job_type_for_dataset_check}".'
        )
    if (
        not submission.get("dataset")
        and job_type_for_dataset_check in _pretrained_optional_types
    ):
        if not job_params.get("pretrained_model_path"):
            errors.append(
                '"dataset" is required in "submission" when '
                '"pretrained_model_path" is not set.'
            )

    job_type = job_params.get("job_type")
    if not job_type:
        errors.append(
            '"job_type" is missing from the "job" block. '
            f"Valid options: {sorted(_VALID_JOB_TYPES)}"
        )
    elif job_type not in _VALID_JOB_TYPES:
        errors.append(
            f'"job_type" is "{job_type}", which is not recognised. '
            f"Valid options: {sorted(_VALID_JOB_TYPES)}"
        )

    if errors:
        _raise(config_path, errors)

    # --- non-preprocess jobs must have a single dataset ---
    if job_type != "preprocess" and len(datasets) > 1:
        errors.append(
            f'job_type="{job_type}" requires a single dataset, but '
            f'"dataset" is a list of {len(datasets)} entries.  '
            "Use a single string for non-preprocessing jobs."
        )

    # --- required job fields ---
    required = _REQUIRED_JOB.get(job_type, {"job_type"})
    missing_job = required - set(job_params.keys())
    if missing_job:
        errors.append(
            f'Missing required field(s) in "job" for job_type="{job_type}": '
            + ", ".join(f'"{k}"' for k in sorted(missing_job))
        )

    # --- graphinvent path ---
    graphinvent_path = Path(submission.get("graphinvent_path", "./src/graphinvent"))
    if not (graphinvent_path / "main.py").exists():
        errors.append(
            f'"graphinvent_path" points to "{graphinvent_path}", but '
            f'"{graphinvent_path / "main.py"}" does not exist. '
            'Check that "graphinvent_path" is set to the graphinvent/ source directory.'
        )

    # --- preprocessing-specific checks (one check per dataset) ---
    if job_type == "preprocess":
        split_type = job_params.get("split_type", "random")
        if split_type not in _VALID_SPLIT_TYPES:
            errors.append(
                f'"split_type" is "{split_type}". '
                f"Valid options: {sorted(_VALID_SPLIT_TYPES)}"
            )

        train_frac = float(job_params.get("train_frac", 0.8))
        valid_frac = float(job_params.get("valid_frac", 0.1))
        if train_frac + valid_frac > 1.0:
            errors.append(
                f'"train_frac" ({train_frac}) + "valid_frac" ({valid_frac}) = '
                f"{train_frac + valid_frac:.3f}, which exceeds 1.0."
            )

        for data_path, dataset, sf in zip(data_paths, datasets, smiles_files):
            dataset_dir = Path(data_path) / dataset
            if sf is not None:
                # Mode A: the source SMILES file must exist
                smi_path = Path(sf)
                if not smi_path.exists():
                    errors.append(
                        f'Dataset "{dataset}": "smiles_file" is set to "{sf}", '
                        "but that file does not exist. Check the path."
                    )
            else:
                # Mode B: all three split files must be present
                missing_smi = [
                    name
                    for name in ("train.smi", "valid.smi", "test.smi")
                    if not (dataset_dir / name).exists()
                ]
                if missing_smi:
                    errors.append(
                        f'Dataset "{dataset}": directory "{dataset_dir}" must '
                        f"contain train.smi, valid.smi, and test.smi — "
                        f"missing: {', '.join(missing_smi)}.\n"
                        f'  Place the pre-split files there, or set "smiles_file" '
                        f"to a .smi path for automatic splitting."
                    )

    # --- goal_directed-specific checks ---
    if job_type == "goal_directed":
        components = job_params.get("score_components", [])
        thresholds = job_params.get("score_thresholds", [])
        if len(components) != len(thresholds):
            errors.append(
                f'"score_components" has {len(components)} item(s) but '
                f'"score_thresholds" has {len(thresholds)}. '
                "They must have the same length."
            )
        max_n = job_params.get("max_n_nodes", 0)
        # If max_n_nodes isn't in the job params, try to resolve it from the
        # pretrained model's params_all.json or the dataset's preprocessing_params.json.
        if not max_n:
            _pth = job_params.get("pretrained_model_path", "")
            if _pth:
                _pa = Path(_pth).parent / "params_all.json"
                if _pa.exists():
                    with open(_pa) as _f:
                        max_n = json.load(_f).get("max_n_nodes", 0)
        for comp in components:
            if comp.startswith("target_size="):
                try:
                    n = int(comp.split("=")[1])
                    if max_n and n >= max_n:
                        errors.append(
                            f'"score_components" contains "{comp}", but the target '
                            f'size ({n}) must be strictly less than "max_n_nodes" '
                            f"({max_n}). Use target_size={max_n - 1} or smaller."
                        )
                except ValueError:
                    errors.append(f'Could not parse target size in "{comp}".')

    # --- pretrained model / resume_from checks ---

    # unconditional and conditional: resume_from is optional; validate if set.
    if job_type in ("unconditional", "conditional"):
        resume = job_params.get("resume_from")
        if resume and not Path(resume).exists():
            errors.append(
                f'"resume_from" is set to "{resume}", but that file does not exist. '
                'Check the path or set "resume_from" to null to train from scratch.'
            )

    # goal_directed and generate always require a pretrained model.
    if job_type in ("goal_directed", "generate"):
        model_path = job_params.get("pretrained_model_path", "")
        model_dir = job_params.get("pretrained_model_dir", "")
        epoch = job_params.get("generation_epoch")
        if model_path:
            if not Path(model_path).exists():
                errors.append(
                    f'Checkpoint "{model_path}" does not exist. '
                    f'Check "pretrained_model_path".'
                )
        else:
            if job_type == "goal_directed":
                if not model_dir or epoch is None:
                    errors.append(
                        'Specify either "pretrained_model_path" (direct path to a .pth file) '
                        'or both "pretrained_model_dir" and "generation_epoch".'
                    )
                else:
                    checkpoint = Path(model_dir) / f"model_restart_{epoch}.pth"
                    if not checkpoint.exists():
                        errors.append(
                            f'Checkpoint "{checkpoint}" does not exist. '
                            f'Check "pretrained_model_dir" ("{model_dir}") and '
                            f'"generation_epoch" ({epoch}).'
                        )
            elif job_type == "generate":
                if model_dir and epoch is not None:
                    checkpoint = Path(model_dir) / f"model_restart_{epoch}.pth"
                    if not checkpoint.exists():
                        errors.append(
                            f'Checkpoint "{checkpoint}" does not exist. '
                            f'Check "pretrained_model_dir" ("{model_dir}") and '
                            f'"generation_epoch" ({epoch}).'
                        )
                else:
                    errors.append(
                        'Specify "pretrained_model_path" (direct path to a .pth file) '
                        'in the "job" block.'
                    )

    if errors:
        _raise(config_path, errors)


def _raise(config_path: str, errors: list) -> None:
    lines = [f"\nConfig validation failed for: {config_path}\n"]
    for i, err in enumerate(errors, 1):
        lines.append(f"  {i}. {err}")
    lines.append("")
    raise ValueError("\n".join(lines))


# ---------------------------------------------------------------------------
# Output directory creation
# ---------------------------------------------------------------------------


def create_output_directories(dataset: str, job_type: str) -> tuple:
    """Create the output directory for this job.

    TensorBoard logs are written to ``<job_dir>/tensorboard/`` (i.e. inside
    the job directory) rather than a sibling ``tensorboard/`` folder, so each
    run is self-contained in a single directory.
    """
    base_path = Path("output") / dataset / job_type
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"* Output directory: {base_path}", flush=True)
    # tensorboard_path is set per-job inside submit_jobs(); return a sentinel.
    return base_path, None


# ---------------------------------------------------------------------------
# Single-job submission
# ---------------------------------------------------------------------------


def submit_jobs(
    submission: dict,
    job_params: dict,
    base_path: Path,
    tensorboard_path: Path,  # kept for API compatibility; value is ignored
    dataset_dir: Path,
    extra_params: dict = None,
) -> None:
    """Build a job directory and launch (or schedule) one job."""
    job_name = submission.get("job_name", "job")
    job_dir = base_path / job_name
    tb_dir = job_dir / "tensorboard"

    job_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    params = dict(job_params)
    if extra_params:
        params.update(extra_params)

    params["job_dir"] = str(job_dir) + "/"
    params["tensorboard_dir"] = str(tb_dir) + "/"
    params["dataset_dir"] = str(dataset_dir) + "/"

    smiles_file = submission.get("smiles_file") or None
    if smiles_file is not None:
        params["smiles_file"] = str(Path(smiles_file))

    params_path = job_dir / "params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"* Created job directory: {job_dir}", flush=True)
    _submit_single_job(submission, job_dir)


def _submit_single_job(submission: dict, job_dir: Path) -> None:
    python_path = submission.get("python_path", "python")
    graphinvent_path = Path(submission.get("graphinvent_path", "./src/graphinvent"))
    main_py = graphinvent_path / "main.py"

    if submission.get("use_slurm", False):
        script_path = _write_submission_script(submission, job_dir, main_py)
        print("* Submitting job to SLURM.", flush=True)
        subprocess.run(["sbatch", str(script_path)], check=True)
    else:
        print("* Running job directly.", flush=True)
        subprocess.run(
            [python_path, str(main_py), "--job-dir", str(job_dir) + "/"],
            check=True,
        )


def _write_submission_script(submission: dict, job_dir: Path, main_py: Path) -> Path:
    slurm = submission.get("slurm", {})
    python_path = submission.get("python_path", "python")
    script_path = job_dir / "submit.sh"
    output_log = job_dir / "output.o${SLURM_JOB_ID}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH -A {slurm.get('account', 'ACCOUNT')}",
        f"#SBATCH --job-name={submission.get('dataset', 'graphinvent')}",
        f"#SBATCH --time={slurm.get('run_time', '0-06:00:00')}",
    ]
    if "gpus_per_node" in slurm:
        lines.append(f"#SBATCH --gpus-per-node={slurm['gpus_per_node']}")
    lines += [
        "hostname",
        "export QT_QPA_PLATFORM='offscreen'",
        f"({python_path} {main_py} --job-dir {job_dir}/ > {output_log})",
    ]

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("* Wrote SLURM submission script.", flush=True)
    return script_path


# ---------------------------------------------------------------------------
# Multi-dataset preprocessing
# ---------------------------------------------------------------------------


def _compute_union_vocab(
    dataset_dirs: list,
    smiles_files: list,
    graphinvent_path: str,
    use_explicit_H: bool = False,
    ignore_H: bool = False,
) -> dict:
    """
    Scans SMILES from every dataset and returns the union feature vocabulary
    as a dict with keys: atom_types, formal_charge, imp_H, max_n_nodes.

    For Mode A datasets (smiles_files[i] is not None) the raw source file is
    scanned directly.  For Mode B datasets the pre-split train/valid/test.smi
    files in the dataset directory are scanned.
    """
    tools_dir = str(Path(graphinvent_path) / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from scan_features import scan_features  # noqa: PLC0415

    all_paths = []
    for dataset_dir, sf in zip(dataset_dirs, smiles_files):
        if sf is not None:
            # Mode A: scan the raw source file before splitting
            all_paths.append(str(sf))
        else:
            # Mode B: scan the pre-split files
            for name in ("train.smi", "valid.smi", "test.smi"):
                p = dataset_dir / name
                if p.exists():
                    all_paths.append(str(p))

    return scan_features(all_paths, use_explicit_H=use_explicit_H, ignore_H=ignore_H)


def submit_multi_preprocess(
    submission: dict,
    job_params: dict,
    datasets: list,
    dataset_dirs: list,
    smiles_files: list,
) -> None:
    """
    Compute the union feature vocabulary across all datasets, then launch one
    preprocessing job per dataset using that shared vocabulary.

    Each dataset may independently use Mode A (smiles_files[i] is a path) or
    Mode B (smiles_files[i] is None).
    """
    use_explicit_H = job_params.get("use_explicit_H", False)
    ignore_H = job_params.get("ignore_H", False)
    graphinvent_path = submission.get("graphinvent_path", "./src/graphinvent")

    print(
        f"* Multi-dataset preprocessing: computing union vocabulary "
        f"across {len(datasets)} datasets...",
        flush=True,
    )
    vocab = _compute_union_vocab(
        dataset_dirs=dataset_dirs,
        smiles_files=smiles_files,
        graphinvent_path=graphinvent_path,
        use_explicit_H=use_explicit_H,
        ignore_H=ignore_H,
    )
    print(f"  atom_types    : {vocab['atom_types']}", flush=True)
    print(f"  formal_charge : {vocab['formal_charge']}", flush=True)
    if not use_explicit_H and not ignore_H:
        print(f"  imp_H         : {vocab['imp_H']}", flush=True)
    print(f"  max_n_nodes   : {vocab['max_n_nodes']}", flush=True)

    # Build the shared vocab params to inject into every per-dataset job.
    # Set auto_detect_features=False so constants.py uses these values as-is.
    vocab_params = {
        "auto_detect_features": False,
        "atom_types": vocab["atom_types"],
        "formal_charge": vocab["formal_charge"],
        "max_n_nodes": vocab["max_n_nodes"],
    }
    if not use_explicit_H and not ignore_H:
        vocab_params["imp_H"] = vocab["imp_H"]
    if job_params.get("use_chirality", False):
        vocab_params["chirality"] = ["None", "R", "S"]

    for dataset, dataset_dir, sf in zip(datasets, dataset_dirs, smiles_files):
        print(f"\n* Preprocessing dataset: {dataset}", flush=True)
        base_path, tensorboard_path = create_output_directories(
            dataset, job_params["job_type"]
        )
        # Give each per-dataset submission its own dataset name and smiles_file
        # so that SLURM job names and Mode A splitting are handled correctly.
        single_submission = {**submission, "dataset": dataset, "smiles_file": sf}
        submit_jobs(
            submission=single_submission,
            job_params=job_params,
            base_path=base_path,
            tensorboard_path=tensorboard_path,
            dataset_dir=dataset_dir,
            extra_params=vocab_params,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    config = load_config(args.config)

    submission = config["submission"]
    job_params = config["job"]

    if job_params.get("job_type") in ("rl", "transfer", "generate"):
        _resolve_dataset_from_pretrained(submission, job_params)

    datasets, data_paths, smiles_files = _normalize_datasets(submission)
    dataset_dirs = [
        Path(dp) / ds if (dp and ds) else Path("")
        for dp, ds in zip(data_paths, datasets)
    ]

    validate_config(
        args.config, submission, job_params, datasets, data_paths, smiles_files
    )

    is_multi_preprocess = (
        job_params.get("job_type") == "preprocess" and len(datasets) > 1
    )

    if is_multi_preprocess:
        submit_multi_preprocess(
            submission, job_params, datasets, dataset_dirs, smiles_files
        )
    else:
        base_path, tensorboard_path = create_output_directories(
            datasets[0], job_params["job_type"]
        )
        submit_jobs(
            submission=submission,
            job_params=job_params,
            base_path=base_path,
            tensorboard_path=tensorboard_path,
            dataset_dir=dataset_dirs[0],
        )


if __name__ == "__main__":
    main()
