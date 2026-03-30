"""
GraphINVENT2 job submission script.

Each job type has a dedicated config directory under jobs/:
    jobs/preprocess/params.json   -- data preprocessing
    jobs/pretrain/params.json     -- prior model training
    jobs/transfer/params.json     -- supervised fine-tuning
    jobs/rl/params.json           -- RL fine-tuning
    jobs/sample/params.json       -- molecule sampling / generation

Edit the relevant params.json, then run:
    python submit.py --config jobs/preprocess/params.json

Dataset input modes (preprocessing only)
-----------------------------------------
Mode A — single SMILES file:
    Set "smiles_file": "<path/to/molecules.smi>" in the submission block.
    The preprocessing workflow will split it into train/valid/test according
    to "split_type", "train_frac", and "valid_frac" in the job block.

Mode B — pre-split directory (default):
    Leave "smiles_file" absent (or null).  The dataset directory must already
    contain train.smi, valid.smi, and test.smi; an error is raised otherwise.
"""
import argparse
import json
import subprocess
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
# Config validation
# ---------------------------------------------------------------------------

_VALID_JOB_TYPES = {"preprocess", "pretrain", "transfer", "rl", "generate"}

_VALID_SPLIT_TYPES = {"random", "butina", "custom"}

# Fields required in the submission block for every job
_REQUIRED_SUBMISSION = {"data_path", "dataset"}

# Fields required in the job block for every job type
_REQUIRED_JOB: dict = {
    "preprocess": {"job_type"},
    "pretrain":   {"job_type", "atom_types", "formal_charge", "imp_H", "max_n_nodes"},
    "transfer":   {"job_type", "atom_types", "formal_charge", "imp_H", "max_n_nodes",
                   "pretrained_model_dir", "generation_epoch"},
    "rl":         {"job_type", "atom_types", "formal_charge", "imp_H", "max_n_nodes",
                   "pretrained_model_dir", "generation_epoch",
                   "score_components", "score_thresholds"},
    "generate":   {"job_type", "atom_types", "formal_charge", "imp_H", "max_n_nodes",
                   "generation_epoch"},
}


def validate_config(config_path: str, submission: dict, job_params: dict) -> None:
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
            f'Missing required field(s) in "submission": '
            + ", ".join(f'"{k}"' for k in sorted(missing_submission))
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

    # Bail early if we can't even determine job_type
    if errors:
        _raise(config_path, errors)

    # --- required job fields ---
    required = _REQUIRED_JOB.get(job_type, {"job_type"})
    missing_job = required - set(job_params.keys())
    if missing_job:
        errors.append(
            f'Missing required field(s) in "job" for job_type="{job_type}": '
            + ", ".join(f'"{k}"' for k in sorted(missing_job))
        )

    # --- path existence checks ---
    data_path   = submission.get("data_path", "")
    dataset     = submission.get("dataset", "")
    dataset_dir = Path(data_path) / dataset

    graphinvent_path = Path(submission.get("graphinvent_path", "./graphinvent"))
    if not (graphinvent_path / "main.py").exists():
        errors.append(
            f'"graphinvent_path" points to "{graphinvent_path}", but '
            f'"{graphinvent_path / "main.py"}" does not exist. '
            "Check that \"graphinvent_path\" is set to the graphinvent/ source directory."
        )

    # --- preprocessing-specific checks ---
    if job_type == "preprocess":
        smiles_file = submission.get("smiles_file") or None
        split_type  = job_params.get("split_type", "random")

        if split_type not in _VALID_SPLIT_TYPES:
            errors.append(
                f'"split_type" is "{split_type}". '
                f"Valid options: {sorted(_VALID_SPLIT_TYPES)}"
            )

        if smiles_file:
            # Mode A: single file must exist
            smi_path = Path(smiles_file)
            if not smi_path.exists():
                errors.append(
                    f'"smiles_file" is set to "{smiles_file}", but that file '
                    "does not exist. Check the path."
                )
        else:
            # Mode B: all three split files must already be in dataset_dir
            missing_smi = [
                name for name in ("train.smi", "valid.smi", "test.smi")
                if not (dataset_dir / name).exists()
            ]
            if missing_smi:
                missing_list = ", ".join(missing_smi)
                errors.append(
                    f'"smiles_file" is not set, so the dataset directory '
                    f'"{dataset_dir}" must already contain train.smi, valid.smi, '
                    f"and test.smi — but the following are missing: {missing_list}.\n"
                    "\n"
                    "  Fix option A — point to your SMILES file for automatic splitting:\n"
                    '    In params.json, add to the "submission" block:\n'
                    '      "smiles_file": "path/to/your/molecules.smi"\n'
                    "\n"
                    "  Fix option B — provide the pre-split files yourself:\n"
                    f"    Place train.smi, valid.smi, and test.smi in:\n"
                    f"      {dataset_dir}/"
                )

        train_frac = float(job_params.get("train_frac", 0.8))
        valid_frac = float(job_params.get("valid_frac", 0.1))
        if train_frac + valid_frac > 1.0:
            errors.append(
                f'"train_frac" ({train_frac}) + "valid_frac" ({valid_frac}) = '
                f"{train_frac + valid_frac:.3f}, which exceeds 1.0. "
                "Reduce one or both values."
            )

    # --- RL-specific checks ---
    if job_type == "rl":
        components = job_params.get("score_components", [])
        thresholds = job_params.get("score_thresholds", [])
        if len(components) != len(thresholds):
            errors.append(
                f'"score_components" has {len(components)} item(s) but '
                f'"score_thresholds" has {len(thresholds)}. '
                "They must have the same length."
            )
        max_n = job_params.get("max_n_nodes", 0)
        for comp in components:
            if comp.startswith("target_size="):
                try:
                    n = int(comp.split("=")[1])
                    if n >= max_n:
                        errors.append(
                            f'"score_components" contains "{comp}", but the target '
                            f"size ({n}) must be strictly less than \"max_n_nodes\" "
                            f"({max_n}). Use target_size={max_n - 1} or smaller."
                        )
                except ValueError:
                    errors.append(f'Could not parse target size in "{comp}".')

    # --- generation / training: pretrained model checks ---
    if job_type in ("transfer", "rl", "generate"):
        model_dir = job_params.get("pretrained_model_dir", "")
        epoch     = job_params.get("generation_epoch")
        if model_dir and epoch is not None:
            checkpoint = Path(model_dir) / f"model_restart_{epoch}.pth"
            if not checkpoint.exists():
                errors.append(
                    f'Checkpoint "{checkpoint}" does not exist. '
                    f"Check \"pretrained_model_dir\" (\"{model_dir}\") and "
                    f"\"generation_epoch\" ({epoch})."
                )

    if errors:
        _raise(config_path, errors)


def _raise(config_path: str, errors: list) -> None:
    lines = [f"\nConfig validation failed for: {config_path}\n"]
    for i, err in enumerate(errors, 1):
        lines.append(f"  {i}. {err}")
    lines.append("")
    raise ValueError("\n".join(lines))


def create_output_directories(submission: dict, job_params: dict) -> tuple:
    """Create the output and tensorboard directories for this job."""
    dataset  = submission["dataset"]
    job_type = job_params["job_type"]

    base_path        = Path("output") / dataset / job_type
    tensorboard_path = base_path / "tensorboard"

    base_path.mkdir(parents=True, exist_ok=True)
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    print(f"* Output directory: {base_path}", flush=True)
    return base_path, tensorboard_path


def submit_jobs(
    submission: dict,
    job_params: dict,
    base_path: Path,
    tensorboard_path: Path,
) -> None:
    n_jobs    = submission.get("n_jobs", 1)
    start_idx = submission.get("jobdir_start_idx", 0)

    for job_idx in range(start_idx, start_idx + n_jobs):
        job_dir = base_path / f"job_{job_idx}"
        tb_dir  = tensorboard_path / f"job_{job_idx}"

        job_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)

        # Build the flat params dict that graphinvent/main.py will read.
        # Path() normalizes slashes; the trailing "/" is added so downstream
        # code can safely concatenate file names.
        dataset_dir = Path(submission["data_path"]) / submission["dataset"]

        params = dict(job_params)
        params["job_dir"]         = str(job_dir) + "/"
        params["tensorboard_dir"] = str(tb_dir) + "/"
        params["dataset_dir"]     = str(dataset_dir) + "/"

        # Pass the smiles_file from the submission block into the job params
        # so the preprocessing workflow can pick it up.
        smiles_file = submission.get("smiles_file") or None
        if smiles_file is not None:
            params["smiles_file"] = str(Path(smiles_file))

        params_path = job_dir / "params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

        print(f"* Created job directory: {job_dir}", flush=True)
        _submit_single_job(submission, job_dir)


def _submit_single_job(submission: dict, job_dir: Path) -> None:
    python_path      = submission.get("python_path", "python")
    graphinvent_path = Path(submission.get("graphinvent_path", "./graphinvent"))
    main_py          = graphinvent_path / "main.py"

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


def _write_submission_script(
    submission: dict, job_dir: Path, main_py: Path
) -> Path:
    slurm       = submission.get("slurm", {})
    python_path = submission.get("python_path", "python")
    script_path = job_dir / "submit.sh"
    output_log  = job_dir / "output.o${SLURM_JOB_ID}"

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


def main():
    args   = parse_args()
    config = load_config(args.config)

    submission = config["submission"]
    job_params = config["job"]

    validate_config(args.config, submission, job_params)

    base_path, tensorboard_path = create_output_directories(submission, job_params)
    submit_jobs(submission, job_params, base_path, tensorboard_path)


if __name__ == "__main__":
    main()
