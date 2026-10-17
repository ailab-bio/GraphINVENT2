"""
Run a GraphINVENT2 job.

    python submit.py experiments/<exp>/jobs/<job>
    python submit.py experiments/<exp>/jobs/<job> --slurm
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
# The editable install writes a .pth file that does not survive a repository
# path containing spaces, so the source directories go on the path here.
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "src" / "graphinvent")]

from parameters import config  # noqa: E402

TEMPLATE = REPO_ROOT / "config" / "slurm.sh.template"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GraphINVENT2 job.")
    parser.add_argument(
        "job_dir",
        type=Path,
        help="job directory, e.g. experiments/my_experiment/jobs/my_job",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="submit to SLURM instead of running here",
    )
    args = parser.parse_args()

    params = config.resolve(args.job_dir)

    job_dir = Path(params["job_dir"])
    for name in ("results", "checkpoints", "logs"):
        (job_dir / name).mkdir(parents=True, exist_ok=True)

    # Derived here rather than inside the job so that the recorded config holds
    # the feature vocabulary the job will actually use, not the empty
    # placeholders the merge leaves behind.
    constants = config.derive(params)
    resolved_path = job_dir / "results" / "config.resolved.yaml"
    with open(resolved_path, "w") as f:
        yaml.safe_dump(
            {key: getattr(constants, key) for key in params},
            f,
            sort_keys=True,
            default_flow_style=False,
        )
    print(f"* Resolved configuration written to {resolved_path}", flush=True)

    if args.slurm:
        script_path = write_slurm_script(params)
        print(f"* Submitting {script_path}", flush=True)
        subprocess.run(["sbatch", str(script_path)], check=True)
    else:
        import main as graphinvent_main

        graphinvent_main.run(constants)


def render_slurm_script(params: dict) -> str:
    """Render the one SLURM template for this job."""
    slurm = params["slurm"]
    job_dir = Path(params["job_dir"]).resolve()

    if not slurm["account"]:
        raise ValueError(
            "slurm.account is unset, and SLURM rejects a job without one.\n"
            "Set it in config/defaults.yaml, or per experiment:\n"
            "    slurm:\n"
            "      account: naiss2026-1-234\n"
            "A GPU job usually needs the -gpu suffix on the account name."
        )

    partition = slurm["partition"] or (
        slurm["partition_gpu"] if slurm["gpus"] else slurm["partition_cpu"]
    )

    extra = []
    if slurm["gpus"]:
        extra.append(f"#SBATCH --gpus={slurm['gpus']}")
    if slurm["memory"]:
        extra.append(f"#SBATCH --mem={slurm['memory']}")

    return TEMPLATE.read_text().format(
        account=slurm["account"],
        partition=partition,
        job_name=job_dir.name,
        time=slurm["time"],
        ntasks=slurm["ntasks"],
        cpus_per_task=slurm["cpus_per_task"],
        resources="".join(line + "\n" for line in extra),
        # Absolute: sbatch resolves the log paths before the script runs cd.
        logs_dir=f"{job_dir}/logs/",
        modules="\n".join(f"module load {m}" for m in slurm["modules"]),
        # shlex.quote because a repository path may contain spaces.
        repo_root=shlex.quote(str(REPO_ROOT)),
        python=shlex.quote(sys.executable),
        job_dir=shlex.quote(str(job_dir)),
    )


def write_slurm_script(params: dict) -> Path:
    """
    Write the rendered script to the job's logs/ and to run.sbatch beside the
    config, so the submitted script can be read or resubmitted by hand.
    """
    script = render_slurm_script(params)
    job_dir = Path(params["job_dir"])

    script_path = job_dir / "logs" / "submit.sh"
    script_path.write_text(script)
    (job_dir / "run.sbatch").write_text(script)
    return script_path


if __name__ == "__main__":
    main()
