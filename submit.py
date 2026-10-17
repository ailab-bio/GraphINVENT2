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
        submit_to_slurm(params)
    else:
        import main as graphinvent_main

        graphinvent_main.run(constants)


def submit_to_slurm(params: dict) -> None:
    """Render the SLURM template into the job's logs/ and submit it."""
    slurm = params["slurm"]
    job_dir = Path(params["job_dir"])

    directives = [f"#SBATCH --account={slurm['account']}"] if slurm["account"] else []
    if slurm["partition"]:
        directives.append(f"#SBATCH --partition={slurm['partition']}")
    if slurm["gpus"]:
        directives.append(f"#SBATCH --gpus={slurm['gpus']}")
    if slurm["memory"]:
        directives.append(f"#SBATCH --mem={slurm['memory']}")

    template = (REPO_ROOT / "config" / "slurm.sh.template").read_text()
    script = template.format(
        job_name=job_dir.name,
        time=slurm["time"],
        # Absolute: sbatch resolves the log paths before the script runs cd.
        logs_dir=f"{job_dir.resolve()}/logs/",
        resources="\n".join(directives) + "\n" if directives else "",
        # shlex.quote because this repository's own path contains spaces.
        repo_root=shlex.quote(str(REPO_ROOT)),
        python=shlex.quote(sys.executable),
        job_dir=shlex.quote(str(job_dir)),
    )

    script_path = job_dir / "logs" / "submit.sh"
    script_path.write_text(script)
    print(f"* SLURM script written to {script_path}", flush=True)
    subprocess.run(["sbatch", str(script_path)], check=True)


if __name__ == "__main__":
    main()
