"""
Render a job's `run.sbatch` without submitting it and without requiring its
inputs to exist yet.

`submit.py --slurm` renders and submits in one step, and `config.resolve`
validates that every input is already on disk -- a checkpoint to resume from, a
preprocessed dataset. That is right for submission and wrong for drafting: the
whole point of a drafted experiment directory is that a person can read the
sbatch script for job 5 before job 4 has run.

So this merges the three config layers with the same functions `resolve` uses,
skips only the existence checks, and renders through the same
`render_slurm_script` and the same `config/slurm.sh.template`. There is still
one template and one renderer; this is a second entry point to them, not a
second source of truth.

    python scripts/render-sbatch.py experiments/<exp>/jobs/<job> [...]
    python scripts/render-sbatch.py --all          # every job under experiments/

A rendered script is a draft. It carries the walltime from the job's config,
which is an estimate until someone has run the job once.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "src" / "graphinvent")]

import importlib.util  # noqa: E402

from parameters import config  # noqa: E402

_spec = importlib.util.spec_from_file_location("_submit", REPO_ROOT / "submit.py")
_submit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_submit)


def merge_without_validation(job_dir: Path) -> dict:
    """The three-layer merge from `config.resolve`, minus the existence checks."""
    job_dir = Path(job_dir)
    job_config = job_dir / "config.yaml"
    if not job_config.exists():
        raise SystemExit(f"no config.yaml in {job_dir}")

    params = config._read_yaml(REPO_ROOT / "config" / "defaults.yaml")
    experiment_yaml = job_dir.parent.parent / "experiment.yaml"
    if experiment_yaml.exists():
        params = config._merge(params, config._read_yaml(experiment_yaml))
    params = config._merge(params, config._read_yaml(job_config))
    params["job_dir"] = str(job_dir)
    return params


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path, nargs="*", help="job directories")
    parser.add_argument(
        "--all", action="store_true", help="every experiments/*/jobs/* directory"
    )
    args = parser.parse_args()

    targets = list(args.job_dir)
    if args.all:
        targets = sorted(
            p.parent for p in (REPO_ROOT / "experiments").glob("*/jobs/*/config.yaml")
        )
    if not targets:
        raise SystemExit("nothing to render: pass job directories or --all")

    written, failed = 0, 0
    for job_dir in targets:
        try:
            params = merge_without_validation(job_dir)
            script = _submit.render_slurm_script(params)
        except Exception as error:
            print(f"  FAILED  {job_dir}: {error}")
            failed += 1
            continue
        (Path(job_dir) / "run.sbatch").write_text(script)
        for name in ("results", "checkpoints", "logs"):
            (Path(job_dir) / name).mkdir(parents=True, exist_ok=True)
            (Path(job_dir) / name / ".gitkeep").touch()
        print(f"  wrote   {job_dir}/run.sbatch")
        written += 1

    print(f"\n{written} rendered, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
