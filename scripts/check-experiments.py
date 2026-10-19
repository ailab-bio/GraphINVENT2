#!/usr/bin/env python3
"""
Resolve every job config through the three-layer merge and report what is
missing. Rerun after any config change.

    python3 scripts/check-experiments.py            # all ten experiments
    python3 scripts/check-experiments.py --tutorials  # include the tutorials

Three things are checked that nothing else checks:
  - every config resolves at all, which catches a key the merge rejects
  - every input path a job names exists, which catches the join between the
    acquisition script and the experiment that consumes its output
  - no experiment silently inherits a parameter it must set itself
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "src" / "graphinvent")]

# Parameters an experiment must set in its own config or experiment.yaml. A
# value inherited from config/defaults.yaml is a number nobody chose.
MUST_NOT_INHERIT = {
    "goal_directed": (
        "success_threshold",
        "success_definition",
        "rl_steps",
        "rl_batch_size",
        "property_window",
    ),
    "generate": ("n_samples", "property_window"),
    "unconditional": ("epochs", "property_window"),
    "conditional": ("epochs", "property_window"),
    "preprocess": ("split_type", "max_n_nodes"),
}


def own_keys(job_dir: Path) -> set:
    """Keys this job or its experiment sets, as opposed to inheriting."""
    import yaml

    keys: set = set()
    for path in (job_dir / "config.yaml", job_dir.parents[1] / "experiment.yaml"):
        if path.exists():
            keys |= set((yaml.safe_load(path.read_text()) or {}).keys())
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutorials", action="store_true")
    args = ap.parse_args()

    from parameters import config

    jobs = sorted(REPO.glob("experiments/*/jobs/*/config.yaml"))
    if not args.tutorials:
        jobs = [j for j in jobs if not j.parts[-4].startswith("tutorial_")]

    rows, failures, inherited, missing_inputs = [], [], [], []
    for cfg in jobs:
        job_dir = cfg.parent
        name = f"{job_dir.parts[-3]}/{job_dir.name}"
        try:
            params = config.resolve(job_dir)
        except Exception as exc:
            failures.append(
                (name, f"{type(exc).__name__}: {' '.join(str(exc).split())[:150]}")
            )
            rows.append((name, "?", "RESOLVE FAILED"))
            continue

        job_type = params["job_type"]
        rows.append((name, job_type, "resolves"))

        for key in MUST_NOT_INHERIT.get(job_type, ()):
            if key not in own_keys(job_dir):
                inherited.append((name, job_type, key))

        for key in ("smiles_file", "pretrained_model_path", "resume_from"):
            value = params.get(key)
            if value and not (REPO / str(value)).exists():
                missing_inputs.append((name, key, str(value)))
        if job_type != "preprocess":
            ddir = REPO / "data" / "datasets" / str(params.get("dataset_dir", ""))
            if not ddir.is_dir():
                missing_inputs.append(
                    (name, "dataset_dir", str(ddir.relative_to(REPO)))
                )

    print(f"{'JOB':48} {'TYPE':14} RESULT")
    print("-" * 84)
    for name, job_type, result in rows:
        print(f"{name:48} {job_type:14} {result}")

    print(f"\n=== {len(failures)} configs failed to resolve ===")
    for name, why in failures:
        print(f"  {name}\n      {why}")

    print(f"\n=== {len(missing_inputs)} inputs a job names that do not exist ===")
    for name, key, value in missing_inputs:
        print(f"  {name:46} {key:22} {value}")

    print(
        f"\n=== {len(inherited)} parameters silently inherited that should be set per experiment ==="
    )
    for name, job_type, key in inherited:
        print(f"  {name:46} {job_type:14} {key}")

    return 1 if (failures or missing_inputs) else 0


if __name__ == "__main__":
    sys.exit(main())
