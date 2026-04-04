"""
Main entry point for GraphINVENT2 jobs.

Supported job types (set via `job_type` in params.json):
  preprocess      -- convert SMILES datasets to HDF5 format
  unconditional   -- train/fine-tune an unconditional generative model
  conditional     -- train/fine-tune a conditioning-aware generative model
  goal_directed   -- RL fine-tuning with optional oracle budget
  generate        -- generate molecules or evaluate a trained model

Backward-compatible aliases (print a deprecation warning):
  pretrain       → unconditional
  transfer       → unconditional  (with resume_from set)
  rl             → goal_directed
  constrained_rl → goal_directed  (with oracle_budget set)
  sample         → generate
  test           → generate       (with sample_mode="evaluate")

Usage:
  python graphinvent/main.py --job-dir path/to/job_dir/
"""

import datetime
import random

import numpy as np
import torch
import util
from parameters.config import constants
from Workflow import Workflow

util.suppress_warnings()

# ---------------------------------------------------------------------------
# Backward-compatible alias map
# ---------------------------------------------------------------------------

_DEPRECATED_JOB_TYPES = {
    "pretrain": "unconditional",
    "transfer": "unconditional",
    "rl": "goal_directed",
    "constrained_rl": "goal_directed",
    "sample": "generate",
    "test": "generate",
}


def main():
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"* Job started at: {start_time}", flush=True)

    seed = getattr(constants, "seed", 0)
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"* Random seed set to {seed}", flush=True)

    workflow = Workflow(constants=constants)
    job_type = constants.job_type

    # Backward-compatibility: remap legacy job type names with a warning
    if job_type in _DEPRECATED_JOB_TYPES:
        new_name = _DEPRECATED_JOB_TYPES[job_type]
        print(
            f"* Warning: job_type '{job_type}' is deprecated. "
            f"Use '{new_name}' in new configs.",
            flush=True,
        )
        job_type = new_name

    print(f"* Run mode: '{job_type}'", flush=True)

    if job_type == "preprocess":
        util.write_preprocessing_parameters(params=constants)
        workflow.preprocess_phase()

    elif job_type == "unconditional":
        util.write_job_parameters(params=constants)
        workflow.unconditional_training_phase()

    elif job_type == "conditional":
        util.write_job_parameters(params=constants)
        workflow.conditional_training_phase()

    elif job_type == "goal_directed":
        util.write_job_parameters(params=constants)
        workflow.goal_directed_training_phase()

    elif job_type == "generate":
        util.write_job_parameters(params=constants)
        workflow.sample_phase()

    else:
        raise NotImplementedError(
            f"Unknown job_type '{job_type}'. "
            "Valid options: preprocess, unconditional, conditional, goal_directed, generate."
        )


if __name__ == "__main__":
    main()
