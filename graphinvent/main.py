"""
Main entry point for GraphINVENT2 jobs.

Supported job types (set via `job_type` in params.json):
  preprocess  -- convert SMILES datasets to HDF5 format
  pretrain    -- train a generative model from random weight initialization
  transfer    -- fine-tune a pretrained model with supervised learning on a new dataset
  generate    -- sample molecules from a trained model
  test        -- evaluate a trained model on the test set
  rl          -- optimize a pretrained model via policy-gradient reinforcement learning

Usage:
  python graphinvent/main.py --job-dir path/to/job_dir/

The job directory must contain a params.json file written by submit.py.
"""

import datetime

import util
from parameters.constants import constants
from Workflow import Workflow

util.suppress_warnings()


def main():
    _ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    workflow = Workflow(constants=constants)
    job_type = constants.job_type
    print(f"* Run mode: '{job_type}'", flush=True)

    if job_type == "preprocess":
        util.write_preprocessing_parameters(params=constants)
        workflow.preprocess_phase()

    elif job_type == "pretrain":
        util.write_job_parameters(params=constants)
        workflow.training_phase()

    elif job_type == "transfer":
        util.write_job_parameters(params=constants)
        workflow.training_phase()

    elif job_type == "generate":
        util.write_job_parameters(params=constants)
        workflow.generation_phase()

    elif job_type == "test":
        util.write_job_parameters(params=constants)
        workflow.testing_phase()

    elif job_type == "rl":
        util.write_job_parameters(params=constants)
        workflow.rl_training_phase()

    else:
        raise NotImplementedError(
            f"Unknown job_type '{job_type}'. "
            "Valid options: preprocess, pretrain, transfer, generate, test, rl."
        )


if __name__ == "__main__":
    main()
