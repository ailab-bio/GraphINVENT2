"""
Run one GraphINVENT2 job.

``submit.py`` resolves the configuration and calls :func:`run`. Nothing else
calls into this module.
"""

import datetime
import random

import numpy as np
import torch


def run(constants) -> None:
    """Run the job that *constants* describes."""
    import util
    from Workflow import Workflow

    util.suppress_warnings()

    print(
        f"* Job started at: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        flush=True,
    )

    if constants.seed > 0:
        random.seed(constants.seed)
        np.random.seed(constants.seed)
        torch.manual_seed(constants.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(constants.seed)
        print(f"* Random seed set to {constants.seed}", flush=True)

    workflow = Workflow(constants=constants)
    print(f"* Run mode: '{constants.job_type}'", flush=True)

    if constants.job_type == "preprocess":
        workflow.preprocess_phase()
        # Written only after the run succeeds. Persisting it first meant a
        # failed auto-detection left an empty vocabulary on disk, which every
        # later run then loaded in preference to re-detecting, failing far away
        # with "Input C not in allowable set []".
        util.write_preprocessing_parameters(
            params=constants, n_oversized=workflow.n_oversized
        )
    else:
        {
            "unconditional": workflow.unconditional_training_phase,
            "conditional": workflow.conditional_training_phase,
            "goal_directed": workflow.goal_directed_training_phase,
            "generate": workflow.sample_phase,
        }[constants.job_type]()
