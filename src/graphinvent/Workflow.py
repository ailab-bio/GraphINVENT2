"""
Orchestrates all training, generation, and evaluation jobs in GraphINVENT2.

The `Workflow` class is the single entry point for every run mode supported
by the framework.  Each public method corresponds to one job type:

  preprocess_phase  -- convert SMILES files to HDF5 format for efficient loading
  training_phase    -- supervised (KL-divergence) training from random weights
                       (pretrain) or from a pretrained checkpoint (transfer)
  generation_phase  -- sample new molecules from a trained model
  testing_phase     -- evaluate a trained model on the held-out test set
  rl_training_phase -- optimize a pretrained model via policy-gradient RL
"""

# load general packages and functions
import datetime
import json
import math
import os
import pickle
import shutil
import time
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
from typing import Tuple, Union

import gnn.mpnn
import numpy as np
import torch
import torch.utils.tensorboard
import util
import yaml

# load GraphINVENT-specific functions
from Analyzer import Analyzer
from BlockDatasetLoader import BlockDataLoader, HDFDataset
from DataProcessor import DataProcessor, split_smiles_file
from GraphGenerator import GraphGenerator
from GraphGeneratorRL import GraphGeneratorRL
from rdkit import Chem
from ScoringFunction import ScoringFunction
from tqdm import tqdm


class Workflow:
    """
    Orchestrates all job types for the GraphINVENT2 molecular generation framework.

    Job types
    ---------
    preprocess  -- Reads a SMILES file, encodes each molecule as a sequence of
                   subgraphs (the decoding route), and writes node features, edge
                   features, and action probabilities targets to an HDF5 file for fast batch loading.

    pretrain    -- Trains a GGNN model from random initialization using supervised
                   learning: the model is trained to reproduce the target action probabilities at
                   each step of the decoding route (KL-divergence loss).

    transfer    -- Same supervised training loop as pretrain, but the model is
                   initialized from a pretrained checkpoint rather than random
                   weights.  Useful for fine-tuning on a new chemical series.

    generate    -- Uses a trained model to autoregressively sample new molecular
                   graphs by repeatedly drawing actions from the predicted action probabilities.

    test        -- Evaluates a trained model on the held-out test set and reports
                   NLL and related metrics.

    rl          -- Fine-tunes a pretrained model with augmented log-likelihood
                   policy-gradient RL.  Three model copies are maintained:
                   the agent (being optimized), a frozen prior (KL anchor), and
                   the best-agent-so-far (BASF, used to update the prior when the
                   agent improves).

    Args:
        constants: Experiment constants namedtuple loaded from params.json.
    """

    def __init__(self, constants: namedtuple) -> None:

        self.start_time = time.time()
        self.constants = constants

        # define number of accumulation steps
        self.accumulation_steps = self.constants.accumulation_steps

        # define path variables for various datasets
        self.test_h5_path = self.constants.test_set[:-3] + "h5"
        self.train_h5_path = self.constants.training_set[:-3] + "h5"
        self.valid_h5_path = self.constants.validation_set[:-3] + "h5"

        self.test_smi_path = self.constants.test_set
        self.train_smi_path = self.constants.training_set
        self.valid_smi_path = self.constants.validation_set

        # general paramters (placeholders)
        self.optimizer = None
        self.scheduler = None
        self.analyzer = None
        self.current_epoch = None
        self.restart_epoch = None

        # non-reinforcement learning parameters (placeholders)
        self.model = None
        self.training_set_properties = None
        self.test_dataloader = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.likelihood_per_action = None

        # reinforcement learning parameters (placeholders)
        self.agent_model = None
        self.prior_model = None
        self.best_agent_model = None  # tracks the highest-scoring model seen during RL
        self.best_avg_score = 0.0
        # Every molecule the agent generated, with its score, for the
        # goal-directed metrics at the end of the run.
        self.rl_smiles: list = []
        self.rl_scores: list = []
        # Graphs the agent generated, including those that are not molecules.
        # This is N for the validity denominator; counting only the ones that
        # converted would measure validity of the already-valid.
        self.rl_n_generated: int = 0
        self.scoring_function = None

    def preprocess_test_data(self) -> None:
        """
        Converts test dataset to HDF file format.
        """
        print("* Preprocessing test data.", flush=True)
        test_set_preprocesser = DataProcessor(path=self.constants.test_set)
        test_set_preprocesser.preprocess()

        self.print_time_elapsed()

    def preprocess_train_data(self) -> None:
        """
        Converts training dataset to HDF file format.
        """
        print("* Preprocessing training data.", flush=True)
        train_set_preprocesser = DataProcessor(
            path=self.constants.training_set, is_training_set=True
        )
        train_set_preprocesser.preprocess()

        self.print_time_elapsed()

    def preprocess_valid_data(self) -> None:
        """
        Converts validation dataset to HDF file format.
        """
        print("* Preprocessing validation data.", flush=True)
        valid_set_preprocesser = DataProcessor(path=self.constants.validation_set)
        valid_set_preprocesser.preprocess()

        self.print_time_elapsed()

    def get_dataloader(
        self, hdf_path: str, data_description: Union[str, None] = None
    ) -> torch.utils.data.DataLoader:
        """
        Loads preprocessed data (training, validation, or test set) into a
        PyTorch Dataloader.

        Args:
        ----
            data_path (str)        : Path to HDF data to be read.
            data_description (str) : Used for printing status (e.g. "test data").

        Returns:
        -------
            dataloader (torch.utils.data.DataLoader) : PyTorch Dataloader.
        """
        if data_description is None:
            data_description = "data"

        print(f"* Loading preprocessed {data_description}.", flush=True)
        dataset = HDFDataset(hdf_path)
        # pin_memory speeds up CPU→GPU transfers; only beneficial when using CUDA
        pin_memory = self.constants.device == "cuda"
        dataloader = BlockDataLoader(
            dataset=dataset,
            batch_size=self.constants.batch_size,
            block_size=self.constants.block_size,
            shuffle=True,
            n_workers=self.constants.n_workers,
            pin_memory=pin_memory,
        )
        self.print_time_elapsed()

        return dataloader

    def load_training_set_properties(self) -> None:
        """
        Loads the training sets properties from CSV into a dictionary. The
        training set properties are used during model evaluation.
        """
        filename = self.constants.training_set[:-3] + "csv"
        self.training_set_properties = util.load_training_set_properties(
            csv_path=filename
        )

    def define_model_and_optimizer(self) -> Tuple[int, int]:
        """
        Defines the model, optimizer, and scheduler, depending on the type of
        job, i.e., a regular training job, a restart job, or a fine-tuning job.

        Returns:
        -------
            start_epoch (int) : Epoch at which to start training.
            end_epoch (int)   : Epoch at which to end training.
        """

        job_type = self.constants.job_type

        if job_type == "goal_directed":
            # Reinforcement learning: load a pretrained checkpoint then set up
            # three model copies (agent, frozen prior, best-agent-so-far).
            print("* Defining models for RL fine-tuning.", flush=True)
            self.agent_model = self.create_model()
            self.prior_model = self.create_model()
            self.best_agent_model = self.create_model()

            self.restart_epoch = util.get_restart_epoch()

            prior_checkpoint = self.constants.pretrained_model_path

            _opt_state = None
            _sched_state = None

            if self.constants.restart:
                # Restart: resume the agent from the RL job checkpoint.
                print("-- Resuming RL agent from checkpoint.", flush=True)
                _ckpt_path = f"{self.constants.checkpoints_dir}model_restart_{self.restart_epoch}.pth"
                _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(_ckpt, dict) and "model" in _ckpt:
                    # New-format checkpoint with optimizer/scheduler states
                    self.agent_model.load_state_dict(_ckpt["model"])
                    _opt_state = _ckpt.get("optimizer")
                    _sched_state = _ckpt.get("scheduler")
                else:
                    # Legacy checkpoint: plain state_dict
                    self.agent_model.load_state_dict(_ckpt)
                # The prior is always the original pretrained model.
                print("-- Loading frozen prior from pretrained checkpoint.", flush=True)
                self.prior_model = util.load_saved_model(
                    model=self.prior_model, path=prior_checkpoint
                )
                self._freeze(self.prior_model)
            else:
                # Fresh RL start: load pretrained checkpoint as both agent and prior.
                print("-- Loading pretrained model checkpoint.", flush=True)
                self.agent_model = util.load_saved_model(
                    model=self.agent_model, path=prior_checkpoint
                )
                self.prior_model = self._freeze(deepcopy(self.agent_model))

            self.best_agent_model = self._freeze(deepcopy(self.agent_model))

            print("-- Defining optimizer.", flush=True)
            self.optimizer = torch.optim.Adam(
                params=self.agent_model.parameters(), lr=self.constants.init_lr
            )

            start_epoch = self.restart_epoch + 1
            end_epoch = start_epoch + self.constants.epochs

            print("-- Defining scheduler.", flush=True)
            # The number of RL steps depends on which loop will run:
            # `rl_training_phase` runs `epochs` steps, while the budget-capped
            # `constrained_rl_training_phase` runs until the oracle budget is
            # spent (one batch of molecules scored per step).  Sizing the cycle
            # from `epochs` in the budget-capped case makes scheduler.step()
            # overrun total_steps and raise part-way through training.
            if self.constants.oracle_budget is not None:
                n_rl_steps = math.ceil(
                    self.constants.oracle_budget / self.constants.batch_size
                )
            else:
                n_rl_steps = self.constants.epochs
            # ceil division: accounts for the final flush step when
            # n_rl_steps % accumulation_steps != 0 (avoids OneCycleLR ValueError).
            n_optimizer_steps = max(
                1, math.ceil(n_rl_steps / self.constants.accumulation_steps)
            )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.constants.max_rel_lr * self.constants.init_lr,
                div_factor=self.constants.max_rel_lr,
                final_div_factor=1.0 / self.constants.min_rel_lr,
                pct_start=0.05,
                total_steps=n_optimizer_steps,
            )

            if _opt_state is not None:
                print("-- Restoring optimizer state from checkpoint.", flush=True)
                self.optimizer.load_state_dict(_opt_state)
            if _sched_state is not None:
                print("-- Restoring scheduler state from checkpoint.", flush=True)
                self.scheduler.load_state_dict(_sched_state)

        elif (
            job_type in ("unconditional", "conditional") and self.constants.resume_from
        ):
            # Transfer learning or unconditional with resume_from: load a pretrained checkpoint,
            # then continue with supervised (KL-divergence) training on a new dataset.
            # Epoch counter resets to 1 for the new training run.
            print("* Defining model for transfer learning.", flush=True)
            self.model = self.create_model()

            print("-- Loading pretrained model checkpoint.", flush=True)
            self.model = util.load_saved_model(
                model=self.model, path=self.constants.resume_from
            )

            print("-- Defining optimizer.", flush=True)
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(), lr=self.constants.init_lr
            )

            self.restart_epoch = 0
            start_epoch = 1
            end_epoch = start_epoch + self.constants.epochs

            print("-- Defining scheduler.", flush=True)
            self.scheduler = self._make_supervised_scheduler()

        elif self.constants.restart:
            # Resume a previously interrupted pretrain or transfer job from
            # the last saved checkpoint in the same job directory.
            print("* Defining model (resuming from checkpoint).", flush=True)
            self.model = self.create_model()

            print("-- Loading model from previous checkpoint.", flush=True)
            self.restart_epoch = util.get_restart_epoch()
            _ckpt_path = f"{self.constants.checkpoints_dir}model_restart_{self.restart_epoch}.pth"
            self.model = util.load_saved_model(model=self.model, path=_ckpt_path)
            _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
            _opt_state = _ckpt.get("optimizer") if isinstance(_ckpt, dict) else None
            _sched_state = _ckpt.get("scheduler") if isinstance(_ckpt, dict) else None

            print("-- Defining optimizer.", flush=True)
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(), lr=self.constants.init_lr
            )

            # On a restart `epochs` is the TOTAL for the run, not a count of
            # additional epochs. Treating it as additional ran
            # restart_epoch + epochs epochs against a one-cycle schedule sized
            # for epochs, and OneCycleLR raised "Tried to step N times. The
            # specified number of total steps is M" part-way through the first
            # resumed epoch. A one-cycle schedule spans the whole run; there is
            # no second cycle to extend it with.
            start_epoch = self.restart_epoch + 1
            end_epoch = self.constants.epochs + 1
            if start_epoch >= end_epoch:
                raise ValueError(
                    f"restart: true, and the last checkpoint is at epoch "
                    f"{self.restart_epoch} of the configured epochs: "
                    f"{self.constants.epochs}, so there is nothing left to "
                    "run.\n`epochs` is the total for the run rather than a "
                    "number of additional epochs, because the learning-rate "
                    "schedule is one cycle over the whole run. To train "
                    "longer, raise `epochs` above "
                    f"{self.restart_epoch} or start a new job directory."
                )

            print("-- Defining scheduler.", flush=True)
            # Sized from the same total as the original run. The restored state
            # below carries total_steps too and wins, which is correct: the
            # cycle belongs to the run, not to this invocation.
            self.scheduler = self._make_supervised_scheduler()

            if _opt_state is not None:
                print("-- Restoring optimizer state from checkpoint.", flush=True)
                self.optimizer.load_state_dict(_opt_state)
            if _sched_state is not None:
                print("-- Restoring scheduler state from checkpoint.", flush=True)
                self.scheduler.load_state_dict(_sched_state)
                consumed = _sched_state.get("_step_count", 0)
                total = _sched_state.get("total_steps")
                if total is not None and consumed > total:
                    raise ValueError(
                        f"The checkpoint's learning-rate schedule is already "
                        f"complete: {consumed - 1} of {total} steps were taken "
                        "before it was saved.\nThat checkpoint came from a run "
                        "that finished its schedule, so there is no cycle left "
                        "to resume. Start a new job directory, or use "
                        "resume_from to fine-tune from this checkpoint with a "
                        "fresh schedule."
                    )
            else:
                print(
                    "-- Warning: checkpoint has no optimizer/scheduler state "
                    "(saved by an older version); both restart from scratch.",
                    flush=True,
                )

        else:
            # Pretraining from scratch (random weight initialization).
            print("* Defining model for pretraining from scratch.", flush=True)
            self.model = self.create_model()
            self.restart_epoch = 0

            print("-- Defining optimizer.", flush=True)
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(), lr=self.constants.init_lr
            )

            start_epoch = 1
            end_epoch = start_epoch + self.constants.epochs

            print("-- Defining scheduler.", flush=True)
            self.scheduler = self._make_supervised_scheduler()

        return start_epoch, end_epoch

    def _make_supervised_scheduler(
        self, total_epochs: int | None = None
    ) -> torch.optim.lr_scheduler.OneCycleLR:
        """
        Builds the OneCycle learning-rate schedule for supervised training.

        The number of scheduler steps must match how often `train_epoch`
        actually calls `scheduler.step()` -- once per `accumulation_steps`
        batches, plus one flush for the remainder -- not once per batch.
        Sizing the cycle by batch count instead makes the schedule advance only
        1/accumulation_steps of the way, so the LR never reaches `max_lr` and
        never anneals back down.

        `div_factor` / `final_div_factor` are set so the cycle starts at
        `init_lr`, peaks at `max_rel_lr * init_lr`, and ends at
        `min_rel_lr * init_lr`, matching the documented meaning of those
        parameters (and the RL branch above).

        `total_epochs` is the length of the WHOLE cycle, which on a restart is
        not the number of epochs still to run. One cycle spans the entire
        training run, so a resumed job restores a scheduler that has already
        consumed the steps of the epochs it completed; sizing the cycle for
        only the remaining epochs then overruns it and raises "Tried to step N
        times. The specified number of total steps is M" part-way through the
        first resumed epoch.
        """
        n_batches = len(self.train_dataloader)
        steps_per_epoch = max(1, math.ceil(n_batches / self.accumulation_steps))
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.constants.max_rel_lr * self.constants.init_lr,
            div_factor=self.constants.max_rel_lr,
            final_div_factor=1.0 / self.constants.min_rel_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=total_epochs or self.constants.epochs,
        )

    @staticmethod
    def _freeze(model: torch.nn.Module) -> torch.nn.Module:
        """Disable gradient tracking for all parameters of a frozen model."""
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def create_model(self) -> torch.nn.Module:
        """
        Initializes the model to be trained. Only the GGNN option is possible in
        GraphINVENT2.

        Returns:
        -------
            net (torch.nn.Module) : Neural net model.
        """
        net = gnn.mpnn.GGNN(constants=self.constants)

        if self.constants.device != "cpu":
            net = net.to(self.constants.device)

        return net

    def _backup_stale_preprocessing_files(self, mode_a: bool = False) -> None:
        """
        If .h5 or .h5.chunked files left over from a previous (failed or
        interrupted) preprocessing run are present in the dataset directory,
        move them into a timestamped backup subdirectory so the new run can
        start cleanly.

        When ``mode_a`` is True (single SMILES file with auto-split), any
        existing train/valid/test .smi files are also backed up, since they
        will be overwritten by the new split.
        """
        split_dir = self.constants.split_dir
        candidates = [
            "train.h5",
            "valid.h5",
            "test.h5",
            "train.h5.chunked",
            "valid.h5.chunked",
            "test.h5.chunked",
        ]
        if mode_a:
            candidates += ["train.smi", "valid.smi", "test.smi"]

        stale = [
            split_dir + name for name in candidates if os.path.exists(split_dir + name)
        ]
        if not stale:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = split_dir + f"_previous_run_{timestamp}/"
        os.makedirs(backup_dir, exist_ok=True)

        print(
            f"* Found {len(stale)} leftover file(s) from a previous "
            "preprocessing run.  Moving them to a backup directory before "
            "starting fresh.",
            flush=True,
        )
        print(f"  Backup location: {backup_dir}", flush=True)
        for path in stale:
            name = os.path.basename(path)
            shutil.move(path, backup_dir + name)
            print(f"  Moved: {name}", flush=True)

    def _backup_stale_generation_files(self) -> None:
        """
        Before a fresh generation run, move any existing output (generation/
        directory, *_samples.* files) to a timestamped backup
        subdirectory, matching the pattern used by preprocessing and training.
        """
        job_dir = self.constants.job_dir
        results_dir = self.constants.results_dir

        candidates = []
        gen_dir = results_dir + "generation"
        if os.path.isdir(gen_dir):
            candidates.append(gen_dir)
        for ext in (".smi", ".likelihood", ".valid"):
            f = results_dir + "molecules" + ext
            if os.path.exists(f):
                candidates.append(f)

        if not candidates:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = job_dir + f"_previous_run_{timestamp}/"
        os.makedirs(backup_dir, exist_ok=True)

        print(
            f"* Found {len(candidates)} file(s)/dir(s) from a previous generation "
            "run — moving to backup before starting fresh.",
            flush=True,
        )
        print(f"  Backup location: {backup_dir}", flush=True)
        for path in candidates:
            name = os.path.basename(path)
            shutil.move(path, backup_dir + name)
            print(f"  Moved: {name}", flush=True)

    def _concatenate_generation_batches(self) -> None:
        """
        After all generation batches are written, concatenate the per-batch
        batch_N.{smi,likelihood,valid} files into a single trio named
        {n_samples}_samples.{smi,likelihood,valid}, preserving line order
        (i.e. each line index corresponds to the same molecule across files).
        The individual batch files are removed afterwards.
        """
        import re as _re

        gen_dir = self.constants.results_dir + "generation/"

        batch_smis = sorted(
            Path(gen_dir).glob("batch_*.smi"),
            key=lambda p: int(_re.search(r"batch_(\d+)\.smi", p.name).group(1)),
        )
        if not batch_smis:
            return

        out_base = self.constants.results_dir + "molecules"
        for ext in (".smi", ".likelihood", ".valid"):
            with open(out_base + ext, "w") as out_f:
                for batch_smi in batch_smis:
                    batch_path = gen_dir + batch_smi.stem + ext
                    if not os.path.exists(batch_path):
                        continue
                    with open(batch_path) as in_f:
                        out_f.writelines(in_f.readlines())

        # remove individual batch files
        for batch_smi in batch_smis:
            for ext in (".smi", ".likelihood", ".valid"):
                p = gen_dir + batch_smi.stem + ext
                if os.path.exists(p):
                    os.remove(p)

        Path(gen_dir).rmdir()

        print(f"* Generated molecules written to: {out_base}.smi", flush=True)

    def _backup_stale_job_files(self) -> None:
        """
        If output files from a previous (non-restart) training, transfer, or RL
        job exist in the job directory, move them to a timestamped backup
        subdirectory so the new run starts cleanly.

        Backs up: convergence.log, validation.log,
        score.log (RL only), all model_restart_*.pth checkpoints, and
        the generation/ subdirectory.
        """
        job_dir = self.constants.job_dir
        results_dir = self.constants.results_dir
        logs_dir = self.constants.logs_dir
        checkpoints_dir = self.constants.checkpoints_dir
        job_type = self.constants.job_type

        # generation.log is not here: nothing writes it any more, metrics.json
        # superseded it, and a backup list naming a file that cannot exist
        # drifts from the set of files a job actually produces.
        log_names = ["convergence.log", "validation.log"]
        if job_type == "goal_directed":
            log_names.append("score.log")

        stale = [
            logs_dir + name for name in log_names if os.path.exists(logs_dir + name)
        ]
        # Collect any saved model checkpoints.
        stale += [str(p) for p in Path(checkpoints_dir).glob("model_restart_*.pth")]
        # Include the generation/ subdirectory if it exists.
        generation_dir = results_dir + "generation"
        if os.path.isdir(generation_dir):
            stale.append(generation_dir)
        # For constrained_rl / goal_directed with oracle_budget: also back up oracle checkpoint files and eval CSV.
        if job_type == "goal_directed":
            stale += [
                str(p) for p in Path(checkpoints_dir).glob("checkpoint_oracle_*.pth")
            ]
            if os.path.exists(results_dir + "oracle_eval.csv"):
                stale.append(results_dir + "oracle_eval.csv")

        if not stale:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = job_dir + f"_previous_run_{timestamp}/"
        os.makedirs(backup_dir, exist_ok=True)

        print(
            f"* Found {len(stale)} file(s)/dir(s) from a previous job run. "
            "Moving them to a backup directory before starting fresh.",
            flush=True,
        )
        print(f"  Backup location: {backup_dir}", flush=True)
        for path in stale:
            name = os.path.basename(path)
            shutil.move(path, backup_dir + name)
            print(f"  Moved: {name}", flush=True)

    # Changing one of these changes WHICH MOLECULES are in each split, so
    # rebuilding in place would leave a model already trained here evaluated
    # against a different test set. Fatal.
    #
    # max_n_nodes and use_chirality are here and not below because both act
    # before the split rather than after it: molecules above the cap are
    # dropped, and deduplication is keyed on the representation the model sees,
    # so lowering the cap or dropping the chiral tag moves molecules between
    # splits as surely as changing the seed does.
    SPLIT_IDENTITY_KEYS = (
        "strategy",
        "seed",
        "train_frac",
        "valid_frac",
        "max_n_nodes",
        "use_chirality",
    )

    # Changing one of these changes the HDF contents but not which molecule is
    # in which split. A rebuild from the same .smi files is correct and harms
    # nothing; only resuming a half-built HDF5 is wrong, so this returns False
    # rather than raising.
    VOCABULARY_KEYS = (
        "atom_types",
        "formal_charge",
        "imp_H",
        "chirality",
        "use_aromatic_bonds",
        "use_canon",
        "use_explicit_H",
        "ignore_H",
        "decoding_route",
    )

    def _guard_existing_split(self, split_dir: str) -> bool:
        """
        Decide whether an existing split set may be resumed, and raise if the
        split itself would be silently redefined.

        Returns False when there is no recorded split, which is a first build,
        or when the feature vocabulary has changed, which makes a half-built
        HDF5 unresumable but is otherwise harmless. Returns True when the
        recorded split and vocabulary both match, which is the legitimate
        re-run.

        Raises when a split-identity key differs. Rebuilding a split set under
        different parameters used to move the previous splits into
        ``_previous_run_<timestamp>/`` and write new ones, so a model trained
        against one split was then evaluated against another's test set with
        nothing failing. A split set is identified by split_type, seed,
        train_frac and valid_frac; changing any of them means a new dataset.
        """
        recorded = Path(split_dir) / "preprocessing.yaml"
        if not recorded.exists():
            return False

        with open(recorded) as f:
            saved = yaml.safe_load(f)
        saved.update(saved.pop("split", {}))

        # `strategy` is this file's name for split_type; the rest read straight
        # off constants. The file also holds bookkeeping written after the fact
        # (`run_info`, the split counts), which has no counterpart on
        # `constants` and would make every re-run look like a mismatch.
        current = {"strategy": self.constants.split_type}

        def differing(keys) -> dict:
            return {
                key: (saved[key], current.get(key, getattr(self.constants, key, None)))
                for key in keys
                if key in saved
                and saved[key] != current.get(key, getattr(self.constants, key, None))
            }

        redefined = differing(self.SPLIT_IDENTITY_KEYS)
        if redefined:
            lines = "\n".join(
                f"  {k}: recorded={v[0]!r}, this job={v[1]!r}"
                for k, v in redefined.items()
            )
            raise ValueError(
                f"{recorded} records a different split set than this job "
                "declares, and rebuilding it in place would leave every model "
                "already trained on it evaluated against a different test "
                f"split.\n  Mismatched keys:\n{lines}\n"
                "A split set is identified by split_type, seed, train_frac and "
                "valid_frac. Point this job at a new dataset_dir, or restore "
                "the parameters this split was built with."
            )

        stale_vocabulary = differing(self.VOCABULARY_KEYS)
        if stale_vocabulary:
            lines = "\n".join(
                f"  {k}: recorded={v[0]!r}, this job={v[1]!r}"
                for k, v in stale_vocabulary.items()
            )
            print(
                "* The split is unchanged but the feature vocabulary is not, so "
                "the existing HDF5 cannot be resumed and will be rebuilt from "
                f"the same splits.\n  Changed keys:\n{lines}",
                flush=True,
            )
            return False

        return True

    def preprocess_phase(self) -> None:
        """
        Preprocesses all the datasets (validation, training, and testing).

        If ``constants.smiles_file`` is set, the single SMILES file is first
        split into train/valid/test using the chosen strategy
        (``constants.split_type``).  Otherwise the three .smi files are expected
        to already exist in the dataset directory (Mode B).
        """
        split_dir = self.constants.split_dir
        os.makedirs(split_dir, exist_ok=True)

        # Only Mode A splits, so only Mode A can drop anything for being too
        # large; the other routes take the .smi files as given.
        self.n_oversized = 0

        smiles_file = getattr(self.constants, "smiles_file", None)

        # Runs unconditionally, not only under restart: a mismatch against an
        # existing split set is fatal however the job was launched, because the
        # harm is overwriting a split other jobs have already been evaluated
        # against. `restart` then decides only whether to resume this split's
        # half-built HDF files or rebuild them.
        existing = self._guard_existing_split(split_dir)
        restart = self.constants.restart and existing
        if self.constants.restart and not existing:
            print(
                "* restart=True but no preprocessing.yaml found in "
                f"{split_dir} — starting fresh.",
                flush=True,
            )

        if not restart:
            # Move any leftover files from a previous run out of the way.
            # In Mode A also back up existing .smi files since they'll be overwritten.
            self._backup_stale_preprocessing_files(mode_a=bool(smiles_file))

            # --- Mode A: split a single SMILES file ---
            if smiles_file:
                self.n_oversized = split_smiles_file(
                    smiles_file=smiles_file,
                    split_dir=split_dir,
                    split_type=self.constants.split_type,
                    train_frac=self.constants.train_frac,
                    valid_frac=self.constants.valid_frac,
                    seed=self.constants.seed,
                    max_n_nodes=self.constants.max_n_nodes,
                    condition_dim=self.constants.condition_dim,
                    conditioning=getattr(self.constants, "conditioning", None),
                    use_chirality=self.constants.use_chirality,
                )
            else:
                # --- Mode B: verify all three .smi files are present ---
                missing = [
                    name
                    for name in ("train.smi", "valid.smi", "test.smi")
                    if not os.path.exists(split_dir + name)
                ]
                if missing:
                    raise FileNotFoundError(
                        f"The following required file(s) are missing from "
                        f"{split_dir}:\n"
                        + "".join(f"  - {m}\n" for m in missing)
                        + "\nSet 'smiles_file' in your params.json for "
                        "automatic splitting, or provide the missing files."
                    )
            if os.path.exists(self.valid_smi_path):
                self.preprocess_valid_data()
            if os.path.exists(self.test_smi_path):
                self.preprocess_test_data()
            if os.path.exists(self.train_smi_path):
                self.preprocess_train_data()

        else:  # resume an interrupted preprocessing job with matching params
            # Determine where to resume based on which HDF files already exist.
            if os.path.exists(self.train_h5_path):
                print(
                    "-- All three HDF files appear complete. Nothing to resume.",
                    flush=True,
                )
            elif os.path.exists(self.train_h5_path + ".chunked") or os.path.exists(
                self.test_h5_path
            ):
                print(
                    "-- Resuming preprocessing from 'train.h5' "
                    "(valid.h5 and test.h5 appear complete).",
                    flush=True,
                )
                if os.path.exists(self.train_smi_path):
                    self.preprocess_train_data()
            elif os.path.exists(self.test_h5_path + ".chunked") or os.path.exists(
                self.valid_h5_path
            ):
                print(
                    "-- Resuming preprocessing from 'test.h5' "
                    "(valid.h5 appears complete).",
                    flush=True,
                )
                if os.path.exists(self.test_smi_path):
                    self.preprocess_test_data()
                if os.path.exists(self.train_smi_path):
                    self.preprocess_train_data()
            elif os.path.exists(self.valid_h5_path + ".chunked"):
                print("-- Resuming preprocessing from 'valid.h5'.", flush=True)
                if os.path.exists(self.valid_smi_path):
                    self.preprocess_valid_data()
                if os.path.exists(self.test_smi_path):
                    self.preprocess_test_data()
                if os.path.exists(self.train_smi_path):
                    self.preprocess_train_data()
            else:
                raise ValueError(
                    "restart=True but no in-progress HDF files were found in "
                    f"{split_dir}. Set 'restart': false to start from scratch."
                )

        self._preprocess_figure()

    def _preprocess_figure(self) -> None:
        """
        The heavy-atom distribution with `max_nodes` marked, and the molecule
        count per split.

        `max_nodes` is the one preprocessing parameter taken from the data
        rather than chosen, so the figure that shows where it falls in the
        distribution is the one that justifies it -- a tail past the line is
        molecules the representation cannot hold.

        A `preprocess` job writes no metrics.json, so this is the only output it
        produces beyond the HDF5 and the splits.
        """
        from metrics import _figures
        from metrics._run import read_smiles

        splits = {}
        for name, path in (
            ("train", self.train_smi_path),
            ("validation", self.valid_smi_path),
            ("test", self.test_smi_path),
        ):
            splits[name] = read_smiles(path) if os.path.exists(path) else []

        for figure in _figures.preprocess_figures(
            self.constants.results_dir,
            splits,
            self.constants.max_n_nodes,
            self.constants.split_type,
            self.constants.seed,
        ):
            print(f"* Figure written to {figure}", flush=True)

    def training_phase(self) -> None:
        """
        Trains model and generates graphs.
        """
        print("* Setting up training job.", flush=True)
        self.train_dataloader = self.get_dataloader(
            hdf_path=self.train_h5_path, data_description="training set"
        )
        self.valid_dataloader = self.get_dataloader(
            hdf_path=self.valid_h5_path, data_description="validation set"
        )

        self.load_training_set_properties()
        self.analyzer = Analyzer(
            valid_dataloader=self.valid_dataloader,
            train_dataloader=self.train_dataloader,
            start_time=self.start_time,
            create_tensorboard=self.constants.use_tensorboard,
        )
        if not self.constants.restart:
            self._backup_stale_job_files()
        self.create_output_files()

        start_epoch, end_epoch = self.define_model_and_optimizer()

        print("* Beginning training.", flush=True)
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            avg_train_loss = self.train_epoch()
            avg_valid_loss = self.validation_epoch()

            util.write_training_status(
                tb_writer=self.analyzer.tb_writer,
                epoch=self.current_epoch,
                lr=self.optimizer.param_groups[0]["lr"],
                training_loss=avg_train_loss,
                validation_loss=avg_valid_loss,
            )

            _ = self.evaluate_model(model_to_evaluate=self.model)

        # Checkpoints are otherwise written only by `evaluate_model`, on the
        # `sample_every` schedule, so a run with fewer epochs than
        # `sample_every` trained a model, reported success and saved nothing.
        # CLAUDE.md says this job type produces a checkpoint; this is the one
        # that guarantees it exists.
        final_path = (
            f"{self.constants.checkpoints_dir}model_restart_{self.current_epoch}.pth"
        )
        if not os.path.exists(final_path):
            print(
                f"* Saving final model state at Epoch {self.current_epoch}.", flush=True
            )
            torch.save(
                obj={
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                f=final_path,
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )

        from metrics import _run as metrics_run

        metrics_run.evaluate_training(
            self.constants,
            model=self.model,
            train_wall_clock_s=time.time() - self.start_time,
        )

        self.print_time_elapsed()

    def generation_phase(self) -> None:
        """
        Generates molecules using a pre-trained model.
        """
        print("* Setting up generation job.", flush=True)
        self._backup_stale_generation_files()
        self.load_training_set_properties()
        self.analyzer = Analyzer(
            valid_dataloader=None, train_dataloader=None, start_time=self.start_time
        )

        model_path = self.constants.pretrained_model_path

        os.makedirs(self.constants.results_dir + "generation/", exist_ok=True)

        print(f"* Loading model from: {model_path}", flush=True)
        self.model = self.create_model()
        self.model = util.load_saved_model(model=self.model, path=model_path)

        self.model.eval()
        generation_started = time.time()
        with torch.no_grad():
            self.sample_molecules(n_samples=self.constants.n_samples)

        self._concatenate_generation_batches()
        generation_seconds = time.time() - generation_started

        # Everything already written is safe at this point: the molecules are
        # on disk and this is the last step of the job.
        # delta_nll scores held-out molecules under each condition, which is a
        # model property and needs the model still loaded. Outside the
        # inference timing on purpose: it is not part of generating a sample
        # and would inflate inference_s_per_10k by the cost of k passes over
        # the test split.
        per_class_nll: dict = {}
        if getattr(self.constants, "condition_classes", None):
            with torch.no_grad():
                per_class_nll = self.per_class_nll()

        from metrics import _run as metrics_run

        metrics_run.evaluate_generation(
            self.constants,
            generated_smiles=metrics_run.read_smiles(
                f"{self.constants.results_dir}molecules.smi"
            ),
            model=self.model,
            inference_seconds=generation_seconds,
            n_generated_raw=getattr(self, "n_generated_raw", None),
            per_class_nll=per_class_nll,
        )

        self.print_time_elapsed()

    def testing_phase(self) -> None:
        """
        Evaluates model using test set data.
        """
        self.train_dataloader = self.get_dataloader(self.train_h5_path, "training set")
        self.valid_dataloader = self.get_dataloader(
            self.valid_h5_path, "validation set"
        )
        self.test_dataloader = self.get_dataloader(self.test_h5_path, "test set")
        self.load_training_set_properties()
        self.analyzer = Analyzer(
            valid_dataloader=self.valid_dataloader,
            train_dataloader=self.train_dataloader,
            start_time=self.start_time,
        )
        model_path = self.constants.pretrained_model_path
        print(f"* Loading model from: {model_path}", flush=True)
        self.model = self.create_model()
        self.model = util.load_saved_model(model=self.model, path=model_path)

        self.model.eval()
        with torch.no_grad():
            self.sample_molecules(n_samples=self.constants.n_samples, evaluation=True)

            print("* Evaluating model.", flush=True)
            self.analyzer.model = self.model
            self.analyzer.test_dataloader = self.test_dataloader
            scores = self.analyzer.evaluate_model(
                likelihood_per_action=self.likelihood_per_action
            )

        from metrics import _run as metrics_run

        metrics_run.evaluate_model_scoring(
            self.constants,
            nll={
                key: scores[key] for key in ("nll_train", "nll_validation", "nll_test")
            },
            uc_jsd=scores["uc_jsd"],
            model=self.model,
        )

        self.print_time_elapsed()

    def evaluate_model(
        self, model_to_evaluate: torch.nn.Module, label: str = ""
    ) -> Union[float, None]:
        """
        Evaluates model.

        For regular training jobs, evaluates the model every `sample_every`
        epochs by calculating the UC-JSD from generated structures. Saves model
        scores in `validation.log` and then saves model state.

        For fine-tuning jobs, evaluates the model every time function is called
        (i.e., every fine-tuning step) by computing the score of molecules
        generated by the specified model.

        Args:
        ----
            model_to_evaluate (torch.nn.Module) : Specific model to evaluate
                                                  (e.g. SummationMPNN).
            label (str)                         : Label to use for saving generated
                                                  structures from a specific graph
                                                  generation step.

        Returns:
        -------
            score (float) : Model score for fine-tuning job, otherwise simply None.
        """
        if self.constants.job_type == "goal_directed":
            model_to_evaluate.eval()
            with torch.no_grad():
                _, score = self.sample_molecules_rl(
                    model_a=model_to_evaluate,
                    model_b=self.prior_model,
                    tb_writer=self.analyzer.tb_writer,
                    is_agent=True,
                    model_a_label=label,
                )

                print(
                    f"* Saving model state at Epoch {self.current_epoch}.", flush=True
                )
                # `pickle.HIGHEST_PROTOCOL` good for large objects
                model_path = (
                    f"{self.constants.checkpoints_dir}"
                    f"model_restart_{self.current_epoch}.pth"
                )
                torch.save(
                    obj={
                        "model": model_to_evaluate.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    },
                    f=model_path,
                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                )

        elif self.current_epoch % self.constants.sample_every == 0:
            model_to_evaluate.eval()  # sets layers to eval mode (e.g. norm, dropout)
            with torch.no_grad():  # deactivates autograd engine
                self.sample_molecules(
                    n_samples=self.constants.n_samples, evaluation=True
                )
                score = None

                print(
                    f"* Saving model state at Epoch {self.current_epoch}.", flush=True
                )
                # `pickle.HIGHEST_PROTOCOL` good for large objects
                model_path = (
                    f"{self.constants.checkpoints_dir}"
                    f"model_restart_{self.current_epoch}.pth"
                )
                # Save optimizer/scheduler alongside the weights so a restart
                # resumes Adam's moments and the LR cycle instead of silently
                # resetting both (the RL branch above already does this).
                torch.save(
                    obj={
                        "model": model_to_evaluate.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    },
                    f=model_path,
                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                )

                print("* Evaluating model.", flush=True)
                self.analyzer.model = model_to_evaluate
                self.analyzer.evaluate_model(
                    likelihood_per_action=self.likelihood_per_action
                )

        else:
            # score not computer, so use placeholder
            util.write_training_status(tb_writer=self.analyzer.tb_writer, score="NA")
            score = None

        return score

    def rl_training_phase(self) -> None:
        """
        Fine-tunes model (`self.prior_model`) via policy gradient reinforcement
        learning (`self.agent_model`).
        """
        print("* Setting up RL fine-tuning job.", flush=True)
        self.load_training_set_properties()

        self.analyzer = Analyzer(
            valid_dataloader=None,
            train_dataloader=None,
            start_time=self.start_time,
            create_tensorboard=self.constants.use_tensorboard,
        )

        if not self.constants.restart:
            self._backup_stale_job_files()
        self.create_output_files()

        # define the scoring function to be used
        self.scoring_function = ScoringFunction(constants=self.constants)

        start_step, end_step = self.define_model_and_optimizer()

        # set current_epoch before the pre-fine-tuning evaluation so that the
        # checkpoint is saved with the correct step number (not None)
        self.current_epoch = start_step

        # evaluate model before fine-tuning
        score = self.evaluate_model(
            model_to_evaluate=self.agent_model, label="pre-fine-tuning"
        )

        # for fresh starts, create the log; for restarts, append to it
        self.analyzer.save_metrics(
            step=start_step, score=score, append=self.constants.restart
        )

        print("* Begin learning.", flush=True)

        accum = self.constants.accumulation_steps
        self.agent_model.zero_grad()
        self.optimizer.zero_grad()
        accum_counter = 0

        for step in range(start_step, end_step):
            self.current_epoch = step
            loss, score_a = self.rl_training_step()

            # scale loss and accumulate gradients
            (loss / accum).backward()
            accum_counter += 1

            if accum_counter % accum == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                accum_counter = 0

            util.write_training_status(
                tb_writer=self.analyzer.tb_writer,
                epoch=step,
                lr=self.optimizer.param_groups[0]["lr"],
                training_loss=loss.detach(),
                score=torch.mean(score_a).item(),
            )

            # Evaluate when either cadence is due: the logged trajectory
            # follows `rl_eval_every` and the best agent follows
            # `rl_best_agent_every`, and the two are independent settings.
            due_eval = step % self.constants.sample_every == 0
            due_best = step % int(self.constants.rl_best_agent_every) == 0
            if due_eval or due_best:
                score = self.evaluate_model(
                    model_to_evaluate=self.agent_model, label="eval"
                )
                if due_eval:
                    self.analyzer.save_metrics(step=step, score=score)
                self._maybe_update_best_agent(step, score)

        # flush any remaining accumulated gradients at end of training
        if accum_counter > 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self._write_optimization_log()
        self._write_goal_directed_metrics()

        self.print_time_elapsed()

    def _write_goal_directed_metrics(self) -> None:
        """metrics.json and report.html for a goal-directed run."""
        from metrics import _run as metrics_run

        log_path = Path(self.constants.results_dir) / "optimization_log.jsonl"
        optimization_log: list = []
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    optimization_log.append((entry["oracle_calls"], entry["score"]))

        metrics_run.evaluate_goal_directed(
            self.constants,
            generated_smiles=self.rl_smiles,
            n_generated=self.rl_n_generated,
            scores=self.rl_scores,
            optimization_log=optimization_log,
            model=self.agent_model,
            train_wall_clock_s=time.time() - self.start_time,
        )

    def constrained_rl_training_phase(self) -> None:
        """
        Oracle-budget-constrained RL fine-tuning.

        Runs the same augmented log-likelihood RL loop as rl_training_phase() but
        counts oracle calls instead of epochs. Saves checkpoints at milestones
        specified by ``constants.checkpoint_oracle_counts`` and stops once the
        total oracle calls exceed ``constants.oracle_budget``. After training,
        evaluates all saved checkpoints with a full metrics suite.

        Oracle calls are counted as: one oracle call = one molecule scored by the
        surrogate model during a training step (agent batch only).
        """
        print("* Setting up constrained_rl fine-tuning job.", flush=True)
        self.load_training_set_properties()

        self.analyzer = Analyzer(
            valid_dataloader=None,
            train_dataloader=None,
            start_time=self.start_time,
            create_tensorboard=self.constants.use_tensorboard,
        )

        if not self.constants.restart:
            self._backup_stale_job_files()
        self.create_output_files()

        self.scoring_function = ScoringFunction(constants=self.constants)

        start_step, _ = self.define_model_and_optimizer()

        # Oracle call counter
        oracle_calls = 0

        # Sorted list of checkpoint milestones still to save
        checkpoint_milestones = sorted(self.constants.checkpoint_oracle_counts)
        remaining_milestones = list(checkpoint_milestones)

        self.current_epoch = 0

        # Pre-fine-tuning evaluation
        score = self.evaluate_model(
            model_to_evaluate=self.agent_model, label="pre-fine-tuning"
        )
        self.analyzer.save_metrics(step=0, score=score, append=self.constants.restart)

        print("* Begin constrained_rl learning.", flush=True)
        print(
            f"* Oracle budget: {self.constants.oracle_budget} calls, "
            f"batch_size: {self.constants.batch_size}",
            flush=True,
        )

        accum = self.constants.accumulation_steps
        self.agent_model.zero_grad()
        self.optimizer.zero_grad()
        accum_counter = 0
        step = 0

        while oracle_calls < self.constants.oracle_budget:
            self.current_epoch = step
            loss, score_a = self.rl_training_step()

            # Ask the scoring function how many evaluations were actually
            # consumed.  Adding `batch_size` per step both ignored cache hits
            # and missed the second (best-agent-so-far) batch each step scores.
            oracle_calls = self.scoring_function.oracle_calls

            (loss / accum).backward()
            accum_counter += 1

            if accum_counter % accum == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                accum_counter = 0

            util.write_training_status(
                tb_writer=self.analyzer.tb_writer,
                epoch=step,
                lr=self.optimizer.param_groups[0]["lr"],
                training_loss=loss.detach(),
                score=torch.mean(score_a).item(),
            )

            # Periodic evaluation, on either cadence. See the supervised RL
            # loop above: the best agent has its own interval.
            due_eval = step % self.constants.sample_every == 0
            due_best = step % int(self.constants.rl_best_agent_every) == 0
            if due_eval or due_best:
                score = self.evaluate_model(
                    model_to_evaluate=self.agent_model, label="eval"
                )
                if due_eval:
                    self.analyzer.save_metrics(step=step, score=score)
                self._maybe_update_best_agent(step, score)

            # Save checkpoint when a milestone is crossed
            while remaining_milestones and oracle_calls >= remaining_milestones[0]:
                milestone = remaining_milestones.pop(0)
                ckpt_path = (
                    f"{self.constants.checkpoints_dir}checkpoint_oracle_{milestone}.pth"
                )
                print(
                    f"* Saving oracle checkpoint at {milestone} oracle calls "
                    f"(step {step}).",
                    flush=True,
                )
                torch.save(
                    obj={
                        "model": self.agent_model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "oracle_calls": oracle_calls,
                        "step": step,
                    },
                    f=ckpt_path,
                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                )
                self._write_optimization_log()

            step += 1

        # Flush remaining accumulated gradients
        if accum_counter > 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        print(
            f"* Oracle budget of {self.constants.oracle_budget} calls exhausted "
            f"after {step} steps ({oracle_calls} oracle calls).",
            flush=True,
        )
        self._write_optimization_log()

        # Post-training: evaluate all saved checkpoints
        self.constrained_rl_evaluate_checkpoints()
        self._write_goal_directed_metrics()

        self.print_time_elapsed()

    def _write_optimization_log(self) -> None:
        """
        Persists the oracle's (call_count, score) curve to
        ``optimization_log.jsonl`` in the job directory.

        The curve exists only in memory on the `CachedOracle`, so without this
        the PMO-style AUC Top-k cannot be computed after the run: the evaluation
        script found no file and silently reported 0.0.  Nothing is written when
        the scoring function uses no external oracle, since the call-indexed
        curve the metric integrates over is then undefined.
        """
        log = self.scoring_function.optimization_log
        if not log:
            return
        path = self.constants.results_dir + "optimization_log.jsonl"
        with open(path, "w") as f:
            for call_count, score in log:
                f.write(json.dumps({"oracle_calls": call_count, "score": score}) + "\n")
        print(
            f"* Wrote oracle optimization log ({len(log)} entries) to {path}.",
            flush=True,
        )

    def constrained_rl_evaluate_checkpoints(self) -> None:
        """
        After constrained_rl training completes, loads each saved checkpoint,
        generates ``eval_sample_size`` molecules, computes the full metrics suite,
        and writes results to ``oracle_eval.csv``.
        """
        print(
            "* Starting oracle checkpoint evaluation. These scorings are NOT "
            "charged to the oracle budget.",
            flush=True,
        )
        csv_path = self.constants.results_dir + "oracle_eval.csv"
        checkpoint_milestones = sorted(self.constants.checkpoint_oracle_counts)

        eval_model = self.create_model()
        eval_model.eval()

        for oracle_count in checkpoint_milestones:
            ckpt_path = (
                f"{self.constants.checkpoints_dir}checkpoint_oracle_{oracle_count}.pth"
            )
            if not os.path.exists(ckpt_path):
                print(
                    f"-- Checkpoint not found: {ckpt_path}, skipping.",
                    flush=True,
                )
                continue

            print(
                f"* Evaluating checkpoint at {oracle_count} oracle calls.",
                flush=True,
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                eval_model.load_state_dict(ckpt["model"])
            else:
                eval_model.load_state_dict(ckpt)

            if self.constants.device != "cpu":
                eval_model = eval_model.to(self.constants.device)

            # Generate eval_sample_size molecules in batches
            n_samples = self.constants.eval_sample_size
            batch_size = self.constants.batch_size
            n_batches = math.ceil(n_samples / batch_size)

            all_graphs = []
            all_terminations = []

            print(
                f"  Generating {n_batches} batches ({batch_size} molecules each)...",
                flush=True,
            )
            generator = GraphGenerator(model=eval_model, batch_size=batch_size)
            with torch.no_grad():
                for b in range(n_batches):
                    batch_graphs, _, _, termination = generator.sample()
                    all_graphs.extend(batch_graphs)
                    all_terminations.append(termination)
                    if (b + 1) % 10 == 0:
                        print(f"  Batch {b + 1}/{n_batches}", flush=True)

            # Truncate to exact n_samples
            all_graphs = all_graphs[:n_samples]
            termination_cat = torch.cat(all_terminations)[:n_samples]

            # Compute validity/uniqueness
            _, validity_tensor, uniqueness_tensor = util.write_graphs_to_smi(
                # Name matches what `experiments/goal_directed/evaluate_results.py`
                # globs for; it previously wrote generation/oracle_eval_<N>.smi
                # and the evaluator found nothing.
                smi_filename=(
                    f"{self.constants.results_dir}checkpoint_{oracle_count}_samples.smi"
                ),
                molecular_graphs_list=all_graphs,
                write=True,
            )

            # Compute scores with components, unmetered: this sample is
            # drawn from a budget checkpoint for reporting and is not part of
            # what the campaign spent.
            with self.scoring_function.unmetered():
                scores, component_scores = (
                    self.scoring_function.compute_score_with_components(
                        graphs=all_graphs,
                        termination=termination_cat,
                        validity=validity_tensor,
                        uniqueness=uniqueness_tensor,
                    )
                )

            # Full extended evaluation
            self.analyzer.evaluate_checkpoint_molecules(
                generated_graphs=all_graphs,
                validity_tensor=validity_tensor,
                scores=scores,
                component_scores=component_scores,
                oracle_count=oracle_count,
                csv_path=csv_path,
            )

        print(
            f"* Oracle checkpoint evaluation complete. Results in: {csv_path}",
            flush=True,
        )

    def rl_training_step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the RL loss for one training step (one molecule batch).

        Does NOT call backward or update model weights; gradient accumulation
        and the optimizer step are handled by `rl_training_phase`.

        Returns:
        -------
            loss    (torch.Tensor) : Unscaled scalar RL loss for this step.
            score_a (torch.Tensor) : Per-molecule scores from the agent batch.
        """
        print(f"* Learning step {self.current_epoch}.", flush=True)
        self.agent_model.train()
        self.prior_model.eval()
        self.best_agent_model.eval()

        # generate molecules with agent model
        loss_a, score_a = self.sample_molecules_rl(
            model_a=self.agent_model,
            model_b=self.prior_model,
            tb_writer=self.analyzer.tb_writer,
            is_agent=True,
            model_a_label="agent",
        )

        # generate molecules with best agent so far ("basf")
        loss_b, _ = self.sample_molecules_rl(
            model_a=self.best_agent_model,
            model_b=self.agent_model,
            tb_writer=self.analyzer.tb_writer,
            is_agent=False,
            model_a_label="BASF",
        )

        loss = (1 - self.constants.alpha) * loss_a + self.constants.alpha * loss_b
        return loss, score_a

    def create_output_files(self) -> None:
        """
        Creates output files (with appropriate headers) for new (i.e.
        non-restart) jobs. If restart a job, all new output will be appended
        to existing output files.
        """
        if not self.constants.restart:
            print("* Touching output files.", flush=True)
            # begin writing `convergence.log` file
            util.write_training_status(tb_writer=self.analyzer.tb_writer, append=False)

            # create `generation/` subdirectory to write generation output to
            os.makedirs(self.constants.results_dir + "generation/", exist_ok=True)

    def _maybe_update_best_agent(self, step: int, score) -> None:
        """
        Update the best-agent-so-far on ITS OWN cadence, `rl_best_agent_every`.

        CLAUDE.md fixes "best agent updated every 5 learning steps" as a
        hyperparameter, and this used to follow `rl_eval_every` instead,
        because the update sat inside the evaluation branch. `rl_best_agent_every`
        was declared in the defaults, documented, and read by nothing. The two
        coincide at their defaults of 5, so the behavior was correct and the
        parameter was dead; raising `rl_eval_every` to cut evaluation cost
        would silently have changed a fixed hyperparameter.
        """
        if step % int(self.constants.rl_best_agent_every) != 0:
            return
        if score > self.best_avg_score:
            self.best_avg_score = score
            self.best_agent_model = self._freeze(deepcopy(self.agent_model))
            print("-- Updated best model.", flush=True)

    def _drop_conditions(self, condition_vector):
        """
        Replace the condition with zeros for a fraction of the batch's rows.

        NOT A REGULARIZER, though it looks like one. Every conditional metric
        is reported against an unconditional baseline, which means the same
        model sampled with no condition set. With a one-hot condition, "no
        condition" is the all-zeros vector, and a model trained only on
        one-hot inputs has never seen it: the baseline would then measure what
        the model does when asked a question outside its training
        distribution, and the gap between conditional and baseline -- which is
        the entire claim -- would be partly an artifact of that.

        Dropping the condition on `condition_dropout` of the training examples
        makes the zero vector a legitimate input meaning "unspecified", so the
        baseline becomes a real measurement and the model keeps a usable
        unconditional mode.

        Training only. Validation keeps its conditions, because model
        selection should choose the best conditional model rather than the
        best average of two modes.

        The rows are drawn from a CPU generator seeded once from the job seed,
        so a rerun of the same job drops the same examples.
        """
        rate = float(getattr(self.constants, "condition_dropout", 0.0) or 0.0)
        if rate <= 0.0 or condition_vector is None:
            return condition_vector
        if getattr(self, "_condition_rng", None) is None:
            self._condition_rng = torch.Generator()
            self._condition_rng.manual_seed(int(self.constants.seed))
        keep = (
            torch.rand(condition_vector.size(0), generator=self._condition_rng) >= rate
        )
        keep = keep.to(condition_vector.device).view(-1, 1).to(condition_vector.dtype)
        return condition_vector * keep

    def _condition_requests(
        self, prop_names: list, sample_conds: dict, n_batches: int
    ) -> list:
        """
        The condition values to request, one dict per generation batch.

        A property whose configured value is the string "from_train" is spread
        across the deciles of its training distribution, so every decile of
        `condition_coverage_<property>` is exercised. Any other value is used
        as given for every batch.

        The requests are written to `results/condition_requests.json` so the
        metrics can attribute each generated molecule to what was asked of it.
        """
        import json as _json

        import numpy as _np

        from metrics._run import read_smiles_with_property

        # A categorical panel asked for as "all_classes" cycles the one-hot
        # across batches. CLAUDE.md reports the categorical metrics as a
        # macro-average over the panel, which needs molecules for every class,
        # and one class per job would mean one job per class and no way to
        # average them in a single metrics.json.
        if isinstance(sample_conds, str):
            if sample_conds.strip().lower() != "all_classes":
                raise ValueError(
                    f"sample_conditions: {sample_conds!r} is not a condition "
                    "request. Give a value per property, or 'all_classes' for "
                    "a categorical panel."
                )
            classes = list(getattr(self.constants, "condition_classes", None) or [])
            if not classes:
                raise ValueError(
                    "sample_conditions: all_classes needs condition_classes "
                    "set to the one-hot panel; without it there is no panel to "
                    "cycle."
                )
            requests = [
                {name: float(name == classes[idx % len(classes)]) for name in classes}
                for idx in range(max(1, n_batches))
            ]
            path = Path(self.constants.results_dir) / "condition_requests.json"
            path.write_text(_json.dumps(requests, indent=2) + "\n")
            return requests

        spread: dict = {}
        for name in prop_names:
            if str(sample_conds[name]).strip().lower() != "from_train":
                continue
            _, values = read_smiles_with_property(self.constants.training_set, name)
            finite = [v for v in values if v is not None and _np.isfinite(v)]
            if len(finite) < 10:
                raise ValueError(
                    f"sample_conditions[{name}] is 'from_train', but the "
                    f"training split at {self.constants.training_set} has "
                    f"{len(finite)} usable {name} values and ten deciles need "
                    "at least ten. Give an explicit value instead."
                )
            # Decile midpoints, so each request sits inside its own bucket.
            spread[name] = [
                float(_np.quantile(finite, q)) for q in _np.arange(0.05, 1.0, 0.1)
            ]

        requests = []
        for idx in range(max(1, n_batches)):
            request = {}
            for name in prop_names:
                if name in spread:
                    request[name] = spread[name][idx % len(spread[name])]
                else:
                    request[name] = float(sample_conds[name])
            requests.append(request)

        path = Path(self.constants.results_dir) / "condition_requests.json"
        path.write_text(_json.dumps(requests, indent=2) + "\n")
        return requests

    def per_class_nll(self) -> dict:
        """
        The model's NLL for each held-out molecule under each class in the panel.

        This is what `delta_nll` aggregates, and it is the one categorical
        metric that is a property of the model rather than of a sample: it asks
        whether a molecule of class p is more likely under condition p than
        under the competing conditions, which needs k forward passes over the
        same molecule with k different condition vectors.

        Returns the shape `metrics._run` expects, or an empty dict when the
        inputs are absent: a preprocessed test HDF5 carrying `condition_vector`
        and a declared `condition_classes`.

        The loader is built unshuffled on purpose. A molecule's NLL is the sum
        over its construction actions, the subgraphs of one molecule are
        contiguous in the HDF5, and the terminate column marks the last of
        them -- so the segmentation only works in file order, and the shuffled
        training loader would silently sum fragments of different molecules.
        """
        classes = list(getattr(self.constants, "condition_classes", None) or [])
        if not classes:
            return {}
        if not os.path.exists(self.test_h5_path):
            print(
                f"-- delta_nll needs a preprocessed test split at "
                f"{self.test_h5_path}, which does not exist; skipping.",
                flush=True,
            )
            return {}

        loader = BlockDataLoader(
            dataset=HDFDataset(self.test_h5_path),
            batch_size=self.constants.batch_size,
            block_size=self.constants.block_size,
            shuffle=False,
            n_workers=0,
            pin_memory=False,
        )
        softmax = torch.nn.Softmax(dim=1)
        one_hots = {
            name: torch.tensor(
                [[float(name == other) for other in classes]],
                dtype=torch.float32,
                device=self.constants.device,
            )
            for name in classes
        }

        per_molecule: dict = {}
        true_labels: list = []
        alternatives: dict = {}
        running = {name: 0.0 for name in classes}
        running_hot: list = []
        index = 0

        print(
            f"* Scoring the test split under each of {len(classes)} conditions "
            "for delta_nll.",
            flush=True,
        )
        with torch.no_grad():
            for batch in loader:
                if self.constants.device != "cpu":
                    batch = [b.to(self.constants.device) for b in batch]
                if len(batch) != 4:
                    print(
                        "-- delta_nll needs a test HDF5 carrying "
                        "condition_vector, and this one yields 3-tuples; "
                        "skipping.",
                        flush=True,
                    )
                    return {}
                nodes, edges, target_output, condition_vector = batch
                renormalized = target_output / torch.sum(
                    target_output, dim=1, keepdim=True
                ).clamp(min=1e-12)
                terminates = target_output[:, -1] > 0

                # One forward pass per class, over the whole batch. The stored
                # condition is replaced rather than ignored: the question is
                # what this molecule's NLL would be had the model been asked
                # for class q.
                nll_by_class = {}
                for name, one_hot in one_hots.items():
                    output = softmax(
                        self.model(
                            nodes,
                            edges,
                            condition_vector=one_hot.expand(nodes.size(0), -1),
                        )
                    )
                    correct = torch.sum(torch.mul(renormalized, output), dim=1)
                    nll_by_class[name] = (
                        -1 * torch.log(correct.clamp(min=1e-12))
                    ).cpu()

                hot = (condition_vector > 0.5).cpu()
                for row in range(nodes.size(0)):
                    for name in classes:
                        value = float(nll_by_class[name][row])
                        if value == value:  # not NaN
                            running[name] += value
                    running_hot.append(hot[row])
                    if not bool(terminates[row]):
                        continue
                    # The molecule ends here: its label is the class its stored
                    # condition vector marks, shared by all of its subgraphs.
                    marks = running_hot[0]
                    labels = [name for i, name in enumerate(classes) if bool(marks[i])]
                    if labels:
                        per_molecule[index] = dict(running)
                        true_labels.append(labels[0])
                        if len(labels) > 1:
                            alternatives[index] = labels[1:]
                        index += 1
                    running = {name: 0.0 for name in classes}
                    running_hot = []

        if not per_molecule:
            return {}
        return {
            "per_molecule": per_molecule,
            "true_labels": true_labels,
            "alternatives": alternatives,
            "n_alternatives_excluded": sum(len(v) for v in alternatives.values()),
        }

    def sample_molecules(self, n_samples: int, evaluation: bool = False) -> None:
        """
        Generates molecular graphs and evaluates them. Generates the graphs in
        batches of either the size of the mini-batches or `n_samples`, whichever
        is smaller.

        Args:
        ----
            n_samples (int)   : How many graphs to generate.
            evaluation (bool) : Indicates whether the model will be evaluated,
                                in which case we will also need the NLL per
                                action for the generated graphs.
        """
        print(f"* Generating {n_samples} molecules.", flush=True)
        # Every caller writes per-batch files here; `generation_phase` created
        # the directory but `testing_phase` did not, so sample_mode="evaluate"
        # died on a missing path after loading the model.
        os.makedirs(self.constants.results_dir + "generation/", exist_ok=True)
        generation_batch_size = min(self.constants.batch_size, n_samples)
        n_generation_batches = math.ceil(n_samples / generation_batch_size)

        # Build condition_vector from sample_conditions when conditioning is active.
        condition_vector = None
        if getattr(self.constants, "condition_dim", 0) > 0:
            sample_conds = getattr(self.constants, "sample_conditions", None)
            if sample_conds is None:
                raise ValueError(
                    "This is a conditional model (condition_dim="
                    f"{self.constants.condition_dim}) but 'sample_conditions' is "
                    'not set; add e.g. {"pLogS": -1.5} to the job config. '
                    "Without it the model would be sampled unconditionally, "
                    "silently ignoring the conditioning it was trained with."
                )
            # Order by the property list the model was trained on -- relying on
            # the JSON key order instead would silently permute the condition
            # vector if the user wrote the keys in a different order.
            conditioning = getattr(self.constants, "conditioning", None) or {}
            prop_names = conditioning.get("properties") or (
                list(getattr(self.constants, "condition_classes", None) or [])
                if isinstance(sample_conds, str)
                else list(sample_conds)
            )
            missing = (
                []
                if isinstance(sample_conds, str)
                else [k for k in prop_names if k not in sample_conds]
            )
            if missing:
                raise ValueError(
                    f"'sample_conditions' is missing value(s) for {missing}; "
                    f"this model was conditioned on {prop_names}."
                )
            # A property may ask for a single value, or for "from_train", which
            # spreads the requests across the deciles of the training
            # distribution. CLAUDE.md defines condition_coverage_<property> as a
            # per-decile success rate reported as ten numbers, and a single
            # broadcast value can only ever populate one decile and leave nine
            # null -- so the metric designed to expose "works only near the
            # middle of the distribution" could never fire.
            per_batch_requests = self._condition_requests(
                prop_names, sample_conds, n_generation_batches
            )
            cond_arr = np.array(
                [float(per_batch_requests[0][k]) for k in prop_names],
                dtype=np.float32,
            )
            condition_vector = torch.tensor(
                cond_arr, device=self.constants.device
            ).unsqueeze(0)

        generator = GraphGenerator(
            model=self.model,
            batch_size=generation_batch_size,
            condition_vector=condition_vector,
        )

        # A batch can come back larger than `batch_size`: observed 30 and 32
        # graphs from a batch_size of 20, so `n_generation_batches` batches can
        # exceed `n_samples`. Every funnel metric divides by `n_samples`, so
        # left untruncated a model with high validity reports validity above
        # 1.0 -- latent at the low validity of a briefly-trained model and
        # firing on exactly the runs worth reporting. Truncate to exactly
        # `n_samples` before anything measures the sample, and keep the raw
        # count so the over-production stays visible rather than hidden.
        n_emitted = 0
        self.n_generated_raw = 0

        # generate graphs in batches
        for idx in range(0, n_generation_batches):
            print("Batch", idx, "of", n_generation_batches)
            if condition_vector is not None:
                # One request per batch, so a stratified run covers the whole
                # distribution without the generator having to carry a
                # different condition per row through a shrinking batch.
                generator.condition_vector = torch.tensor(
                    np.array(
                        [float(per_batch_requests[idx][k]) for k in prop_names],
                        dtype=np.float32,
                    ),
                    device=self.constants.device,
                ).unsqueeze(0)

            # generate one batch of graphs
            graphs, action_likelihoods, final_loglikelihoods, termination = (
                generator.sample()
            )

            self.n_generated_raw += len(graphs)
            room = n_samples - n_emitted
            if len(graphs) > room:
                graphs = graphs[:room]
                final_loglikelihoods = final_loglikelihoods[:room]
                termination = termination[:room]
            n_emitted += len(graphs)

            # analyze properties of new graphs and save results
            self.analyzer.evaluate_generated_graphs(
                generated_graphs=graphs,
                loglikelihoods=final_loglikelihoods,
                generation_batch_idx=idx,
            )

            # keep track of NLLs per action; note that only NLLs for the first
            # batch are kept, as only a few are needed to evaluate the model
            # (more efficient than saving all)
            if evaluation and idx == 0:
                self.likelihood_per_action = action_likelihoods

            if n_emitted >= n_samples:
                break

    def sample_molecules_rl(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        tb_writer,
        is_agent: bool = False,
        model_a_label: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates molecular graphs during fine-tuning using two different models;
        these can be any pair of models, including the "agent", the "prior", or
        the "best-agent-so-far". The generated structures are then evaluated in
        terms of their validity and uniqueness, for use in the loss function
        (loss is not updated for invalid/duplicate molecules).

        Args:
        ----
            model_a (torch.nn.Module) : The first model, which is used for
                                        generating new molecular graphs.
            model_b (torch.nn.Module) : The second model, which is used only to
                                        compute the likelihood.
            is_agent (bool)           : Indicates if `model_a` is the agent model.
            model_a_label (str)       : Label to use when saving structures generated
                                        from `model_a`.

        Returns:
        -------
            loss_component (torch.Tensor) : Contribution to the loss from the
                                            sampled molecules.
            torch.Tensor : Average score for the sampled molecules.
        """
        n_samples = self.constants.batch_size  # how many molecules to generate

        print(f"* Generating {n_samples} molecules.", flush=True)
        if model_a_label != "":
            print(f"-- Model: {model_a_label}", flush=True)

        # GraphGeneratorRL uses agent_model/prior_model passed to .sample();
        # the constructor's `model` argument is unused during RL generation.
        generator = GraphGeneratorRL(model=None, batch_size=self.constants.batch_size)

        # generate one batch of graphs using `model_a`
        if is_agent:
            graphs, model_a_loglikelihoods, model_b_loglikelihoods, termination = (
                generator.sample(agent_model=model_a, prior_model=model_b)
            )
        else:
            graphs, model_b_loglikelihoods, model_a_loglikelihoods, termination = (
                generator.sample(agent_model=model_a, prior_model=model_b)
            )

        # The property analysis and its plots cost more than the learning step
        # itself and run twice per step, so they happen on the same schedule as
        # model evaluation. Validity and uniqueness are not optional: the loss
        # needs them every step, so they are always computed, and only the
        # files and the analysis are skipped.
        analyze = self.current_epoch % self.constants.sample_every == 0

        _, validity, uniqueness = util.write_molecules(
            molecules=graphs,
            final_likelihoods=model_a_loglikelihoods,
            epoch=f"Step {self.current_epoch} {model_a_label}",
            write=analyze,
            label=model_a_label,
        )

        # Compute scores with per-component breakdown
        scores, component_scores = self.scoring_function.compute_score_with_components(
            graphs=graphs,
            termination=termination,
            validity=validity,
            uniqueness=uniqueness,
        )

        # Analyze properties and save results (skip re-writing molecules)
        if analyze:
            validity, uniqueness = self.analyzer.evaluate_generated_graphs_rl(
                generated_graphs=graphs,
                termination=termination,
                agent_loglikelihoods=model_a_loglikelihoods,
                prior_loglikelihoods=model_b_loglikelihoods,
                training_set_properties=self.training_set_properties,
                step=self.current_epoch,
                is_agent=is_agent,
                label=model_a_label,
                scores=scores,
                component_scores=component_scores,
                precomputed_validity=validity,
                precomputed_uniqueness=uniqueness,
            )

        # Accumulated across the whole run, not just this step: the
        # goal-directed metrics are about everything the optimization
        # produced, and `best_at_<budget>` is cumulative by definition.
        # Learning-step batches only -- the evaluation and pre-fine-tuning
        # batches are the agent measuring itself, and folding them in would
        # mix two different samples into one reported number.
        if is_agent and model_a_label not in ("eval", "pre-fine-tuning"):
            self.rl_n_generated += len(graphs)
            for graph, score in zip(graphs, scores.tolist()):
                molecule = graph.get_molecule()
                # An invalid graph is not a molecule, and skipping it here is
                # how validity is measured: `rl_n_generated` counts everything
                # the policy produced and is the denominator N, while
                # `rl_smiles` holds only what became a molecule. This skip is
                # load-bearing, not defensive.
                #
                # The zero-atom case is the same thing wearing a disguise. A
                # graph that terminates before its first atom comes back as an
                # empty RWMol rather than None, and the old code dropped it
                # only because MolToSmiles returns "" for it and "" is falsy.
                # That happened to be the right outcome for the wrong reason;
                # it is stated here instead.
                if molecule is None or molecule.GetNumAtoms() == 0:
                    continue
                # Anything past that point is a sanitized RDKit molecule, so a
                # failure to write SMILES for it is a real defect rather than
                # an invalid generation. It used to be swallowed, which dropped
                # the molecule from the trajectory after `rl_n_generated` had
                # already counted it and so skewed every trajectory numerator
                # by a silent amount. Let it raise.
                smiles = Chem.MolToSmiles(molecule)
                if not smiles:
                    raise ValueError(
                        "MolToSmiles returned an empty string for a sanitized "
                        f"molecule with {molecule.GetNumAtoms()} atoms at RL "
                        f"step {self.current_epoch}. That is a bug in the graph "
                        "construction, not an invalid generation, which is "
                        "already handled above."
                    )
                self.rl_smiles.append(smiles)
                self.rl_scores.append(float(score))

        if is_agent:
            util.log_likelihoods_to_tensorboard(
                tb_writer=tb_writer,
                step=self.current_epoch,
                agent_loglikelihoods=-torch.clone(model_a_loglikelihoods),
                prior_loglikelihoods=-torch.clone(model_b_loglikelihoods),
            )
        else:
            uniqueness = torch.where(
                scores > self.best_avg_score,
                uniqueness,
                torch.zeros(len(scores), device=self.constants.device),
            )

        loss_component = torch.mean(
            self.compute_loss_component(
                scores=scores,
                agent_loglikelihoods=model_a_loglikelihoods,
                prior_loglikelihoods=model_b_loglikelihoods,
                uniqueness=uniqueness,
            )
        )

        return loss_component, torch.mean(scores)

    def print_time_elapsed(self) -> None:
        """
        Prints elapsed time since the program started running.
        """
        stop_time = time.time()
        elapsed_time = stop_time - self.start_time
        print(f"-- time elapsed: {elapsed_time:.5f} s", flush=True)

    def train_epoch(
        self,
    ) -> float:
        """
        Performs one training epoch.

        Returns:
        -------
            torch.Tensor : Average training loss.
        """
        print(f"* Training epoch {self.current_epoch}.", flush=True)
        training_loss_tensor = torch.zeros(
            len(self.train_dataloader), device=self.constants.device
        )

        self.model.train()  # ensure model is in train mode
        self.model.zero_grad()
        self.optimizer.zero_grad()
        accumulation_counter = 0  # initialize the accumulation counter
        for batch_idx, batch in tqdm(
            enumerate(self.train_dataloader), total=len(self.train_dataloader)
        ):
            if self.constants.device != "cpu":
                batch = [b.to(self.constants.device, non_blocking=True) for b in batch]

            if len(batch) == 4:
                nodes, edges, target_output, condition_vector = batch
                condition_vector = self._drop_conditions(condition_vector)
                output = self.model(nodes, edges, condition_vector=condition_vector)
            else:
                nodes, edges, target_output = batch
                output = self.model(nodes, edges)

            batch_loss = self.loss(output=output, target_output=target_output)
            training_loss_tensor[batch_idx] = (
                batch_loss.item()
            )  # use .item() to detach the loss value from the computation graph

            batch_loss = (
                batch_loss / self.accumulation_steps
            )  # scale the loss down by the number of accumulation steps
            batch_loss.backward()  # accumulate gradients

            accumulation_counter += 1
            if accumulation_counter % self.accumulation_steps == 0:
                self.optimizer.step()  # update parameters only after `accumulation_steps` batches
                self.scheduler.step()  # update the learning rate
                self.optimizer.zero_grad()  # clear gradients after updating
                accumulation_counter = 0  # reset the counter

        # ensure any remaining gradients are applied
        if accumulation_counter != 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return torch.mean(training_loss_tensor)

    def validation_epoch(self) -> torch.Tensor:
        """
        Performs one validation epoch.

        Args:
        ----
            return_batch (bool) : If True, returns the validation loss
                                  tensor for the entire batch.

        Returns:
        -------
            torch.Tensor : Average validation loss.
        """
        print(f"* Evaluating epoch {self.current_epoch}.", flush=True)
        validation_loss_tensor = torch.zeros(
            len(self.valid_dataloader), device=self.constants.device
        )

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self.valid_dataloader), total=len(self.valid_dataloader)
            ):
                if self.constants.device != "cpu":
                    batch = [
                        b.to(self.constants.device, non_blocking=True) for b in batch
                    ]

                if len(batch) == 4:
                    nodes, edges, target_output, condition_vector = batch
                    output = self.model(nodes, edges, condition_vector=condition_vector)
                else:
                    nodes, edges, target_output = batch
                    output = self.model(nodes, edges)

                batch_loss = self.loss(output=output, target_output=target_output)
                validation_loss_tensor[batch_idx] = batch_loss.item()

        return torch.mean(validation_loss_tensor)

    def loss(self, output: torch.Tensor, target_output: torch.Tensor) -> torch.Tensor:
        """
        The graph generation loss is the KL divergence between the target and
        predicted actions.

        Args:
        ----
            output (torch.Tensor)        : Predicted action probabilities tensor.
            target_output (torch.Tensor) : Target action probabilities tensor.

        Returns:
        -------
            loss (torch.Tensor) : Average loss for this output.
        """
        # define activation function; note that one must use the softmax in the
        # KLDiv, never the sigmoid, as the distribution must sum to 1
        output = torch.nn.functional.log_softmax(output, dim=1)

        # normalize the target output (as can contain information on > 1 graph);
        # clamp the denominator so an all-zero target row yields zeros rather
        # than NaNs, which would otherwise poison the whole batch's gradients
        target_output = target_output / torch.sum(
            target_output, dim=1, keepdim=True
        ).clamp(min=1e-12)

        return torch.nn.functional.kl_div(
            input=output, target=target_output, reduction="batchmean"
        )

    def compute_loss_component(
        self,
        scores: torch.Tensor,
        agent_loglikelihoods: torch.Tensor,
        prior_loglikelihoods: torch.Tensor,
        uniqueness: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the contributions to the loss from the log-likelihoods/scores
        of the two input models.

        Args:
        ----
            scores (torch.Tensor)               : Scores for sampled molecules based
                                                  on the user-defined scoring function.
            agent_loglikelihoods (torch.Tensor) : Log-likelihoods of generating
                                                  the sampled structures using the
                                                  agent model.
            prior_loglikelihoods (torch.Tensor) : Log-likelihoods of generating
                                                  the sampled structures using the
                                                  prior model.
            uniqueness (torch.Tensor)           : Vector specifying the uniqueness
                                                  of each sampled structure (1 -->
                                                  unique, 0 --> duplicate).

        Returns:
        -------
            torch.Tensor: The loss contributions from the input log-likelihoods.
        """
        augmented_prior_loglikelihoods = (
            prior_loglikelihoods + self.constants.sigma * scores
        )

        difference = agent_loglikelihoods - augmented_prior_loglikelihoods
        loss = difference * difference
        mask = (uniqueness != 0).int()
        loss = loss * mask

        # Loss modulation: reweight how much each molecule contributes to the
        # gradient by how much its property predictions can be trusted, leaving
        # the objective itself unchanged (Medina & Janet, arXiv:2606.24990,
        # Eq. 8).  The weights are already normalized to mean 1, so this
        # redistributes learning signal rather than rescaling it -- an
        # unnormalised weighting would act as a covert learning-rate change.
        weights = self.scoring_function.loss_weights(len(loss))
        if weights is not None:
            loss = loss * torch.tensor(weights, device=loss.device, dtype=loss.dtype)

        return loss

    def unconditional_training_phase(self) -> None:
        """
        Trains or fine-tunes an unconditional model.

        Replaces the former `pretrain` and `transfer` job types.
        When ``constants.resume_from`` is set, loads that checkpoint before
        training (equivalent to the old transfer job type).
        """
        self.training_phase()

    def goal_directed_training_phase(self) -> None:
        """
        RL fine-tuning with optional oracle-budget cap.

        Replaces the former `rl` (no budget) and `constrained_rl` (with budget)
        job types.  Set ``constants.oracle_budget`` to an integer to enable the
        budget-capped variant.
        """
        if self.constants.oracle_budget is None:
            self.rl_training_phase()
        else:
            self.constrained_rl_training_phase()

    def sample_phase(self) -> None:
        """
        Generates molecules or evaluates a trained model.

        `sample_mode` selects which.
        """
        mode = self.constants.sample_mode
        if mode == "generate":
            self.generation_phase()
        elif mode == "evaluate":
            self.testing_phase()
        else:
            raise ValueError(
                f"Unknown sample_mode '{mode}'. Must be 'generate' or 'evaluate'."
            )

    def conditional_training_phase(self) -> None:
        """
        Trains or fine-tunes a property-conditioning-aware generative model.

        The training data must have been preprocessed with ``condition_dim > 0``
        so that the HDF5 files contain a ``condition_vector`` dataset.  At each
        training step the condition vector is encoded into a virtual seed node
        embedding and injected into the GGNN message-passing rounds.

        All other aspects (optimizer, scheduler, evaluation, TensorBoard logging)
        are identical to ``training_phase``.
        """
        if self.constants.condition_dim <= 0:
            raise ValueError(
                "conditional job requires condition_dim > 0. "
                "Ensure the dataset was preprocessed with 'conditioning' enabled "
                "and that condition_dim is set in the job config."
            )

        # Validate that the training HDF5 has a condition_vector dataset.
        import h5py as _h5py

        with _h5py.File(self.train_h5_path, "r") as _f:
            if "condition_vector" not in _f:
                raise ValueError(
                    f"The training HDF5 file '{self.train_h5_path}' does not "
                    "contain a 'condition_vector' dataset. Re-run preprocessing "
                    "with condition_dim > 0 and a tab-separated input file."
                )

        # The standard training_phase already handles 4-tuple batches
        # (nodes, edges, target_output, condition_vector) transparently via the
        # HDFDataset / train_epoch / validation_epoch code paths.
        self.training_phase()
