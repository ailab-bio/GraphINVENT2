"""
Orchestrates all training, generation, and evaluation jobs in GraphINVENT2.

The `Workflow` class is the single entry point for every run mode supported
by the framework.  Each public method corresponds to one job type:

  preprocess_phase  -- convert SMILES files to HDF5 format for efficient loading
  training_phase    -- supervised (KL-divergence) training from random weights
                       (pretrain) or from a pretrained checkpoint (transfer)
  generation_phase  -- sample new molecules from a trained model
  testing_phase     -- evaluate a trained model on the held-out test set
  rl_training_phase -- optimise a pretrained model via policy-gradient RL
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
import torch
import torch.utils.tensorboard
import util

# load GraphINVENT-specific functions
from Analyzer import Analyzer
from BlockDatasetLoader import BlockDataLoader, HDFDataset
from DataProcessor import DataProcessor, split_smiles_file
from GraphGenerator import GraphGenerator
from GraphGeneratorRL import GraphGeneratorRL
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

    pretrain    -- Trains a GGNN model from random initialisation using supervised
                   learning: the model is trained to reproduce the target action probabilities at
                   each step of the decoding route (KL-divergence loss).

    transfer    -- Same supervised training loop as pretrain, but the model is
                   initialised from a pretrained checkpoint rather than random
                   weights.  Useful for fine-tuning on a new chemical series.

    generate    -- Uses a trained model to autoregressively sample new molecular
                   graphs by repeatedly drawing actions from the predicted action probabilities.

    test        -- Evaluates a trained model on the held-out test set and reports
                   NLL and related metrics.

    rl          -- Fine-tunes a pretrained model with augmented log-likelihood
                   policy-gradient RL.  Three model copies are maintained:
                   the agent (being optimised), a frozen prior (KL anchor), and
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

        job_dir = self.constants.job_dir
        job_type = self.constants.job_type

        if job_type in ("rl", "constrained_rl"):
            # Reinforcement learning: load a pretrained checkpoint then set up
            # three model copies (agent, frozen prior, best-agent-so-far).
            print("* Defining models for RL fine-tuning.", flush=True)
            self.agent_model = self.create_model()
            self.prior_model = self.create_model()
            self.best_agent_model = self.create_model()

            self.restart_epoch = util.get_restart_epoch()

            if self.constants.pretrained_model_path:
                prior_checkpoint = self.constants.pretrained_model_path
            else:
                prior_checkpoint = (
                    f"{self.constants.pretrained_model_dir}"
                    f"model_restart_{self.constants.generation_epoch}.pth"
                )

            _opt_state = None
            _sched_state = None

            if self.constants.restart:
                # Restart: resume the agent from the RL job checkpoint.
                print("-- Resuming RL agent from checkpoint.", flush=True)
                agent_dir = self.constants.job_dir
                _ckpt_path = f"{agent_dir}model_restart_{self.restart_epoch}.pth"
                _ckpt = torch.load(_ckpt_path, map_location="cpu")
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
                try:
                    self.prior_model = util.load_saved_model(
                        model=self.prior_model, path=prior_checkpoint
                    )
                except FileNotFoundError:
                    self.prior_model = util.load_saved_model(
                        model=self.prior_model,
                        path=f"{self.constants.dataset_dir}pretrained_model.pth",
                    )
                self._freeze(self.prior_model)
            else:
                # Fresh RL start: load pretrained checkpoint as both agent and prior.
                print("-- Loading pretrained model checkpoint.", flush=True)
                try:
                    self.agent_model = util.load_saved_model(
                        model=self.agent_model, path=prior_checkpoint
                    )
                except FileNotFoundError:
                    self.agent_model = util.load_saved_model(
                        model=self.agent_model,
                        path=f"{self.constants.dataset_dir}pretrained_model.pth",
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
            # ceil division: accounts for the final flush step when
            # epochs % accumulation_steps != 0 (avoids OneCycleLR ValueError).
            n_optimizer_steps = max(
                1,
                (self.constants.epochs + self.constants.accumulation_steps - 1)
                // self.constants.accumulation_steps,
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

        elif job_type in ("transfer", "unconditional") and getattr(
            self.constants, "resume_from", None
        ):
            # Transfer learning or unconditional with resume_from: load a pretrained checkpoint,
            # then continue with supervised (KL-divergence) training on a new dataset.
            # Epoch counter resets to 1 for the new training run.
            print("* Defining model for transfer learning.", flush=True)
            self.model = self.create_model()

            print("-- Loading pretrained model checkpoint.", flush=True)
            _resume_path = (
                getattr(self.constants, "resume_from", None)
                or self.constants.pretrained_model_path
            )
            if not _resume_path:
                _resume_path = (
                    f"{self.constants.pretrained_model_dir}"
                    f"model_restart_{self.constants.generation_epoch}.pth"
                )
            self.model = util.load_saved_model(model=self.model, path=_resume_path)

            print("-- Defining optimizer.", flush=True)
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(), lr=self.constants.init_lr
            )

            self.restart_epoch = 0
            start_epoch = 1
            end_epoch = start_epoch + self.constants.epochs

            print("-- Defining scheduler.", flush=True)
            n_batches = len(self.train_dataloader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.constants.max_rel_lr * self.constants.init_lr,
                steps_per_epoch=n_batches,
                epochs=self.constants.epochs,
            )

        elif self.constants.restart:
            # Resume a previously interrupted pretrain or transfer job from
            # the last saved checkpoint in the same job directory.
            print("* Defining model (resuming from checkpoint).", flush=True)
            self.model = self.create_model()

            print("-- Loading model from previous checkpoint.", flush=True)
            self.restart_epoch = util.get_restart_epoch()
            self.model = util.load_saved_model(
                model=self.model,
                path=f"{job_dir}model_restart_{self.restart_epoch}.pth",
            )

            print("-- Defining optimizer.", flush=True)
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(), lr=self.constants.init_lr
            )

            start_epoch = self.restart_epoch + 1
            end_epoch = start_epoch + self.constants.epochs

            print("-- Defining scheduler.", flush=True)
            n_batches = len(self.train_dataloader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.constants.max_rel_lr * self.constants.init_lr,
                steps_per_epoch=n_batches,
                epochs=self.constants.epochs,
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
            n_batches = len(self.train_dataloader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.constants.max_rel_lr * self.constants.init_lr,
                steps_per_epoch=n_batches,
                epochs=self.constants.epochs,
            )

        return start_epoch, end_epoch

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
        dataset_dir = self.constants.dataset_dir
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
            dataset_dir + name
            for name in candidates
            if os.path.exists(dataset_dir + name)
        ]
        if not stale:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = dataset_dir + f"_previous_run_{timestamp}/"
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
        directory, *_samples.* files, generation.log) to a timestamped backup
        subdirectory, matching the pattern used by preprocessing and training.
        """
        job_dir = self.constants.job_dir

        candidates = []
        gen_dir = job_dir + "generation"
        if os.path.isdir(gen_dir):
            candidates.append(gen_dir)
        for p in Path(job_dir).glob("*_samples.smi"):
            stem = p.stem
            for ext in (".smi", ".likelihood", ".valid"):
                f = job_dir + stem + ext
                if os.path.exists(f):
                    candidates.append(f)
        if os.path.exists(job_dir + "generation.log"):
            candidates.append(job_dir + "generation.log")

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

        gen_dir = self.constants.job_dir + "generation/"
        n = self.constants.n_samples

        batch_smis = sorted(
            Path(gen_dir).glob("batch_*.smi"),
            key=lambda p: int(_re.search(r"batch_(\d+)\.smi", p.name).group(1)),
        )
        if not batch_smis:
            return

        out_base = self.constants.job_dir + f"{n}_samples"
        for ext in (".smi", ".likelihood", ".valid"):
            with open(out_base + ext, "w") as out_f:
                for batch_smi in batch_smis:
                    batch_path = gen_dir + batch_smi.stem + ext
                    if os.path.exists(batch_path):
                        with open(batch_path) as in_f:
                            out_f.write(in_f.read())

        # remove individual batch files
        for batch_smi in batch_smis:
            for ext in (".smi", ".likelihood", ".valid"):
                p = gen_dir + batch_smi.stem + ext
                if os.path.exists(p):
                    os.remove(p)

        # move features.png to the job root then delete the now-empty generation/ dir
        features_src = Path(gen_dir) / "features.png"
        if features_src.exists():
            shutil.move(str(features_src), self.constants.job_dir + "features.png")
        try:
            Path(gen_dir).rmdir()  # only removes if empty; safe to ignore if not
        except OSError:
            pass

        print(f"* Generated molecules written to: {out_base}.smi", flush=True)

    def _backup_stale_job_files(self) -> None:
        """
        If output files from a previous (non-restart) training, transfer, or RL
        job exist in the job directory, move them to a timestamped backup
        subdirectory so the new run starts cleanly.

        Backs up: convergence.log, generation.log, validation.log,
        score.log (RL only), all model_restart_*.pth checkpoints, and
        the generation/ subdirectory.
        """
        job_dir = self.constants.job_dir
        job_type = self.constants.job_type

        log_names = ["convergence.log", "generation.log", "validation.log"]
        if job_type in ("rl", "constrained_rl", "goal_directed"):
            log_names.append("score.log")

        stale = [job_dir + name for name in log_names if os.path.exists(job_dir + name)]
        # Collect any saved model checkpoints.
        stale += [str(p) for p in Path(job_dir).glob("model_restart_*.pth")]
        # Include the generation/ subdirectory if it exists.
        generation_dir = job_dir + "generation"
        if os.path.isdir(generation_dir):
            stale.append(generation_dir)
        # For constrained_rl / goal_directed with oracle_budget: also back up oracle checkpoint files and eval CSV.
        if job_type in ("constrained_rl", "goal_directed"):
            stale += [str(p) for p in Path(job_dir).glob("checkpoint_oracle_*.pth")]
            if os.path.exists(job_dir + "oracle_eval.csv"):
                stale.append(job_dir + "oracle_eval.csv")

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

    def _check_restart_params_match(self, dataset_dir: str) -> bool:
        """
        Returns True if ``preprocessing_params.json`` in ``dataset_dir`` exists
        and all keys it contains match the current constants.

        If the file is absent or any value differs, prints a warning and returns
        False so the caller can fall back to a fresh start.
        """
        preproc_json = Path(dataset_dir) / "preprocessing_params.json"
        if not preproc_json.exists():
            print(
                "* restart=True but no preprocessing_params.json found in "
                f"{dataset_dir} — starting fresh.",
                flush=True,
            )
            return False

        with open(preproc_json) as f:
            saved = json.load(f)

        mismatches = {
            key: (saved[key], getattr(self.constants, key, None))
            for key in saved
            if saved[key] != getattr(self.constants, key, None)
        }
        if mismatches:
            lines = "\n".join(
                f"  {k}: saved={v[0]!r}, current={v[1]!r}"
                for k, v in mismatches.items()
            )
            print(
                "* restart=True but current parameters differ from the saved "
                "preprocessing — starting fresh instead.\n"
                f"  Mismatched keys:\n{lines}",
                flush=True,
            )
            return False

        print(
            "* restart=True and parameters match — resuming previous preprocessing.",
            flush=True,
        )
        return True

    def preprocess_phase(self) -> None:
        """
        Preprocesses all the datasets (validation, training, and testing).

        If ``constants.smiles_file`` is set, the single SMILES file is first
        split into train/valid/test using the chosen strategy
        (``constants.split_type``).  Otherwise the three .smi files are expected
        to already exist in the dataset directory (Mode B).
        """
        dataset_dir = self.constants.dataset_dir

        smiles_file = getattr(self.constants, "smiles_file", None)
        restart = self.constants.restart

        # If restart=True, validate that current params match the saved
        # preprocessing_params.json.  If they don't match, the dataset was
        # built with different settings — force a fresh start instead.
        if restart:
            restart = self._check_restart_params_match(dataset_dir)

        if not restart:
            # Move any leftover files from a previous run out of the way.
            # In Mode A also back up existing .smi files since they'll be overwritten.
            self._backup_stale_preprocessing_files(mode_a=bool(smiles_file))

            # --- Mode A: split a single SMILES file ---
            if smiles_file:
                split_smiles_file(
                    smiles_file=smiles_file,
                    dataset_dir=dataset_dir,
                    split_type=getattr(self.constants, "split_type", "random"),
                    train_frac=getattr(self.constants, "train_frac", 0.8),
                    valid_frac=getattr(self.constants, "valid_frac", 0.1),
                )
            else:
                # --- Mode B: verify all three .smi files are present ---
                missing = [
                    name
                    for name in ("train.smi", "valid.smi", "test.smi")
                    if not os.path.exists(dataset_dir + name)
                ]
                if missing:
                    raise FileNotFoundError(
                        f"The following required file(s) are missing from "
                        f"{dataset_dir}:\n"
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

            util.update_preprocessing_stats(dataset_dir)

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
                    f"{dataset_dir}. Set 'restart': false to start from scratch."
                )

            util.update_preprocessing_stats(dataset_dir)

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

        if self.constants.pretrained_model_path:
            model_path = self.constants.pretrained_model_path
        else:
            self.restart_epoch = self.constants.generation_epoch
            model_path = (
                f"{self.constants.job_dir}" f"model_restart_{self.restart_epoch}.pth"
            )

        os.makedirs(self.constants.job_dir + "generation/", exist_ok=True)

        print(f"* Loading model from: {model_path}", flush=True)
        self.model = self.create_model()
        self.model = util.load_saved_model(model=self.model, path=model_path)

        self.model.eval()
        with torch.no_grad():
            self.sample_molecules(n_samples=self.constants.n_samples)

        self._concatenate_generation_batches()

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
        self.restart_epoch = util.get_restart_epoch()

        print(
            f"* Loading model from previous saved state (Epoch "
            f"{self.restart_epoch}).",
            flush=True,
        )
        model_path = (
            f"{self.constants.job_dir}" f"model_restart_{self.restart_epoch}.pth"
        )
        self.model = self.create_model()
        self.model = util.load_saved_model(model=self.model, path=model_path)

        self.model.eval()
        with torch.no_grad():
            self.sample_molecules(n_samples=self.constants.n_samples, evaluation=True)

            print("* Evaluating model.", flush=True)
            self.analyzer.model = self.model
            self.analyzer.evaluate_model(
                likelihood_per_action=self.likelihood_per_action
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
        if self.constants.job_type in ("rl", "constrained_rl", "goal_directed"):
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
                    f"{self.constants.job_dir}"
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
                    f"{self.constants.job_dir}"
                    f"model_restart_{self.current_epoch}.pth"
                )
                torch.save(
                    obj=model_to_evaluate.state_dict(),
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

            # evaluate model every `sample_every` steps (not every step)
            if step % self.constants.sample_every == 0:
                score = self.evaluate_model(
                    model_to_evaluate=self.agent_model, label="eval"
                )

                # save the score to the analyzer
                self.analyzer.save_metrics(step=step, score=score)

                # check if agent's score is better than best score so far
                if score > self.best_avg_score:
                    self.best_avg_score = score

                    # update the best agent so far ("basf")
                    self.best_agent_model = self._freeze(deepcopy(self.agent_model))
                    print("-- Updated best model.", flush=True)

        # flush any remaining accumulated gradients at end of training
        if accum_counter > 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.print_time_elapsed()

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

            # Count oracle calls from the agent batch
            oracle_calls += self.constants.batch_size

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

            # Periodic evaluation
            if step % self.constants.sample_every == 0:
                score = self.evaluate_model(
                    model_to_evaluate=self.agent_model, label="eval"
                )
                self.analyzer.save_metrics(step=step, score=score)

                if score > self.best_avg_score:
                    self.best_avg_score = score
                    self.best_agent_model = self._freeze(deepcopy(self.agent_model))
                    print("-- Updated best model.", flush=True)

            # Save checkpoint when a milestone is crossed
            while remaining_milestones and oracle_calls >= remaining_milestones[0]:
                milestone = remaining_milestones.pop(0)
                ckpt_path = (
                    f"{self.constants.job_dir}" f"checkpoint_oracle_{milestone}.pth"
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

            step += 1

        # Flush remaining accumulated gradients
        if accum_counter > 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        print(
            f"* Oracle budget of {self.constants.oracle_budget} calls exhausted "
            f"after {step} steps.",
            flush=True,
        )

        # Post-training: evaluate all saved checkpoints
        self.constrained_rl_evaluate_checkpoints()

        self.print_time_elapsed()

    def constrained_rl_evaluate_checkpoints(self) -> None:
        """
        After constrained_rl training completes, loads each saved checkpoint,
        generates ``eval_sample_size`` molecules, computes the full metrics suite,
        and writes results to ``oracle_eval.csv``.
        """
        print("* Starting oracle checkpoint evaluation.", flush=True)
        csv_path = self.constants.job_dir + "oracle_eval.csv"
        checkpoint_milestones = sorted(self.constants.checkpoint_oracle_counts)

        eval_model = self.create_model()
        eval_model.eval()

        for oracle_count in checkpoint_milestones:
            ckpt_path = (
                f"{self.constants.job_dir}" f"checkpoint_oracle_{oracle_count}.pth"
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
            ckpt = torch.load(ckpt_path, map_location="cpu")
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
                f"  Generating {n_batches} batches "
                f"({batch_size} molecules each)...",
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
                smi_filename=(
                    f"{self.constants.job_dir}"
                    f"generation/oracle_eval_{oracle_count}.smi"
                ),
                molecular_graphs_list=all_graphs,
                write=True,
            )

            # Compute scores with components
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
            # begin writing `generation.log` file
            csv_path_and_filename = self.constants.job_dir + "generation.log"
            util.properties_to_csv(
                prop_dict=self.training_set_properties,
                csv_filename=csv_path_and_filename,
                epoch_key="Training set",
                tb_writer=self.analyzer.tb_writer,
                append=False,
            )

            # begin writing `convergence.log` file
            util.write_training_status(tb_writer=self.analyzer.tb_writer, append=False)

            # create `generation/` subdirectory to write generation output to
            os.makedirs(self.constants.job_dir + "generation/", exist_ok=True)

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
        generation_batch_size = min(self.constants.batch_size, n_samples)
        n_generation_batches = math.ceil(n_samples / generation_batch_size)

        # Build condition_vector from sample_conditions when conditioning is active.
        condition_vector = None
        if getattr(self.constants, "condition_dim", 0) > 0:
            sample_conds = getattr(self.constants, "sample_conditions", None)
            if sample_conds is not None:
                import numpy as _np

                cond_arr = _np.array(
                    [float(v) for v in sample_conds.values()], dtype=_np.float32
                )
                condition_vector = torch.tensor(
                    cond_arr, device=self.constants.device
                ).unsqueeze(0)

        generator = GraphGenerator(
            model=self.model,
            batch_size=generation_batch_size,
            condition_vector=condition_vector,
        )

        # generate graphs in batches
        for idx in range(0, n_generation_batches):
            print("Batch", idx, "of", n_generation_batches)

            # generate one batch of graphs
            graphs, action_likelihoods, final_loglikelihoods, termination = (
                generator.sample()
            )

            # analyze properties of new graphs and save results
            self.analyzer.evaluate_generated_graphs(
                generated_graphs=graphs,
                termination=termination,
                loglikelihoods=final_loglikelihoods,
                training_set_properties=self.training_set_properties,
                generation_batch_idx=idx,
            )

            # keep track of NLLs per action; note that only NLLs for the first
            # batch are kept, as only a few are needed to evaluate the model
            # (more efficient than saving all)
            if evaluation and idx == 0:
                self.likelihood_per_action = action_likelihoods

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

        # Compute validity/uniqueness first (needed for scoring)
        _, validity, uniqueness = util.write_molecules(
            molecules=graphs,
            final_likelihoods=model_a_loglikelihoods,
            epoch=f"Step {self.current_epoch} {model_a_label}",
            write=True,
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
    ) -> (
        float
    ):  # TODO here come back and create a separate function for when batch size is > than acc steps
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
                batch = [b.to(self.constants.device) for b in batch]

            if len(batch) == 4:
                nodes, edges, target_output, condition_vector = batch
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
                    batch = [b.to(self.constants.device) for b in batch]

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
        LogSoftmax = torch.nn.LogSoftmax(dim=1)
        output = LogSoftmax(output)

        # normalize the target output (as can contain information on > 1 graph)
        target_output = target_output / torch.sum(target_output, dim=1, keepdim=True)

        # define loss function and calculate the los
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        loss = criterion(target=target_output, input=output)

        return loss

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

        Replaces the former `generate` (sample_mode='generate') and `test`
        (sample_mode='evaluate') job types.
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
