"""
The `Analyzer` evaluates molecular sets produced at various stages of training.

Responsibilities
----------------
- Compute and log per-epoch metrics (NLL, UC-JSD, validity, uniqueness) for
  the validation set and the most recently generated molecules.
- Compute summary statistics of the training set (used as a reference during
  model evaluation and for normalizing scoring functions).
- Write scalar metrics and molecular property histograms to TensorBoard.
"""

# load general packages and functions
import os
import time
from typing import Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rdkit
import torch
import util

# load GraphINVENT-specific functions
from parameters.config import constants
from torch.utils.tensorboard import SummaryWriter

from metrics import (
    compute_core_metrics,
    compute_internal_diversity,
    compute_test_set_similarity,
)

# Pairwise Tanimoto over the generated set is O(n^2), so it runs on a
# subsample once the set is larger than this.
DIVERSITY_MAX_MOLECULES = 10000

# K for the top-K nearest-neighbor similarity against the test set.
TEST_SIMILARITY_TOP_K = 10


def _is_number(value) -> bool:
    """Whether a configured condition value is a number rather than a sentinel."""
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


class Analyzer:
    """
    Evaluates and logs molecular generation quality throughout training.

    The Analyzer is used in two contexts:

    1. **During training** (pass all four constructor args): after each epoch,
       `evaluate_model()` computes NLL on the validation and training sets,
       NLL on the freshly generated molecules, and the UC-JSD between the
       training and generated distributions.  All scalars are written to
       TensorBoard and to `validation.log`.

    2. **During preprocessing** (no constructor args needed): `evaluate_training_set()`
       computes summary statistics (property histograms, scaffold counts, etc.)
       for the training set molecules; these are saved alongside the HDF5 data
       and used as the reference distribution during evaluation.

    Args:
        valid_dataloader:   DataLoader for the validation split.
        train_dataloader:   DataLoader for the training split.
        start_time:         Wall-clock start time of the run (for elapsed-time logs).
        create_tensorboard: If True, open a TensorBoard `SummaryWriter`.
    """

    def __init__(
        self,
        valid_dataloader: Union[torch.utils.data.DataLoader, None] = None,
        train_dataloader: Union[torch.utils.data.DataLoader, None] = None,
        start_time: Union[time.time, None] = None,
        create_tensorboard: bool = False,
    ) -> None:

        self.valid_dataloader = valid_dataloader
        self.train_dataloader = train_dataloader
        self.start_time = start_time
        self.create_tensorboard = create_tensorboard
        if self.create_tensorboard:
            self.tb_writer = SummaryWriter(
                log_dir=constants.tensorboard_dir, flush_secs=10
            )
        else:
            self.tb_writer = None

        self.model = None  # placeholder

    def evaluate_model(self, likelihood_per_action: torch.Tensor) -> None:
        """
        Calculates the model score, which is the UC-JSD. Also calculates the
        mean NLL per action of the validation, training, and generated sets.
        Writes the scores to `validation.log`.

        Args:
        ----
            likelihood_per_action (torch.Tensor) : Contains NLLs per action for a
              batch of generated graphs.
        """

        def _uc_jsd(
            likelihood_valid: torch.Tensor,
            likelihood_train: torch.Tensor,
            likelihood_sampled: torch.Tensor,
        ) -> float:
            """
            Computes the UC-JSD (metric used for the benchmark of generative
            models in Arús-Pous, J. et al., J. Chem. Inf., 2019, 1-13).

            Args:
            ----
                likelihood_valid (torch.Tensor)   : NLLs for correct actions in
                                                    validation set.
                likelihood_train (torch.Tensor)   : NLLs for correct actions in
                                                    training set.
                likelihood_sampled (torch.Tensor) : NLLs for sampled actions in
                                                    the generated set.

            Returns:
            -------
                uc_jsd (float) : UC-JSD.
            """
            min_len = min(
                len(likelihood_valid), len(likelihood_sampled), len(likelihood_train)
            )

            # make all the distributions the same length (dim=0)
            likelihood_valid_norm = likelihood_valid[:min_len] / torch.sum(
                likelihood_valid[:min_len]
            )
            likelihood_train_norm = likelihood_train[:min_len] / torch.sum(
                likelihood_train[:min_len]
            )
            likelihood_sampled_norm = likelihood_sampled[:min_len] / torch.sum(
                likelihood_sampled[:min_len]
            )

            likelihood_sum = (
                likelihood_valid_norm + likelihood_train_norm + likelihood_sampled_norm
            ) / 3

            log_likelihood_sum = torch.log(likelihood_sum + 1e-8)
            uc_jsd = (
                torch.nn.functional.kl_div(
                    log_likelihood_sum, likelihood_valid_norm, reduction="sum"
                )
                + torch.nn.functional.kl_div(
                    log_likelihood_sum, likelihood_train_norm, reduction="sum"
                )
                + torch.nn.functional.kl_div(
                    log_likelihood_sum, likelihood_sampled_norm, reduction="sum"
                )
            ) / 3

            return float(uc_jsd)

        epoch_key = util.get_last_epoch()

        print("-- Calculating NLL statistics for validation set.", flush=True)
        valid_likelihood_list, avg_valid_likelihood = self.get_validation_likelihood(
            dataset="validation"
        )

        print("-- Calculating NLL statistics for training set.", flush=True)
        train_likelihood_list, avg_train_likelihood = self.get_validation_likelihood(
            dataset="training"
        )

        # Average final NLL for the generation set.  `likelihood_per_action`
        # holds raw action *probabilities*, whereas the train/valid figures are
        # NLLs, so it must be negated-log'd before the three are compared or
        # written side by side.  It is also captured for the first generation
        # batch only, so the divisor is that batch's size, not `n_samples`.
        likelihood_per_action = -torch.log(
            torch.as_tensor(likelihood_per_action).clamp(min=1e-12)
        )
        n_generated = min(constants.batch_size, constants.n_samples)
        avg_gen_likelihood = torch.sum(likelihood_per_action) / n_generated

        # initialize dictionary with NLL statistics
        model_scores = {
            "likelihood_val": valid_likelihood_list,
            "avg_likelihood_val": avg_valid_likelihood,
            "likelihood_train": train_likelihood_list,
            "avg_likelihood_train": avg_train_likelihood,
            "likelihood_gen": likelihood_per_action,
            "avg_likelihood_gen": avg_gen_likelihood,
        }

        # get the UC-JSD and add it to the dictionary
        model_scores["UC-JSD"] = _uc_jsd(
            likelihood_valid=model_scores["likelihood_val"],
            likelihood_train=model_scores["likelihood_train"],
            likelihood_sampled=model_scores["likelihood_gen"],
        )

        # write results to disk
        import os as _os

        _val_log = constants.logs_dir + "validation.log"
        util.write_validation_scores(
            output_dir=constants.logs_dir,
            epoch_key=epoch_key,
            model_scores=model_scores,
            tb_writer=self.tb_writer,
            append=_os.path.exists(_val_log),
        )
        util.write_training_status(
            tb_writer=self.tb_writer, score=model_scores["UC-JSD"]
        )

        # Returned, not just logged: metrics.json needs these under CLAUDE.md's
        # names, and recomputing them there would be a second implementation.
        nll_test = None
        if getattr(self, "test_dataloader", None) is not None:
            print("-- Calculating NLL statistics for test set.", flush=True)
            _, nll_test = self.get_validation_likelihood(dataset="test")
        return {
            "nll_train": float(avg_train_likelihood),
            "nll_validation": float(avg_valid_likelihood),
            "nll_test": None if nll_test is None else float(nll_test),
            "uc_jsd": float(model_scores["UC-JSD"]),
        }

    def evaluate_generated_graphs(
        self,
        generated_graphs: list,
        loglikelihoods: torch.Tensor,
        generation_batch_idx: int,
    ) -> None:
        """
        Write one batch of generated graphs to disk as the aligned
        `.smi`/`.likelihood`/`.valid` trio.

        This used to also compute molecular properties and append a row to
        `logs/generation.log`, and it did both for the first batch only, because
        the properties were expensive. That made every logged rate describe the
        first `batch_size` molecules while the `.smi` file held the whole
        sample: at the reporting convention of N = 10,000 with a batch of 1,000,
        a tenth of the sample reported as the run's result. `results/metrics.json`
        now computes every metric once over the whole sample, so the log, its
        three first-batch guards and the two figures drawn from it are gone
        rather than fixed.

        What remains is the part nothing else does: writing the molecules.

        Args:
        ----
            generated_graphs (list)    : Contains `GenerationGraph`s.
            loglikelihoods (torch.Tensor) : Final NLL of each graph.
            generation_batch_idx (int) : Index of this batch in the sample.
        """
        epoch_key = util.get_last_epoch()

        if constants.job_type == "generate":
            label = f"batch_{generation_batch_idx}"
        else:
            label = f"epoch_{epoch_key[6:]}_batch_{generation_batch_idx}"

        util.write_molecules(
            molecules=generated_graphs,
            final_likelihoods=loglikelihoods,
            epoch=epoch_key,
            write=True,
            label=label,
            # Every batch holds the same number of graphs, so this is the
            # index of its first graph in the whole generated set.
            index_offset=generation_batch_idx * len(generated_graphs),
        )

    def evaluate_generated_graphs_rl(
        self,
        generated_graphs: list,
        termination: torch.Tensor,
        agent_loglikelihoods: torch.Tensor,
        prior_loglikelihoods: torch.Tensor,
        training_set_properties: dict,
        step: int,
        is_agent: bool = False,
        label: str = "",
        scores=None,
        component_scores=None,
        precomputed_validity=None,
        precomputed_uniqueness=None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """
        Computes molecular properties for input set of generated graphs, saves
        results to CSV, and writes `generated_graphs` to disk as a SMILES file.
        Properties are expensive to calculate, so only done for the first batch
        of generated molecules.

        Args:
        ----
            generated_graphs (list)             : Contains `GenerationGraph`s.
            termination (torch.Tensor)          : Molecular termination details;
                                                  contains 1 at index if graph from
                                                  `generated_graphs` was "properly"
                                                  terminated, 0 otherwise.
            agent_loglikelihoods (torch.Tensor) : Contains final NLL of each item
                                                  in `generated_graphs` (agent).
            prior_loglikelihoods (torch.Tensor) : Contains final NLL of each item
                                                  in `generated_graphs` (prior).
            training_set_properties (dict)      : Contains training set properties.
            step (int)                          : Training step.
            is_agent (bool)                     : Indicates whether the `agent_loglikelihoods`
                                                  correspond to the agent.
            label (str)                         : Label to use for saving files.
            scores                              : Optional pre-computed score tensor.
            component_scores                    : Optional dict of per-component score tensors.
            precomputed_validity                : Optional pre-computed validity tensor.
            precomputed_uniqueness              : Optional pre-computed uniqueness tensor.

        Returns:
        -------
            validity (torch.Tensor)   : Indicates the validity of the generated
                                        structures with a 1 for valid, 0 for invalid.
            uniqueness (torch.Tensor) : Indicates the uniqueness of the generated
                                        structures with a 1 for unique (and or first
                                        duplicate), and 0 for duplicate.
        """
        # epoch_key = util.get_last_epoch()
        epoch_key = f"Step {step} {label}"

        # calculate molecular properties of generated set
        prop_dict = self.get_molecular_properties(
            molecules=generated_graphs, epoch_key=epoch_key, termination=termination
        )

        # add a few additional properties to the propery dictionary
        prop_dict[(epoch_key, "final_agent_loglikelihood")] = agent_loglikelihoods
        prop_dict[(epoch_key, "final_prior_loglikelihood")] = prior_loglikelihoods
        prop_dict[(epoch_key, "run_time")] = round(time.time() - self.start_time, 2)

        if precomputed_validity is not None:
            validity = precomputed_validity
            uniqueness = precomputed_uniqueness
            n = len(generated_graphs)
            fraction_valid = float(validity.float().sum().item()) / n if n > 0 else 0.0
        else:
            fraction_valid, validity, uniqueness = util.write_molecules(
                molecules=generated_graphs,
                final_likelihoods=agent_loglikelihoods,
                epoch=epoch_key,
                write=True,
                label=label,
            )
        prop_dict[(epoch_key, "fraction_valid")] = fraction_valid
        prop_dict[(epoch_key, "validity_tensor")] = validity
        prop_dict[(epoch_key, "uniqueness_tensor")] = uniqueness

        # compute ±std across 3 molecule subsets
        stds = self._compute_error_bars(generated_graphs, termination)
        for k, v in stds.items():
            prop_dict[(epoch_key, k)] = v

        # Compute extended metrics (novelty, SA score, success rate) for agent batch
        _extra_cols = [
            "novelty",
            "sa_score_mean",
            "sa_score_median",
        ]
        _extra_cols += [
            "internal_diversity",
            "mean_internal_similarity",
            "max_internal_similarity",
            "internal_diversity_successful",
        ]
        if is_agent:
            ext = self._compute_extended_metrics(
                generated_graphs=generated_graphs,
                validity_tensor=validity,
                scores=scores,
            )
            for k, v in ext.items():
                prop_dict[(epoch_key, k)] = v
            if component_scores is not None:
                for comp_name, comp_tensor in component_scores.items():
                    _comp_key = f"score_{comp_name}_mean"
                    prop_dict[(epoch_key, _comp_key)] = float(
                        comp_tensor.float().mean().item()
                    )
                    if _comp_key not in _extra_cols:
                        _extra_cols.append(_comp_key)

        # The per-step series goes to TensorBoard, which CLAUDE.md names as the
        # place to watch a run, rather than to logs/generation.log, which is
        # gone. Every scalar in prop_dict is written, not the five
        # properties_to_csv used to forward: internal_diversity, novelty and
        # fraction_unique are the ones that matter, because the collapse of
        # diversity over optimization steps is what the metric suite exists to
        # make visible, and they were CSV-only before.
        self._rl_step_scalars_to_tensorboard(prop_dict, epoch_key)

        return validity, uniqueness

    def _rl_step_scalars_to_tensorboard(self, prop_dict: dict, epoch_key: str) -> None:
        """
        Write every scalar for one RL evaluation step to TensorBoard.

        `use_tensorboard` is false by default, so this is silent unless the job
        asks for it. The three goal-directed experiment directories set it to
        true for that reason: without it an optimization run keeps no per-step
        record of validity, uniqueness, novelty or internal diversity at all,
        and those four are the trajectory the diversity-collapse figure is
        drawn from.
        """
        if self.tb_writer is None:
            return

        # "Step 12 agent" -> 12. The label carries the model as well as the
        # step, so the model becomes part of the tag and the step the x-axis.
        parts = epoch_key.split()
        try:
            step = int(parts[1])
        except (IndexError, ValueError):
            return
        model_label = "_".join(parts[2:]) or "agent"

        for (key_epoch, name), value in prop_dict.items():
            if key_epoch != epoch_key:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self.tb_writer.add_scalar(f"RL/{model_label}/{name}", float(value), step)

    def evaluate_training_set(self, preprocessing_graphs: list) -> dict:
        """
        Computes molecular properties for structures in training set.

        Args:
        ----
            preprocessing_graphs (list) : Contains `PreprocessingGraph`s.

        Returns:
        -------
            training_set_properties (dict) : Dictionary of training set
                                             molecular properties.
        """
        training_set_properties = self.get_molecular_properties(
            molecules=preprocessing_graphs, epoch_key="Training set"
        )
        return training_set_properties

    def get_molecular_properties(
        self,
        molecules: list,
        epoch_key: str,
        termination: Union[torch.Tensor, None] = None,
    ) -> dict:
        """
        Calculates properties for input `molecules` (`list` of
        `MolecularGraph`s). Properties include the distribution in number of
        nodes per molecule, the distribution of atom types, the distribution of
        edge features (bond types), the distribution of the chirality (if used),
        and the fraction of unique molecules.

        Args:
        ----
            molecules (list)           : `PreprocessingGraph`s or `GenerationGraph`s,
                                          depending on job type.
            epoch_key (str)            : For example, "Training set" or "Epoch {n}".
            termination (torch.Tensor) : If specified, contains molecular termination
                                         details for generated graphs; contains 1
                                         at index if graph was "properly" terminated,
                                         0 otherwise.

        Returns:
        -------
            properties (dict) : Contains properties of generated and training
                                set molecules. Keys are string tuples, e.g. ("Training set",
                                "{property}") or ("Epoch {n}", "{property}").
        """

        def _get_n_edges_distribution(
            molecular_graphs: list, n_edges_to_bin: int = 10
        ) -> Tuple[torch.Tensor, float]:
            """
            Returns a histogram of the number of edges per node present in the
            `molecular_graphs`. The histogram is a `list` where the first item
            corresponds to the count of the number of nodes with one edge, the
            second item to the count of the number of nodes with two edges, etc,
            up until the count of the number of nodes with `n_edges_to_bin`
            edges. Also returns the average number of edges per node.
            """
            # initialize and populate histogram (last bin is for # num edges >
            # `n_edges_to_bin`)
            n_edges_histogram = torch.zeros(n_edges_to_bin, device=constants.device)
            for molecular_graph in molecular_graphs:
                edges = molecular_graph.edge_features
                for node_idx in range(molecular_graph.n_nodes):
                    n_edges = 0
                    for bond_type in range(constants.n_edge_features):
                        try:
                            n_edges += int(torch.sum(edges[node_idx, :, bond_type]))
                        except TypeError:  # if edges is `np.ndarray`
                            n_edges += int(np.sum(edges[node_idx, :, bond_type]))
                    if n_edges == 0:
                        # an isolated node has no bond to bin; `n_edges - 1`
                        # would index -1, i.e. the ">= n_edges_to_bin" bin
                        continue
                    if n_edges > n_edges_to_bin:
                        n_edges = n_edges_to_bin

                    n_edges_histogram[n_edges - 1] += 1

            # compute average number of edges per node
            sum_n_edges = 0
            for n_edges, count in enumerate(n_edges_histogram, start=1):
                sum_n_edges += n_edges * count

            try:
                avg_n_edges = sum_n_edges / torch.sum(n_edges_histogram, dim=0)
            except ValueError:
                avg_n_edges = 0

            return n_edges_histogram, avg_n_edges

        def _get_n_nodes_distribution(
            molecular_graphs: list,
        ) -> Tuple[torch.Tensor, float]:
            """
            Returns a histogram of the number of nodes per graph present in the
            `molecular_graphs`. The histogram is a `list` where the first item
            corresponds to the count of the number of graphs with one node, the
            second item corresponds to the count of the number of graphs with
            two nodes, etc, up until the count of the number of graphs with the
            largest number of nodes. Also returns the average number of nodes
            per graph.
            """
            # initialize histogram
            n_nodes_histogram = torch.zeros(
                constants.max_n_nodes + 1, device=constants.device
            )

            # populate histogram
            for molecular_graph in molecular_graphs:
                n_nodes = molecular_graph.n_nodes
                n_nodes_histogram[n_nodes] += 1

            # compute the average number of nodes per graph
            sum_n_nodes = 0
            for key, count in enumerate(n_nodes_histogram):
                n_nodes = key
                sum_n_nodes += n_nodes * count

            avg_n_nodes = sum_n_nodes / len(molecular_graphs)

            return n_nodes_histogram, avg_n_nodes

        def _get_node_feature_distribution(
            molecular_graphs: list,
        ) -> Tuple[Union[torch.Tensor, np.ndarray], ...]:
            """
            Returns a `tuple` of histograms (`torch.Tensor`s) for atom types,
            formal charges, number of implicit Hs, and chiral states that are
            present in the input `molecular_graphs`. Each histogram is a `list`
            where the nth item corresponds to the count of the nth property in
            `atom_types`, `formal_charge`, `imp_H`, and `chirality`.
            """
            # sum up all node feature vectors to get an un-normalized histogram
            if isinstance(molecular_graphs[0].node_features, torch.Tensor):
                nodes_hist = torch.zeros(
                    constants.n_node_features, device=constants.device
                )
            else:
                nodes_hist = np.zeros(constants.n_node_features)

            # loop over all the node feature matrices of the input `TrainingGraph`s
            for molecular_graph in molecular_graphs:
                try:
                    nodes_hist += torch.sum(molecular_graph.node_features, dim=0)
                except TypeError:
                    nodes_hist += np.sum(molecular_graph.node_features, axis=0)

            idc = util.get_feature_vector_indices()  # **note: "idc" == "indices"

            # split up `nodes_hist` into atom types hist, formal charge hist, etc
            atom_type_histogram = nodes_hist[: idc[0]]
            formal_charge_histogram = nodes_hist[idc[0] : idc[1]]
            if not constants.use_explicit_H and not constants.ignore_H:
                numh_histogram = nodes_hist[idc[1] : idc[2]]
            else:
                numh_histogram = [0] * constants.n_imp_H
            if constants.use_chirality:
                correction = int(
                    not constants.use_explicit_H and not constants.ignore_H
                )
                chirality_histogram = nodes_hist[
                    idc[1 + correction] : idc[2 + correction]
                ]
            else:
                chirality_histogram = [0] * constants.n_chirality

            return (
                atom_type_histogram,
                formal_charge_histogram,
                numh_histogram,
                chirality_histogram,
            )

        def _get_edge_feature_distribution(molecular_graphs: list) -> torch.Tensor:
            """
            Returns a histogram of edge features present in the input
            `molecular_graphs`. The histogram is a `torch.Tensor` where the
            first item corresponds to the count of the first edge type, etc. The
            edge types correspond to those defined in `BONDTYPE_TO_INT`.
            """
            # initialize and populate the histogram
            edge_feature_hist = torch.zeros(
                constants.n_edge_features, device=constants.device
            )

            for molecular_graph in molecular_graphs:
                edges = molecular_graph.edge_features
                for edge in range(constants.n_edge_features):
                    try:  # `GenerationGraph`s
                        edge_feature_hist[edge] += torch.sum(edges[:, :, edge]) / 2
                    except TypeError:  # `PreprocessingGraph`s
                        edge_feature_hist[edge] += np.sum(edges[:, :, edge]) / 2
            return edge_feature_hist

        def _get_fraction_unique(molecular_graphs: list) -> float:
            """
            Returns the fraction of unique graphs in `molecular_graphs`by
            comparing their canonical SMILES strings.
            """
            smiles_list = []
            for molecular_graph in molecular_graphs:
                smiles = molecular_graph.get_smiles()
                smiles_list.append(smiles)
            smiles_set = set(smiles_list)
            try:
                smiles_set.remove(None)  # remove placeholder for invalid SMILES
            except KeyError:  # no invalid SMILES in set!
                pass
            n_repeats = len(smiles_set)
            try:
                fraction_unique = n_repeats / len(smiles_list)
            except (ValueError, ZeroDivisionError):
                fraction_unique = 0
            return fraction_unique

        def _get_fraction_valid(
            molecular_graphs: list, termination: torch.Tensor
        ) -> Tuple[float, ...]:
            """
            Determines which graphs in `molecular_graphs` correspond to valid
            molecular structures. Uses RDKit which admittedly isn't perfect.
            `termination` is a `torch.Tensor` containing 0s or 1s corresponding
            to the validity of the structures in `molecular_graphs`.

            Returns:
            -------
                fraction_valid (float)               : Fraction of valid structures
                                                       in the input set.
                fraction_valid_properly_terminated (float) : Fraction of valid structures
                                                             in the input set, excluding
                                                             structures which were
                                                             improperly terminated.
                fraction_properly_terminated (float) : Fraction of generated structures
                                                       which were properly terminated.
            """
            n_invalid = 0  # start counting
            n_valid_and_properly_terminated = 0  # start counting
            n_graphs = len(molecular_graphs)

            for idx, molecular_graph in enumerate(molecular_graphs):
                mol = molecular_graph.get_molecule()
                # determine if valid
                try:
                    rdkit.Chem.SanitizeMol(mol)
                    n_valid_and_properly_terminated += int(termination[idx])
                except (ValueError, RuntimeError):  # invalid molecule
                    n_invalid += 1
            fraction_valid = (n_graphs - n_invalid) / n_graphs
            if 1 in termination:
                fraction_valid_properly_terminated = (
                    n_valid_and_properly_terminated / torch.sum(termination)
                )
            else:
                fraction_valid_properly_terminated = 0.0
            fraction_properly_terminated = torch.sum(termination) / len(termination)
            return (
                fraction_valid,
                fraction_valid_properly_terminated,
                fraction_properly_terminated,
            )

        # get the distribution of the number of atoms per graph
        n_nodes_hist, avg_n_nodes = _get_n_nodes_distribution(
            molecular_graphs=molecules
        )

        # get the distributions of node features (e.g. atom types) in the graphs
        atom_type_hist, formal_charge_hist, numh_hist, chirality_hist = (
            _get_node_feature_distribution(molecular_graphs=molecules)
        )

        # get the distribution of the number of edges per node and the average
        # number of edges per graph
        n_edges_hist, avg_n_edges = _get_n_edges_distribution(
            molecular_graphs=molecules, n_edges_to_bin=10
        )

        # get the distribution of bond types present in the graphs
        edge_feature_hist = _get_edge_feature_distribution(molecular_graphs=molecules)

        # get the fraction of unique molecules in the input graphs
        fraction_unique = _get_fraction_unique(molecular_graphs=molecules)

        if epoch_key == "Training set":
            # for the training set, we assume everything is valid (otherwise,
            # what are you doing)
            fraction_valid, fraction_valid_pt, fraction_pt = 1.0, 1.0, 1.0
        else:
            # get the fraction of valid molecules in the graphs
            (
                fraction_valid,  # fraction valid
                fraction_valid_pt,  # fraction valid and properly terminated
                fraction_pt,  # fraction properly terminated
            ) = _get_fraction_valid(molecular_graphs=molecules, termination=termination)

        properties = {
            (epoch_key, "n_nodes_hist"): n_nodes_hist,
            (epoch_key, "avg_n_nodes"): avg_n_nodes,
            (epoch_key, "atom_type_hist"): atom_type_hist,
            (epoch_key, "formal_charge_hist"): formal_charge_hist,
            (epoch_key, "n_edges_hist"): n_edges_hist,
            (epoch_key, "avg_n_edges"): avg_n_edges,
            (epoch_key, "edge_feature_hist"): edge_feature_hist,
            (epoch_key, "fraction_unique"): fraction_unique,
            (epoch_key, "fraction_valid"): fraction_valid,
            (epoch_key, "fraction_valid_properly_terminated"): fraction_valid_pt,
            (epoch_key, "fraction_properly_terminated"): fraction_pt,
            (epoch_key, "numh_hist"): numh_hist,
            (epoch_key, "chirality_hist"): chirality_hist,
        }

        return properties

    def merge_training_set_properties(
        self, prev_properties: dict, next_properties: dict, weight_next: int
    ) -> dict:
        """
        Averages the properties of `prev_properties` and `next_properties` (both
        dictionaries). This is used when calculating the properties of the
        training set in separate "groups", as is done during preprocessing.

        Args:
        ----
            prev_properties (dict) : Dictionary of old training set properties.
            next_properties (dict) : Dictionary of new training set properties.
            weight_next (int)      : Weight given to `next_properties`, equal to
                                     the number of graphs in the group used to calculate
                                     it (the weight is assumed to be `constants.batch_size`
                                     for `prev_properties`).

        Returns:
        -------
            training_set_properties (dict) : Averaged training set properties from the two
              input dictionaries.
        """
        # convert any CUDA (torch.Tensor)s to CPU tensors
        for dictionary in [prev_properties, next_properties]:
            for key, value in dictionary.items():
                try:
                    if value.is_cuda:
                        dictionary[key] = value.cpu()
                except AttributeError:
                    pass

        # `weight_prev` says how much to weight the properties of the old structures
        weight_prev = constants.batch_size

        # bundle properties in a tuple for some readibility
        bundle_properties = (prev_properties, next_properties, weight_prev, weight_next)

        # take a weighted average of the "old properties" with the "new properties"
        n_nodes_hist = self.weighted_average(b=bundle_properties, key="n_nodes_hist")
        avg_n_nodes = self.weighted_average(b=bundle_properties, key="avg_n_nodes")
        atom_type_hist = self.weighted_average(
            b=bundle_properties, key="atom_type_hist"
        )
        formal_charge_hist = self.weighted_average(
            b=bundle_properties, key="formal_charge_hist"
        )
        n_edges_hist = self.weighted_average(b=bundle_properties, key="n_edges_hist")
        avg_n_edges = self.weighted_average(b=bundle_properties, key="avg_n_edges")
        edge_feature_hist = self.weighted_average(
            b=bundle_properties, key="edge_feature_hist"
        )
        fraction_unique = self.weighted_average(
            b=bundle_properties, key="fraction_unique"
        )
        fraction_valid = self.weighted_average(
            b=bundle_properties, key="fraction_valid"
        )
        numh_hist = self.weighted_average(b=bundle_properties, key="numh_hist")
        chirality_hist = self.weighted_average(
            b=bundle_properties, key="chirality_hist"
        )

        # return the weighted averages in a new dictionary
        training_set_properties = {
            ("Training set", "n_nodes_hist"): n_nodes_hist,
            ("Training set", "avg_n_nodes"): avg_n_nodes,
            ("Training set", "atom_type_hist"): atom_type_hist,
            ("Training set", "formal_charge_hist"): formal_charge_hist,
            ("Training set", "n_edges_hist"): n_edges_hist,
            ("Training set", "avg_n_edges"): avg_n_edges,
            ("Training set", "edge_feature_hist"): edge_feature_hist,
            ("Training set", "fraction_unique"): fraction_unique,
            ("Training set", "fraction_valid"): fraction_valid,
            ("Training set", "numh_hist"): numh_hist,
            ("Training set", "chirality_hist"): chirality_hist,
        }
        return training_set_properties

    def weighted_average(self, b: Tuple[dict, dict, int, int], key: str) -> np.ndarray:
        """
        Takes a weighted average of two training set property dictionaries.

        Args:
        ----
            b (tuple) : Bundle of the following four items:
              p (dict) : "Previous" dictionary.
              n (dict) : "Next" dictionary.
              wp (int) : Weight for `p`.
              wn (int) : Weight for `n`.
            key (str) : 2nd string in the tuple keys.

        Returns:
        -------
            weighted_average (dict) : Dictionary is weighted average of `p` and `n`.
        """
        p, n, wp, wn = b

        def _to_numpy(v):
            if isinstance(v, torch.Tensor):
                return v.cpu().numpy()
            return np.array(v)

        weighted_average = np.around(
            (
                _to_numpy(p[("Training set", key)]) * wp
                + _to_numpy(n[("Training set", key)]) * wn
            )
            / (wp + wn),
            decimals=3,
        )

        return weighted_average

    def get_validation_likelihood(self, dataset: str) -> Tuple[torch.Tensor, float]:
        """
        Computes validation NLL (e.g. the NLL for taking the "correct" action
        for a specific fragment/atom) for graphs in the validation and training
        sets (whichever is specified by the `dataloader`). The subsets are equal
        in size to the number of structures generated per batch (`n_samples`
        below). Note: do not use for generation set structures, as there is no
        "correct" action!

        Returns:
        -------
            likelihoods (torch.Tensor) : Contains all NLLs per action for generating
                                         a set of molecules via the "correct" set
                                         of actions.
            avg_final_likelihood (torch.Tensor) : Contains average final NLLs for
                                                  generating a set of molecules
                                                  via the "correct" set of actions.
        """
        if dataset == "validation":
            dataloader = self.valid_dataloader
        elif dataset == "training":
            dataloader = self.train_dataloader
        elif dataset == "test":
            # Only a generate job with sample_mode: evaluate sets this, which
            # is final evaluation; the test split is not read during training.
            dataloader = self.test_dataloader
        else:
            raise ValueError(
                f"dataset: {dataset} is not a split. "
                "Use one of: training, validation, test."
            )

        Softmax = torch.nn.Softmax(dim=1)
        n_samples = min(100000, constants.n_samples)  # n graphs to evaluate
        likelihoods = torch.zeros(
            n_samples * (constants.max_n_nodes + 5), device=constants.device
        )
        n_structures = torch.zeros(1, device=constants.device)

        # `batch` contains constants.n_samples subgraphs during validation
        for idx, batch in enumerate(dataloader):
            # for really large dataloaders (like that of the training set), the
            # line below ensures that the validation NLL is only calculated
            # until the number of structures analyzed is roughly equivalent to
            # the number of structures generated, purely for speed
            if idx * constants.batch_size > n_samples:
                break

            if constants.device != "cpu":
                batch = [b.to(constants.device) for b in batch]
            # conditional datasets yield (nodes, edges, target, condition)
            if len(batch) == 4:
                nodes, edges, target_output, condition_vector = batch
            else:
                nodes, edges, target_output = batch
                condition_vector = None

            renormalized_target_output = target_output / torch.sum(
                target_output, dim=1, keepdim=True
            ).clamp(min=1e-12)

            # return the output and normalize; the condition must be passed
            # through, otherwise the reported NLL measures an unconditional
            # forward pass rather than the model actually being trained
            normalized_output = Softmax(
                self.model(nodes, edges, condition_vector=condition_vector)
                if condition_vector is not None
                else self.model(nodes, edges)
            )

            # multiplication with `target_output` zeros out the "incorrect" actions
            correct_action_probabilities = torch.mul(
                renormalized_target_output, normalized_output
            )
            likelihood = torch.sum(correct_action_probabilities, dim=1)
            likelihood = likelihood[~torch.isnan(likelihood)]
            # Clamp before the log.  A correct action the model assigns zero
            # probability gives -log(0) = +inf, which propagates through the
            # mean into `inf` and then makes the UC-JSD normalization inf/inf,
            # i.e. nan -- the model score was silently unusable whenever any
            # single action underflowed.
            likelihood = -1 * torch.log(likelihood.clamp(min=1e-12))
            start_idx = idx * constants.batch_size
            end_idx = idx * constants.batch_size + len(likelihood)
            likelihoods[start_idx:end_idx] = likelihood

            # in computing the number of structures, important to use
            # `target_output` and not `renormalized_target_output` (unnormalized
            # means the sum is number of subgraphs)
            n_structures += torch.sum(target_output[:, -1]).unsqueeze(dim=0)

        # A per-molecule average is undefined when the evaluated slice happens
        # to contain no completed molecule (possible on small datasets, since
        # only enough batches to cover `n_samples` are scored).  Report NaN
        # rather than 0.0, which would read as a perfect likelihood.  Both
        # branches are 0-dim: formatting a one-ELEMENT tensor with "{:.5f}"
        # raises TypeError.
        if n_structures[0] > 0:
            avg_final_likelihood = torch.sum(likelihoods, dim=0) / n_structures[0]
        else:
            print(
                f"-- Warning: no complete molecules in the {dataset} slice used "
                "for NLL evaluation; reporting NaN. Increase `n_samples` or use "
                "a larger dataset.",
                flush=True,
            )
            avg_final_likelihood = torch.tensor(float("nan"), device=constants.device)

        return likelihoods, avg_final_likelihood

    def plot_molecular_properties(self, properties: dict, plot_filename: str) -> None:
        """
        Plots a 3 by 3 grid of the histograms in `properties` using separate
        colors for the training set and for each epoch.

        Args:
        ----
            properties (dict) : Contains properties of generated and training
              set molecules. Only plots histogram properties, not averages.
            plot_filename (str) : Full path/filename for saving output PNG.
        """
        # start the grid
        matplotlib.rc("figure", figsize=(8.0, 7.0))
        n_plots_y, n_plots_x = 3, 3
        fig, ax = plt.subplots(n_plots_y, n_plots_x, sharey="all")
        fig.subplots_adjust(hspace=0.6, wspace=0.4)

        ax_nn = ax[0, 0]  # number of nodes
        ax_at = ax[0, 1]  # atom types
        ax_fc = ax[0, 2]  # formal charges
        ax_nh = ax[1, 0]  # num implicit Hs
        ax_ne = ax[1, 1]  # number of edges
        ax_bt = ax[1, 2]  # here, bond type == edge feature
        ax_ct = ax[2, 0]  # chirality

        # get the keys of the properties to plot
        keys_to_plot = list(set([key[0] for key in properties.keys()]))

        # plot the results for the training set and for each epoch
        for epoch_key in keys_to_plot:
            # set the plot labels
            if epoch_key == "Training set":
                m, c, ls = "*", "goldenrod", "-"
            else:
                m, c, ls = "o", "cadetblue", "--"

            # normalize so that all can share one y-axis
            (
                norm_n_nodes_hist,
                norm_atom_type_hist,
                norm_formal_charge_hist,
                norm_numh_hist,
                norm_n_edges_hist,
                norm_edge_feature_hist,
                norm_chirality_hist,
            ) = util.normalize_evaluation_metrics(
                property_histograms=properties, epoch_key=epoch_key
            )

            # plot num nodes histogram
            ax_nn.plot(
                range(1, len(norm_n_nodes_hist) + 1),
                norm_n_nodes_hist,
                color=c,
                label=epoch_key,
                linestyle=ls,
                marker=m,
            )
            ax_nn.set(xlabel="Num nodes per graph")

            # plot atom type histogram
            ax_at.plot(
                range(1, len(norm_atom_type_hist) + 1),
                norm_atom_type_hist,
                color=c,
                label=epoch_key,
                linestyle=ls,
                marker=m,
            )
            xlabel_values = ", ".join(map(str, constants.atom_types))
            ax_at.set(xlabel=f"Atom type ({xlabel_values})")

            # plot formal charge histogram
            ax_fc.plot(
                constants.formal_charge,
                norm_formal_charge_hist,
                color=c,
                label=epoch_key,
                linestyle=ls,
                marker=m,
            )
            xlabel_values = ", ".join(map(str, constants.formal_charge))
            ax_fc.set(xlabel=f"Formal charge ({xlabel_values})")

            # plot num H histogram
            ax_nh.plot(
                constants.imp_H,
                norm_numh_hist,
                color=c,
                label=epoch_key,
                linestyle=ls,
                marker=m,
            )
            xlabel_values = ", ".join(map(str, constants.imp_H))
            ax_nh.set(
                xlabel=f"Num implicit Hs ({xlabel_values})", ylabel="Fractional count"
            )

            # plot n_edges histogram
            ax_ne.plot(
                range(1, len(norm_n_edges_hist) + 1),
                norm_n_edges_hist,
                color=c,
                label=epoch_key,
                linestyle=ls,
                marker=m,
            )
            ax_ne.set(xlabel="Num edges per node")

            # plot bond type/edge feature histogram
            ax_bt.plot(
                range(0, len(norm_edge_feature_hist)),
                norm_edge_feature_hist,
                color=c,
                label=epoch_key,
                linestyle=ls,
                marker=m,
            )
            xlabel_values = ", ".join(map(str, constants.int_to_bondtype))
            ax_bt.set(xlabel=f"Bond type ({xlabel_values})")

            # plot chirality histogram
            ax_ct.plot(
                range(1, len(norm_chirality_hist) + 1),
                norm_chirality_hist,
                color=c,
                label=epoch_key,
                linestyle=ls,
                marker=m,
            )
            xlabel_values = ", ".join(map(str, constants.chirality))
            ax_ct.set(xlabel=f"Chirality ({xlabel_values})")

            # put the legend in the bottom right corner regardless
            ax_ct.legend(loc="upper right", prop={"size": 6})

        ax = util.turn_off_empty_axes(n_plots_y, n_plots_x, ax)

        fig.savefig(plot_filename)
        plt.close()

    def save_metrics(self, step, score, append=True) -> None:
        """
        Save the fine-tuning metrics, such as the model score, to a log file.

        Args:
        ----
            step (int)              : The fine-tuning step.
            score (float)           : Model score.
            append (bool, optional) : Indicates whether to append to an existing
                                      file, or create a new file. Defaults to True.
        """
        score_val = float(score)
        if not append:
            with open(constants.logs_dir + "score.log", "w") as output_file:
                output_file.write("Step, Score\n")
                output_file.write(f"Step {step}, {score_val:.8f}\n")
        else:
            with open(constants.logs_dir + "score.log", "a") as output_file:
                output_file.write(f"Step {step}, {score_val:.8f}\n")

        if self.create_tensorboard:
            self.tb_writer.add_scalar("Evaluation/score", score, step)

    # ------------------------------------------------------------------
    # Error-bar helpers
    # ------------------------------------------------------------------

    def _scalar_metrics(
        self,
        molecules: list,
        termination: torch.Tensor,
    ) -> dict:
        """
        Compute lightweight scalar generation metrics for a subset of molecules.
        Mirrors the logic in ``get_molecular_properties`` but skips histograms.

        Returns:
            dict with keys: fraction_valid, fraction_valid_pt, fraction_pt,
                            avg_n_nodes, avg_n_edges, fraction_unique
        """
        n = len(molecules)
        if n == 0:
            return {
                k: 0.0
                for k in (
                    "fraction_valid",
                    "fraction_valid_pt",
                    "fraction_pt",
                    "avg_n_nodes",
                    "avg_n_edges",
                    "fraction_unique",
                )
            }

        n_invalid = 0
        n_valid_pt = 0
        total_nodes = 0
        total_node_edge_sum = 0
        smiles_list = []

        for idx, graph in enumerate(molecules):
            mol = graph.get_molecule()
            try:
                rdkit.Chem.SanitizeMol(mol)
                n_valid_pt += int(termination[idx])
            except (ValueError, RuntimeError):
                n_invalid += 1

            nn = graph.n_nodes
            total_nodes += nn
            edges = graph.edge_features
            for node_idx in range(nn):
                ne = 0
                for bond_type in range(constants.n_edge_features):
                    try:
                        ne += int(torch.sum(edges[node_idx, :, bond_type]))
                    except TypeError:
                        ne += int(np.sum(edges[node_idx, :, bond_type]))
                total_node_edge_sum += ne

            smiles_list.append(graph.get_smiles())

        n_pt = int(torch.sum(termination))
        frac_valid = (n - n_invalid) / n
        frac_pt = n_pt / n
        frac_valid_pt = n_valid_pt / n_pt if n_pt > 0 else 0.0
        avg_n_nodes = total_nodes / n
        avg_n_edges = total_node_edge_sum / total_nodes if total_nodes > 0 else 0.0
        valid_smiles = [s for s in smiles_list if s is not None]
        frac_unique = len(set(valid_smiles)) / n

        return {
            "fraction_valid": frac_valid,
            "fraction_valid_pt": frac_valid_pt,
            "fraction_pt": frac_pt,
            "avg_n_nodes": avg_n_nodes,
            "avg_n_edges": avg_n_edges,
            "fraction_unique": frac_unique,
        }

    def _compute_error_bars(
        self,
        molecules: list,
        termination: torch.Tensor,
        n_splits: int = 3,
    ) -> dict:
        """
        Split ``molecules`` into ``n_splits`` equal subsets, compute scalar
        generation metrics per subset, and return the standard deviation across
        subsets as a proxy for uncertainty.

        Returns:
            dict with keys: fraction_valid_std, fraction_valid_pt_std,
                            fraction_pt_std, avg_n_nodes_std,
                            avg_n_edges_std, fraction_unique_std
        """
        n = len(molecules)
        if n < n_splits:
            return {
                "fraction_valid_std": 0.0,
                "fraction_valid_pt_std": 0.0,
                "fraction_pt_std": 0.0,
                "avg_n_nodes_std": 0.0,
                "avg_n_edges_std": 0.0,
                "fraction_unique_std": 0.0,
            }

        split_indices = np.array_split(np.arange(n), n_splits)
        metrics_per_split = []
        for idx_arr in split_indices:
            sub_mols = [molecules[i] for i in idx_arr]
            # Convert numpy indices to a torch LongTensor to avoid unreliable
            # numpy-array indexing of torch.Tensor across different PyTorch versions.
            idx_t = torch.from_numpy(idx_arr.astype(np.int64))
            sub_term = termination[idx_t]
            metrics_per_split.append(self._scalar_metrics(sub_mols, sub_term))

        result = {}
        for key in metrics_per_split[0]:
            values = [m[key] for m in metrics_per_split]
            result[f"{key}_std"] = float(np.std(values))
        return result

    def _load_test_smiles(
        self,
    ) -> tuple:
        """
        Reads the test-set SMILES file and returns a (smiles, conditions) pair.

        For plain ``.smi`` files returns ``(list[str], None)``.
        For tab-separated files with a header row returns
        ``(list[str], list[dict[str, float]])`` where the second element is a
        parallel list of per-molecule property dicts.

        Results are cached after the first call.
        """
        if hasattr(self, "_test_smiles_cache"):
            return self._test_smiles_cache

        path = constants.test_set
        smiles: list = []
        conditions: list = []
        has_conditions = False

        if os.path.exists(path):
            with open(path) as fh:
                first = fh.readline()
                cols = first.strip().split("\t")
                if cols[0].upper() == "SMILES" and len(cols) > 1:
                    # Tab-separated with property header
                    has_conditions = True
                    prop_names = cols[1:]
                    for line in fh:
                        parts = line.strip().split("\t")
                        if not parts or not parts[0]:
                            continue
                        smiles.append(parts[0])
                        cond: dict = {}
                        for i, name in enumerate(prop_names):
                            try:
                                cond[name] = float(parts[i + 1])
                            except (IndexError, ValueError):
                                pass
                        conditions.append(cond)
                else:
                    # Plain SMILES, first line may be SMILES or a molecule
                    smi0 = cols[0]
                    if smi0.upper() != "SMILES":
                        smiles.append(smi0)
                    for line in fh:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        smi = parts[0]
                        if smi.upper() != "SMILES":
                            smiles.append(smi)

        result = (smiles, conditions if has_conditions else None)
        self._test_smiles_cache = result
        return result

    def _load_training_smiles(self) -> set:
        """
        Reads the training set SMILES file and returns a set of SMILES strings.
        Result is cached after the first call.
        """
        if hasattr(self, "_training_smiles_cache"):
            return self._training_smiles_cache
        path = constants.training_set
        smiles_set = set()
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    smi = parts[0]
                    if smi.upper() != "SMILES":  # skip header
                        smiles_set.add(smi)
        self._training_smiles_cache = smiles_set
        return smiles_set

    def _compute_extended_metrics(
        self,
        generated_graphs: list,
        validity_tensor,
        scores=None,
        include_expensive: bool = False,
        reference_smiles: list = None,
    ) -> dict:
        """
        Computes novelty, SA score, success rate, and (optionally) diversity,
        FCD, and rediscovery rate.

        Args:
            generated_graphs  : list of GenerationGraph objects.
            validity_tensor   : binary torch.Tensor (1=valid, 0=invalid).
            scores            : optional 1-D score tensor for success_rate.
            include_expensive : if True, also compute diversity, FCD,
                                rediscovery_rate.
            reference_smiles  : list of reference SMILES for FCD/rediscovery.

        Returns:
            dict with float values (or None for optional metrics that failed).
        """
        all_smiles = [g.get_smiles() for g in generated_graphs]

        training_smiles = self._load_training_smiles()

        # One implementation per metric. This used to call a second copy in
        # src/graphinvent/_metrics.py whose definitions had drifted from
        # src/metrics/: success_rate compared with a strict `>` where the
        # config key reads as "at least", internal diversity subsampled to
        # 1,000 against the specified 10,000, and SA score averaged over valid
        # rather than unique valid molecules. CLAUDE.md requires exactly one
        # implementation, so the numbers come from src/metrics/_core.py.
        #
        # Four keys this function used to return are gone rather than ported:
        # success_rate, rediscovery_rate, sa_score_std and
        # internal_diversity_successful. The first is defined in
        # src/metrics/_budget.py and reported in metrics.json, which is where
        # CLAUDE.md puts it; the other three are not in the specification at
        # all. Reporting them from here was the duplication.
        core = compute_core_metrics(
            all_smiles,
            n_sampled=len(generated_graphs),
            training_smiles=sorted(training_smiles),
            isomeric=bool(getattr(constants, "use_chirality", False)),
        )
        core.pop("_filter_unavailable", None)
        result = {
            "novelty": core["novelty"],
            "sa_score_mean": core["sa_score_mean"],
            "sa_score_median": core["sa_score_median"],
            "internal_diversity": core["internal_diversity"],
        }

        # mean and max internal similarity are not part of the core table, so
        # they come from the one implementation that does define them.
        valid_smiles = [s for s in all_smiles if s is not None]
        if len(valid_smiles) >= 2:
            div = compute_internal_diversity(
                sorted(set(valid_smiles)), max_mols=DIVERSITY_MAX_MOLECULES
            )
            result["mean_internal_similarity"] = div["mean_internal_similarity"]
            result["max_internal_similarity"] = div["max_internal_similarity"]
        else:
            result["mean_internal_similarity"] = float("nan")
            result["max_internal_similarity"] = float("nan")

        if include_expensive and reference_smiles is not None:
            from metrics._unconditional import _compute_fcd

            isomeric = bool(getattr(constants, "use_chirality", False))
            from metrics._core import canonical_smiles as _canon

            fcd_result = _compute_fcd(
                [x for x in _canon(valid_smiles, isomeric=isomeric) if x],
                [x for x in _canon(reference_smiles, isomeric=isomeric) if x],
            )
            result["fcd"] = fcd_result["value"]
            result["fcd_n_generated"] = fcd_result["n_generated"]
            result["fcd_n_reference"] = fcd_result["n_reference"]

        # --- Test-set similarity ---
        # The test split is the reference set for final evaluation and is read
        # only there: a `generate` job, or the budget-capped checkpoint
        # evaluation that sets `include_expensive`. It used to be loaded
        # unconditionally, which made every goal_directed learning step read it
        # during training. CLAUDE.md holds the split back until the model is
        # fixed, because a number computed during training is a choice made
        # during development, and the split stops being held out once read.
        final_evaluation = include_expensive or constants.job_type == "generate"
        valid_smiles_for_sim = [s for s in all_smiles if s is not None]
        test_smi_list, test_cond_list = (
            self._load_test_smiles() if final_evaluation else ([], None)
        )
        if test_smi_list and valid_smiles_for_sim:
            # Build condition_filter from sample_conditions for conditional jobs
            cond_filter = None
            sample_conds = constants.sample_conditions
            if sample_conds and test_cond_list is not None:
                # A stratified job records "from_train" rather than a number,
                # because it asks a different value of each batch. There is no
                # single value to filter the reference set by, so skip those.
                cond_filter = {
                    name: {"value": float(val), "tolerance": 0.3}
                    for name, val in sample_conds.items()
                    if _is_number(val)
                }
            top_k = TEST_SIMILARITY_TOP_K
            max_refs = None
            sim = compute_test_set_similarity(
                valid_smiles_for_sim,
                test_smi_list,
                top_k=top_k,
                condition_filter=cond_filter,
                test_conditions=test_cond_list,
                max_refs=max_refs,
            )
            result["sim_mean"] = sim["mean_similarity"]
            result["sim_median"] = sim["median_similarity"]
            result[f"sim_top{top_k}"] = sim["top_k_similarity"]
            result["sim_gt_0_4"] = sim["sim_gt_0_4"]
            result["sim_gt_0_6"] = sim["sim_gt_0_6"]
            result["sim_gt_0_8"] = sim["sim_gt_0_8"]
            result["sim_gt_0_9"] = sim["sim_gt_0_9"]
            result["exact_rediscovery_count"] = sim["exact_rediscovery_count"]

        return result

    def evaluate_checkpoint_molecules(
        self,
        generated_graphs: list,
        validity_tensor,
        scores,
        component_scores: dict,
        oracle_count: int,
        csv_path: str,
    ) -> None:
        """
        Runs the full evaluation suite for a budget-capped RL checkpoint and writes
        one row to oracle_eval.csv.

        Args:
            generated_graphs : list of GenerationGraph objects.
            validity_tensor  : binary torch.Tensor (1=valid, 0=invalid).
            scores           : 1-D final score tensor.
            component_scores : dict mapping score component name -> score tensor.
            oracle_count     : oracle call milestone this checkpoint corresponds to.
            csv_path         : path to oracle_eval.csv.
        """
        # Load test set SMILES for FCD / rediscovery rate
        test_smiles = []
        if os.path.exists(constants.test_set):
            with open(constants.test_set) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    smi = parts[0]
                    if smi.upper() != "SMILES":
                        test_smiles.append(smi)

        ext = self._compute_extended_metrics(
            generated_graphs=generated_graphs,
            validity_tensor=validity_tensor,
            scores=scores,
            include_expensive=True,
            reference_smiles=test_smiles if test_smiles else None,
        )

        # Basic validity / uniqueness rates
        n = len(generated_graphs)
        n_valid = int(validity_tensor.sum().item())
        all_smiles = [g.get_smiles() for g in generated_graphs]
        # Only count SMILES for molecules where validity_tensor[i] == 1.
        # Using get_smiles() alone can miss the edge case where validity_tensor
        # and get_smiles() disagree (different sanitization code paths).
        unique_valid = {
            all_smiles[i]
            for i in range(n)
            if validity_tensor[i] == 1 and all_smiles[i] is not None
        }
        n_unique = len(unique_valid)
        fraction_valid = n_valid / n if n > 0 else 0.0
        fraction_unique = n_unique / n_valid if n_valid > 0 else 0.0

        row = {
            "oracle_count": oracle_count,
            "fraction_valid": round(fraction_valid, 5),
            "fraction_unique": round(fraction_unique, 5),
            "novelty": round(ext.get("novelty", float("nan")), 5),
            "sa_score_mean": round(ext.get("sa_score_mean", float("nan")), 5),
            "sa_score_median": round(ext.get("sa_score_median", float("nan")), 5),
            "sa_score_std": round(ext.get("sa_score_std", float("nan")), 5),
            "success_rate": round(ext.get("success_rate", float("nan")), 5),
            "diversity": round(ext.get("diversity", float("nan")), 5),
            "fcd": ext.get("fcd"),
            "rediscovery_rate": ext.get("rediscovery_rate"),
        }
        for comp_name, comp_tensor in component_scores.items():
            row[f"score_{comp_name}_mean"] = round(
                float(comp_tensor.float().mean().item()), 5
            )

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as fh:
            if write_header:
                fh.write(",".join(row.keys()) + "\n")
            vals = []
            for v in row.values():
                if v is None or (isinstance(v, float) and v != v):
                    vals.append("NA")
                else:
                    vals.append(str(v))
            fh.write(",".join(vals) + "\n")

    # ------------------------------------------------------------------
    # Progress plot
    # ------------------------------------------------------------------
