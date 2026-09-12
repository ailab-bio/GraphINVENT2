"""
Scoring functions used during reinforcement learning fine-tuning.

`ScoringFunction` combines one or more component scores into a single scalar
reward per generated molecule.  Two kinds of component exist.

Built-ins, computed directly from the molecule:

  QED                    -- Quantitative Estimate of Drug-likeness (RDKit)
  target_size=<int>      -- Windowed score peaking at a heavy-atom count
  logp_target=<float>    -- Windowed score peaking at a Crippen logP
  <name>_activity        -- A pickled QSAR model listed in `qsar_models`

Oracles, declared by name in the `oracles` config block and referenced from
`score_components`.  These are user-supplied: a surrogate trained on the user's
own data, an AutoDock Vina docking run against a chosen receptor, or an
arbitrary Python callable.  Each carries its own transform onto [0, 1] and a
direction, so "bind this target but avoid that one" is two oracles over the
same kind of quantity with opposite directions.  See :mod:`oracles`.

Components combine either as a product (`score_type: "continuous"`) or as an
AND over per-component thresholds (`score_type: "binary"`), so a molecule must
satisfy every criterion simultaneously to score well.

When an oracle reports predictive uncertainty, that uncertainty can damp the
reward or the gradient contribution per component; see
:mod:`oracles._uncertainty`.
"""

# load general packages and functions
from collections import namedtuple

import numpy as np
import sklearn
import torch
from rdkit import DataStructs
from rdkit.Chem import QED, AllChem, Crippen


def _ensure_src_on_path() -> None:
    """
    Make the sibling ``oracles`` package importable.

    `main.py` runs as a script from ``src/graphinvent/``, so ``src`` is not on
    the path, and the editable install does not work from a path containing
    spaces (see CLAUDE.md).
    """
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


class ScoringFunction:
    """
    Combines multiple property scores into a single RL reward per molecule.

    Each generated molecule is evaluated against the score components listed in
    ``constants.score_components``.  Component scores are normalised to [0, 1],
    then multiplied together (after thresholding) to produce a final scalar.

    The score is set to 0 for molecules that were not properly terminated, are
    chemically invalid, or fail any of the per-component thresholds.
    """

    def __init__(self, constants: namedtuple) -> None:
        """
        Args:
        ----
            constants (namedtuple) : Contains job parameters as well as global
                                     constants.
        """
        self.score_components = constants.score_components  # list
        self.score_type = constants.score_type  # list
        self.qsar_models = constants.qsar_models  # dict
        self.device = constants.device
        self.max_n_nodes = constants.max_n_nodes
        self.score_thresholds = constants.score_thresholds

        self.n_graphs = None  # placeholder
        self.constants = constants

        assert len(self.score_components) == len(
            self.score_thresholds
        ), "`score_components` and `score_thresholds` do not match."

        # Oracles named in the `oracles` config block.  These are the
        # user-supplied objectives (a trained surrogate, a docking run, an
        # arbitrary callable); `score_components` references them by name.
        self._oracles: dict = {}
        # Fallback budget counter for components with no external oracle
        # (QED, target_size, logp_target).
        self.n_molecules_scored: int = 0
        # Per-molecule uncertainty from the most recent batch, keyed by
        # component; only populated for oracles that report one.
        self._component_uncertainty: dict = {}
        self._init_oracles()
        self._modulation = self._init_modulation()

    def _init_modulation(self):
        """
        Build the uncertainty-modulation policy from the job config.

        Validated at construction so a malformed spec fails before any
        molecules are generated.
        """
        _ensure_src_on_path()
        from oracles import UncertaintyModulation

        modulation = UncertaintyModulation(
            getattr(self.constants, "uncertainty_modulation", None)
        )
        if modulation.enabled:
            unsupported = [
                name
                for name in modulation.components
                if modulation.is_configured(name)
                and not (
                    name in self._oracles and self._oracles[name].supports_uncertainty
                )
            ]
            if unsupported:
                print(
                    "-- Warning: uncertainty_modulation is configured for "
                    f"{unsupported}, but those components report no uncertainty "
                    "(not an oracle, or a non-ensemble model). They will be "
                    "left unmodulated.",
                    flush=True,
                )
        return modulation

    def reliability_weights_per_component(self) -> dict:
        """
        Reliability weights from the most recent batch, keyed by component.

        Empty when nothing reported an uncertainty, which the callers treat as
        "no modulation" rather than as an error.
        """
        return {
            component: self._modulation.weights_for(component, uncertainties)
            for component, uncertainties in self._component_uncertainty.items()
        }

    def loss_weights(self, n_molecules: int):
        """
        Per-molecule loss weights for the most recent batch.

        Returns None when loss modulation is off or nothing reported an
        uncertainty, so the caller can skip the reweighting entirely.
        """
        if not self._modulation.modulates_loss():
            return None
        reliability = self.reliability_weights_per_component()
        if not reliability:
            return None

        _ensure_src_on_path()
        from oracles import combine_loss_weights, modulate_loss_weights

        combined = combine_loss_weights(reliability, n_molecules)
        return modulate_loss_weights(combined)

    def _init_oracles(self) -> None:
        """
        Build every oracle named in ``constants.oracles``.

        Constructed eagerly so a missing model file, an unreadable receptor, or
        a typo in a spec fails at job start rather than after the first batch
        of molecules has been generated.
        """
        specs = dict(getattr(self.constants, "oracles", None) or {})
        if not specs:
            self._check_components_resolvable()
            return

        _ensure_src_on_path()
        from oracles import OracleFactory

        self._oracles = OracleFactory.from_specs(specs)
        self._check_components_resolvable()

    def _check_components_resolvable(self) -> None:
        """
        Fail fast on a score component nothing can compute.

        Without this the run raises `NotImplementedError` from deep inside
        scoring, part-way through the first training step and after the models
        have already been loaded.
        """
        unresolved = [
            component
            for component in self.score_components
            if not self._is_builtin(component) and component not in self._oracles
        ]
        if unresolved:
            declared = sorted(self._oracles) or "none"
            raise ValueError(
                f"Score component(s) {unresolved} are neither built-in nor "
                f"declared in the 'oracles' config block (declared: {declared}). "
                "Built-ins are: QED, target_size=<int>, logp_target=<float>, "
                "and <name>_activity with a matching entry in 'qsar_models'."
            )

    def _is_builtin(self, component: str) -> bool:
        """Whether a component is computed directly rather than by an oracle."""
        return (
            component == "QED"
            or component.startswith("target_size=")
            or component.startswith("logp_target=")
            or ("activity" in component and component in self.qsar_models)
        )

    @property
    def oracle_calls(self) -> int:
        """
        Number of oracle evaluations consumed so far.

        With a cached oracle this is the deduplicated unique-molecule count,
        which is the meaningful unit for a budget: re-proposing a molecule the
        agent has already seen costs nothing.  With no external oracle there is
        nothing to meter, so the number of molecules scored is reported instead.
        """
        if self._oracles:
            return max(o.call_count for o in self._oracles.values())
        return self.n_molecules_scored

    @property
    def optimization_log(self) -> list:
        """
        Chronological (cumulative_call_count, score) record from the oracles.

        Empty without an external oracle, since only a real oracle defines the
        call-indexed curve the AUC Top-k metric integrates over.
        """
        if not self._oracles:
            return []
        busiest = max(self._oracles.values(), key=lambda o: o.call_count)
        return busiest.optimization_log

    def compute_score(
        self,
        graphs: list,
        termination: torch.Tensor,
        validity: torch.Tensor,
        uniqueness: torch.Tensor,
    ) -> torch.Tensor:
        """Same interface as before — returns only the final score tensor."""
        final_score, _ = self.compute_score_with_components(
            graphs=graphs,
            termination=termination,
            validity=validity,
            uniqueness=uniqueness,
        )
        return final_score

    def compute_score_with_components(
        self,
        graphs: list,
        termination: torch.Tensor,
        validity: torch.Tensor,
        uniqueness: torch.Tensor,
    ) -> tuple:
        """
        Computes the overall score and returns per-component scores.

        Args:
        ----
            graphs (list)              : Contains molecular graphs to evaluate.
            termination (torch.Tensor) : Termination status of input molecular graphs.
            validity (torch.Tensor)    : Validity of input molecular graphs.
            uniqueness (torch.Tensor)  : Uniqueness of input molecular graphs.

        Returns:
        -------
            final_score (torch.Tensor)      : Final score per graph, shape (n_graphs,).
            component_scores (dict)         : Maps each score component name to its
                                              raw score tensor before masking.
        """
        self.n_graphs = len(graphs)
        self.n_molecules_scored += len(graphs)
        # Reset per-batch uncertainty before scoring; `get_contributions_to_score`
        # fills it for the oracles that report one.
        self._component_uncertainty = {}
        contributions_to_score = self.get_contributions_to_score(graphs=graphs)

        # Build component dict keyed by score component name
        component_scores = {
            name: contributions_to_score[i]
            for i, name in enumerate(self.score_components)
        }

        if self.score_type == "continuous":
            final_score = contributions_to_score[0]
            for component in contributions_to_score[1:]:
                final_score = final_score * component

        elif self.score_type == "binary":
            component_masks = []
            for idx, score_component in enumerate(contributions_to_score):
                component_mask = torch.where(
                    score_component > self.score_thresholds[idx],
                    torch.ones(self.n_graphs, device=self.device, dtype=torch.uint8),
                    torch.zeros(self.n_graphs, device=self.device, dtype=torch.uint8),
                )
                component_masks.append(component_mask)

            final_score = component_masks[0]
            for mask in component_masks[1:]:
                final_score = final_score * mask
                final_score = final_score.float()

        else:
            raise NotImplementedError

        # Score modulation: fold predictive reliability into the objective, so
        # a molecule the surrogate cannot vouch for is worth less as a molecule
        # (Medina & Janet, arXiv:2606.24990, Eq. 6).
        if self._modulation.modulates_score():
            reliability = self.reliability_weights_per_component()
            if reliability:
                from oracles import combine_score_weights

                factor = combine_score_weights(reliability, self.n_graphs)
                final_score = final_score * torch.tensor(
                    factor, device=self.device, dtype=torch.float32
                )

        # remove contribution of duplicate molecules to the score
        final_score = final_score * uniqueness

        # remove contribution of invalid molecules to the score
        final_score = final_score * validity

        # remove contribution of improperly-terminated molecules to the score
        final_score = final_score * termination

        return final_score, component_scores

    def get_contributions_to_score(self, graphs: list) -> list:
        """
        Returns the different elements of the score.

        Args:
        ----
            graphs (list) : Contains molecular graphs to evaluate.

        Returns:
        -------
            contributions_to_score (list) : Contains elements of the score due to
                                            each scoring function component.
        """
        contributions_to_score = []

        for score_component in self.score_components:
            if score_component.startswith("target_size="):

                target_size = int(score_component.split("=", 1)[1])

                assert (
                    target_size < self.max_n_nodes
                ), "Target size must be strictly less than `max_n_nodes` (equal causes division by zero)."
                assert 0 < target_size, "Target size must be greater than 0."

                target_size *= torch.ones(self.n_graphs, device=self.device)
                n_nodes = torch.tensor(
                    [graph.n_nodes for graph in graphs], device=self.device
                )
                max_nodes = self.max_n_nodes
                # Clamped at 0: the unclamped expression is unbounded below, so
                # a 1-atom graph scored -2.0, and with two negative components
                # the product flips positive -- a bad molecule earning a good
                # reward.  A score is a value in [0, 1]; distance beyond the
                # window is simply "no credit".
                score = torch.clamp(
                    torch.ones(self.n_graphs, device=self.device)
                    - torch.abs(n_nodes - target_size) / (max_nodes - target_size),
                    min=0.0,
                )

                contributions_to_score.append(score)

            elif score_component.startswith("logp_target="):
                # Crippen logP driven toward a target value rather than
                # maximised.  Lipophilicity is not monotonically desirable:
                # degraders in particular tend to sit well above the drug-like
                # window, so the useful objective is to reach a value, not to
                # exceed it.  The +/- 5 log-unit window is a convention, not a
                # principled choice; it is wide enough that a randomly
                # initialised model still receives gradient signal.
                target_logp = float(score_component.split("=", 1)[1])
                window = 5.0

                values = []
                for graph in graphs:
                    try:
                        values.append(Crippen.MolLogP(graph.molecule))
                    except (ValueError, RuntimeError, AttributeError, TypeError):
                        # invalid graphs decode to None; they are zeroed by the
                        # validity mask anyway
                        values.append(target_logp - window)
                logp = torch.tensor(values, device=self.device, dtype=torch.float32)
                score = torch.clamp(
                    1.0 - torch.abs(logp - target_logp) / window, min=0.0
                )

                contributions_to_score.append(score)

            elif score_component == "QED":
                mols = [graph.molecule for graph in graphs]

                # compute the QED score for each molecule (if possible)
                qed = []
                for mol in mols:
                    try:
                        qed.append(QED.qed(mol))
                    except (ValueError, RuntimeError):
                        qed.append(0.0)
                score = torch.tensor(qed, device=self.device)

                contributions_to_score.append(score)

            elif "activity" in score_component and score_component in self.qsar_models:
                # Membership in `qsar_models` is part of the condition, matching
                # `_is_builtin`.  Matching on the substring alone would capture
                # an oracle named e.g. "EGFR_activity" -- which passes component
                # validation, since it IS a declared oracle -- and then raise a
                # KeyError here on the first scored batch.
                mols = [graph.molecule for graph in graphs]
                qsar_model = self.qsar_models[score_component]
                score = self.compute_activity(mols, qsar_model)

                contributions_to_score.append(score)

            elif score_component in self._oracles:
                oracle = self._oracles[score_component]
                smiles = self._graphs_to_smiles(graphs)
                # Only pay for the uncertainty estimate when something will use
                # it: for a docking oracle it costs several extra pose searches
                # per molecule.
                if (
                    self._modulation.enabled
                    and self._modulation.is_configured(score_component)
                    and oracle.supports_uncertainty
                ):
                    scores_list, uncertainties = oracle.predict_with_uncertainty(smiles)
                    self._component_uncertainty[score_component] = uncertainties
                else:
                    scores_list = oracle(smiles)
                score = torch.tensor(
                    scores_list, device=self.device, dtype=torch.float32
                )
                contributions_to_score.append(score)

            else:
                raise ValueError(
                    f"Score component '{score_component}' is not defined. "
                    "Built-ins are QED, target_size=<int>, logp_target=<float>, "
                    "and <name>_activity; anything else must be declared in the "
                    "'oracles' block of the job config."
                )

        return contributions_to_score

    @staticmethod
    def _graphs_to_smiles(graphs: list) -> list:
        """
        Canonical SMILES for each graph, with None for those that fail.

        Oracles take SMILES rather than graphs so that a user-supplied scoring
        function needs to know nothing about GraphINVENT's internals.
        """
        from rdkit.Chem import MolToSmiles

        smiles = []
        for graph in graphs:
            try:
                molecule = graph.molecule
                smiles.append(MolToSmiles(molecule) if molecule else None)
            except (ValueError, RuntimeError, AttributeError, TypeError):
                smiles.append(None)
        return smiles

    def compute_activity(self, mols: list, activity_model: sklearn.svm.SVC) -> list:
        """
        Note: this function may have to be tuned/replicated depending on how
        the activity model is saved.

        Args:
        ----
            mols (list) : Contains `rdkit.Mol` objects corresponding to molecular
                          graphs sampled.
            activity_model (sklearn.svm.classes.SVC) : Pre-trained QSAR model.

        Returns:
        -------
            activity (list) : Contains predicted activities for input molecules.
        """
        n_mols = len(mols)
        activity = torch.zeros(n_mols, device=self.device)

        for idx, mol in enumerate(mols):
            try:
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                ecfp4 = np.zeros((2048,))
                DataStructs.ConvertToNumpyArray(fingerprint, ecfp4)
                activity[idx] = activity_model.predict_proba([ecfp4])[0][1]
            except (ValueError, RuntimeError, AttributeError, TypeError):
                # TypeError covers Boost's ArgumentError, raised when `mol` is
                # None -- which is routine, since invalid graphs decode to None.
                pass  # activity[idx] will remain 0.0

        return activity
