"""
What each job type writes to results/metrics.json, and the report beside it.

One function per job type rather than one function with a parameter for every
job type's inputs: they need different things, and a signature that accepts all
of them accepts nonsense too.

Every one of these collects, writes metrics.json, then renders report.html from
that file. The report is last, so a failure rendering it cannot cost anything
already saved.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import _budget, _core, _efficiency, _html, _json

# CLAUDE.md's goal-directed table combines multiple property thresholds with
# AND; with one component the job's own threshold is the success threshold.
DEFAULT_SUCCESS_THRESHOLD = 0.0


def read_smiles(path: str) -> list:
    """
    SMILES from a .smi or tab-separated file, skipping a header and blanks.

    The generated .smi holds one line per valid molecule and nothing for the
    rest, so a blank line is not expected here, but the split files may carry a
    property header and tab-separated columns.
    """
    out: list = []
    try:
        with open(path) as handle:
            for i, line in enumerate(handle):
                token = line.split("\t")[0].split()[0] if line.strip() else ""
                if not token:
                    continue
                if i == 0 and token.upper() == "SMILES":
                    continue
                out.append(token)
    except FileNotFoundError:
        return []
    return out


def read_smiles_with_property(path: str, property_name: str):
    """
    SMILES and one property column from a tab-separated .smi, by header name.

    Returns (smiles, values). A dataset without that column returns empty
    lists, which the caller reports as a missing input rather than a zero.
    """
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return [], []
    if not lines:
        return [], []
    header = lines[0].split("\t")
    if header[0].strip().upper() != "SMILES" or property_name not in [
        h.strip() for h in header
    ]:
        return [], []
    column = [h.strip() for h in header].index(property_name)
    smiles: list = []
    values: list = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) <= column or not fields[0].strip():
            continue
        try:
            values.append(float(fields[column]))
        except ValueError:
            continue
        smiles.append(fields[0].strip())
    return smiles, values


def _isomeric(constants) -> bool:
    """
    Whether this job's molecules should be compared with stereochemistry.

    The graph representation only carries a chiral tag when `use_chirality` is
    set, so comparing isomerically on a flat representation makes every
    generated molecule look novel. One accessor, so the funnel, the condition
    metrics and the report grid cannot disagree about it.
    """
    return bool(getattr(constants, "use_chirality", False))


def _shared_run_fields(constants) -> dict:
    return {
        "job_name": str(constants.job_dir).rstrip("/").split("/")[-1],
        "job_type": constants.job_type,
        "dataset": str(constants.dataset_dir).rstrip("/"),
        "seed": constants.seed,
        "device": constants.device,
    }


def _finish(
    constants,
    metrics: dict,
    unavailable: dict,
    *,
    grid_smiles: list | None = None,
    scored: list | None = None,
    conditions: list | None = None,
) -> None:
    payload = _json.collect(
        **_shared_run_fields(constants), metrics=metrics, unavailable=unavailable
    )
    path = _json.write(payload, constants.results_dir)
    print(f"* Metrics written to {path}", flush=True)
    # The report is rendered from the written payload, last, so a failure here
    # cannot cost the metrics, the checkpoints or the molecules.
    report = _html.render(
        payload=payload,
        results_dir=constants.results_dir,
        job_dir=constants.job_dir,
        grid_smiles=grid_smiles,
        scored=scored,
        conditions=conditions,
    )
    print(f"* Report written to {report}", flush=True)


def evaluate_generation(
    constants,
    *,
    generated_smiles: list,
    model=None,
    inference_seconds: float | None = None,
    n_generated_raw: int | None = None,
) -> None:
    """
    metrics.json and report.html for a `generate` job.

    FCD belongs here and not only in the budget-capped checkpoint evaluation:
    CLAUDE.md calls it the headline number for unconditional generation, and
    the test split is its reference set.
    """
    metrics: dict = {}
    unavailable: dict = {}

    training_smiles = read_smiles(constants.training_set)
    test_smiles = read_smiles(constants.test_set)

    core = _core.compute_core_metrics(
        generated_smiles,
        n_sampled=constants.n_samples,
        training_smiles=training_smiles,
        isomeric=_isomeric(constants),
    )
    unavailable.update(core.pop("_filter_unavailable", {}))
    metrics.update(core)

    # How many graphs the generator actually produced before truncation to
    # n_samples. A batch can come back larger than batch_size, so this is
    # normally above n_sampled; the gap is what used to push validity over 1.0.
    if n_generated_raw is not None:
        metrics["validity_n_generated_raw"] = int(n_generated_raw)

    # Whole-PROTAC synthesizability, for bifunctional datasets only.
    _retrotac_metrics(constants, generated_smiles, metrics, unavailable)

    # fcd, against the test split.
    from ._unconditional import _compute_fcd

    if test_smiles:
        # Both sides through the same canonicalization with the same isomeric
        # flag. ChemNet reads SMILES characters, so a flattened generated set
        # against a raw stereo-bearing reference is a distribution shift rather
        # than a comparison.
        isomeric = _isomeric(constants)
        result = _compute_fcd(
            [
                s
                for s in _core.canonical_smiles(generated_smiles, isomeric=isomeric)
                if s
            ],
            [s for s in _core.canonical_smiles(test_smiles, isomeric=isomeric) if s],
        )
        metrics["fcd"] = result["value"]
        metrics["fcd_n_generated"] = result["n_generated"]
        metrics["fcd_n_reference"] = result["n_reference"]
        if result["value"] is None:
            unavailable["fcd"] = result["reason"]
        elif result["biased"]:
            unavailable["fcd_comparability"] = (
                f"insufficient_sample: computed from {result['n_generated']} "
                f"generated against {result['n_reference']} reference molecules, "
                "which is above the floor but below the 10,000 published values "
                "use. Comparable between our own runs, not to a published number."
            )
    else:
        metrics["fcd"] = None
        metrics["fcd_n_generated"] = 0
        metrics["fcd_n_reference"] = 0
        unavailable["fcd"] = (
            f"not_applicable_to_job_type: no test split at {constants.test_set}"
        )

    # Conditional, continuous properties: only when this checkpoint was
    # conditioned and the job said what to ask of it.
    conditions_for_report: list = []
    requested_conditions = getattr(constants, "sample_conditions", None)
    if getattr(constants, "condition_dim", 0) > 0 and requested_conditions:
        metrics.update(
            _condition_metrics(
                constants,
                generated_smiles,
                requested_conditions,
                unavailable,
                conditions_for_report,
            )
        )

    _blocked(metrics, unavailable)

    _efficiency_metrics(
        metrics,
        unavailable,
        model=model,
        inference_seconds=inference_seconds,
        n_sampled=constants.n_samples,
    )

    unique_valid = sorted(
        {
            s
            for s in _core.canonical_smiles(
                generated_smiles, isomeric=_isomeric(constants)
            )
            if s
        }
    )
    _finish(
        constants,
        metrics,
        unavailable,
        grid_smiles=unique_valid,
        conditions=conditions_for_report or None,
    )


def requested_per_molecule(constants, property_name: str, n_molecules: int) -> list:
    """
    The value requested of each generated molecule, in generation order.

    A stratified run asks for a different value per batch, so the request is
    not a single scalar and `condition_coverage_<property>` needs to know which
    molecule was asked for what. `results/condition_requests.json` records one
    request per batch and `.valid` records each written molecule's index in the
    generated set, so the two together recover it.

    Falls back to the single configured value when the job did not stratify.
    """
    requests_path = Path(constants.results_dir) / "condition_requests.json"
    valid_path = Path(constants.results_dir) / "molecules.valid"
    if not requests_path.exists() or not valid_path.exists():
        return []
    try:
        requests = json.loads(requests_path.read_text())
        indices = [int(line) for line in valid_path.read_text().split()]
    except (json.JSONDecodeError, ValueError):
        return []
    if not requests or len(indices) != n_molecules:
        return []

    batch_size = max(1, int(constants.batch_size))
    out: list = []
    for index in indices:
        request = requests[min(index // batch_size, len(requests) - 1)]
        value = request.get(property_name)
        out.append(float(value) if value is not None else None)
    return out


def _condition_metrics(
    constants, generated_smiles, requested_conditions, unavailable, out_conditions
) -> dict:
    """
    `condition_success_rate_<property>` and `condition_coverage_<property>`,
    with the unconditional baseline when one is declared.
    """
    from ._conditions import MissingInput, compute_continuous_condition_metrics
    from ._properties import PROPERTY_REGISTRY

    tolerances = getattr(constants, "condition_tolerance", None) or {}
    baseline_dir = getattr(constants, "baseline_results_dir", None)
    baseline_smiles = read_smiles(f"{baseline_dir}") if baseline_dir else []
    if baseline_smiles:
        _check_baseline_comparable(constants, baseline_dir)

    results: dict = {}
    # In generation order, not sorted, because each molecule's requested value
    # is looked up by its position. Deduplicated while keeping the first
    # occurrence: CLAUDE.md denominates every other property rate over unique
    # valid molecules, and counting duplicates would let a model lift the rate
    # by repeating one on-target molecule.
    ordered = [
        s
        for s in _core.canonical_smiles(generated_smiles, isomeric=_isomeric(constants))
        if s
    ]
    seen: set = set()
    keep_positions: list = []
    achieved_smiles: list = []
    for position, smi in enumerate(ordered):
        if smi in seen:
            continue
        seen.add(smi)
        keep_positions.append(position)
        achieved_smiles.append(smi)
    from rdkit import Chem

    for name, requested_value in requested_conditions.items():
        key = f"condition_success_rate_{name}"
        fn = PROPERTY_REGISTRY.get(name.lower())
        if fn is None:
            results[key] = None
            unavailable[key] = (
                f"no property function named {name}; known: "
                f"{', '.join(sorted(PROPERTY_REGISTRY))}"
            )
            continue
        tolerance = tolerances.get(name) if isinstance(tolerances, dict) else tolerances
        if tolerance is None:
            results[key] = None
            unavailable[key] = (
                f"condition_tolerance is unset for {name}; the metric is defined "
                "as the fraction within a tolerance of the requested value"
            )
            continue

        achieved = []
        for smi in achieved_smiles:
            mol = Chem.MolFromSmiles(smi)
            achieved.append(fn(mol) if mol is not None else None)

        per_molecule = requested_per_molecule(constants, name, len(ordered))
        if per_molecule:
            requested = [per_molecule[i] for i in keep_positions]
        else:
            # The job asked one value of everything, so every molecule shares it.
            requested = [float(requested_value)] * len(achieved)
        for smi, want, got in zip(achieved_smiles, requested, achieved):
            if got is not None and want is not None:
                out_conditions.append((smi, float(want), float(got)))

        _, training_values = read_smiles_with_property(constants.training_set, name)
        if not training_values:
            unavailable[f"condition_coverage_{name}"] = (
                f"the training split at {constants.training_set} carries no "
                f"{name} column, so its deciles are unknown"
            )
        try:
            results.update(
                compute_continuous_condition_metrics(
                    name,
                    requested,
                    achieved,
                    float(tolerance),
                    training_values,
                    baseline_requested=(
                        [float(requested_value)] * len(baseline_smiles)
                        if baseline_smiles
                        else None
                    ),
                    baseline_achieved=(
                        [
                            (
                                fn(Chem.MolFromSmiles(s))
                                if Chem.MolFromSmiles(s) is not None
                                else None
                            )
                            for s in baseline_smiles
                        ]
                        if baseline_smiles
                        else None
                    ),
                )
            )
        except MissingInput as error:
            results[key] = None
            unavailable[key] = " ".join(str(error).split())
        reason = results.pop("_coverage_unavailable_reason", None)
        if reason:
            unavailable[f"condition_coverage_{name}"] = reason
        if not baseline_smiles:
            unavailable[f"baseline_condition_success_rate_{name}"] = (
                "no unconditional baseline sample is declared; set "
                "baseline_results_dir to the .smi of the same model trained "
                "without conditioning and sampled the same way"
            )
            results[f"baseline_condition_success_rate_{name}"] = None
    return results


def evaluate_goal_directed(
    constants,
    *,
    generated_smiles: list,
    n_generated: int,
    scores: list,
    optimization_log: list,
    model=None,
    train_wall_clock_s: float | None = None,
) -> None:
    """metrics.json and report.html for a `goal_directed` job."""
    metrics: dict = {}
    unavailable: dict = {}

    training_smiles = read_smiles(constants.training_set)
    core = _core.compute_core_metrics(
        generated_smiles,
        n_sampled=n_generated,
        training_smiles=training_smiles,
        isomeric=_isomeric(constants),
    )
    unavailable.update(core.pop("_filter_unavailable", {}))
    metrics.update(core)

    # Unique valid molecules with their best score, which is the denominator
    # CLAUDE.md gives success_rate.
    best_by_smiles: dict = {}
    for smi, score in zip(
        _core.canonical_smiles(generated_smiles, isomeric=_isomeric(constants)), scores
    ):
        if smi is None:
            continue
        if smi not in best_by_smiles or score > best_by_smiles[smi]:
            best_by_smiles[smi] = float(score)
    scored = sorted(best_by_smiles.items())

    retrotac_by_smiles = _retrotac_metrics(
        constants, generated_smiles, metrics, unavailable
    )

    threshold = _success_threshold(constants)
    metrics.update(
        _budget.compute_set_level_metrics(
            scored,
            threshold,
            retrotac_by_smiles=retrotac_by_smiles or None,
        )
    )
    metrics["success_threshold"] = threshold

    try:
        metrics.update(_budget.compute_at_budget_metrics(optimization_log))
        metrics["auc_top10"] = _budget.compute_auc_top10(optimization_log)
    except _budget.MissingInput as error:
        reason = " ".join(str(error).split())
        for label in _budget.BUDGETS:
            for key in (
                f"best_at_{label}",
                f"top10_mean_at_{label}",
                f"top100_mean_at_{label}",
            ):
                metrics[key] = None
                unavailable[key] = reason
        metrics["auc_top10"] = None
        unavailable["auc_top10"] = reason

    # Group 7: every one of these waits on an input no experiment declares yet.
    for key, reason in _group7_reasons().items():
        metrics[key] = None
        unavailable[key] = reason

    _blocked(metrics, unavailable, goal_directed=True)

    _efficiency_metrics(
        metrics, unavailable, model=model, train_wall_clock_s=train_wall_clock_s
    )

    _finish(
        constants,
        metrics,
        unavailable,
        grid_smiles=[smi for smi, _ in scored],
        scored=scored,
    )


def evaluate_training(
    constants,
    *,
    model=None,
    train_wall_clock_s: float | None = None,
    nll: dict | None = None,
    uc_jsd: float | None = None,
) -> None:
    """
    metrics.json and report.html for an `unconditional` or `conditional` job.

    A training job samples molecules only to score itself, so the molecule-level
    tables belong to the `generate` job that follows it. What belongs here is
    the model: its size, what it cost, and the likelihood numbers.

    `nll_train`, `nll_validation` and `nll_test` are **per molecule, summed
    over construction actions** -- the molecule's negative log-likelihood under
    the model, not a per-action or per-token average. They are comparable
    across the three splits of one dataset, which is what the three keys are
    for, and not across datasets whose molecules differ in size.
    """
    metrics: dict = {}
    unavailable: dict = {}

    _efficiency_metrics(
        metrics, unavailable, model=model, train_wall_clock_s=train_wall_clock_s
    )

    for key in ("nll_train", "nll_validation", "nll_test"):
        value = (nll or {}).get(key)
        metrics[key] = value
        if value is None:
            unavailable[key] = (
                "per-split NLL is computed by a generate job with "
                "sample_mode: evaluate, not during training"
            )
    metrics["uc_jsd"] = uc_jsd
    if uc_jsd is None:
        unavailable["uc_jsd"] = (
            "uc_jsd is computed by a generate job with sample_mode: evaluate"
        )

    _finish(constants, metrics, unavailable)


def evaluate_model_scoring(
    constants,
    *,
    nll: dict,
    uc_jsd: float | None,
    model=None,
) -> None:
    """metrics.json and report.html for `generate` with `sample_mode: evaluate`."""
    metrics: dict = dict(nll)
    metrics["uc_jsd"] = uc_jsd
    unavailable: dict = {}
    _efficiency_metrics(metrics, unavailable, model=model)
    _finish(constants, metrics, unavailable)


def _check_baseline_comparable(constants, baseline_smi_path: str) -> None:
    """
    Fail unless the baseline sample was drawn the same way as this one.

    A baseline drawn at a different sample size or under a different seed
    silently changes every conditional comparison, and nothing else in the
    pipeline would notice: the numbers stay plausible and the conclusion moves.
    """
    baseline_metrics = Path(baseline_smi_path).parent / "metrics.json"
    if not baseline_metrics.exists():
        raise ValueError(
            f"baseline_results_dir points at {baseline_smi_path}, but there is "
            f"no metrics.json beside it at {baseline_metrics}.\n"
            "The baseline has to be a completed generate job, so its sample "
            "size and seed can be checked against this one."
        )
    baseline = json.loads(baseline_metrics.read_text())
    their_n = baseline["metrics"].get("n_sampled")
    their_seed = baseline["run"].get("seed")
    if their_n != constants.n_samples or their_seed != constants.seed:
        raise ValueError(
            "The unconditional baseline was not sampled like this job, so the "
            "conditional comparison would not be like for like.\n"
            f"  this job: n_samples={constants.n_samples}, seed={constants.seed}\n"
            f"  baseline: n_samples={their_n}, seed={their_seed}\n"
            f"  baseline: {baseline_metrics}\n"
            "Re-run the baseline generate job with the same n_samples and seed."
        )


def _efficiency_metrics(
    metrics: dict,
    unavailable: dict,
    *,
    model=None,
    train_wall_clock_s: float | None = None,
    inference_seconds: float | None = None,
    n_sampled: int | None = None,
) -> None:
    """
    The four model-efficiency keys, always all four.

    An undefined one records null with its reason rather than being left out,
    because an absent key cannot be told apart from a metric nobody attempted.
    """
    metrics["n_parameters"] = None if model is None else _efficiency.n_parameters(model)
    if model is None:
        unavailable["n_parameters"] = "no model in scope for this job"

    metrics["peak_memory_mb"] = _efficiency.peak_memory_mb()

    metrics["train_wall_clock_s"] = (
        None if train_wall_clock_s is None else float(train_wall_clock_s)
    )
    if train_wall_clock_s is None:
        unavailable["train_wall_clock_s"] = "this job type does not train a model"

    metrics["inference_s_per_10k"] = (
        _efficiency.inference_s_per_10k(inference_seconds, n_sampled)
        if inference_seconds is not None and n_sampled
        else None
    )
    if metrics["inference_s_per_10k"] is None:
        unavailable["inference_s_per_10k"] = (
            "this job type does not sample molecules, so there is no generation "
            "time to rescale"
        )


def _retrotac_metrics(constants, generated_smiles, metrics, unavailable) -> dict:
    """
    `retrotac_mean`/`retrotac_median`, for bifunctional datasets only.

    Gated on `bifunctional` because the score has no meaning off the
    distribution it was fitted to: RetroTAC predicts a route-derived score for
    PROTAC-sized molecules, and running it over drug-like output would produce
    a number that looks comparable and is not. SA score is reported either way
    and remains the measure for molecular glues and every drug-like dataset.

    Returns the score per molecule so the goal-directed quality gate can reuse
    it. An empty dict means RetroTAC did not run.
    """
    if not getattr(constants, "bifunctional", False):
        return {}

    unique_valid = sorted(
        {
            s
            for s in _core.canonical_smiles(
                generated_smiles, isomeric=_isomeric(constants)
            )
            if s
        }
    )
    if not unique_valid:
        for key in ("retrotac_mean", "retrotac_median"):
            metrics[key] = None
            unavailable[key] = "no unique valid molecule to score"
        return {}

    from ._retrotac import MissingInput as RetroTACMissing
    from ._retrotac import compute_retrotac_metrics

    try:
        aggregate, per_molecule = compute_retrotac_metrics(unique_valid)
    except RetroTACMissing as error:
        reason = " ".join(str(error).split())
        for key in ("retrotac_mean", "retrotac_median"):
            metrics[key] = None
            unavailable[key] = reason
        return {}

    metrics.update(aggregate)
    for key, value in aggregate.items():
        if value is None:
            unavailable[key] = "RetroTAC scored none of the unique valid molecules"
    return per_molecule


def _success_threshold(constants) -> float:
    """
    The score at or above which a molecule counts as a success.

    CLAUDE.md defines `success_rate` against `success_threshold`, so that is
    the key read here. It used to read `score_thresholds[0]`, which is the RL
    reward-shaping threshold and a different quantity that merely happens to
    share the default value of 0.5.
    """
    configured = getattr(constants, "success_threshold", None)
    if configured is not None:
        return float(configured)
    return DEFAULT_SUCCESS_THRESHOLD


def _blocked(metrics: dict, unavailable: dict, *, goal_directed: bool = False) -> None:
    """
    The three metrics whose definitions CLAUDE.md does not pin down, plus the
    one that depends on them.

    They record null and this reason rather than a number computed from a
    threshold nobody approved: these go in a paper, and a plausible wrong
    number is worse than an absent one.
    """
    # filter_pass_rate and Ro5_pass_rate are implemented; bRo5_pass_rate is
    # not, because its definition could not be verified at the source.
    # quality_filtered_top10_mean is computed in _budget.compute_set_level_metrics,
    # which returns None when nothing survives the filters -- a real outcome for a
    # reward-hacked run, so it gets a reason rather than being left bare.
    if goal_directed and metrics.get("quality_filtered_top10_mean") is None:
        metrics["quality_filtered_top10_mean"] = None
        unavailable.setdefault(
            "quality_filtered_top10_mean",
            "no unique valid molecule passed the structural filters with an SA "
            "score below 4.5",
        )


def _group7_reasons() -> dict:
    return {
        "vs_baseline_top10_at_1k": "no screening library is declared by any experiment",
        "vs_baseline_top10_at_5k": "no screening library is declared by any experiment",
        "vs_baseline_top10_at_10k": "no screening library is declared by any experiment",
        "vs_delta_top10_at_1k": "needs vs_baseline_top10_at_1k",
        "vs_delta_top10_at_5k": "needs vs_baseline_top10_at_5k",
        "vs_delta_top10_at_10k": "needs vs_baseline_top10_at_10k",
        "eval_oracle_top10_mean": "no evaluation surrogate is trained yet",
        "top100_outside_domain_rate": "no optimization surrogate training set is available yet",
    }
