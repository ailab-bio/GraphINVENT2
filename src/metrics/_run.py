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

# CLAUDE.md: class_match_rate is the fraction above a stated threshold of the
# maximum ECFP4 Tanimoto to the requested class's training molecules. 0.4 is
# the conventional "same chemical series" floor for ECFP4 and is where a
# warhead analog sits while an unrelated molecule does not; it is a constant
# rather than a config key because no planned experiment varies it, and it is
# emitted as `class_match_threshold` so a reader never has to find it here.
CLASS_MATCH_THRESHOLD = 0.4


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


def read_smiles_with_labels(path: str, class_names: list) -> tuple:
    """
    SMILES and the requested class of each, from a one-hot column block.

    Returns `(smiles, labels)`. A row whose one-hot is not exactly one class is
    carried with a label of None, so a multi-label molecule is visible to the
    caller rather than dropped: `delta_nll` needs it in order to exclude the
    molecule's legitimate alternative classes, and the similarity metrics need
    it excluded.
    """
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return [], []
    if not lines:
        return [], []
    header = [h.strip() for h in lines[0].split("\t")]
    if header[0].upper() != "SMILES":
        return [], []
    missing = [name for name in class_names if name not in header]
    if missing:
        return [], []
    columns = {name: header.index(name) for name in class_names}

    smiles: list = []
    labels: list = []
    for line in lines[1:]:
        fields = line.split("\t")
        if not fields or not fields[0].strip():
            continue
        hot = []
        for name, column in columns.items():
            if len(fields) <= column:
                continue
            try:
                if float(fields[column]) > 0.5:
                    hot.append(name)
            except ValueError:
                continue
        smiles.append(fields[0].strip())
        labels.append(hot[0] if len(hot) == 1 else None)
    return smiles, labels


def read_multilabel(path: str, class_names: list) -> dict:
    """
    `{smiles: set of classes}` for every molecule carrying more than one.

    `delta_nll` excludes a multi-label molecule's other legitimate classes from
    the competing set, and the number excluded is reported: a panel where every
    molecule belongs to three classes has very little left to compete against,
    and the metric has to be read knowing that.
    """
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return {}
    if not lines:
        return {}
    header = [h.strip() for h in lines[0].split("\t")]
    if header[0].upper() != "SMILES" or any(n not in header for n in class_names):
        return {}
    columns = {name: header.index(name) for name in class_names}
    out: dict = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if not fields or not fields[0].strip():
            continue
        hot = set()
        for name, column in columns.items():
            if len(fields) > column:
                try:
                    if float(fields[column]) > 0.5:
                        hot.add(name)
                except ValueError:
                    continue
        if len(hot) > 1:
            out[fields[0].strip()] = hot
    return out


def read_novelty_reference(constants) -> tuple:
    """
    The molecules novelty is measured against, and how many corpora they span.

    Every training split in the checkpoint's lineage, not only the task
    corpus: a transfer-learned model that reproduces a base-corpus molecule
    has not produced a novel molecule, and scoring it as one biases the
    transfer arm of the transfer-versus-scratch comparison.
    """
    paths = list(
        getattr(constants, "novelty_reference_sets", None) or [constants.training_set]
    )
    molecules, found = [], []
    for path in paths:
        smiles = read_smiles(path)
        if smiles:
            molecules += smiles
            found.append(path)
    return molecules, found


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
    draw=None,
) -> None:
    payload = _json.collect(
        **_shared_run_fields(constants), metrics=metrics, unavailable=unavailable
    )
    path = _json.write(payload, constants.results_dir)
    print(f"* Metrics written to {path}", flush=True)
    # Figures after metrics.json and before the report: the report embeds the
    # PNGs, and anything that fails here has already cost nothing, because the
    # numbers are on disk. A figure whose input is missing says so and is
    # skipped; anything else is a bug and crashes rather than writing a report
    # that quietly has no figures in it.
    if draw is not None:
        for figure in draw(payload["metrics"]):
            print(f"* Figure written to {figure}", flush=True)
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
    per_class_nll: dict | None = None,
) -> None:
    """
    metrics.json and report.html for a `generate` job.

    FCD belongs here and not only in the budget-capped checkpoint evaluation:
    CLAUDE.md calls it the headline number for unconditional generation, and
    the test split is its reference set.
    """
    metrics: dict = {}
    unavailable: dict = {}

    training_smiles, reference_corpora = read_novelty_reference(constants)
    test_smiles = read_smiles(constants.test_set)

    core = _core.compute_core_metrics(
        generated_smiles,
        n_sampled=constants.n_samples,
        training_smiles=training_smiles,
        isomeric=_isomeric(constants),
        property_window=getattr(constants, "property_window", None),
    )
    unavailable.update(core.pop("_filter_unavailable", {}))
    metrics.update(core)

    # Which corpora novelty was measured against. A novelty of 0.9 against one
    # corpus and against three are different numbers.
    metrics["novelty_reference_corpora"] = [
        str(Path(p).parent.name) for p in reference_corpora
    ]
    metrics["novelty_n_reference_molecules"] = len(training_smiles)

    # How many graphs the generator actually produced before truncation to
    # n_samples. A batch can come back larger than batch_size, so this is
    # normally above n_sampled; the gap is what used to push validity over 1.0.
    if n_generated_raw is not None:
        metrics["validity_n_generated_raw"] = int(n_generated_raw)

    # Whole-PROTAC synthesizability, for bifunctional datasets only. The
    # per-molecule scores come back so the triage stack can reuse them rather
    # than paying for a second pass.
    retrotac_by_smiles = _retrotac_metrics(
        constants, generated_smiles, metrics, unavailable
    )

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

    # Whole-sample component metrics, for bifunctional datasets only.
    _component_metrics(constants, generated_smiles, metrics, unavailable)

    # Conditional metrics: only when this checkpoint was conditioned and the
    # job said what to ask of it. Which set is defined depends on what the
    # condition is. `condition_classes` declares the condition to be a class
    # label, and CLAUDE.md gives the continuous metrics no definition in that
    # case: a success rate within a tolerance of 1.0 on a one-hot column
    # measures whether the model echoed its own input.
    conditions_for_report: list = []
    requested_conditions = getattr(constants, "sample_conditions", None)
    if getattr(constants, "condition_dim", 0) > 0 and requested_conditions:
        if getattr(constants, "condition_classes", None):
            metrics.update(
                _categorical_condition_metrics(
                    constants,
                    generated_smiles,
                    unavailable,
                    per_class_nll=per_class_nll,
                )
            )
        else:
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

    # Triage, and every metric again over what survives it. Both sets are
    # reported and neither replaces the other: the unfiltered set says what the
    # model produces, which is what a generative result is, and the filtered
    # set says what survives triage, which is what a project would carry
    # forward. A paper quoting only the filtered numbers describes the filter;
    # one quoting only the unfiltered numbers describes something nobody would
    # make.
    _filtered_metrics(
        constants,
        unique_valid,
        metrics,
        unavailable,
        training_smiles=training_smiles,
        retrotac_by_smiles=retrotac_by_smiles,
    )

    def draw(written: dict) -> list:
        from . import _figures

        paths = _figures.generate_figures(
            constants.results_dir,
            written,
            unique_valid,
            training_smiles,
            test_smiles,
        )
        if getattr(constants, "condition_dim", 0) > 0:
            paths += _figures.conditional_figures(
                constants.results_dir, written, conditions_for_report
            )
        return paths

    _finish(
        constants,
        metrics,
        unavailable,
        grid_smiles=unique_valid,
        conditions=conditions_for_report or None,
        draw=draw,
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

    def unavailable_group(name: str, category: str, reason: str) -> None:
        """
        Record the whole continuous group for one property as a categorized
        null.

        Every bail-out used to set only `condition_success_rate_<property>`, so
        a property that could not be measured left `condition_mae_` and
        `condition_coverage_` absent from metrics.json rather than present and
        null. A key that is missing cannot be told from a key nobody defined,
        which is what the envelope exists to prevent.
        """
        for metric in (
            f"condition_success_rate_{name}",
            f"condition_mae_{name}",
            f"condition_coverage_{name}",
        ):
            results[metric] = None
            unavailable[metric] = f"{category}: {reason}"

    for name, requested_value in requested_conditions.items():
        fn = PROPERTY_REGISTRY.get(name.lower())
        if fn is None:
            unavailable_group(
                name,
                "not_applicable_to_job_type",
                f"no property function named {name}, so the achieved value "
                f"cannot be computed; known: {', '.join(sorted(PROPERTY_REGISTRY))}. "
                "A one-hot class label is a categorical condition, and the "
                "continuous metrics have no definition for it.",
            )
            continue
        tolerance = tolerances.get(name) if isinstance(tolerances, dict) else tolerances
        if tolerance is None:
            unavailable_group(
                name,
                "not_applicable_to_job_type",
                f"condition_tolerance is unset for {name}; the metric is "
                "defined as the fraction within a tolerance of the requested "
                "value",
            )
            continue

        achieved = []
        for smi in achieved_smiles:
            mol = Chem.MolFromSmiles(smi)
            achieved.append(fn(mol) if mol is not None else None)

        per_molecule = requested_per_molecule(constants, name, len(ordered))
        if per_molecule:
            requested = [per_molecule[i] for i in keep_positions]
        elif isinstance(requested_value, (int, float)):
            # The job asked one value of everything, so every molecule shares it.
            requested = [float(requested_value)] * len(achieved)
        else:
            # A sentinel such as `from_train` means the value differs per batch
            # and is recoverable only from results/condition_requests.json
            # together with molecules.valid. When those do not line up, there
            # is no requested value to compare against, and coercing the
            # sentinel to a float raised instead of saying so.
            unavailable_group(
                name,
                "not_applicable_to_job_type",
                f"sample_conditions.{name} is '{requested_value}', so the "
                "value asked of each molecule comes from "
                "results/condition_requests.json, and that could not be "
                "matched to the generated molecules. It needs one request per "
                "batch and a molecules.valid whose length equals the number "
                "of valid molecules.",
            )
            continue
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
                        if baseline_smiles and isinstance(requested_value, (int, float))
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
            unavailable_group(name, "insufficient_sample", " ".join(str(error).split()))
        reason = results.pop("_coverage_unavailable_reason", None)
        if reason:
            unavailable[f"condition_coverage_{name}"] = reason
        if not baseline_smiles:
            baseline_reason = (
                "no unconditional baseline sample is declared; set "
                "baseline_results_dir to the .smi of the same model trained "
                "without conditioning and sampled the same way"
            )
            for baseline_key in (
                f"baseline_condition_success_rate_{name}",
                f"baseline_condition_mae_{name}",
            ):
                results[baseline_key] = None
                unavailable[baseline_key] = baseline_reason
    return results


def _requested_labels(constants, class_names: list, n_molecules: int) -> list:
    """
    The class asked of each generated molecule, in generation order.

    A stratified categorical run cycles the one-hot across batches, so the
    request is not one class for the whole sample and every per-class number
    depends on recovering which molecule was asked for what.
    `results/condition_requests.json` holds one request per batch and
    `molecules.valid` each written molecule's index, which together give the
    per-molecule answer; a job that asked one class of everything falls back to
    that class.
    """
    from ._conditions import requested_label

    per_class = {
        name: requested_per_molecule(constants, name, n_molecules)
        for name in class_names
    }
    if all(per_class[name] for name in class_names):
        out: list = []
        for index in range(n_molecules):
            one_hot = {name: per_class[name][index] for name in class_names}
            out.append(requested_label(one_hot, class_names))
        return out

    # `all_classes` is recoverable only from condition_requests.json together
    # with molecules.valid, and we are here because those did not line up --
    # usually because the run produced no valid molecule at all. There is no
    # class to fall back to, so every molecule is unlabelled and the caller
    # reports the block as a categorized null rather than attributing the whole
    # sample to one class.
    one_hot = getattr(constants, "sample_conditions", None)
    if not isinstance(one_hot, dict):
        return [None] * n_molecules
    return [requested_label(one_hot, class_names)] * n_molecules


def _categorical_condition_metrics(
    constants,
    generated_smiles: list,
    unavailable: dict,
    *,
    per_class_nll: dict | None = None,
) -> dict:
    """
    The categorical conditional block: the macro-averaged metrics, the
    per-class breakdown, and both baselines.

    CLAUDE.md's continuous metrics have no definition for a class label, so
    this replaces them rather than running alongside: a success rate within a
    tolerance of 1.0 on a one-hot column measures whether the model echoed its
    own input, not whether it generated that class's chemistry.
    """
    from ._components import (
        COMPONENT_SAMPLE_SIZE,
        compute_class_warhead_metrics,
        warhead_records,
    )
    from ._components import MissingInput as ComponentMissing
    from ._conditions import (
        MissingInput,
        categorical_block,
        delta_nll,
        posthoc_filter_baseline,
    )

    class_names = list(getattr(constants, "condition_classes", None) or [])
    isomeric = _isomeric(constants)
    results: dict = {}

    ordered = [
        s for s in _core.canonical_smiles(generated_smiles, isomeric=isomeric) if s
    ]
    labels = _requested_labels(constants, class_names, len(ordered))

    # Deduplicated keeping the first occurrence, as the continuous path does:
    # every rate here is over unique valid molecules, and counting duplicates
    # would let a model lift class_match_rate by repeating one on-class
    # molecule.
    seen: set = set()
    sample_smiles: list = []
    sample_labels: list = []
    for smi, label in zip(ordered, labels):
        if smi in seen:
            continue
        seen.add(smi)
        sample_smiles.append(smi)
        sample_labels.append(label)

    train_smiles, train_labels = read_smiles_with_labels(
        constants.training_set, class_names
    )
    training_by_label: dict = {name: [] for name in class_names}
    for smi, label in zip(train_smiles, train_labels):
        if label in training_by_label:
            training_by_label[label].append(smi)

    test_smiles, test_labels = read_smiles_with_labels(constants.test_set, class_names)
    reference_pool = [
        (smi, label) for smi, label in zip(test_smiles, test_labels) if label
    ]

    # delta_nll first, and outside everything that depends on the generated
    # sample. It is a property of the model and the held-out split: it asks
    # whether a test molecule of class p is more likely under condition p
    # than under the competing conditions, and a run that generated nothing
    # valid still has that number. Reporting it as unavailable because the
    # sample was empty would hide the one measurement that survived.
    #
    # It needs the model's NLL for a held-out molecule under each candidate
    # class, which is k forward passes per molecule. The generation phase
    # computes it and hands it over, because metrics/ holds no model.
    if per_class_nll and per_class_nll.get("per_molecule"):
        results["delta_nll"] = delta_nll(
            {int(k): v for k, v in per_class_nll["per_molecule"].items()},
            per_class_nll["true_labels"],
            alternative_labels={
                int(k): set(v)
                for k, v in (per_class_nll.get("alternatives") or {}).items()
            },
        )
        results["delta_nll_n_molecules"] = len(per_class_nll["per_molecule"])
        results["delta_nll_n_alternatives_excluded"] = int(
            per_class_nll.get("n_alternatives_excluded", 0)
        )
        if results["delta_nll"] is None:
            unavailable["delta_nll"] = (
                "insufficient_sample: no held-out molecule had both its own "
                "class and a competing class to compare against"
            )
    else:
        results["delta_nll"] = None
        unavailable["delta_nll"] = (
            "insufficient_sample: delta_nll needs the model's NLL for each "
            "held-out molecule under each class in the panel, and the "
            "generation phase produced none. It requires a test split carrying "
            "the class columns and a preprocessed test HDF5."
        )

    def group_unavailable(category: str, reason: str) -> dict:
        """
        Every sample-dependent key in the block as a categorized null.

        It updates `results` and returns it rather than returning only the
        nulls: returning a fresh dictionary discarded `delta_nll` and its two
        supporting fields, which are computed above and do not depend on the
        sample at all.
        """
        for key in (
            "class_match_median",
            "class_match_rate",
            "class_selectivity_margin",
            "delta_nll",
            "warhead_novelty_in_class",
            *[f"knn_purity_{k}" for k in (5, 10, 20)],
        ):
            if results.get(key) is not None:
                continue
            results[key] = None
            unavailable[key] = f"{category}: {reason}"
        return results

    if not class_names:
        return group_unavailable(
            "not_applicable_to_job_type",
            "condition_classes is unset, so the panel's classes are unknown",
        )
    if not any(training_by_label.values()):
        return group_unavailable(
            "insufficient_sample",
            f"the training split at {constants.training_set} carries no "
            f"single-label molecule for any of {class_names}; the categorical "
            "metrics are all measured against that class's training chemistry",
        )
    if not any(label for label in sample_labels):
        return group_unavailable(
            "insufficient_sample",
            f"none of the {len(sample_smiles)} unique valid molecules could be "
            "attributed to a requested class. Every categorical metric is "
            "per-class, so there is nothing to compute. A run that produced no "
            "valid molecule lands here, as does one whose "
            "results/condition_requests.json and molecules.valid do not line up.",
        )
    if not reference_pool:
        unavailable["knn_reference_pool_size"] = (
            "insufficient_sample: the test split at "
            f"{constants.test_set} carries no labelled molecule, so there is "
            "no held-out pool for knn_purity"
        )

    threshold = float(
        getattr(constants, "class_match_threshold", CLASS_MATCH_THRESHOLD)
    )
    metrics, block_unavailable, per_class = categorical_block(
        sample_smiles,
        sample_labels,
        class_names,
        training_by_label,
        reference_pool,
        threshold=threshold,
        training_smiles=read_novelty_reference(constants)[0],
        isomeric=isomeric,
    )
    results.update(metrics)
    unavailable.update(block_unavailable)

    # The per-class warhead table, for a bifunctional dataset. This is the half
    # of warhead_* that is conditional: the whole-sample half is in
    # _component_metrics and stays whole-sample.
    warhead_rows: dict = {}
    if getattr(constants, "bifunctional", False):
        warheads_by_label: dict = {name: [] for name in class_names}
        try:
            for name in class_names:
                for record in warhead_records(
                    _subsample(training_by_label[name], constants)
                ):
                    if record["status"] == "ok":
                        warheads_by_label[name].append(record["warhead"])
            for name in class_names:
                molecules = [
                    smi
                    for smi, label in zip(sample_smiles, sample_labels)
                    if label == name
                ]
                records = warhead_records(_subsample(molecules, constants))
                row, row_unavailable = compute_class_warhead_metrics(
                    records,
                    warheads_by_label[name],
                    n_unique_valid=len(set(molecules)),
                    isomeric=isomeric,
                )
                warhead_rows[name] = row
                # The row's keys are already CLAUDE.md's names, so the class
                # suffix is all that is added. Prefixing them a second time
                # produced warhead_warhead_usable_rate_<class>.
                for key, value in row.items():
                    results[f"{key}_{name}"] = value
                for key, reason in row_unavailable.items():
                    unavailable[f"{key}_{name}"] = reason

            # Only `warhead_novelty_in_class` gets a macro-average under its
            # bare name. The other four rates already have that name at the
            # whole-sample level, computed over the whole sample rather than as
            # a mean over classes, and two different numbers under one key is
            # the thing the flat dictionary cannot survive. Their per-class
            # values are there with the suffix; their panel-level value is the
            # whole-sample one.
            key = "warhead_novelty_in_class"
            values = [
                row[key] for row in warhead_rows.values() if row.get(key) is not None
            ]
            results[key] = float(sum(values) / len(values)) if values else None
            if not values:
                unavailable[key] = (
                    "insufficient_sample: no class in the panel produced a "
                    f"value for {key}"
                )
            results["component_sample_size"] = COMPONENT_SAMPLE_SIZE
            results["component_sample_seed"] = int(getattr(constants, "seed", 0) or 0)
        except ComponentMissing as error:
            reason = " ".join(str(error).split())
            for name in class_names:
                for key in (
                    "split_success_rate",
                    "warhead_usable_rate",
                    "warhead_uniqueness",
                    "warhead_diversity",
                    "warhead_novelty_in_class",
                ):
                    results[f"{key}_{name}"] = None
                    unavailable[f"{key}_{name}"] = reason
            results["warhead_novelty_in_class"] = None
            unavailable["warhead_novelty_in_class"] = reason

    # Baseline 1: the training molecules of the requested label, sampled to the
    # same N per class. The retrieval ceiling.
    import random as _random

    rng = _random.Random(int(getattr(constants, "seed", 0) or 0))
    counts = {
        name: sum(1 for label in sample_labels if label == name) for name in class_names
    }
    lookup_smiles: list = []
    lookup_labels: list = []
    for name in class_names:
        pool = training_by_label[name]
        want = min(counts[name], len(pool))
        for smi in rng.sample(pool, want) if want else []:
            lookup_smiles.append(smi)
            lookup_labels.append(name)
    if lookup_smiles:
        lookup, lookup_unavailable, lookup_per_class = categorical_block(
            lookup_smiles,
            lookup_labels,
            class_names,
            training_by_label,
            reference_pool,
            threshold=threshold,
            training_smiles=read_novelty_reference(constants)[0],
            isomeric=isomeric,
        )
        for key, value in lookup.items():
            results[f"baseline_lookup_{key}"] = value
        for key, reason in lookup_unavailable.items():
            unavailable[f"baseline_lookup_{key}"] = reason
    else:
        lookup_per_class = {}
        results["baseline_lookup_class_selectivity_margin"] = None
        unavailable["baseline_lookup_class_selectivity_margin"] = (
            "insufficient_sample: no training molecule carries any class in the "
            "panel, so there is nothing to draw the lookup baseline from"
        )

    # Baseline 2: the unconditional model sampled the same way and reranked per
    # class. This is the one that can falsify the conditional claim.
    baseline_dir = getattr(constants, "baseline_results_dir", None)
    baseline_smiles = read_smiles(f"{baseline_dir}") if baseline_dir else []
    posthoc_per_class: dict = {}
    if baseline_smiles:
        _check_baseline_comparable(constants, baseline_dir)
        try:
            posthoc_smiles, posthoc_labels = posthoc_filter_baseline(
                baseline_smiles, class_names, training_by_label, counts
            )
            posthoc, posthoc_unavailable, posthoc_per_class = categorical_block(
                posthoc_smiles,
                posthoc_labels,
                class_names,
                training_by_label,
                reference_pool,
                threshold=threshold,
                training_smiles=read_novelty_reference(constants)[0],
                isomeric=isomeric,
            )
            for key, value in posthoc.items():
                results[f"baseline_posthoc_{key}"] = value
            for key, reason in posthoc_unavailable.items():
                unavailable[f"baseline_posthoc_{key}"] = reason
        except MissingInput as error:
            results["baseline_posthoc_class_selectivity_margin"] = None
            unavailable["baseline_posthoc_class_selectivity_margin"] = (
                "insufficient_sample: " + " ".join(str(error).split())
            )
    else:
        for key in (
            "baseline_posthoc_class_match_median",
            "baseline_posthoc_class_match_rate",
            "baseline_posthoc_class_selectivity_margin",
        ):
            results[key] = None
            unavailable[key] = (
                "not_applicable_to_job_type: the post-hoc filter baseline needs "
                "an unconditional sample from the same architecture to rerank; "
                "set baseline_results_dir to its molecules.smi. Without it the "
                "conditional numbers cannot be falsified, only reported."
            )

    # The per-class table in full, beside metrics.json rather than inside it.
    table = {
        "class_panel": class_names,
        "class_match_threshold": threshold,
        "conditional": per_class,
        "baseline_lookup": lookup_per_class,
        "baseline_posthoc": posthoc_per_class,
        "warheads": warhead_rows,
    }
    path = Path(constants.results_dir) / "condition_per_class.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    return results


def _subsample(smiles: list, constants) -> list:
    """
    At most `COMPONENT_SAMPLE_SIZE` molecules, drawn with the job's own seed.

    Decomposition costs seconds per molecule, so a component rate over a full
    10,000-molecule sample is hours of wall clock for a number that a seeded
    2,000-molecule subsample pins to well under a percentage point. The size
    and seed are reported next to the rates.
    """
    import random as _random

    from ._components import COMPONENT_SAMPLE_SIZE

    unique = sorted(set(s for s in smiles if s))
    if len(unique) <= COMPONENT_SAMPLE_SIZE:
        return unique
    return _random.Random(int(getattr(constants, "seed", 0) or 0)).sample(
        unique, COMPONENT_SAMPLE_SIZE
    )


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

    training_smiles, reference_corpora = read_novelty_reference(constants)
    core = _core.compute_core_metrics(
        generated_smiles,
        n_sampled=n_generated,
        training_smiles=training_smiles,
        isomeric=_isomeric(constants),
        property_window=getattr(constants, "property_window", None),
    )
    unavailable.update(core.pop("_filter_unavailable", {}))
    metrics.update(core)

    # Which corpora novelty was measured against. A novelty of 0.9 against one
    # corpus and against three are different numbers.
    metrics["novelty_reference_corpora"] = [
        str(Path(p).parent.name) for p in reference_corpora
    ]
    metrics["novelty_n_reference_molecules"] = len(training_smiles)

    # The same supporting field the generate path records. Without it the RL
    # funnel's denominator cannot be reconstructed from metrics.json, and the
    # trajectory over-produces for exactly the reason the generate path does.
    metrics["validity_n_generated_raw"] = int(n_generated)

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
    # The words behind the threshold. CLAUDE.md requires it as a supporting
    # field, five job configs set it, and nothing carried it into metrics.json,
    # so a table showed a success rate with no statement of what was counted.
    definition = getattr(constants, "success_definition", None)
    metrics["success_definition"] = definition or None
    if not definition:
        unavailable["success_definition"] = (
            "not_implemented: this experiment's config does not set "
            "success_definition, so nothing records in words what success_rate "
            "counted"
        )

    # The whole-sample component metrics. Computed over a fresh sample from the
    # best agent, not over the optimization trajectory, because CLAUDE.md takes
    # generation-quality metrics from the model you would deploy; the top-100
    # keys below come from the trajectory, which is what spending oracle calls
    # means.
    _component_metrics(constants, generated_smiles, metrics, unavailable)

    # top100_warhead_scaffold_count, for a bifunctional dataset only. Emitted
    # nowhere else, as with the retrotac_* keys: off a bifunctional dataset
    # there is nothing to decompose, and a key reporting null for every
    # drug-like run would say nothing. The whole-molecule
    # top100_scaffold_count stays beside it and is expected to be
    # uninformative there, which is the reason this key exists.
    if getattr(constants, "bifunctional", False):
        from ._components import MissingInput as ComponentMissing
        from ._components import compute_top100_warhead_scaffolds

        try:
            warhead_scaffolds = compute_top100_warhead_scaffolds(
                [smi for smi, _ in _budget.top100(scored)]
            )
            metrics.update(warhead_scaffolds)
            if warhead_scaffolds["top100_warhead_scaffold_count"] is None:
                unavailable["top100_warhead_scaffold_count"] = (
                    "PROTAC-Splitter decomposed none of the "
                    f"{min(len(scored), 100)} highest-scoring molecules, so "
                    "there "
                    "are no target-binding components to count scaffolds over"
                )
        except ComponentMissing as error:
            split_reason = " ".join(str(error).split())
            for key in (
                "top100_warhead_scaffold_count",
                "top100_split_success_rate",
            ):
                metrics[key] = None
                unavailable[key] = split_reason

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

    _screening_baseline(constants, metrics, unavailable)

    # Group 7: every one of these waits on an input no experiment declares yet.
    for key, reason in _group7_reasons().items():
        metrics[key] = None
        unavailable[key] = reason

    _blocked(metrics, unavailable, goal_directed=True)

    _efficiency_metrics(
        metrics, unavailable, model=model, train_wall_clock_s=train_wall_clock_s
    )

    def draw(written: dict) -> list:
        from . import _figures

        return _figures.goal_directed_figures(
            constants.results_dir,
            written,
            str(Path(constants.logs_dir) / "score.log"),
            optimization_log,
        )

    _finish(
        constants,
        metrics,
        unavailable,
        grid_smiles=[smi for smi, _ in scored],
        scored=scored,
        draw=draw,
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

    # The fraction of training examples trained with the condition dropped.
    # Supporting field, not a metric: the unconditional baseline of every
    # conditional number is only a real measurement when this is non-zero, so
    # a reader comparing two runs has to be able to see it without opening the
    # config.
    if getattr(constants, "condition_dim", 0) > 0:
        metrics["condition_dropout"] = float(
            getattr(constants, "condition_dropout", 0.0) or 0.0
        )

    def draw(_written: dict) -> list:
        from . import _figures

        # A transfer job overlays the from-scratch control, which is the only
        # way the transfer claim is legible: the same final loss reached in
        # fewer epochs is the claim, and one curve cannot show it. The control
        # is the job named by `control_job_dir`, if the experiment declared one.
        control = getattr(constants, "control_job_dir", None)
        return _figures.training_figure(
            constants.results_dir,
            str(Path(constants.logs_dir) / "convergence.log"),
            control_log=(
                str(Path(control) / "logs" / "convergence.log") if control else None
            ),
        )

    _finish(constants, metrics, unavailable, draw=draw)


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
        unavailable["train_wall_clock_s"] = (
            "not_applicable_to_job_type: this job type does not train a model"
        )

    metrics["inference_s_per_10k"] = (
        _efficiency.inference_s_per_10k(inference_seconds, n_sampled)
        if inference_seconds is not None and n_sampled
        else None
    )
    if metrics["inference_s_per_10k"] is None:
        unavailable["inference_s_per_10k"] = (
            "not_applicable_to_job_type: this job type does not sample "
            "molecules, so there is no generation time to rescale"
        )


# The whole-sample component keys, so an unavailable block can name all of
# them with one reason rather than leaving a reader to guess which are defined.
COMPONENT_KEYS = (
    "split_success_rate",
    "warhead_usable_rate",
    "e3_recognizable_rate",
    "warhead_validity",
    "warhead_uniqueness",
    "warhead_novelty",
    "warhead_internal_diversity",
    "warhead_scaffold_diversity",
    "warhead_sa_score_mean",
    "warhead_sa_score_median",
)


def _filtered_metrics(
    constants,
    unique_valid: list,
    metrics: dict,
    unavailable: dict,
    *,
    training_smiles: list,
    retrotac_by_smiles: dict | None = None,
) -> None:
    """
    The triage pass rates, and the core and component tables recomputed over
    what survives.

    Every surviving key carries a `_filtered` suffix, so the two sets are
    unmistakable in one table. The denominator stays `n_samples`, which is what
    makes `validity_filtered` read as "the fraction of the sample that both
    parsed and survived triage" rather than as a rate over a set whose size
    depends on the filter.

    Not everything has a filtered twin, and the ones that do not are model
    properties rather than sample properties: `nll_*`, `uc_jsd`, the efficiency
    keys and `delta_nll` do not change when molecules are removed from a
    sample they were never computed over.
    """
    from ._filters import triage

    bifunctional = bool(getattr(constants, "bifunctional", False))

    # The warhead of each molecule, for the bifunctional SA gate. Served from
    # the splitter cache, which the component metrics have already filled.
    warhead_by_smiles: dict = {}
    if bifunctional and unique_valid:
        from ._components import MissingInput as ComponentMissing
        from ._components import warhead_records

        try:
            for record in warhead_records(_subsample(unique_valid, constants)):
                if record["status"] == "ok":
                    warhead_by_smiles[record["molecule"]] = record["warhead"]
        except ComponentMissing as error:
            reason = " ".join(str(error).split())
            metrics["triage_pass_rate"] = None
            unavailable["triage_pass_rate"] = reason
            return

    kept, rates = triage(
        unique_valid,
        bifunctional=bifunctional,
        retrotac_by_smiles=retrotac_by_smiles,
        warhead_by_smiles=warhead_by_smiles,
    )
    metrics.update(rates)

    if not kept:
        metrics["validity_filtered"] = 0.0
        unavailable["validity_filtered"] = (
            "insufficient_sample: no unique valid molecule survived triage "
            f"({' then '.join(rates['triage_stages'])}), which on a weak model "
            "is a real outcome rather than a bug"
        )
        return

    core = _core.compute_core_metrics(
        kept,
        n_sampled=constants.n_samples,
        training_smiles=training_smiles,
        isomeric=_isomeric(constants),
        property_window=getattr(constants, "property_window", None),
    )
    for key, reason in core.pop("_filter_unavailable", {}).items():
        unavailable[f"{key}_filtered"] = reason
    for key, value in core.items():
        metrics[f"{key}_filtered"] = value

    if bifunctional:
        from ._components import MissingInput as ComponentMissing
        from ._components import compute_component_metrics

        try:
            component, _ = compute_component_metrics(
                _subsample(kept, constants),
                n_sampled=len(_subsample(kept, constants)),
                training_warheads=_training_warheads(constants),
                isomeric=_isomeric(constants),
            )
        except ComponentMissing as error:
            reason = " ".join(str(error).split())
            for key in COMPONENT_KEYS:
                metrics[f"{key}_filtered"] = None
                unavailable[f"{key}_filtered"] = reason
            return
        for key, value in component.items():
            metrics[f"{key}_filtered"] = value


def _training_warheads(constants) -> list:
    """
    The reference warheads `warhead_novelty` is measured against.

    Derived by decomposing a seeded subsample of the training split rather than
    read from a file, so there is nothing to keep in sync and no way for the
    reference set to be the one a previous dataset version produced. On a
    bifunctional dataset the `warhead_scaffold` split has already decomposed
    every training molecule, so the splitter cache serves this and it costs
    nothing.
    """
    from ._components import warhead_records

    records = warhead_records(
        _subsample(read_smiles(constants.training_set), constants)
    )
    return [r["warhead"] for r in records if r["status"] == "ok"]


def _component_metrics(constants, generated_smiles, metrics, unavailable) -> None:
    """
    `split_success_rate`, `warhead_usable_rate`, `warhead_*` and
    `e3_recognizable_rate` over the whole sample, for bifunctional datasets.

    Gated on `bifunctional`, which is what gates every metric in CLAUDE.md's
    component section. On a drug-like dataset the keys are absent rather than
    null: a warhead is not defined for a molecule that has no components, and a
    null would read as a decomposition that failed.

    The sample is subsampled to `COMPONENT_SAMPLE_SIZE` unique valid molecules
    with the job's own seed, because decomposition costs seconds per molecule.
    The size and the seed are reported next to the rates, since a rate over a
    subsample is a different number from a rate over the sample.
    """
    if not getattr(constants, "bifunctional", False):
        return

    from ._components import COMPONENT_SAMPLE_SIZE, compute_component_metrics
    from ._components import MissingInput as ComponentMissing

    sample = _subsample(
        _core.canonical_smiles(generated_smiles, isomeric=_isomeric(constants)),
        constants,
    )
    if not sample:
        for key in COMPONENT_KEYS:
            metrics[key] = None
            unavailable[key] = (
                "insufficient_sample: no unique valid molecule to decompose"
            )
        return

    training_warheads = _training_warheads(constants)
    try:
        component, records = compute_component_metrics(
            sample,
            n_sampled=len(sample),
            training_warheads=training_warheads,
            isomeric=_isomeric(constants),
        )
    except ComponentMissing as error:
        reason = " ".join(str(error).split())
        for key in COMPONENT_KEYS:
            metrics[key] = None
            unavailable[key] = reason
        return

    metrics.update(component)
    metrics["component_sample_size"] = min(COMPONENT_SAMPLE_SIZE, len(sample))
    metrics["component_sample_seed"] = int(getattr(constants, "seed", 0) or 0)
    if not training_warheads:
        for key in ("warhead_novelty", "warhead_near_duplicate_rate"):
            metrics[key] = None
            unavailable[key] = (
                "insufficient_sample: no training molecule decomposed into a "
                f"usable warhead, from the split at {constants.training_set}, "
                "so a generated warhead has nothing to be novel against"
            )

    # The per-molecule record, beside metrics.json. Both denominators and every
    # processing failure are recoverable from it without re-running the
    # splitter, which costs hours.
    path = Path(constants.results_dir) / "component_records.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n")


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


def _screening_baseline(constants, metrics: dict, unavailable: dict) -> None:
    """
    `vs_baseline_top10_at_*` and `vs_delta_top10_at_*`.

    The screening baseline is produced by `scripts/screen-library.py`, which
    spends the same oracle and the same budget on a declared library under the
    same seeds. This reads its output and subtracts: the delta is the headline
    comparison, because a generative method that does not beat random library
    screening at matched budget has not demonstrated anything.
    """
    path = getattr(constants, "screening_baseline_results", None)
    if not path:
        for label in _budget.BUDGETS:
            for key in (f"vs_baseline_top10_at_{label}", f"vs_delta_top10_at_{label}"):
                metrics[key] = None
                unavailable[key] = (
                    "not_applicable_to_job_type: this experiment declares no "
                    "screening library. Run scripts/screen-library.py against "
                    "one and set screening_baseline_results to its output."
                )
        return

    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"screening_baseline_results {path} could not be read: {error}.\n"
            "It is written by scripts/screen-library.py. Remove the key to "
            "report the comparison as unavailable rather than wrong."
        ) from error

    baseline = payload.get("metrics", {})
    metrics["screening_library"] = payload.get("run", {}).get("library")
    metrics["screening_library_size"] = payload.get("run", {}).get("library_size")
    for label in _budget.BUDGETS:
        key = f"vs_baseline_top10_at_{label}"
        entry = baseline.get(key)
        value = entry.get("mean") if isinstance(entry, dict) else entry
        metrics[key] = value
        ours = metrics.get(f"top10_mean_at_{label}")
        if value is None or ours is None:
            metrics[f"vs_delta_top10_at_{label}"] = None
            unavailable[f"vs_delta_top10_at_{label}"] = (
                "insufficient_sample: needs both the generative and the "
                f"screening top-10 mean at {label}"
            )
            if value is None:
                unavailable[key] = (
                    f"insufficient_sample: the screening baseline reports no "
                    f"top-10 mean at {label}"
                )
        else:
            metrics[f"vs_delta_top10_at_{label}"] = ours - value


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
    # The screening baseline keys are handled by _screening_baseline, which
    # reports a value when a library is declared.
    return {
        "eval_oracle_top10_mean": (
            "not_implemented: no evaluation surrogate is trained yet"
        ),
        "top100_outside_domain_rate": (
            "not_implemented: no optimization surrogate training set is available yet"
        ),
    }
