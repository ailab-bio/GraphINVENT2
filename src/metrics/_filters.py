"""
Structural-alert and property-window filters.

`filter_pass_rate` is MCF and PAINS only, which is what CLAUDE.md defines.
MOSES's own `get_all_metrics` filter additionally requires every atom to be in
an allowed element set and the molecule to be a single fragment; we deliberately
do not include those two, so our number is not MOSES's number and should not be
compared to one without saying so.

Sources:
  MCF   22 SMARTS vendored from the MOSES project's `moses/metrics/mcf.csv`,
        kept in `tools/data/mcf.csv` with its provenance in tools/README.md.
        Polykovskiy et al., Front. Pharmacol. 11:565644 (2020).
  PAINS RDKit's own FilterCatalog, Baell & Holloway, J. Med. Chem. 53:2719
        (2010).
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

# tools/data/, not src/: it is externally sourced data, inventoried in
# tools/README.md alongside every other external dependency.
MCF_PATH = Path(__file__).resolve().parents[2] / "tools" / "data" / "mcf.csv"

# Lipinski et al., Adv. Drug Deliv. Rev. 46:3 (2001).
RO5_LIMITS = {"mw": 500.0, "logp": 5.0, "hbd": 5, "hba": 10}


@lru_cache(maxsize=1)
def _mcf_patterns() -> tuple:
    """The MCF SMARTS, compiled once."""
    from rdkit import Chem

    if not MCF_PATH.exists():
        raise FileNotFoundError(
            f"{MCF_PATH} is missing, so filter_pass_rate cannot be computed.\n"
            "It is the MCF SMARTS set vendored from MOSES; run tools/install.sh."
        )
    patterns = []
    with open(MCF_PATH) as handle:
        for row in csv.DictReader(handle):
            query = Chem.MolFromSmarts(row["smarts"])
            if query is None:
                raise ValueError(f"MCF entry {row['names']} is not valid SMARTS")
            patterns.append(query)
    return tuple(patterns)


@lru_cache(maxsize=1)
def _pains_catalog():
    """RDKit's PAINS filter catalog, built once."""
    from rdkit.Chem import FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


def passes_filters(mol) -> bool:
    """True when the molecule matches no MCF pattern and no PAINS alert."""
    if any(mol.HasSubstructMatch(query) for query in _mcf_patterns()):
        return False
    return not _pains_catalog().HasMatch(mol)


@lru_cache(maxsize=200_000)
def passes_filters_smiles(smiles: str) -> bool:
    """
    `passes_filters`, memoized on the SMILES.

    A sampling job asks this of the same molecule at least three times: once in
    triage, once for `filter_pass_rate` over the sample, and once for
    `filter_pass_rate_filtered` over the survivors. Measured at reporting scale
    the alert scan is about five seconds a pass, and the answer cannot change
    between them.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return bool(mol is not None and passes_filters(mol))


def ro5_violations(mol) -> int:
    """
    How many of Lipinski's four criteria the molecule violates.

    MW <= 500, logP <= 5, HBD <= 5, HBA <= 10; Lipinski et al., Adv. Drug
    Deliv. Rev. 46:3 (2001).
    """
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    violations = 0
    violations += Descriptors.MolWt(mol) > RO5_LIMITS["mw"]
    violations += Crippen.MolLogP(mol) > RO5_LIMITS["logp"]
    violations += Lipinski.NumHDonors(mol) > RO5_LIMITS["hbd"]
    violations += Lipinski.NumHAcceptors(mol) > RO5_LIMITS["hba"]
    return int(violations)


def passes_ro5(mol) -> bool:
    """
    True when the molecule violates **at most one** of Lipinski's criteria.

    This is the rule as written: Lipinski et al., Adv. Drug Deliv. Rev. 46:3
    (2001) state that poor absorption is more likely when *two or more*
    criteria are violated, so a single violation is a pass. Much of the
    generative-modeling literature instead reports all-four-satisfied, which is
    stricter; our number is not that number and is not comparable to it.
    """
    return ro5_violations(mol) <= 1


def compute_filter_metrics(unique_valid_smiles: list) -> tuple:
    """
    `filter_pass_rate` and `Ro5_pass_rate` over the unique valid molecules,
    which is the denominator CLAUDE.md gives both.

    Returns (metrics, unavailable) so the caller can record why a metric is
    null rather than guessing.
    """
    from rdkit import Chem

    metrics: dict = {}
    unavailable: dict = {}

    parsed = [smi for smi in unique_valid_smiles if Chem.MolFromSmiles(smi) is not None]
    mols = [Chem.MolFromSmiles(smi) for smi in parsed]
    if not mols:
        metrics["filter_pass_rate"] = None
        metrics["Ro5_pass_rate"] = None
        unavailable["filter_pass_rate"] = "no unique valid molecules to filter"
        unavailable["Ro5_pass_rate"] = "no unique valid molecules to filter"
    else:
        metrics["filter_pass_rate"] = sum(
            1 for smi in parsed if passes_filters_smiles(smi)
        ) / len(mols)
        metrics["Ro5_pass_rate"] = sum(1 for mol in mols if passes_ro5(mol)) / len(mols)

    return metrics, unavailable


def compute_property_window_rate(unique_valid_smiles: list, bounds: dict) -> tuple:
    """
    `property_window_rate`: the fraction of unique valid molecules whose every
    named property falls inside its stated numeric window.

    This replaces `bRo5_pass_rate`, which was ill-posed. "Beyond Rule of 5"
    names the region compounds fall *into* by exceeding Ro5, not an envelope
    they stay *within*, so a pass rate against it has no definition. One
    parameterized window generalizes instead: Ro5 bounds for drug-like
    datasets, a degrader window for bifunctionals, the property set of the
    objective for the battery program. The bounds are echoed into metrics.json
    beside the value as `property_window_rate_bounds`, because the rate means
    nothing without them and a table that omits them cannot be compared with
    another.

    `bounds` is {property_name: [low, high]}, where either end may be null for
    an open interval.
    """
    from rdkit import Chem

    from ._properties import PROPERTY_REGISTRY

    metrics: dict = {"property_window_rate_bounds": bounds or None}
    unavailable: dict = {}

    if not bounds:
        metrics["property_window_rate"] = None
        unavailable["property_window_rate"] = (
            "not_applicable_to_job_type: no property_window is set for this "
            "experiment, and the rate is undefined without explicit bounds"
        )
        return metrics, unavailable

    unknown = sorted(set(bounds) - set(PROPERTY_REGISTRY))
    if unknown:
        raise ValueError(
            f"property_window names {unknown}, which are not property "
            f"functions. Known: {', '.join(sorted(PROPERTY_REGISTRY))}."
        )

    parsed = [smi for smi in unique_valid_smiles if Chem.MolFromSmiles(smi) is not None]
    mols = [Chem.MolFromSmiles(smi) for smi in parsed]
    if not mols:
        metrics["property_window_rate"] = None
        unavailable["property_window_rate"] = (
            "insufficient_sample: no unique valid molecules to measure"
        )
        return metrics, unavailable

    inside = 0
    for mol in mols:
        if all(
            (low is None or PROPERTY_REGISTRY[name](mol) >= low)
            and (high is None or PROPERTY_REGISTRY[name](mol) <= high)
            for name, (low, high) in bounds.items()
        ):
            inside += 1
    metrics["property_window_rate"] = inside / len(mols)
    return metrics, unavailable


# The triage stack, in the order it is applied. Named in the output so a
# reader never has to infer which gates a `_filtered` number passed.
TRIAGE_STAGES_DRUGLIKE = ("structural_alerts", "synthesizability")
TRIAGE_STAGES_BIFUNCTIONAL = (
    "structural_alerts",
    "retrotac",
    "warhead_synthesizability",
)


def triage(
    unique_valid_smiles: list,
    *,
    bifunctional: bool = False,
    retrotac_by_smiles: dict | None = None,
    warhead_by_smiles: dict | None = None,
) -> tuple:
    """
    The molecules that survive triage, and the pass rate at each stage.

    Returns `(kept, rates)`. `rates` carries one `<stage>_pass_rate` per stage,
    each denominated over the molecules that *reached* that stage, plus
    `triage_pass_rate` end to end over the unique valid molecules and
    `triage_stages` naming the stack in order.

    Two stacks, because synthesizability is not one measurement. A drug-like
    molecule is gated on its own SA score. A bifunctional one cannot be: SA
    score reads a PROTAC's size as difficulty and rejects chemistry that
    demonstrably exists, so the whole molecule is gated on RetroTAC and the SA
    gate moves to the warhead, where the fragment is drug-sized and SA behaves.

    A molecule with no RetroTAC score or no usable warhead fails that stage
    rather than passing it. The stage rates make that visible: a collapsing
    `warhead_synthesizability_pass_rate` with a collapsing
    `split_success_rate` beside it is a decomposition problem, not a chemistry
    one.
    """
    from rdkit import Chem

    from ._budget import QUALITY_RETROTAC_MIN, QUALITY_SA_MAX
    from ._properties import sa_score

    stages = TRIAGE_STAGES_BIFUNCTIONAL if bifunctional else TRIAGE_STAGES_DRUGLIKE
    rates: dict = {"triage_stages": list(stages)}
    kept = [smi for smi in unique_valid_smiles if smi]
    n_entered = len(kept)

    def record(stage: str, survivors: list, reached: int) -> list:
        rates[f"{stage}_pass_rate"] = len(survivors) / reached if reached else None
        rates[f"{stage}_n_reached"] = reached
        return survivors

    reached = len(kept)
    survivors = [smi for smi in kept if passes_filters_smiles(smi)]
    kept = record("structural_alerts", survivors, reached)

    if bifunctional:
        reached = len(kept)
        scores = retrotac_by_smiles or {}
        kept = record(
            "retrotac",
            [
                s
                for s in kept
                if (scores.get(s) is not None and scores[s] >= QUALITY_RETROTAC_MIN)
            ],
            reached,
        )
        reached = len(kept)
        warheads = warhead_by_smiles or {}
        survivors = []
        for smi in kept:
            warhead = warheads.get(smi)
            mol = Chem.MolFromSmiles(warhead) if warhead else None
            if mol is not None and sa_score(mol) < QUALITY_SA_MAX:
                survivors.append(smi)
        kept = record("warhead_synthesizability", survivors, reached)
    else:
        reached = len(kept)
        survivors = []
        for smi in kept:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and sa_score(mol) < QUALITY_SA_MAX:
                survivors.append(smi)
        kept = record("synthesizability", survivors, reached)

    rates["triage_pass_rate"] = len(kept) / n_entered if n_entered else None
    rates["triage_n_entered"] = n_entered
    rates["triage_sa_max"] = QUALITY_SA_MAX
    if bifunctional:
        rates["triage_retrotac_min"] = QUALITY_RETROTAC_MIN
    return kept, rates
