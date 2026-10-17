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
    `filter_pass_rate`, `Ro5_pass_rate` and `bRo5_pass_rate` over the unique
    valid molecules, which is the denominator CLAUDE.md gives all three.

    Returns (metrics, unavailable) so the caller can record why a metric is
    null rather than guessing.
    """
    from rdkit import Chem

    metrics: dict = {}
    unavailable: dict = {}

    mols = [Chem.MolFromSmiles(smi) for smi in unique_valid_smiles]
    mols = [mol for mol in mols if mol is not None]
    if not mols:
        metrics["filter_pass_rate"] = None
        metrics["Ro5_pass_rate"] = None
        unavailable["filter_pass_rate"] = "no unique valid molecules to filter"
        unavailable["Ro5_pass_rate"] = "no unique valid molecules to filter"
    else:
        metrics["filter_pass_rate"] = sum(
            1 for mol in mols if passes_filters(mol)
        ) / len(mols)
        metrics["Ro5_pass_rate"] = sum(1 for mol in mols if passes_ro5(mol)) / len(mols)

    # bRo5 has no implementable definition yet. The property limits circulating
    # informally could not be confirmed against Doak et al., Chem. Biol.
    # 21:1115 (2014): the paper is paywalled, and its abstract and the reviews
    # citing it describe bRo5 as the region compounds fall *into* by exceeding
    # Ro5, not an envelope they stay *within* -- so "passing bRo5" needs a
    # definition before it needs thresholds.
    metrics["bRo5_pass_rate"] = None
    unavailable["bRo5_pass_rate"] = (
        "no verified definition: Doak et al., Chem. Biol. 21:1115 (2014) could "
        "not be obtained, and the literature defines bRo5 as the space beyond "
        "Ro5 rather than a window to pass, so the sense of the rate is unsettled"
    )
    return metrics, unavailable
