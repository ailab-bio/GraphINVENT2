"""
Goal-directed metrics: oracle score statistics at fixed budgets, and the
diversity and quality checks that detect reward hacking.

Two different inputs are needed here and they are not interchangeable.

The budget-indexed metrics (`best_at_*`, `top10_mean_at_*`, `top100_mean_at_*`,
`auc_top10`) need the oracle call curve: the chronological list of
(cumulative_call_count, score). Only an oracle routed through
``oracles.CachedOracle`` produces one, because only the cache knows which
submissions were new molecules and which were free hits.

The set-level metrics (`success_rate`, `top100_scaffold_count`,
`top100_internal_diversity`) need only the scored molecules, so they work for
any scoring function.
"""

from __future__ import annotations

import numpy as np

from ._core import bemis_murcko
from ._internal_diversity import compute_internal_diversity

# CLAUDE.md tabulates these three budgets for the at-budget statistics.
BUDGETS = {"1k": 1000, "5k": 5000, "10k": 10000}

# CLAUDE.md: quality_filtered_top10_mean keeps molecules below this SA score.
# Correct for drug-like molecules and molecular glues; 93.8% of drug-like ZINC
# molecules pass it.
QUALITY_SA_MAX = 4.5

# The same gate for bifunctional molecules, where SA score reads size as
# difficulty and so cannot be used at the whole-PROTAC level.
#
# Approved 2026-10-18. Deliberately NOT the paper's 0.7: RetroTAC's
# authors state that 0.7 is "a reporting convention, not a calibrated decision
# boundary", and measured on 500 real TPD-DB degraders -- compounds that were
# made and tested -- only 18.8% reach 0.7, against 41.4% that pass SA < 4.5.
# Adopting 0.7 would therefore make the headline goal-directed score *stricter*
# than the gate it replaces, and reject four fifths of demonstrably real
# chemistry.
#
# 0.30 is chosen to match the permissiveness of the SA gate on the corpus each
# one is appropriate for: SA < 4.5 passes 93.8% of drug-like ZINC molecules,
# and retrotac >= 0.30 passes 94.0% of real degraders. That makes the two gates
# reject comparable tails rather than comparable numbers.
QUALITY_RETROTAC_MIN = 0.30


class MissingInput(Exception):
    """A metric's input is absent. The message says which and where from."""


def _top_mean(scores: list, k: int) -> float | None:
    """Mean of the k highest scores, or of all of them when fewer than k exist."""
    if not scores:
        return None
    return float(np.mean(sorted(scores, reverse=True)[:k]))


def compute_at_budget_metrics(optimization_log: list) -> dict:
    """
    Oracle score statistics at the budgets CLAUDE.md tabulates.

    Parameters
    ----------
    optimization_log
        Chronological (cumulative_oracle_calls, score) pairs, as written to
        ``results/optimization_log.jsonl``.

    Raises
    ------
    MissingInput
        When the log is empty. A goal-directed run whose score components are
        all built-in property functions produces no log, because those are
        evaluated inline and never reach the oracle cache that counts calls.
    """
    if not optimization_log:
        raise MissingInput(
            "The oracle call curve is empty, so no at-budget metric can be "
            "computed.\nIt is written to results/optimization_log.jsonl by a "
            "scoring function whose components go through oracles.CachedOracle. "
            "Built-in components (QED, target_size, logp_target) are evaluated "
            "inline and are never counted as oracle calls."
        )

    results: dict = {}
    for label, budget in BUDGETS.items():
        within = [score for calls, score in optimization_log if calls <= budget]
        results[f"best_at_{label}"] = float(max(within)) if within else None
        results[f"top10_mean_at_{label}"] = _top_mean(within, 10)
        results[f"top100_mean_at_{label}"] = _top_mean(within, 100)
    return results


def compute_auc_top10(optimization_log: list, *, finish: bool = True) -> float:
    """
    `auc_top10`: area under the top-10 mean against oracle calls, to 10,000.

    Delegates to the PMO implementation in ``oracles``; this exists so the
    metric has one call site with CLAUDE.md's name and budget on it.
    """
    if not optimization_log:
        raise MissingInput(
            "The oracle call curve is empty, so auc_top10 cannot be computed. "
            "See compute_at_budget_metrics for why."
        )
    from oracles import compute_auc_top_k

    return float(compute_auc_top_k(optimization_log, k=10, budget=10000, finish=finish))


def compute_set_level_metrics(
    scored: list,
    success_threshold: float,
    *,
    subsample: int = 10000,
    retrotac_by_smiles: dict | None = None,
) -> dict:
    """
    The goal-directed metrics that need only scored molecules.

    Parameters
    ----------
    scored
        (canonical_smiles, score) for each **unique valid** molecule. The
        denominator of `success_rate` is this list's length, per CLAUDE.md.
    success_threshold
        A molecule succeeds at or above this score.
    retrotac_by_smiles
        RetroTAC score per canonical SMILES, for a bifunctional dataset. When
        given, `quality_filtered_top10_mean` gates on it instead of on SA
        score, because SA score reads a PROTAC's size as difficulty. None means
        the dataset is not bifunctional and the SA gate applies.
    """
    results: dict = {}
    if not scored:
        return {
            "success_rate": None,
            "top100_scaffold_count": None,
            "top100_internal_diversity": None,
            "quality_filtered_top10_mean": None,
        }

    scores = [s for _, s in scored]
    n_unique_valid = len(scored)
    results["success_rate"] = (
        sum(1 for s in scores if s >= success_threshold) / n_unique_valid
    )

    # The top 100 by score. This is where reward hacking shows: overall
    # diversity stays high while the winning set collapses onto one scaffold,
    # so these two are reported over the top 100 and not over everything.
    top100 = sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:100]
    top100_smiles = [smi for smi, _ in top100]
    scaffolds = {
        s for s in (bemis_murcko(smi) for smi in top100_smiles) if s is not None
    }
    results["top100_scaffold_count"] = len(scaffolds)
    results["top100_internal_diversity"] = (
        compute_internal_diversity(sorted(top100_smiles), max_mols=subsample)[
            "internal_diversity"
        ]
        if len(top100_smiles) >= 2
        else None
    )

    # quality_filtered_top10_mean, which CLAUDE.md calls the headline
    # goal-directed score. A plain top-10 mean rewards a model that found ten
    # unmakeable molecules the oracle happens to like, which is the default
    # outcome of RL against a cheap surrogate. Restricting to molecules that
    # pass the structural alerts and sit below the SA ceiling is what makes the
    # number mean "ten compounds someone could actually order".
    from rdkit import Chem

    from ._filters import passes_filters
    from ._properties import sa_score

    qualified: list = []
    for smi, score in scored:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if not passes_filters(mol):
            continue
        if retrotac_by_smiles is None:
            if sa_score(mol) >= QUALITY_SA_MAX:
                continue
        else:
            # A molecule RetroTAC could not score is not silently kept: it has
            # no synthesizability evidence, so it fails the gate like any other
            # molecule without one.
            value = retrotac_by_smiles.get(smi)
            if value is None or value < QUALITY_RETROTAC_MIN:
                continue
        qualified.append(score)
    results["quality_filtered_top10_mean"] = _top_mean(qualified, 10)
    return results
