#!/usr/bin/env python3
"""
Virtual screening baseline: spend an oracle budget on a declared library.

    python scripts/screen-library.py experiments/<exp>/jobs/<rl_job> \
        --library data/datasets/tpddb_protac/train.smi --seeds 1 2 3

A generative method that does not beat random library screening at matched
budget has not demonstrated anything, so this runs for every optimization
experiment. It draws molecules uniformly at random from the library, scores
them through the same `oracles.CachedOracle` the optimization job uses, and
writes the at-budget statistics in the same `metrics.json` shape, so the
comparison needs no special casing later.

The oracle, the budget and the seeds come from the optimization job's own
resolved configuration, which is what "matched" means: the same scoring
components, the same cache semantics where one oracle call is one unique
canonical SMILES, and the same seed list.

The output goes to `<job_dir>/results/screening_<library>.json` and is read by
the optimization job through `screening_baseline_results`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "src" / "graphinvent")]

BUDGETS = {"1k": 1000, "5k": 5000, "10k": 10000}


def read_library(path: Path) -> list:
    """Canonical SMILES from a .smi or TSV, header and blanks skipped."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out, seen = [], set()
    for i, line in enumerate(path.read_text().splitlines()):
        token = line.split("\t")[0].split()[0] if line.strip() else ""
        if not token or (i == 0 and token.upper() == "SMILES"):
            continue
        mol = Chem.MolFromSmiles(token)
        if mol is None:
            continue
        # Deduplicated on the canonical form, because an oracle call is one
        # unique molecule and a library with duplicates would otherwise buy
        # fewer distinct molecules than its size suggests.
        canonical = Chem.MolToSmiles(mol)
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def build_oracle(params: dict):
    """The optimization job's scoring function, as a single callable."""
    from oracles import CachedOracle, OracleFactory

    specs = params.get("oracles") or {}
    components = list(params.get("score_components") or [])
    named = [c for c in components if c in specs]
    if not named:
        raise SystemExit(
            f"This job's score_components are {components} and none of them is "
            "declared in its `oracles` block.\nA screening baseline has to "
            "spend the same oracle the optimization job spends, and a built-in "
            "component is evaluated inline rather than through the cache, so "
            "there is no budget to match. Declare the objective as an oracle."
        )
    cached = {
        name: CachedOracle(OracleFactory.create(name, specs[name])) for name in named
    }

    def score(smiles: list) -> list:
        # Components multiply, as `score_type: continuous` does in
        # ScoringFunction. A geometric-mean objective is zeroed by one failed
        # component, which is the behaviour the generative arm is scored under.
        totals = [1.0] * len(smiles)
        for oracle in cached.values():
            for i, value in enumerate(oracle(smiles)):
                totals[i] *= float(value)
        return totals

    return score, cached


def top_mean(scores: list, k: int) -> float | None:
    if not scores:
        return None
    return float(sum(sorted(scores, reverse=True)[:k]) / min(k, len(scores)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "job_dir",
        type=Path,
        help="the optimization job whose oracle and budget to match",
    )
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--budget", type=int, default=10000)
    args = ap.parse_args()

    from parameters import config

    params = config.resolve(args.job_dir)
    library = read_library(args.library)
    print(f"* Library {args.library}: {len(library)} unique molecules", flush=True)
    if len(library) < args.budget:
        print(
            f"  the budget is {args.budget} and the library holds "
            f"{len(library)}, so the whole library is screened and the "
            "at-budget points above its size repeat the final value.",
            flush=True,
        )

    per_seed: dict = {}
    for seed in args.seeds:
        score, cached = build_oracle(params)
        order = list(library)
        random.Random(seed).shuffle(order)
        drawn = order[: args.budget]
        print(f"* seed {seed}: scoring {len(drawn)} molecules", flush=True)
        scores = score(drawn)
        calls = max(o.call_count for o in cached.values())
        row = {"n_screened": len(drawn), "oracle_calls": calls}
        for label, budget in BUDGETS.items():
            within = scores[: min(budget, len(scores))]
            row[f"vs_baseline_top10_at_{label}"] = top_mean(within, 10)
            row[f"vs_baseline_best_at_{label}"] = max(within) if within else None
        per_seed[seed] = row
        print(
            f"  oracle calls {calls}, top-10 mean at 1k "
            f"{row['vs_baseline_top10_at_1k']}",
            flush=True,
        )

    keys = [k for k in per_seed[args.seeds[0]] if k.startswith("vs_baseline_")]
    aggregate = {}
    for key in keys:
        values = [per_seed[s][key] for s in args.seeds if per_seed[s][key] is not None]
        if not values:
            aggregate[key] = None
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        aggregate[key] = {
            "mean": mean,
            "std": var**0.5,
            "per_seed": {str(s): per_seed[s][key] for s in args.seeds},
        }

    out_dir = args.job_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"screening_{args.library.stem}.json"
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run": {
                    "kind": "virtual_screening_baseline",
                    "matched_job": str(args.job_dir),
                    "library": str(args.library),
                    "library_size": len(library),
                    "budget": args.budget,
                    "seeds": args.seeds,
                },
                "metrics": aggregate,
                "per_seed": per_seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"* Wrote {out}", flush=True)
    print(
        "  Point the optimization job's screening_baseline_results at it to "
        "get vs_delta_top10_at_*.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
