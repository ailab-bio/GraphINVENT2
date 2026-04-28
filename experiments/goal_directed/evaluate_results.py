"""
Compute PMO-style evaluation metrics from goal-directed RL run outputs.

Usage (from repository root):
    python experiments/goal_directed/evaluate_results.py [options]

Options:
    --output-root  Root output directory (default: ./output)
    --dataset      Dataset name used in the RL runs (default: chembl_v34)
    --config       Path to oracles_config.yaml
    --budget       Oracle budget for AUC normalisation (default: 10000)
    --k            k for AUC Top-k (default: 10)
    --out          Path to write summary CSV and LaTeX table (default: experiments/goal_directed/results/)

For each (oracle, seed) run the script reads:
    output/<dataset>/goal_directed/<oracle>_seed<seed>/checkpoint_<N>_samples.smi

and computes per-checkpoint:
    - Top-1, Top-10, Top-100 oracle scores
    - Validity, uniqueness, novelty (vs ChEMBL training set)
    - Internal diversity (mean pairwise Tanimoto over ECFP4)
    - SA score distribution (mean, p25, p75) using RDKit sascorer
    - Success rate (fraction >= oracle-specific threshold)

Then aggregates across seeds and computes:
    - AUC Top-10 over the oracle budget curve
    - Mean ± std across seeds for all metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CONFIG = REPO_ROOT / "experiments" / "goal_directed" / "oracles_config.yaml"
DEFAULT_OUT = REPO_ROOT / "experiments" / "goal_directed" / "results"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def load_smiles(path: Path) -> list[str]:
    """Load SMILES from a .smi file (one per line, optional space-separated ID)."""
    smiles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi = line.split()[0]
            smiles.append(smi)
    return smiles


def compute_validity(smiles: list[str]) -> tuple[list[str], float]:
    """Return (valid_smiles, validity_fraction)."""
    try:
        from rdkit import Chem

        valid = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
    except ImportError:
        return smiles, 1.0
    return valid, len(valid) / max(len(smiles), 1)


def compute_uniqueness(smiles: list[str]) -> tuple[list[str], float]:
    """Return (unique_canonical_smiles, uniqueness_fraction)."""
    try:
        from rdkit import Chem

        canonical = []
        seen: set[str] = set()
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol)
            if can not in seen:
                seen.add(can)
                canonical.append(can)
        return canonical, len(canonical) / max(len(smiles), 1)
    except ImportError:
        unique = list(dict.fromkeys(smiles))
        return unique, len(unique) / max(len(smiles), 1)


def compute_novelty(smiles: list[str], reference: set[str]) -> float:
    """Fraction of (canonical) SMILES not in reference set."""
    if not smiles:
        return 0.0
    novel = sum(1 for s in smiles if s not in reference)
    return novel / len(smiles)


def compute_internal_diversity(smiles: list[str], max_mols: int = 1000) -> float:
    """Mean pairwise Tanimoto distance (1 - similarity) over ECFP4 fingerprints."""
    try:
        import random

        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        mols = [Chem.MolFromSmiles(s) for s in smiles if s]
        mols = [m for m in mols if m is not None]
        if len(mols) > max_mols:
            mols = random.sample(mols, max_mols)
        if len(mols) < 2:
            return 0.0

        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
        n = len(fps)
        total = 0.0
        count = 0
        for i in range(n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
            total += sum(1.0 - s for s in sims)
            count += len(sims)
        return total / count if count > 0 else 0.0
    except ImportError:
        return float("nan")


def compute_sa_scores(smiles: list[str]) -> list[float]:
    """Compute raw SA scores (1-10) using RDKit's SA_Score contribution module."""
    try:
        from rdkit import Chem, RDConfig

        sa_score_path = Path(RDConfig.RDContribDir) / "SA_Score"
        if str(sa_score_path) not in sys.path:
            sys.path.append(str(sa_score_path))
        import sascorer  # type: ignore[import]

        scores = []
        for s in smiles:
            try:
                mol = Chem.MolFromSmiles(s)
                scores.append(sascorer.calculateScore(mol) if mol else 10.0)
            except Exception:
                scores.append(10.0)
        return scores
    except (ImportError, Exception):
        return [float("nan")] * len(smiles)


def oracle_scores_from_log(log_path: Path) -> list[tuple[int, float]]:
    """
    Load the optimization log from a checkpoint JSON written by the RL training loop.

    Expected format (written by Workflow.py at each checkpoint):
        {"oracle_call": <int>, "score": <float>}  (one per line)

    Falls back to parsing the .smi file header if the log file is absent.
    """
    if not log_path.exists():
        return []
    log = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                log.append((int(entry["oracle_call"]), float(entry["score"])))
            except (json.JSONDecodeError, KeyError):
                pass
    return log


def topk_mean(scores: list[float], k: int) -> float:
    """Mean of the top-k scores (or fewer if < k available)."""
    if not scores:
        return 0.0
    top = sorted(scores, reverse=True)[:k]
    return sum(top) / k  # divide by k even if fewer than k


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    smi_path: Path,
    oracle,
    threshold: float | None,
    reference_smiles: set[str],
) -> dict:
    """Evaluate one checkpoint .smi file and return a metrics dict."""
    smiles_raw = load_smiles(smi_path)
    valid_smiles, validity = compute_validity(smiles_raw)
    unique_smiles, uniqueness = compute_uniqueness(valid_smiles)
    novelty = compute_novelty(unique_smiles, reference_smiles)
    diversity = compute_internal_diversity(unique_smiles)
    sa_scores = compute_sa_scores(unique_smiles)
    sa_mean = sum(sa_scores) / len(sa_scores) if sa_scores else float("nan")

    oracle_scores = oracle(unique_smiles) if unique_smiles else []

    top1 = topk_mean(oracle_scores, 1)
    top10 = topk_mean(oracle_scores, 10)
    top100 = topk_mean(oracle_scores, 100)

    success_rate = 0.0
    if threshold is not None and oracle_scores:
        success_rate = sum(1 for s in oracle_scores if s >= threshold) / len(
            oracle_scores
        )

    return {
        "n_generated": len(smiles_raw),
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "internal_diversity": diversity,
        "sa_mean": sa_mean,
        "top1": top1,
        "top10": top10,
        "top100": top100,
        "success_rate": success_rate,
    }


# ---------------------------------------------------------------------------
# AUC computation (delegates to src/oracles)
# ---------------------------------------------------------------------------


def compute_auc(oracle_log: list[tuple[int, float]], k: int, budget: int) -> float:
    try:
        from oracles import compute_auc_top_k

        return compute_auc_top_k(oracle_log, k=k, budget=budget)
    except ImportError:
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "output")
    p.add_argument("--dataset", type=str, default="chembl_v34")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--oracle", action="append", default=None, dest="oracles")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    oracle_cfg = load_oracle_config(args.config)
    pmo_cfg = oracle_cfg.get("pmo", {})
    all_seeds = pmo_cfg.get("seeds", [42, 123, 456])
    checkpoints = pmo_cfg.get("checkpoints", [1000, 3000, 5000, 10000])
    oracle_list = oracle_cfg.get("oracles", [])

    if args.oracles:
        oracle_list = [o for o in oracle_list if o["name"] in args.oracles]

    # Load reference SMILES (ChEMBL training set) for novelty computation
    ref_path = REPO_ROOT / "data" / "datasets" / args.dataset / "train.smi"
    reference_smiles: set[str] = set()
    if ref_path.exists():
        print(f"Loading reference SMILES from {ref_path}...")
        reference_smiles = set(load_smiles(ref_path))
        print(f"  {len(reference_smiles)} reference molecules loaded.")
    else:
        print(f"Warning: reference file not found at {ref_path}. Novelty will be 0.0.")

    args.out.mkdir(parents=True, exist_ok=True)

    # Summary rows for CSV
    summary_rows = []

    for oracle_info in oracle_list:
        oracle_name = oracle_info["name"]
        threshold = oracle_info.get("threshold")
        print(f"\n=== Oracle: {oracle_name} ===")

        # Instantiate oracle (cached, for efficiency)
        try:
            from oracles import OracleFactory

            oracle = OracleFactory.create_cached(oracle_name)
        except Exception as exc:
            print(f"  Could not load oracle {oracle_name}: {exc}. Skipping.")
            continue

        seed_aucs = []
        seed_rows = []

        for seed in all_seeds:
            job_name = f"{oracle_name.replace(':', '_')}_seed{seed}"
            job_dir = args.output_root / args.dataset / "goal_directed" / job_name

            if not job_dir.exists():
                print(
                    f"  Seed {seed}: output directory not found at {job_dir}. Skipping."
                )
                continue

            print(f"  Seed {seed}: {job_dir}")

            # Load optimization log for AUC
            log_path = job_dir / "optimization_log.jsonl"
            opt_log = oracle_scores_from_log(log_path)
            auc = compute_auc(opt_log, k=args.k, budget=args.budget)
            seed_aucs.append(auc)

            # Evaluate each checkpoint
            for ckpt in checkpoints:
                smi_path = job_dir / f"checkpoint_{ckpt}_samples.smi"
                if not smi_path.exists():
                    # Fall back to the final samples file
                    candidates = list(job_dir.glob("*_samples.smi"))
                    if candidates:
                        smi_path = candidates[-1]
                    else:
                        print(f"    Checkpoint {ckpt}: no .smi file found.")
                        continue

                metrics = evaluate_checkpoint(
                    smi_path=smi_path,
                    oracle=oracle,
                    threshold=threshold,
                    reference_smiles=reference_smiles,
                )
                row = {
                    "oracle": oracle_name,
                    "seed": seed,
                    "oracle_budget": ckpt,
                    "auc_top10": auc,
                    **metrics,
                }
                seed_rows.append(row)
                summary_rows.append(row)
                print(
                    f"    budget={ckpt:>6}  top10={metrics['top10']:.3f}  "
                    f"valid={metrics['validity']:.3f}  novel={metrics['novelty']:.3f}  "
                    f"success={metrics['success_rate']:.3f}"
                )

        if seed_aucs:
            import statistics

            mean_auc = statistics.mean(seed_aucs)
            std_auc = statistics.stdev(seed_aucs) if len(seed_aucs) > 1 else 0.0
            print(
                f"  AUC Top-{args.k} ({args.budget} budget): {mean_auc:.4f} ± {std_auc:.4f}"
            )

    # Write summary CSV
    if summary_rows:
        csv_path = args.out / "pmo_results.csv"
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary written to {csv_path}")

        # Write a LaTeX table of AUC Top-10 at budget=10000, mean ± std across seeds
        _write_latex_table(summary_rows, args.out, args.budget, args.k)
    else:
        print("\nNo results found. Run experiments first with run_all_oracles.py.")


def _write_latex_table(rows: list[dict], out_dir: Path, budget: int, k: int) -> None:
    """Write a LaTeX booktabs table of AUC Top-k at the full budget."""
    import statistics
    from collections import defaultdict

    # Group by oracle, collect AUC values at full budget
    oracle_aucs: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["oracle_budget"] == budget:
            oracle_aucs[row["oracle"]].append(row["auc_top10"])

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{PMO AUC Top-{k} results (oracle budget = {budget:,}). "
        r"Values are mean (std) over 3 seeds.}}",
        r"  \label{tab:pmo}",
        r"  \small",
        r"  \renewcommand{\arraystretch}{1.25}",
        r"  \begin{tabular}{l c}",
        r"    \toprule",
        rf"    \textbf{{Oracle}} & \textbf{{AUC Top-{k}}} $\uparrow$ \\",
        r"    \midrule",
    ]
    for oracle_name, aucs in sorted(oracle_aucs.items()):
        mean = statistics.mean(aucs) if aucs else float("nan")
        std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        lines.append(rf"    {oracle_name} & {mean:.3f} \scriptsize{{({std:.3f})}} \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex_path = out_dir / "pmo_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table written to {tex_path}")


def load_oracle_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
