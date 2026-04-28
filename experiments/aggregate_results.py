"""
Aggregate results from all four GraphINVENT2 experiments.

Usage (from repository root):
    python experiments/aggregate_results.py [--out experiments/summary/]

Reads:
    experiments/goal_directed/results/pmo_results.csv
    experiments/conditional/results/conditional_results.csv
    output/drd2_actives/unconditional/run/convergence.log

Outputs:
    experiments/summary/pmo_summary.csv   -- PMO AUC Top-10 table
    experiments/summary/cond_summary.csv  -- Conditional accuracy table
    experiments/summary/tables.tex        -- LaTeX tables for the paper
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_OUT = REPO_ROOT / "experiments" / "summary"
PMO_RESULTS = (
    REPO_ROOT / "experiments" / "goal_directed" / "results" / "pmo_results.csv"
)
COND_RESULTS = (
    REPO_ROOT / "experiments" / "conditional" / "results" / "conditional_results.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Warning: {path} not found.")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    import statistics

    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(m: float, s: float, decimals: int = 3) -> str:
    if m != m:  # nan
        return "—"
    return f"{m:.{decimals}f} ({s:.{decimals}f})"


# ---------------------------------------------------------------------------
# PMO summary
# ---------------------------------------------------------------------------


def summarise_pmo(rows: list[dict], budget: int = 10000) -> list[dict]:
    """
    Aggregate PMO rows: mean ± std across seeds at the given budget.
    Returns one row per oracle with all metric means.
    """
    # Filter to the final budget
    budget_rows = [r for r in rows if int(r.get("oracle_budget", 0)) == budget]

    metric_cols = [
        "auc_top10",
        "validity",
        "uniqueness",
        "novelty",
        "internal_diversity",
        "top1",
        "top10",
        "top100",
        "success_rate",
    ]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in budget_rows:
        grouped[r["oracle"]].append(r)

    summary = []
    for oracle, oracle_rows in sorted(grouped.items()):
        row_out: dict = {"oracle": oracle, "n_seeds": len(oracle_rows)}
        for col in metric_cols:
            try:
                vals = [float(r[col]) for r in oracle_rows if r.get(col, "") != ""]
                row_out[f"{col}_mean"] = mean(vals)
                row_out[f"{col}_std"] = std(vals)
            except (ValueError, KeyError):
                row_out[f"{col}_mean"] = float("nan")
                row_out[f"{col}_std"] = 0.0
        summary.append(row_out)
    return summary


# ---------------------------------------------------------------------------
# Conditional summary
# ---------------------------------------------------------------------------


def summarise_conditional(rows: list[dict]) -> list[dict]:
    """Group by (property, range) and return a clean summary."""
    summary = []
    for r in rows:
        try:
            summary.append(
                {
                    "property": r["property"],
                    "range": r["range"],
                    "target": f"[{float(r['target_lo']):.2f}, {float(r['target_hi']):.2f}]",
                    "validity": float(r.get("validity", float("nan"))),
                    "cond_accuracy": float(r.get("cond_accuracy", float("nan"))),
                    "internal_diversity": float(
                        r.get("internal_diversity", float("nan"))
                    ),
                    "mean_prop": float(r.get("mean_prop", float("nan"))),
                    "std_prop": float(r.get("std_prop", float("nan"))),
                }
            )
        except (ValueError, KeyError):
            continue
    return summary


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def pmo_latex_table(pmo_summary: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{PMO benchmark results at oracle budget = 10,000 calls. "
        r"Values are mean (std) over 3 seeds. "
        r"AUC Top-10 is the primary PMO metric (higher is better).}",
        r"  \label{tab:pmo-full}",
        r"  \small",
        r"  \renewcommand{\arraystretch}{1.25}",
        r"  \begin{tabular}{l ccccc}",
        r"    \toprule",
        r"    \textbf{Oracle} & \textbf{AUC Top-10} $\uparrow$ "
        r"& \textbf{Validity} $\uparrow$ & \textbf{Novelty} $\uparrow$ "
        r"& \textbf{Diversity} $\uparrow$ & \textbf{Success} $\uparrow$ \\",
        r"    \midrule",
    ]
    for r in pmo_summary:
        lines.append(
            f"    {r['oracle']} "
            f"& {fmt(r['auc_top10_mean'], r['auc_top10_std'])} "
            f"& {fmt(r['validity_mean'], r['validity_std'])} "
            f"& {fmt(r['novelty_mean'], r['novelty_std'])} "
            f"& {fmt(r['internal_diversity_mean'], r['internal_diversity_std'])} "
            f"& {fmt(r['success_rate_mean'], r['success_rate_std'])} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def conditional_latex_table(cond_summary: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Conditional generation results. "
        r"``Accuracy'' is the fraction of valid generated molecules whose computed "
        r"property value falls within the conditioning target range.}",
        r"  \label{tab:conditional}",
        r"  \small",
        r"  \renewcommand{\arraystretch}{1.25}",
        r"  \begin{tabular}{l l c c c c}",
        r"    \toprule",
        r"    \textbf{Property} & \textbf{Target range} & \textbf{Validity} $\uparrow$ "
        r"& \textbf{Accuracy} $\uparrow$ & \textbf{Diversity} $\uparrow$ "
        r"& \textbf{Mean prop.} \\",
        r"    \midrule",
    ]
    for r in cond_summary:
        mp = r["mean_prop"]
        sp = r["std_prop"]
        mp_str = f"{mp:.3f} ({sp:.3f})" if mp == mp else "—"
        lines.append(
            f"    {r['property']} & {r['target']} "
            f"& {r['validity']:.3f} & {r['cond_accuracy']:.3f} "
            f"& {r['internal_diversity']:.3f} & {mp_str} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pmo-budget", type=int, default=10000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading PMO results...")
    pmo_rows = load_csv(PMO_RESULTS)
    pmo_summary = summarise_pmo(pmo_rows, budget=args.pmo_budget)

    print("Loading conditional results...")
    cond_rows = load_csv(COND_RESULTS)
    cond_summary = summarise_conditional(cond_rows)

    # Write summary CSVs
    if pmo_summary:
        pmo_csv = args.out / "pmo_summary.csv"
        with open(pmo_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pmo_summary[0].keys()))
            writer.writeheader()
            writer.writerows(pmo_summary)
        print(f"PMO summary: {pmo_csv}")

    if cond_summary:
        cond_csv = args.out / "cond_summary.csv"
        with open(cond_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cond_summary[0].keys()))
            writer.writeheader()
            writer.writerows(cond_summary)
        print(f"Conditional summary: {cond_csv}")

    # Write LaTeX tables
    tex_lines = ["% Auto-generated by experiments/aggregate_results.py", ""]
    if pmo_summary:
        tex_lines.append(pmo_latex_table(pmo_summary))
        tex_lines.append("")
    if cond_summary:
        tex_lines.append(conditional_latex_table(cond_summary))

    tex_path = args.out / "tables.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"LaTeX tables: {tex_path}")

    # Print console summary
    if pmo_summary:
        print("\n--- PMO AUC Top-10 ---")
        print(f"{'Oracle':<28} {'AUC Top-10':>12} {'Success':>8}")
        print("-" * 52)
        for r in pmo_summary:
            print(
                f"{r['oracle']:<28} "
                f"{fmt(r['auc_top10_mean'], r['auc_top10_std']):>12} "
                f"{fmt(r['success_rate_mean'], r['success_rate_std']):>8}"
            )

    if cond_summary:
        print("\n--- Conditional Accuracy ---")
        print(f"{'Property':<12} {'Range':<10} {'Accuracy':>10} {'Diversity':>10}")
        print("-" * 46)
        for r in cond_summary:
            acc = r["cond_accuracy"]
            div = r["internal_diversity"]
            print(f"{r['property']:<12} {r['range']:<10} " f"{acc:>10.3f} {div:>10.3f}")


if __name__ == "__main__":
    main()
