"""
Evaluate conditional generation quality across all property target ranges.

Usage (from repository root):
    python experiments/conditional/evaluate_conditional.py [options]

Options:
    --checkpoint  Path to trained conditional model checkpoint (.pth)
    --dataset     Dataset name used during training (default: chembl_v34_cond)
    --n-samples   Molecules to generate per condition range (default: 1000)
    --out         Output directory (default: experiments/conditional/results/)
    --device      cuda or cpu (default: cuda)

For each property range defined in PROPERTY_RANGES, this script:
  1. Calls submit.py to generate n_samples molecules with the target condition.
  2. Loads the generated SMILES file.
  3. Computes the actual property value for each generated molecule.
  4. Reports:
       - Validity
       - Fraction of molecules within the target property range (conditional accuracy)
       - Internal diversity (mean pairwise Tanimoto distance, ECFP4)
       - Mean ± std of the property across the generated set
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_OUT = REPO_ROOT / "experiments" / "conditional" / "results"

# ---------------------------------------------------------------------------
# Property target ranges
# Values should be in the SAME NORMALISED scale used during training.
# QED:      [0, 1] (raw RDKit QED)
# SA_norm:  [0, 1] = (10 - raw_SA) / 9   (1 = easy to synthesize)
# LogP_norm:[0, 1] = (LogP + 3) / 10     (0 = LogP=-3, 1 = LogP=7)
# GSK3B:    [0, 1] (TDC oracle score)
# ---------------------------------------------------------------------------

PROPERTY_RANGES: dict[str, dict[str, list[float]]] = {
    "QED": {
        "low": [0.0, 0.3],
        "medium": [0.4, 0.6],
        "high": [0.7, 1.0],
    },
    "SA_norm": {
        "easy": [0.78, 1.0],  # raw SA ≈ 1.0 – 3.0
        "medium": [0.44, 0.78],  # raw SA ≈ 3.0 – 5.0
        "hard": [0.0, 0.44],  # raw SA ≈ 5.0 – 10.0
    },
    "LogP_norm": {
        "low": [0.0, 0.2],  # LogP in [-3, -1]
        "medium": [0.2, 0.5],  # LogP in [-1, 2]
        "high": [0.5, 0.8],  # LogP in [2, 5]
    },
    "GSK3B": {
        "inactive": [0.0, 0.3],
        "active": [0.5, 1.0],
    },
}


# Property names → midpoint value to use as the conditioning target
def _midpoint(lo: float, hi: float) -> float:
    return (lo + hi) / 2.0


# Column order must match preprocessing (conditioning.properties in preprocess_params.json)
PROPERTY_COLUMN_ORDER = ["QED", "SA_norm", "LogP_norm", "GSK3B"]


# ---------------------------------------------------------------------------
# Property computation (same as compute_properties.py)
# ---------------------------------------------------------------------------


def compute_properties_for_smiles(smiles: list[str]) -> dict[str, list[float]]:
    """Compute QED, SA_norm, LogP_norm for each SMILES. Returns dict of lists."""
    from rdkit import Chem
    from rdkit.Chem import QED as rdkitQED
    from rdkit.Chem import Descriptors, RDConfig

    try:
        sa_path = Path(RDConfig.RDContribDir) / "SA_Score"
        if str(sa_path) not in sys.path:
            sys.path.append(str(sa_path))
        import sascorer  # type: ignore[import]

        has_sa = True
    except Exception:
        has_sa = False

    results: dict[str, list[float]] = {col: [] for col in PROPERTY_COLUMN_ORDER}

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            for col in PROPERTY_COLUMN_ORDER:
                results[col].append(float("nan"))
            continue

        # QED
        try:
            results["QED"].append(float(rdkitQED.qed(mol)))
        except Exception:
            results["QED"].append(float("nan"))

        # SA_norm
        if has_sa:
            try:
                raw_sa = sascorer.calculateScore(mol)
                results["SA_norm"].append(float((10.0 - raw_sa) / 9.0))
            except Exception:
                results["SA_norm"].append(float("nan"))
        else:
            results["SA_norm"].append(float("nan"))

        # LogP_norm
        try:
            logp = Descriptors.MolLogP(mol)
            results["LogP_norm"].append(float((max(-3.0, min(7.0, logp)) + 3.0) / 10.0))
        except Exception:
            results["LogP_norm"].append(float("nan"))

        # GSK3B (TDC oracle — expensive; skip for batch, compute separately if needed)
        results["GSK3B"].append(float("nan"))

    # Fill GSK3B with TDC oracle (batched)
    try:
        from oracles import OracleFactory

        oracle = OracleFactory.create_cached("GSK3B")
        gsk3b_scores = oracle(smiles)
        results["GSK3B"] = [float(s) for s in gsk3b_scores]
    except Exception:
        pass  # Leave as nan

    return results


def fraction_in_range(values: list[float], lo: float, hi: float) -> float:
    """Fraction of non-nan values in [lo, hi]."""
    finite = [v for v in values if v == v]  # filter nan
    if not finite:
        return float("nan")
    return sum(lo <= v <= hi for v in finite) / len(finite)


def internal_diversity(smiles: list[str], max_mols: int = 500) -> float:
    """Mean pairwise Tanimoto distance over ECFP4."""
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
        total, count = 0.0, 0
        for i in range(len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
            total += sum(1.0 - s for s in sims)
            count += len(sims)
        return total / count if count > 0 else 0.0
    except ImportError:
        return float("nan")


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def generate_with_condition(
    checkpoint: Path,
    dataset: str,
    condition_dict: dict[str, float],
    n_samples: int,
    out_dir: Path,
    job_name: str,
    device: str,
) -> Path | None:
    """Call submit.py to generate molecules with a specific condition vector.

    Returns path to the generated .smi file, or None on failure.
    """
    cfg = {
        "submission": {
            "python_path": sys.executable,
            "graphinvent_path": "./src/graphinvent/",
            "data_path": "./data/datasets/",
            "dataset": dataset,
            "job_name": job_name,
            "use_slurm": False,
        },
        "job": {
            "job_type": "generate",
            "sample_mode": "generate",
            "device": device,
            "batch_size": 1000,
            "n_samples": n_samples,
            "n_workers": 0,
            "pretrained_model_path": str(checkpoint),
            "sample_conditions": condition_dict,
        },
    }

    # Write a temporary config
    tmp_cfg = out_dir / f"{job_name}_params.json"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg, "w") as f:
        json.dump(cfg, f, indent=2)

    cmd = [sys.executable, str(REPO_ROOT / "submit.py"), "--config", str(tmp_cfg)]
    print(f"  Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, capture_output=False)
    except subprocess.CalledProcessError as exc:
        print(f"  Generation failed (return code {exc.returncode}).")
        return None

    # Locate generated SMILES file
    output_job_dir = REPO_ROOT / "output" / dataset / "generate" / job_name
    smi_files = list(output_job_dir.glob("*_samples.smi"))
    if not smi_files:
        print(f"  No .smi file found in {output_job_dir}.")
        return None
    return smi_files[0]


def load_smiles(path: Path) -> list[str]:
    smiles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                smiles.append(line.split()[0])
    return smiles


def validity_fraction(smiles: list[str]) -> float:
    try:
        from rdkit import Chem

        valid = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
        return len(valid) / max(len(smiles), 1)
    except ImportError:
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained conditional model checkpoint (.pth)",
    )
    p.add_argument("--dataset", type=str, default="chembl_v34_cond")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--property",
        action="append",
        default=None,
        dest="properties",
        help="Run only this property (can be repeated; default: all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    properties_to_run = args.properties or list(PROPERTY_RANGES.keys())
    rows = []

    for prop_name in properties_to_run:
        if prop_name not in PROPERTY_RANGES:
            print(f"Unknown property {prop_name!r}. Available: {list(PROPERTY_RANGES)}")
            continue

        print(f"\n=== Property: {prop_name} ===")
        ranges = PROPERTY_RANGES[prop_name]

        for range_name, (lo, hi) in ranges.items():
            # Build condition dict: set the target property to midpoint, others to neutral
            target_val = _midpoint(lo, hi)
            condition_dict: dict[str, float] = {}
            for col in PROPERTY_COLUMN_ORDER:
                if col == prop_name:
                    condition_dict[col] = target_val
                elif col == "QED":
                    condition_dict[col] = 0.7  # neutral high QED
                elif col == "SA_norm":
                    condition_dict[col] = 0.75  # neutral easy SA
                elif col == "LogP_norm":
                    condition_dict[col] = 0.35  # neutral LogP ≈ 0.5
                elif col == "GSK3B":
                    condition_dict[col] = 0.0  # neutral (no GSK3B requirement)

            job_name = f"cond_{prop_name}_{range_name}"
            print(
                f"  Range: {range_name}  [{lo:.2f}, {hi:.2f}]  target={target_val:.3f}"
            )

            smi_path = generate_with_condition(
                checkpoint=args.checkpoint,
                dataset=args.dataset,
                condition_dict=condition_dict,
                n_samples=args.n_samples,
                out_dir=args.out / "configs",
                job_name=job_name,
                device=args.device,
            )

            if smi_path is None:
                print(f"  Skipping {prop_name} {range_name}: generation failed.")
                continue

            smiles = load_smiles(smi_path)
            valid = validity_fraction(smiles)
            diversity = internal_diversity(smiles)

            # Compute actual property values
            props = compute_properties_for_smiles(smiles)
            actual = props.get(prop_name, [])
            in_range = fraction_in_range(actual, lo, hi)
            actual_finite = [v for v in actual if v == v]
            mean_val = (
                sum(actual_finite) / len(actual_finite)
                if actual_finite
                else float("nan")
            )
            import statistics

            std_val = statistics.stdev(actual_finite) if len(actual_finite) > 1 else 0.0

            row = {
                "property": prop_name,
                "range": range_name,
                "target_lo": lo,
                "target_hi": hi,
                "target_val": target_val,
                "n_generated": len(smiles),
                "validity": valid,
                "cond_accuracy": in_range,
                "internal_diversity": diversity,
                "mean_prop": mean_val,
                "std_prop": std_val,
            }
            rows.append(row)

            print(
                f"    validity={valid:.3f}  in_range={in_range:.3f}  "
                f"mean={mean_val:.3f}±{std_val:.3f}  diversity={diversity:.3f}"
            )

    if rows:
        csv_path = args.out / "conditional_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {csv_path}")
        _print_summary_table(rows)
    else:
        print("\nNo results generated.")


def _print_summary_table(rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(
        f"{'Property':<12} {'Range':<10} {'Validity':>8} {'In-range':>8} {'Diversity':>10}"
    )
    print("=" * 70)
    for r in rows:
        print(
            f"{r['property']:<12} {r['range']:<10} "
            f"{r['validity']:>8.3f} {r['cond_accuracy']:>8.3f} {r['internal_diversity']:>10.3f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
