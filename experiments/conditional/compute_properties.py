"""
Compute molecular properties and create a TSV file for conditional training.

Usage (from repository root):
    python experiments/conditional/compute_properties.py \
        --smiles data/raw/chembl_v34_filtered.smi \
        --out data/raw/chembl_v34_cond.tsv \
        [--surrogate NAME=PATH ...]

Output TSV format (tab-separated, with header):
    SMILES  QED  SA_norm  LogP_norm  [<surrogate columns>...]

Property definitions:
    QED      : Quantitative Estimate of Drug-likeness (RDKit), range [0, 1].
    SA_norm  : Synthetic accessibility, normalised to [0, 1] where 1 = easy.
               SA_norm = (10 - raw_SA) / 9.0
    LogP_norm: LogP clipped to [-3, 7] then scaled to [0, 1]:
               LogP_norm = (LogP - (-3)) / (7 - (-3)) = (LogP + 3) / 10
               These bounds cover > 99% of drug-like molecules.
    <name>   : Any additional property from a surrogate you supply with
               --surrogate NAME=PATH, where PATH is a pickled scikit-learn
               model over Morgan fingerprints (see
               src/graphinvent/tools/train-surrogate.py).

Notes:
    - Molecules that fail RDKit sanitisation are skipped silently.
    - All property values are rounded to 4 decimal places.
    - The output file can be used directly as smiles_file in preprocess_params.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Property computation
# ---------------------------------------------------------------------------


def compute_qed(mol) -> float:
    from rdkit.Chem import QED

    try:
        return float(QED.qed(mol))
    except Exception:
        return float("nan")


def compute_sa_norm(mol) -> float:
    """Normalised SA score: (10 - SA) / 9, so 1 = easiest, 0 = hardest."""
    try:
        from rdkit import RDConfig

        sa_path = Path(RDConfig.RDContribDir) / "SA_Score"
        if str(sa_path) not in sys.path:
            sys.path.append(str(sa_path))
        import sascorer  # type: ignore[import]

        raw_sa = sascorer.calculateScore(mol)
        return float((10.0 - raw_sa) / 9.0)
    except Exception:
        return float("nan")


def compute_logp_norm(mol) -> float:
    """
    LogP clipped to [-3, 7] then scaled to [0, 1].
    Covers > 99% of drug-like molecules.
    """
    from rdkit.Chem import Descriptors

    try:
        logp = Descriptors.MolLogP(mol)
        logp_clipped = max(-3.0, min(7.0, logp))
        return float((logp_clipped + 3.0) / 10.0)
    except Exception:
        return float("nan")


def load_surrogate(name: str, path: str):
    """
    Load a user-trained surrogate as a scoring oracle.

    Failing loudly here is deliberate: a property column silently filled with
    zeros would be used as a conditioning target and quietly train the model on
    nothing.
    """
    from oracles import OracleFactory

    return OracleFactory.create_cached(name, {"type": "sklearn", "path": path})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--smiles", type=Path, required=True, help="Input SMILES file (one per line)"
    )
    p.add_argument("--out", type=Path, required=True, help="Output TSV file")
    p.add_argument(
        "--surrogate",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add a property column from a pickled scikit-learn surrogate. "
        "Repeatable. Train one with src/graphinvent/tools/train-surrogate.py.",
    )
    p.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for surrogate calls"
    )
    p.add_argument(
        "--max-mols",
        type=int,
        default=None,
        help="Maximum number of molecules to process",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from rdkit import Chem

    # Load SMILES
    print(f"Loading SMILES from {args.smiles}...")
    raw_smiles = []
    with open(args.smiles) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Skip header lines containing "SMILES"
            if line.upper().startswith("SMILES"):
                continue
            smi = line.split()[0]
            raw_smiles.append(smi)
            if args.max_mols and len(raw_smiles) >= args.max_mols:
                break
    print(f"  {len(raw_smiles)} SMILES loaded.")

    # Parse and compute RDKit properties
    print("Computing RDKit properties (QED, SA, LogP)...")
    records = []
    skipped = 0
    for smi in raw_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped += 1
            continue
        canon = Chem.MolToSmiles(mol)
        records.append(
            {
                "SMILES": canon,
                "QED": compute_qed(mol),
                "SA_norm": compute_sa_norm(mol),
                "LogP_norm": compute_logp_norm(mol),
            }
        )
    print(f"  {len(records)} molecules retained ({skipped} skipped).")

    # Surrogate-predicted properties, batched for efficiency
    valid_smiles = [r["SMILES"] for r in records]
    surrogate_cols = []

    for entry in args.surrogate:
        if "=" not in entry:
            raise SystemExit(f"--surrogate expects NAME=PATH, got '{entry}'.")
        name, path = entry.split("=", 1)
        print(f"Computing {name} scores from {path}...")
        oracle = load_surrogate(name, path)
        scores = []
        for i in range(0, len(valid_smiles), args.batch_size):
            scores.extend(oracle(valid_smiles[i : i + args.batch_size]))
            if (i // args.batch_size) % 10 == 0:
                print(f"  {i}/{len(valid_smiles)}...")
        for r, s in zip(records, scores):
            r[name] = s
        surrogate_cols.append(name)

    property_cols = ["QED", "SA_norm", "LogP_norm"] + surrogate_cols

    # Write TSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing TSV to {args.out}...")
    with open(args.out, "w") as f:
        header = "\t".join(["SMILES"] + property_cols)
        f.write(header + "\n")
        for r in records:
            values = [r["SMILES"]] + [f"{r[col]:.4f}" for col in property_cols]
            f.write("\t".join(values) + "\n")

    print(f"Done. {len(records)} molecules written to {args.out}.")
    print(f"Columns: SMILES, {', '.join(property_cols)}")
    print(
        f"\nNext step: run preprocess_params.json with condition_dim={len(property_cols)}"
    )


if __name__ == "__main__":
    main()
