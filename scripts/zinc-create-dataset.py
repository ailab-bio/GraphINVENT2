#!/usr/bin/env python3
"""
Build a GraphINVENT dataset from the ZINC-250k release that download_data.py
fetches.

    python scripts/zinc-create-dataset.py --output data/datasets/zinc250k

This exists because the raw release is not a SMILES file. It is a CSV with the
header `smiles,logP,qed,SAS` whose SMILES field is double-quoted and contains a
trailing newline inside the quotes, so each record spans two physical lines. The
preprocessing job reads whitespace-delimited SMILES, so pointing it at the CSV
makes it read a leading `"` as part of the first molecule and fail on line 1.
Parsing it with the csv module once, here, is the fix; preprocessing keeps its
one input format.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

SOURCE = "data/raw/zinc/250k_rndm_zinc_drugs_clean_3.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path(SOURCE))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--n-molecules",
        type=int,
        default=0,
        help="keep only the first N after curation; 0 keeps all",
    )
    ap.add_argument("--max-heavy-atoms", type=int, default=54)
    args = ap.parse_args()

    if not args.source.exists():
        print(
            f"{args.source} is missing. Run: python scripts/download_data.py "
            "--source zinc",
            file=sys.stderr,
        )
        return 1

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    with args.source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "smiles" not in rows[0]:
        print(
            f"No 'smiles' column in {args.source}; header was "
            f"{list(rows[0]) if rows else 'empty'}",
            file=sys.stderr,
        )
        return 1
    print(f"* {len(rows)} records in {args.source}", flush=True)

    kept: dict = {}
    n_unparseable = n_multifragment = n_too_large = 0
    for row in rows:
        smiles = (row.get("smiles") or "").strip()
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_unparseable += 1
            continue
        # The decoding route cannot order a disconnected graph, so salts and
        # mixtures are dropped rather than silently traversed.
        if len(Chem.GetMolFrags(mol)) > 1:
            n_multifragment += 1
            continue
        if mol.GetNumHeavyAtoms() > args.max_heavy_atoms:
            n_too_large += 1
            continue
        kept[Chem.MolToSmiles(mol)] = mol.GetNumHeavyAtoms()

    print(
        f"  {len(kept)} usable, {n_unparseable} unparseable, "
        f"{n_multifragment} multi-fragment, {n_too_large} over the size cap",
        flush=True,
    )
    if not kept:
        print("No usable molecules.", file=sys.stderr)
        return 1

    # Deterministic order, so the same N come out on every run.
    selected = sorted(kept.items(), key=lambda kv: (kv[1], kv[0]))
    if args.n_molecules:
        selected = selected[: args.n_molecules]

    args.output.mkdir(parents=True, exist_ok=True)
    name = args.output.name
    smi_path = args.output / f"{name}.smi"
    smi_path.write_text("\n".join(smi for smi, _ in selected) + "\n")
    print(f"* Wrote {len(selected)} molecules to {smi_path}", flush=True)

    sizes = [n for _, n in selected]
    (args.output / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "source_file": str(args.source),
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "selection": {
                    "rule": "canonicalized, deduplicated, single-fragment, "
                    "sorted by (heavy atoms, canonical SMILES)",
                    "n_molecules": len(selected),
                    "max_heavy_atoms": args.max_heavy_atoms,
                    "n_candidates_after_curation": len(kept),
                },
                "summary": {
                    "heavy_atoms_min": min(sizes),
                    "heavy_atoms_max": max(sizes),
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  provenance: {args.output / 'PROVENANCE.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
