#!/usr/bin/env python3
"""
Build data/datasets/battp30k/ from data/raw/battslm/Batt-P30K.h5.

    python scripts/battp30k-create-dataset.py

The raw file holds 29,519 molecule groups, each with a SMILES, 3D coordinates
and DFT labels computed at wB97X-V/def2-TZVPPD/SMD(eps=18.5): a `dipole`
3-vector, a `quadrupole` 6-vector, `homo`, `lumo`, `gap`, `ip`, `ea` and three
energies. This writes the SMILES with the scalar labels a conditional job can
condition on, as a tab-separated file whose first column is SMILES.

**The dipole norm is the column that matters, and it is not the quantity the
experiment conditions on.** `battery_conditional` conditions on dielectric
constant, which the source work reaches *from* the dipole: a linear fit of
permittivity against squared dipole moment below 4.0 Debye and an MLR ensemble
predicting that coefficient above it, fitted on a 200-molecule experimental set
(DC-200) that was never released. Section 2.5 of the supporting information
gives the architecture and Figure S15 holds the fitted values as a raster plot,
so the regression cannot be reconstructed from the paper alone. Until it exists,
this script writes `dipole` and the other DFT labels and **no `dc` column**;
adding one by guessing at the regression would put a fabricated number in a
training set.

A second reason the dipole here is not directly the chain's input: the source
work averages the dipole over the three lowest-energy ETKDGv3 conformers,
MMFF94-optimized, because the dipole is strongly conformer-dependent. The label
in this file is for the single conformer whose coordinates it carries. For
conditioning that is the better label, since it is a direct DFT value rather than
a prediction; for comparing against the published chain it is not the same
quantity, and the two must not be mixed in one table.

Source: Z.-Y. Zhang, R. Mercado, T. T. Le, C. Zhang, "Unlocking the Chemical
Space for Rechargeable Batteries with Generative Solvent Design System",
ACS Nano 2026, 20, 20714-20729, doi 10.1021/acsnano.6c06255.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import h5py
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent

CITATION = (
    "Zhang Z-Y, Mercado R, Le TT, Zhang C. Unlocking the Chemical Space for "
    "Rechargeable Batteries with Generative Solvent Design System. ACS Nano "
    "2026;20:20714-20729. doi:10.1021/acsnano.6c06255"
)

# The scalar labels, in the order they are written. `dipole` is the vector norm;
# the rest are read straight out of their one-element datasets.
SCALAR_FIELDS = ("homo", "lumo", "gap", "ip", "ea")

# Same cap as the rest of the battery program, from the corpus criteria rather
# than from anyone's choice.
MAX_HEAVY_ATOMS = 16


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, default=REPO_ROOT / "data/raw/battslm/Batt-P30K.h5"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data/datasets/battp30k"
    )
    args = parser.parse_args()

    if not args.raw.exists():
        print(
            f"{args.raw} is missing. Run "
            "`python scripts/download_data.py --source battslm` first.",
            file=sys.stderr,
        )
        return 1

    print(f"* Reading {args.raw} ({args.raw.stat().st_size} bytes)", flush=True)
    rows: dict = {}
    n_groups = n_unparseable = n_multifragment = n_too_large = 0
    with h5py.File(args.raw, "r") as handle:
        for name in handle:
            group = handle[name]
            n_groups += 1
            mol = Chem.MolFromSmiles(_text(group["smiles"][0]))
            if mol is None:
                n_unparseable += 1
                continue
            if len(Chem.GetMolFrags(mol)) > 1:
                n_multifragment += 1
                continue
            if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
                n_too_large += 1
                continue
            canonical = Chem.MolToSmiles(mol)
            if canonical in rows:
                # A duplicate structure with its own conformer and its own
                # labels. Keeping the first is arbitrary, so record that it
                # happened rather than hiding it.
                continue
            dipole = group["dipole"][:]
            rows[canonical] = [
                round(math.sqrt(float(sum(c * c for c in dipole))), 6)
            ] + [round(float(group[f][0]), 6) for f in SCALAR_FIELDS]

    columns = ("dipole",) + SCALAR_FIELDS
    print(
        f"  {n_groups} groups: {len(rows)} usable, {n_unparseable} unparseable, "
        f"{n_multifragment} multi-fragment, {n_too_large} over "
        f"{MAX_HEAVY_ATOMS} heavy atoms, "
        f"{n_groups - len(rows) - n_unparseable - n_multifragment - n_too_large}"
        " duplicate structures",
        flush=True,
    )

    ordered = sorted(rows.items())
    args.output.mkdir(parents=True, exist_ok=True)

    # Two files, because the conditioning contract is strict: a preprocess job
    # asserts that the SMILES file has exactly condition_dim property columns,
    # in the order conditioning.properties names, since the condition vector is
    # built positionally from them. battery_conditional sets condition_dim 1 on
    # dipole, so the file it reads carries dipole and nothing else; a six-column
    # file would be rejected up front. The remaining DFT labels go to the
    # metadata TSV beside it, the same split the other datasets use.
    destination = args.output / f"{args.output.name}_dipole.smi"
    with open(destination, "w") as out:
        out.write("SMILES\tdipole\n")
        for smiles, values in ordered:
            out.write(f"{smiles}\t{values[0]}\n")
    print(f"* Wrote {len(ordered)} molecules to {destination}", flush=True)

    metadata = args.output / f"{args.output.name}_metadata.tsv"
    with open(metadata, "w") as out:
        out.write("SMILES\t" + "\t".join(columns) + "\n")
        for smiles, values in ordered:
            out.write(smiles + "\t" + "\t".join(str(v) for v in values) + "\n")
    print(f"  all {len(columns)} DFT labels: {metadata}", flush=True)

    dipoles = [values[0] for _, values in ordered]
    print(
        f"  dipole norm: min {min(dipoles):.4f}, max {max(dipoles):.4f}, "
        f"mean {sum(dipoles) / len(dipoles):.4f} Debye",
        flush=True,
    )

    provenance = {
        "source_file": str(args.raw.relative_to(REPO_ROOT)),
        "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "citation": CITATION,
        "dft_level": "wB97X-V/def2-TZVPPD/SMD(eps=18.5)",
        "conditioning_file": f"{args.output.name}_dipole.smi",
        "conditioning_columns": ["dipole"],
        "metadata_file": f"{args.output.name}_metadata.tsv",
        "metadata_columns": list(columns),
        "selection": {
            "rule": (
                "one row per canonical SMILES, parseable and single-fragment, "
                "at most 16 heavy atoms; sorted by canonical SMILES"
            ),
            "n_molecules": len(ordered),
            "n_groups_in_source": n_groups,
            "n_unparseable": n_unparseable,
            "n_multifragment": n_multifragment,
            "n_over_heavy_atom_cap": n_too_large,
        },
        "notes": {
            "dipole": (
                "the norm of the DFT dipole 3-vector for the single conformer "
                "whose coordinates the source file carries, NOT the "
                "three-lowest-conformer average the published permittivity "
                "chain uses"
            ),
            "dielectric_constant": (
                "deliberately absent. It is reached from the dipole through the "
                "LF-MLR-DC-200 piecewise regression, whose fitted coefficients "
                "are published only as a raster figure (SI figure S15), so no "
                "dc column can be written without fabricating one."
            ),
        },
        "regenerate_with": (
            f"python scripts/battp30k-create-dataset.py --output {args.output}/"
        ),
    }
    with open(args.output / "PROVENANCE.json", "w") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"  provenance: {args.output / 'PROVENANCE.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
