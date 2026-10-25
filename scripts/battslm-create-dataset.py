#!/usr/bin/env python3
"""
Build data/datasets/battslm/ from data/raw/battslm/Batt-SLM.smi.

    python scripts/battslm-create-dataset.py

The other two corpora have builders and this one did not, so the battery prior's
training set was reproducible only by hand. The build itself is small: the
upstream file is already filtered, so this canonicalizes, drops what the graph
representation cannot encode, deduplicates, and sorts by (heavy atoms, canonical
SMILES) so ties break identically on every run.

What makes it worth a script is the second half. Batt-SLM's selection criteria
are in section 1 of the source work's supporting information
(manuscript/nn6c06255_si_001.pdf), and this checks the delivered file against
every one of them rather than trusting it. A corpus that does not satisfy its
own published criteria is worth knowing about before a prior is trained on it
for 24 hours, and the counts belong in the manuscript's methods beside the
TPDdb curation counts.

Most criteria are NOT applied as filters. Upstream already applied them, so a
violation generally means our reading of the criteria differs from theirs or the
release differs from the paper -- both findings, not things to silently delete.

Two are enforced, because the molecules that fail them are malformed rather than
merely out of scope, and both would corrupt a comparison rather than just widen
one. Measured on the 2026-10-19 release, 21 molecules carry redundant
natural-abundance isotope labels (`C#CC#[14N]`, `CC(C)=[16O]`, `C[12C](C)=O`),
which RDKit preserves through canonicalization: the same molecule with and
without the label canonicalizes to two different strings, so deduplication and
every novelty comparison against this corpus would silently treat them as
distinct, against the rule that both sides of a comparison are canonicalized
identically. A further 21, disjoint from those, carry degenerate fused-aromatic
ring systems (`c1cc2oc1=2`, `o1c2oc1=2`, `c1cc2nnc1=2`) that RDKit accepts and
that are not chemistry. Forty-two molecules of 115,756, 0.036%.

Source: Z.-Y. Zhang, R. Mercado, T. T. Le, C. Zhang, "Unlocking the Chemical
Space for Rechargeable Batteries with Generative Solvent Design System",
ACS Nano 2026, 20, 20714-20729, doi 10.1021/acsnano.6c06255.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent

CITATION = (
    "Zhang Z-Y, Mercado R, Le TT, Zhang C. Unlocking the Chemical Space for "
    "Rechargeable Batteries with Generative Solvent Design System. ACS Nano "
    "2026;20:20714-20729. doi:10.1021/acsnano.6c06255"
)

# Supporting information section 1, verbatim in substance.
SI_ELEMENTS = frozenset(("H", "C", "N", "O", "F", "P", "S", "Cl"))
SI_MIN_HEAVY, SI_MAX_HEAVY = 3, 16  # "2 < Num(non-H atoms) < 17"
SI_MAX_MW = 600.0
SI_MAX_AROMATIC_RINGS = 1  # "Num(arom. ring) < 2"
SI_MIN_TANIMOTO_TO_HOLDOUT = 0.20

# Aprotic means no O-H, N-H or S-H. Counting hydrogens on those elements covers
# both implicit and explicit cases, which a SMARTS on [OX2H] would not.
PROTIC_ELEMENTS = frozenset(("O", "N", "S"))

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# The two criteria whose violators are malformed rather than out of scope. See
# the module docstring for the measured counts and why these two and no others.
ENFORCED_CRITERIA = frozenset(("isotopic", "aromatic_rings"))


def read_smiles(path: Path) -> list:
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line.split()[0] for line in lines if line and not line.startswith("SMILES")]


def si_violations(mol, holdout_fps: list) -> list:
    """Which of the published selection criteria this molecule fails."""
    failed = []
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if symbols - SI_ELEMENTS:
        failed.append("elements")
    n_heavy = mol.GetNumHeavyAtoms()
    if not SI_MIN_HEAVY <= n_heavy <= SI_MAX_HEAVY:
        failed.append("heavy_atoms")
    if Descriptors.MolWt(mol) >= SI_MAX_MW:
        failed.append("molecular_weight")
    n_aromatic_rings = sum(
        all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        for ring in mol.GetRingInfo().AtomRings()
    )
    if n_aromatic_rings > SI_MAX_AROMATIC_RINGS:
        failed.append("aromatic_rings")
    if Chem.GetFormalCharge(mol) != 0:
        failed.append("not_charge_neutral")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        failed.append("radical")
    if any(atom.GetIsotope() for atom in mol.GetAtoms()):
        failed.append("isotopic")
    if any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds()
    ) or Chem.FindMolChiralCenters(mol, useLegacyImplementation=False):
        failed.append("isomeric")
    if "C" not in symbols:
        failed.append("no_carbon")
    if not symbols & {"O", "N"}:
        failed.append("no_oxygen_or_nitrogen")
    if any(
        atom.GetSymbol() in PROTIC_ELEMENTS and atom.GetTotalNumHs() > 0
        for atom in mol.GetAtoms()
    ):
        failed.append("protic")
    if holdout_fps:
        similarity = max(
            DataStructs.BulkTanimotoSimilarity(_MORGAN.GetFingerprint(mol), holdout_fps)
        )
        if similarity < SI_MIN_TANIMOTO_TO_HOLDOUT:
            failed.append("tanimoto_to_fp_holdout")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=REPO_ROOT / "data/raw/battslm/Batt-SLM.smi",
        help="The upstream Batt-SLM SMILES file.",
    )
    parser.add_argument(
        "--fp-holdout",
        type=Path,
        default=REPO_ROOT / "data/raw/battslm/KBS-FP-174.smi",
        help="The F/P-holdout set, for the Tanimoto criterion.",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data/datasets/battslm"
    )
    args = parser.parse_args()

    if not args.raw.exists():
        print(
            f"{args.raw} is missing. Run "
            "`python scripts/download_data.py --source battslm` first.",
            file=sys.stderr,
        )
        return 1

    raw = args.raw.read_bytes()
    print(f"* Reading {args.raw} ({len(raw)} bytes)", flush=True)
    smiles = read_smiles(args.raw)
    print(f"  {len(smiles)} lines", flush=True)

    holdout_fps = []
    if args.fp_holdout.exists():
        for candidate in read_smiles(args.fp_holdout):
            mol = Chem.MolFromSmiles(candidate)
            if mol is not None:
                holdout_fps.append(_MORGAN.GetFingerprint(mol))
        print(f"  F/P-holdout: {len(holdout_fps)} molecules", flush=True)
    else:
        print(
            f"  {args.fp_holdout} is missing, so the Tanimoto criterion is skipped.",
            flush=True,
        )

    print("* Curating", flush=True)
    kept: dict = {}
    n_unparseable = n_multifragment = n_malformed = 0
    violations: dict = {}
    for candidate in smiles:
        mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            n_unparseable += 1
            continue
        if len(Chem.GetMolFrags(mol)) > 1:
            n_multifragment += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in kept:
            continue
        failed = si_violations(mol, holdout_fps)
        for name in failed:
            violations[name] = violations.get(name, 0) + 1
        if set(failed) & ENFORCED_CRITERIA:
            n_malformed += 1
            continue
        kept[canonical] = mol.GetNumHeavyAtoms()

    print(
        f"  {len(kept)} usable, {n_unparseable} unparseable, "
        f"{n_multifragment} multi-fragment, "
        f"{n_malformed} malformed, "
        f"{len(smiles) - len(kept) - n_unparseable - n_multifragment - n_malformed}"
        " duplicate",
        flush=True,
    )

    print("* Against the published selection criteria (SI section 1)", flush=True)
    if violations:
        for name, count in sorted(violations.items(), key=lambda kv: -kv[1]):
            mark = "REMOVED " if name in ENFORCED_CRITERIA else "reported"
            print(f"    {name:26} {count:7} molecules   {mark}", flush=True)
        print(
            "  Only the two malformed-structure criteria are enforced. The rest "
            "are reported and kept: upstream already applied them, so a "
            "violation means our reading differs from theirs or the release "
            "differs from the paper.",
            flush=True,
        )
    else:
        print("    every molecule satisfies every criterion", flush=True)

    ordered = sorted(kept.items(), key=lambda kv: (kv[1], kv[0]))
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{args.output.name}.smi"
    destination.write_text("".join(f"{s}\n" for s, _ in ordered))
    print(f"* Wrote {len(ordered)} molecules to {destination}", flush=True)

    provenance = {
        "source_file": str(args.raw.relative_to(REPO_ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "citation": CITATION,
        "selection": {
            "rule": (
                "canonicalized, deduplicated, single-fragment, sorted by "
                "(heavy atoms, canonical SMILES)"
            ),
            "n_molecules": len(ordered),
            "n_unparseable": n_unparseable,
            "n_multifragment": n_multifragment,
            "n_malformed": n_malformed,
        },
        "published_criteria": {
            "source": "manuscript/nn6c06255_si_001.pdf section 1",
            "elements": sorted(SI_ELEMENTS),
            "heavy_atoms": [SI_MIN_HEAVY, SI_MAX_HEAVY],
            "max_molecular_weight": SI_MAX_MW,
            "max_aromatic_rings": SI_MAX_AROMATIC_RINGS,
            "charge_neutral_and_non_radical": True,
            "no_isotopic_or_isomeric_smiles": True,
            "contains_carbon_and_oxygen_or_nitrogen": True,
            "aprotic": True,
            "min_tanimoto_to_fp_holdout": SI_MIN_TANIMOTO_TO_HOLDOUT,
            "enforced": sorted(ENFORCED_CRITERIA),
            "violations": dict(sorted(violations.items())),
        },
        "summary": {
            "heavy_atoms_min": ordered[0][1],
            "heavy_atoms_max": ordered[-1][1],
        },
        "regenerate_with": (
            f"python scripts/battslm-create-dataset.py --output {args.output}/"
        ),
    }
    with open(args.output / "PROVENANCE.json", "w") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"  provenance: {args.output / 'PROVENANCE.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
