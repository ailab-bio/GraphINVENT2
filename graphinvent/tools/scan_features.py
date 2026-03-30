"""
Scans one or more SMILES files and reports the molecular feature vocabulary
needed for GraphINVENT2 preprocessing:

  - Unique atom types (element symbols)
  - Unique formal charges
  - Unique implicit hydrogen counts
  - Maximum number of heavy atoms per molecule

These values are automatically detected during a preprocessing run.  Use this
script to inspect them ahead of time, or to verify that two datasets share the
same vocabulary before transfer learning.

Usage:
    python scan_features.py --smi path/to/file.smi [path/to/other.smi ...]
    python scan_features.py --smi train.smi valid.smi test.smi
"""
import argparse
import sys

import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolfiles import SmilesMolSupplier


def load_molecules(path: str) -> SmilesMolSupplier:
    """Returns a SmilesMolSupplier for the given .smi file."""
    with open(path) as f:
        first_line = f.readline()
    has_header = "SMILES" in first_line
    return SmilesMolSupplier(
        path, sanitize=True, nameColumn=-1, titleLine=has_header
    )


def scan_features(
    smi_paths: list,
    use_explicit_H: bool = False,
    ignore_H: bool = False,
) -> dict:
    """
    Scans SMILES files and returns a dict with the detected feature vocabulary.

    Args:
        smi_paths     : Paths to one or more .smi files.
        use_explicit_H: If True, Hs are added explicitly before atom scanning.
        ignore_H      : If True, implicit H counts are not collected.

    Returns:
        A dict with keys: atom_types, formal_charge, imp_H, max_n_nodes.
    """
    atom_types_set    = set()
    formal_charge_set = set()
    imp_H_set         = set()
    max_n_nodes       = 0

    for path in smi_paths:
        supplier = load_molecules(path)
        for mol in supplier:
            if mol is None:
                continue
            if use_explicit_H and not ignore_H:
                mol = Chem.AddHs(mol)
            n = mol.GetNumAtoms()
            if n > max_n_nodes:
                max_n_nodes = n
            for atom in mol.GetAtoms():
                atom_types_set.add(atom.GetSymbol())
                formal_charge_set.add(atom.GetFormalCharge())
                if not use_explicit_H and not ignore_H:
                    imp_H_set.add(atom.GetTotalNumHs())

    return {
        "atom_types"   : sorted(atom_types_set),
        "formal_charge": sorted(formal_charge_set),
        "imp_H"        : sorted(imp_H_set),
        "max_n_nodes"  : max_n_nodes,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smi",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more SMILES files to scan.",
    )
    parser.add_argument(
        "--use_explicit_H",
        action="store_true",
        default=False,
        help="Add explicit Hs before scanning atoms (match use_explicit_H=true in params).",
    )
    parser.add_argument(
        "--ignore_H",
        action="store_true",
        default=False,
        help="Omit implicit H counts from the report (match ignore_H=true in params).",
    )
    args = parser.parse_args()

    if args.use_explicit_H and args.ignore_H:
        print("Error: --use_explicit_H and --ignore_H are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(args.smi)} file(s):", flush=True)
    for path in args.smi:
        print(f"  {path}", flush=True)

    features = scan_features(
        smi_paths=args.smi,
        use_explicit_H=args.use_explicit_H,
        ignore_H=args.ignore_H,
    )

    print()
    print(f"atom_types    : {features['atom_types']}")
    print(f"formal_charge : {features['formal_charge']}")
    if not args.ignore_H:
        print(f"imp_H         : {features['imp_H']}")
    print(f"max_n_nodes   : {features['max_n_nodes']}")
    print()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
