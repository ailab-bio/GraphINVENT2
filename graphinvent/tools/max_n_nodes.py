"""
Gets the maximum number of nodes per molecule present in a set of molecules.

Deprecated: use scan_features.py instead, which collects all feature
vocabulary (atom types, formal charges, implicit H counts, max nodes)
in a single pass.

To use script, run:
    python graphinvent/tools/max_n_nodes.py --smi path/to/file.smi
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_molecules


# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--smi",
                    type=str,
                    default="data/gdb13_1K/train.smi",
                    help="SMILES file containing molecules to analyse.")
args = parser.parse_args()


def get_max_n_atoms(smi_file : str) -> int:
    """
    Determines the maximum number of atoms per molecule in an input SMILES file.

    Args:
    ----
        smi_file (str) : Full path/filename to SMILES file.
    """
    molecules = load_molecules(path=smi_file)

    max_n_atoms = 0
    for mol in molecules:
        n_atoms = mol.GetNumAtoms()

        if n_atoms > max_n_atoms:
            max_n_atoms = n_atoms

    return max_n_atoms


if __name__ == "__main__":
    max_n_atoms = get_max_n_atoms(smi_file=args.smi)
    print("* Max number of atoms in input file:", max_n_atoms, flush=True)
    print("Done.", flush=True)
