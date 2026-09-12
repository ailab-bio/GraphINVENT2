"""
Functions for loading molecules from SMILES, as well as loading the model type.
"""

# load general packages and functions
import rdkit
from rdkit.Chem.rdmolfiles import SmilesMolSupplier


def molecules(path: str) -> rdkit.Chem.rdmolfiles.SmilesMolSupplier:
    """
    Reads a SMILES file (full path/filename specified by `path`) and returns
    `rdkit.Mol` objects.
    """
    # check first line of SMILES file to see if contains header
    with open(path) as smi_file:
        first_line = smi_file.readline()
        has_header = bool("SMILES" in first_line)
    smi_file.close()

    # Property columns are tab-separated, so RDKit must be told the delimiter;
    # with the default it hands the whole "SMILES\tvalue" line to the parser and
    # every molecule comes back None.
    molecule_set = SmilesMolSupplier(
        path,
        sanitize=True,
        nameColumn=-1,
        titleLine=has_header,
        delimiter="\t" if "\t" in first_line else " \t",
    )
    return molecule_set
