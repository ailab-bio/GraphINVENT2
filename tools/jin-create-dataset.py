"""
Filters ChEMBLmolecules based on number of heavy atoms and formal charge.

Data downloaded from:
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_chemreps.txt.gz


To use script, run:
(graphinvent)$ python jin-create-dataset.py
"""
import os
from pathlib import Path
import shutil
import rdkit
from rdkit import Chem

def read_file_as_list(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]


def save_smiles(smi_file : str, smi_list : list) -> None:
    """Saves input list of SMILES to the specified file path."""
    smi_writer = rdkit.Chem.rdmolfiles.SmilesWriter(smi_file)
    for smi in smi_list:
        try:
            mol = rdkit.Chem.MolFromSmiles(smi[0])
            if 150 < mol.GetNumAtoms() < 220:  # filter out small molecules
                save = True
                for atom in mol.GetAtoms():
                    if atom.GetFormalCharge() not in [-1, 0, +1]:  # filter out molecules with large formal charge
                        save = False
                        break
                    if atom.GetSymbol() not in ['C', 'N', 'O', 'P', 'Fe', 'Co', 'Ni', 'Zn']:  # filter out molecules with unacceptable (non-cage) atom types
                        save = False
                        break
                if save:
                    smi_writer.write(mol)
        except:  # likely TypeError or AttributeError e.g. "smi[0]" is "nan"
            continue
    smi_writer.close()


if __name__ == "__main__":
    DATA_PATH = f"./data/pre-training/chembl/"

    print(f"* Re-saving CHEMBL dataset in a format GraphINVENT can parse.")
    print("-- Filtering and splitting data...")
    smiles_list = read_file_as_list(filename=f"{DATA_PATH}train.smi")
    save_smiles(smi_file=f"{DATA_PATH}train.smi", smi_list=smiles_list)

    print("Done.", flush=True)
