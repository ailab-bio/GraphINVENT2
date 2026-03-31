"""
Test configuration for GraphINVENT2 preprocessing checks.

Edit the values below to point at the dataset you want to verify.

DATASET_DIR   : directory that contains train.smi, valid.smi, test.smi,
                train.h5, valid.h5, test.h5, and preprocessing_params.json
                (i.e. the output of a completed preprocessing job).

SMILES_FILE   : path to the *original* SMILES file used as input.
                Set to None if you used pre-split files (Mode B) and
                don't want to run the total-count check.
"""
from pathlib import Path

DATASET_DIR = Path("data/datasets/debug")
SMILES_FILE = Path("data/datasets/debug/debug.smi")
