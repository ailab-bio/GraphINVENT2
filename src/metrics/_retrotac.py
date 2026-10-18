"""
RetroTAC: route-derived synthesizability for whole PROTACs.

Whole-molecule SA score reads a PROTAC's size as difficulty. Measured on 500
real TPD-DB degraders, median SA is 4.71 and **58.6% fail the `sa_score < 4.5`
quality gate**, against 2.87 and 6.2% for drug-like ZINC molecules of median 23
heavy atoms. Those degraders were made and tested, so on bifunctional molecules
that gate rejects the majority of chemistry that demonstrably exists, and
`sa_score_mean` measures size more than synthesizability.

RetroTAC predicts a route-derived score from the SMILES directly, without
calling a retrosynthesis planner at inference, which is what makes it usable
inside an evaluation loop. It replaces SA score at the **whole-PROTAC** level
only; SA score remains the right measure for the decomposed warheads, where
the molecules are drug-sized and it behaves, and for every non-bifunctional
dataset.

Runs in its own interpreter. RetroTAC pins `python ==3.12.*` and
`xgboost>=2.0,<3.0`; this project is 3.13 with `xgboost>=3.2` for aarch64 GPU
support. The two cannot share an environment, so this module shells out, the
same arrangement `_components.py` uses for PROTAC-Splitter.

Two things about the score, both from the authors' own notes:

- **It is not clipped to [0, 1].** A value outside that range is a signal that
  the molecule is far from the training distribution, not a bug, so nothing
  here clamps it.
- **The 0.7 threshold is a reporting convention, not a calibrated decision
  boundary.** The continuous score and its ranking are the primary output. Any
  hard gate built on a threshold inherits that caveat; see
  `QUALITY_RETROTAC_MIN` in `_budget.py`.

The model used is the released final **GNN refit** (held-out R² 0.615, RMSE
0.140), which ships in the repository. RetroTAC proper is a 27-member Caruana
ensemble (R² 0.647, RMSE 0.132) whose members are the per-fold
cross-validation models; those are 1.4 GiB and the release archives omit them,
so running the ensemble would mean retraining the cross-validation first. The
single GNN is what is available without that, and the gap is 0.03 in R².
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "tools"
RETROTAC_ROOT = TOOLS / "RetroTAC"
RETROTAC_PYTHON = RETROTAC_ROOT / ".venv" / "bin" / "python"

# Base path without the extension: the loader appends ".ckpt" itself.
RETROTAC_CHECKPOINT = (
    RETROTAC_ROOT / "outputs/models/gnn_20260828_182305/gnn_20260828_182305_final"
)

# Any model weight this tool downloads lands under tools/, never in a library's
# default cache: a cluster home directory has a small quota.
RETROTAC_HF_HOME = TOOLS / "hf-cache"

# Everything goes over in ONE subprocess call. Loading the checkpoint costs
# about 6 s against 0.05 s to score a molecule, so chunking here would pay that
# load per chunk: at 10,000 molecules in chunks of 256 it would be forty model
# loads, four minutes of pure startup. Chemprop batches internally (64 at a
# time), so a single call is already memory-bounded.
#
# Chemprop prints "Dropping last batch of size 1 to avoid issues with batch
# normalization" when a trailing batch holds one molecule. That is a training
# dataloader message: verified at n = 1, 65, 129 and 300 that `predict` still
# returns one row per input. The length check below would catch it regardless.


class MissingInput(Exception):
    """A metric's input is absent. The message says which and where from."""


_SCORE_SCRIPT = """
import json, sys, os
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from retrotac.models.gnn.model import CheMeleonRegressor

payload = json.load(sys.stdin)
model = CheMeleonRegressor.load(payload["checkpoint"], device="cpu")
preds = model.predict(payload["smiles"])
# Column 0 is the synthesizability target. Not clipped, deliberately.
out = [None if v != v else float(v) for v in preds[:, 0]]
print(json.dumps(out))
"""


def _run_retrotac(smiles: list) -> list:
    """Score `smiles` in RetroTAC's own interpreter, preserving order."""
    if not RETROTAC_PYTHON.exists():
        raise MissingInput(
            f"RetroTAC is not installed: no interpreter at {RETROTAC_PYTHON}. "
            "Run `bash tools/install.sh`."
        )
    if not RETROTAC_CHECKPOINT.with_suffix(".ckpt").exists():
        raise MissingInput(
            f"RetroTAC's GNN checkpoint is missing at "
            f"{RETROTAC_CHECKPOINT.with_suffix('.ckpt')}. It ships gzipped under "
            "tools/RetroTAC/release/; `bash tools/install.sh` unpacks it."
        )

    # Extend the environment rather than replace it. Replacing it drops
    # LD_LIBRARY_PATH, which on Linux is how Lmod- and conda-provided shared
    # objects are found, and HOME, which breaks `~` resolution inside RDKit and
    # HuggingFace. Only HF_HOME is overridden, to pin any download under tools/.
    env = dict(os.environ)
    env["HF_HOME"] = str(RETROTAC_HF_HOME)

    result = subprocess.run(
        [str(RETROTAC_PYTHON), "-c", _SCORE_SCRIPT],
        input=json.dumps({"checkpoint": str(RETROTAC_CHECKPOINT), "smiles": smiles}),
        capture_output=True,
        text=True,
        # cwd matters: RetroTAC resolves its own relative paths from here.
        cwd=str(RETROTAC_ROOT),
        env=env,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise MissingInput(
            f"RetroTAC failed to score {len(smiles)} molecules "
            f"(exit {result.returncode}):\n{tail}"
        )
    try:
        scores = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise MissingInput(
            f"RetroTAC produced output this module could not read: {error}"
        ) from None

    if len(scores) != len(smiles):
        raise MissingInput(
            f"RetroTAC returned {len(scores)} scores for {len(smiles)} molecules; "
            "the two must line up by position."
        )
    return scores


def retrotac_scores(smiles: list) -> list:
    """
    RetroTAC score for each input SMILES, in order, None where it has none.

    Raises `MissingInput` when the tool or its checkpoint is absent, so the
    caller records a reason rather than a bare null.
    """
    wanted = [(i, s) for i, s in enumerate(smiles) if s]
    out: list = [None] * len(smiles)
    if not wanted:
        return out
    for (i, _), value in zip(wanted, _run_retrotac([s for _, s in wanted])):
        out[i] = value
    return out


def compute_retrotac_metrics(unique_valid_smiles: list) -> tuple:
    """
    (`retrotac_mean`/`retrotac_median`, score per molecule).

    The aggregate uses the same denominator as `sa_score_mean`, so the two read
    side by side: this reports whole-molecule synthesizability for bifunctional
    molecules, SA score reports it for everything else, and both are kept
    rather than one overwriting the other.

    The per-molecule dict is returned as well so the goal-directed quality gate
    can reuse these scores instead of paying for a second pass.
    """
    per_molecule = {
        smi: value
        for smi, value in zip(unique_valid_smiles, retrotac_scores(unique_valid_smiles))
        if value is not None
    }
    if not per_molecule:
        return {"retrotac_mean": None, "retrotac_median": None}, {}
    values = list(per_molecule.values())
    return (
        {
            "retrotac_mean": float(np.mean(values)),
            "retrotac_median": float(np.median(values)),
        },
        per_molecule,
    )
