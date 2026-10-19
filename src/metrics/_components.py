"""
Component-level metrics for bifunctional molecules.

Whole-molecule statistics for a bifunctional compound are dominated by the
linker, which is the least informative part, so CLAUDE.md recomputes the core
and conditional metrics over the decomposed target-binding component. That needs
a decomposition, which comes from PROTAC-Splitter (Ribes et al., J. Cheminform.
18:30, 2026), inventoried in tools/README.md.

`split_success_rate` is always reported beside the component metrics, and the
component metrics are computed only over the molecules that split: a falling
split rate otherwise masquerades as improving component quality.

**PROTAC-Splitter runs in its own interpreter**, not in ours. Its XGBoost
edge-classifier was pickled under `xgboost<3`, and loading it under the
`xgboost>=3.2` this project needs for aarch64 GPU support fails with
`'XGBModel' object has no attribute 'feature_weights'`. The two pins are
mutually exclusive, so `tools/install.sh` builds a separate environment and this
module calls into it. Its `model="heuristic"` strategy does load under our
xgboost, but it is not accurate enough to use: on a VH032-based degrader it cut
inside the E3 ligand, assigning the hydroxyproline and methylthiazole to the
linker and calling only the acetyl-tert-leucine cap the E3 ligand.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ._core import ECFP4_RADIUS, bemis_murcko, compute_core_metrics, ecfp4

TOOLS = Path(__file__).resolve().parents[2] / "tools"
SPLITTER_PYTHON = TOOLS / "PROTAC-Splitter" / ".venv" / "bin" / "python"
# Decomposition costs about 2 s per molecule, so a 21,429-molecule corpus is
# roughly 12 hours and a warhead_scaffold split needs every molecule
# decomposed once. The cache is keyed by splitter model and by isomeric
# canonical SMILES, and lives under tools/ rather than in a home directory.
SPLITTER_MODEL = "adaptive"
SPLITTER_CACHE = TOOLS / "splitter-cache"
# Weights land here, never in the library's default cache: a cluster home
# directory has a small quota and a surprise download into one is a failure
# waiting to happen.
SPLITTER_HF_HOME = TOOLS / "hf-cache"

# CLAUDE.md: a predicted E3 ligand is recognizable above a stated ECFP4
# Tanimoto to a known E3 ligand. E3 families are narrow -- cereblon binders are
# thalidomide analogs, VHL binders are VH032-derived hydroxyproline amides -- so
# a genuine family member sits well above this and an unrelated fragment well
# below, which is what makes one threshold workable at all.
E3_SIMILARITY_FLOOR = 0.4

# Run in the splitter's own interpreter, print one JSON line per molecule.
_SPLIT_SCRIPT = """
import json, sys
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from protac_splitter import split_protac
from protac_splitter.evaluation import count_flags, score_split, split_prediction

payload_in = json.load(sys.stdin)
MODEL = payload_in["model"]
smiles = payload_in["smiles"]
out = []
for smi in smiles:
    fragments, reasons = None, "split_raised"
    try:
        prediction = split_protac(smi, model=MODEL, verbose=0)
        key = next((k for k in prediction if k.startswith("default_pred")), None)
        packed = prediction[key] if key else None
        candidate = split_prediction(packed) if packed else None
        if candidate and all(candidate.get(r) for r in ("poi", "linker", "e3")):
            # The library's own reference-free plausibility checks. Without
            # a gate here, a three-atom molecule "splits" into three fragments
            # and counts as a successful decomposition: ethanol does.
            #
            # Gated on the hard structural check and the fragment size and
            # linker topology checks, not on the two *_low_similarity flags.
            # Those compare against the package's representative warhead and
            # E3 sets, and a kinase-inhibitor warhead legitimately resembles
            # neither -- a real CRBN degrader trips poi_low_similarity while
            # being split perfectly correctly.
            scored = score_split(smi, packed)
            blocking = ["structural", "e3_out_of_range", "poi_out_of_range",
                        "linker_too_short", "linker_too_long"]
            tripped = [name for name in blocking if scored.get("flag_" + name)]
            reasons = ";".join(tripped) if tripped else ""
            _, all_reasons = count_flags(scored)
            if not tripped:
                fragments = {r: candidate[r] for r in ("poi", "linker", "e3")}
            reasons = reasons or all_reasons
    except Exception as error:
        reasons = type(error).__name__
    out.append(fragments if fragments else {"failed": reasons})
print(json.dumps(out))
"""


def cap_attachment_points(fragment_smiles: str) -> str | None:
    """
    A fragment's SMILES with its `[*:n]` attachment points replaced by hydrogen.

    PROTAC-Splitter returns fragments carrying the dummy atoms that mark where
    they were cut. Fingerprinting or scoring those directly compares a molecule
    that has extra atoms against reference ligands that do not: measured on a
    real predicted E3 fragment, ECFP4 Tanimoto to thalidomide was 0.711 with the
    dummy and 1.000 once capped, and its SA score 3.558 against 2.597. So
    `e3_recognizable_rate` was computed against a similarity systematically too
    low, and `warhead_sa_score_mean` against a score systematically too high.

    Returns None when the capped fragment will not parse.
    """
    from rdkit import Chem

    if not fragment_smiles:
        return None
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return None
    editable = Chem.RWMol(mol)
    # Reverse order: removing an atom renumbers the ones after it.
    for atom in sorted(
        (a for a in editable.GetAtoms() if a.GetAtomicNum() == 0),
        key=lambda a: a.GetIdx(),
        reverse=True,
    ):
        editable.RemoveAtom(atom.GetIdx())
    try:
        capped = editable.GetMol()
        Chem.SanitizeMol(capped)
    except Exception:
        return None
    return Chem.MolToSmiles(capped) or None


class MissingInput(Exception):
    """A metric's input is absent. The message says which and where from."""


def _cache_path(model: str) -> Path:
    return SPLITTER_CACHE / f"{model}.json"


def _load_cache(model: str) -> dict:
    path = _cache_path(model)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # A truncated cache is a cache, not a crash: rebuild it.
        return {}


def _save_cache(model: str, cache: dict) -> None:
    SPLITTER_CACHE.mkdir(parents=True, exist_ok=True)
    _cache_path(model).write_text(json.dumps(cache, sort_keys=True))


def decompose(smiles_list: list, *, model: str = SPLITTER_MODEL) -> list:
    """
    Decompose each molecule into its POI warhead, linker and E3 ligand.

    Returns one dict per input with keys `poi`, `linker` and `e3`, or None where
    the decomposition failed or did not yield exactly three fragments.

    Only distinct molecules reach the splitter, and results are cached on disk
    under `tools/splitter-cache/<model>.json`, because the call costs seconds
    per molecule and the same corpus is decomposed by both the split and the
    component metrics.

    `poi` is the target-binding component, which is what CLAUDE.md's
    `warhead_*` metrics are computed over. PROTAC-Splitter labels the fragments
    by attachment point, not by position: attachment 1 is the POI side,
    attachment 2 the E3 side, and the fragment carrying both is the linker.
    """
    if not SPLITTER_PYTHON.exists():
        raise MissingInput(
            f"PROTAC-Splitter is not installed at {SPLITTER_PYTHON}.\n"
            "Run tools/install.sh, which builds it its own environment because "
            "its model needs xgboost<3 while this project needs xgboost>=3.2.\n"
            "Without it split_success_rate and every warhead_* metric are "
            "undefined."
        )
    if not smiles_list:
        return []

    cache = _load_cache(model)
    pending = [s for s in dict.fromkeys(smiles_list) if s and s not in cache]
    if not pending:
        return [cache.get(s) for s in smiles_list]
    print(
        f"* PROTAC-Splitter ({model}): {len(pending)} molecules to decompose, "
        f"{len(smiles_list) - len(pending)} served from cache.",
        flush=True,
    )

    # Extend the environment rather than replace it, as _retrotac.py does.
    # Replacing it drops LD_LIBRARY_PATH, which on Linux is how Lmod-provided
    # shared objects are found, and HOME, which breaks `~` resolution inside
    # RDKit and HuggingFace. Only HF_HOME is overridden, to pin any download
    # under tools/ instead of into a quota-limited cluster home directory.
    env = dict(os.environ)
    env["HF_HOME"] = str(SPLITTER_HF_HOME)

    probe = subprocess.run(
        [str(SPLITTER_PYTHON), "-c", _SPLIT_SCRIPT],
        input=json.dumps({"model": model, "smiles": pending}),
        capture_output=True,
        text=True,
        env=env,
    )
    if probe.returncode != 0:
        tail = (probe.stderr or probe.stdout).strip().splitlines()[-5:]
        raise MissingInput("PROTAC-Splitter failed to run:\n    " + "\n    ".join(tail))
    # The last line is the payload; the library prints progress before it.
    for line in reversed(probe.stdout.strip().splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        # A failed split carries its reason rather than a bare None, so a
        # falling split rate can be explained without re-running anything.
        for smi, entry in zip(pending, payload):
            cache[smi] = None if "failed" in entry else entry
        _save_cache(model, cache)
        return [cache.get(s) for s in smiles_list]
    raise MissingInput("PROTAC-Splitter produced no parseable output.")


def compute_component_metrics(
    generated_smiles: list,
    n_sampled: int,
    training_warheads: list,
    known_e3_ligands: list,
    *,
    e3_similarity_floor: float = E3_SIMILARITY_FLOOR,
    isomeric: bool = False,
) -> dict:
    """
    `split_success_rate`, the `warhead_*` recomputation, and
    `e3_recognizable_rate`.

    Consumes the same sample as the core metrics. `e3_max_similarities` comes
    back alongside the rate so the floor can be revisited against real data
    rather than argued about in the abstract.
    """
    decomposed = decompose(generated_smiles)
    split_ok = [d for d in decomposed if d is not None]

    results: dict = {
        "split_success_rate": len(split_ok) / n_sampled if n_sampled else None,
        "n_split_successfully": len(split_ok),
        "e3_similarity_floor": e3_similarity_floor,
        "ecfp4_radius": ECFP4_RADIUS,
    }
    if not split_ok:
        results["e3_recognizable_rate"] = None
        results["e3_max_similarities"] = []
        return results

    # The core table recomputed over target-binding components only. The
    # whole-molecule core metrics stay whole-molecule and are reported
    # separately; these carry the warhead_ prefix CLAUDE.md tabulates.
    # Capped before anything is computed on them: a dummy atom is not part of
    # the warhead. The denominator stays the number that split, so a fragment
    # that will not parse once capped still counts against it.
    warheads = [cap_attachment_points(d["poi"]) for d in split_ok]
    # isomeric must match the generated side. Generated warheads are flat when
    # the model runs without chirality, and the reference warheads come from
    # real degraders that carry it, so the default of True would compare a
    # flattened set against a stereo-bearing one and pin warhead_novelty at
    # 1.0 -- the same asymmetry that was fixed at whole-molecule level.
    warhead_core = compute_core_metrics(
        [w for w in warheads if w],
        n_sampled=len(warheads),
        training_smiles=training_warheads,
        isomeric=isomeric,
    )
    warhead_core.pop("_filter_unavailable", None)
    for key, value in warhead_core.items():
        results[f"warhead_{key}"] = value

    from rdkit import Chem, DataStructs

    known_fps = []
    for smi in known_e3_ligands:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            known_fps.append(ecfp4(mol))
    if not known_fps:
        raise MissingInput(
            "e3_recognizable_rate needs a set of known E3 ligands to compare "
            "against, and got none."
        )

    similarities: list = []
    for d in split_ok:
        capped = cap_attachment_points(d["e3"])
        mol = Chem.MolFromSmiles(capped) if capped else None
        if mol is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(ecfp4(mol), known_fps)
        similarities.append(max(sims) if sims else 0.0)
    results["e3_max_similarities"] = [round(s, 4) for s in similarities]
    results["e3_recognizable_rate"] = (
        sum(1 for s in similarities if s > e3_similarity_floor) / len(similarities)
        if similarities
        else None
    )
    return results


def compute_top100_warhead_scaffolds(top100_smiles: list) -> dict:
    """
    `top100_warhead_scaffold_count`: distinct Bemis-Murcko scaffolds among the
    target-binding components of the 100 highest-scoring molecules.

    This is the reward-hacking detector for a bifunctional dataset, and it
    exists because the whole-molecule `top100_scaffold_count` cannot be one. A
    Bemis-Murcko scaffold of a PROTAC contains every ring system in the
    molecule, so it is nearly unique per compound: a top 100 that has collapsed
    onto a single warhead still reports close to 100 distinct scaffolds as long
    as the linkers differ, which is the failure the metric was supposed to
    catch. Decomposing first and counting scaffolds of the target-binding
    component measures the part the optimization is actually exploiting.

    `top100_split_success_rate` comes back with it and is not optional. The
    count is computed only over the molecules that decomposed, so a falling
    split rate otherwise reads as a collapsing scaffold count: both numbers go
    down together and only one of them means reward hacking.
    """
    results: dict = {
        "top100_warhead_scaffold_count": None,
        "top100_n_split": 0,
        "top100_split_success_rate": None,
    }
    if not top100_smiles:
        return results

    decomposed = decompose(top100_smiles)
    split_ok = [d for d in decomposed if d is not None]
    results["top100_n_split"] = len(split_ok)
    results["top100_split_success_rate"] = len(split_ok) / len(top100_smiles)
    if not split_ok:
        return results

    # Capped before the scaffold is taken: the attachment point is a dummy atom
    # that is not part of the warhead, and a scaffold derived with it in place
    # is the scaffold of a different molecule.
    scaffolds = set()
    for fragments in split_ok:
        capped = cap_attachment_points(fragments["poi"])
        scaffold = bemis_murcko(capped) if capped else None
        # Falsy, not just None: a warhead with no ring system has an empty
        # scaffold, and counting that as one inflates the tally.
        if scaffold:
            scaffolds.add(scaffold)
    results["top100_warhead_scaffold_count"] = len(scaffolds)
    return results
