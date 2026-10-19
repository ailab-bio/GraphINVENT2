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

from ._core import (
    ECFP4_BITS,
    ECFP4_RADIUS,
    bemis_murcko,
    compute_core_metrics,
    ecfp4,
)

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

# `e3_recognizable_rate` is an exact substructure test against the recognized
# E3-binder cores, listed one per line with their citation in
# tools/data/e3_binder_cores.smarts. A similarity floor was the wrong
# instrument: E3 families are defined by a pharmacophore that is either present
# or absent, and a Tanimoto threshold answers "how close is this to some known
# ligand" when the question is "does this contain a recognized binding core".
# The maximum similarity is still computed and reported as a supporting field,
# so the substructure verdict can be read against it.
E3_CORES_FILE = TOOLS / "data" / "e3_binder_cores.smarts"
E3_SIMILARITY_FLOOR = 0.4

# How a target-binding fragment is turned into the molecule that gets
# fingerprinted. PROTAC-Splitter returns fragments carrying the `[*:n]` dummy
# atoms that mark where they were cut, and the two readings of "the warhead"
# differ: `parent_warhead` caps the attachment point with hydrogen, giving the
# ligand as it exists off the degrader, and `attachment_aware` keeps the dummy,
# so two warheads cut at different positions are different molecules. The
# default is the parent, because that is what the reference warheads are and
# comparing a capped fragment against an uncapped one systematically
# understates similarity. Recorded in metrics.json either way, since every
# warhead_* number depends on which was used.
WARHEAD_REPRESENTATION = "parent_warhead"

# The number of unique valid molecules decomposed for the whole-sample
# component metrics. Decomposition costs seconds per molecule, so a 10,000
# molecule sample is hours; 2,000 drawn with the job's own seed bounds the cost
# while keeping the sampling error on a rate below a percentage point. The size
# and the seed are reported, because a rate over a subsample is a different
# number from a rate over the sample.
COMPONENT_SAMPLE_SIZE = 2000


def splitter_workers() -> int:
    """
    How many splitter interpreters to run at once.

    MEASURED, 2026-10-19, on 1,011 uncached PROTAC-sized molecules (median 68
    heavy atoms) through one interpreter on an otherwise idle 10-core Apple
    M-series laptop:

        interpreter to ready        32.6 s  (19.5 s import, ~1.8 s model load)
        n = 1                       11.22 s
        n = 10                       0.456 s per molecule, marginal
        n = 1000                     2.045 s per molecule, marginal

    So the fixed cost is about 21 s and the per-molecule cost about 2.0 s, with
    an order of magnitude of variance between molecules. **Inference dominates
    the fixed cost by a hundred to one**, which is why this is a process pool
    and not a larger batch: batching is already done -- one invocation per call
    -- and at 1,000 molecules it amortizes 21 s over 2,040 s of work, or 1%.
    There is nothing left there. Parallelism is the only lever, and k workers
    turn 21 + 2.0N seconds into 21 + 2.0N/k.

    Sized from the cpuset this job was actually given, never from the node. A
    GH200 node reports 256 cores while the allocation owns 72, and 256
    interpreters each loading the model would oversubscribe the cores and ask
    for 256 copies of its memory.
    """
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and allocated.isdigit() and int(allocated) > 0:
        return max(1, min(int(allocated) - 1, SPLITTER_MAX_WORKERS))
    return max(1, min((os.cpu_count() or 2) - 1, SPLITTER_MAX_WORKERS))


# Each worker is a whole interpreter that imports the library and loads the
# model, so the memory cost is per worker rather than shared. Capped so a
# large allocation does not launch a hundred of them for a 2,000-molecule
# subsample that cannot use them.
SPLITTER_MAX_WORKERS = 8


# The statuses `warhead_records` uses, in the order the pipeline reaches them.
# A warhead that is not usable has exactly one of these reasons, and a falling
# `warhead_usable_rate` is diagnosable from the counts without re-running the
# splitter.
WARHEAD_STATUSES = (
    "ok",
    "no_prediction",
    "malformed_prediction",
    "parse_failed",
    "capping_failed",
    "canonicalization_failed",
)

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
    the decomposition failed or did not yield exactly three fragments. Use
    `decompose_with_reasons` when the reason matters.

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
        # Through `_fragments_or_none` like every other return, because the
        # cache now stores a failure as its reason rather than as None, and a
        # caller handed `{"failed": ...}` would read it as a successful split
        # and then fail on the missing fragment keys.
        return [_fragments_or_none(cache.get(s)) for s in smiles_list]
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

    workers = min(splitter_workers(), len(pending))
    chunks = [pending[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    if workers > 1:
        print(f"  across {len(chunks)} interpreters.", flush=True)

    # Popen rather than a ProcessPoolExecutor: the work runs in a *different*
    # interpreter, so there is nothing for a pool to pickle, and Popen
    # exec's immediately rather than forking this process. That matters
    # because the parent may hold a CUDA context -- a `goal_directed` job
    # computes these metrics with the model still on the GPU -- and a forked
    # child inherits a context it cannot use and may deadlock on.
    running = [
        subprocess.Popen(
            [str(SPLITTER_PYTHON), "-c", _SPLIT_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in chunks
    ]
    # One thread per worker, because `communicate` writes the input and then
    # blocks until that process exits. Calling it in a loop started eight
    # interpreters and then ran them one at a time, each waiting on its own
    # stdin: measured at 2.27 s per molecule against 2.05 s serial, which is
    # slower than no pool at all. The threads only wait on pipes; the work is
    # in the subprocesses.
    from concurrent.futures import ThreadPoolExecutor

    def talk(pair: tuple) -> tuple:
        process, chunk = pair
        stdout, stderr = process.communicate(
            json.dumps({"model": model, "smiles": chunk})
        )
        return process.returncode, chunk, stdout, stderr

    with ThreadPoolExecutor(max_workers=len(running)) as pool:
        finished = list(pool.map(talk, zip(running, chunks)))

    results = []
    for returncode, chunk, stdout, stderr in finished:
        if returncode != 0:
            tail = (stderr or stdout).strip().splitlines()[-5:]
            raise MissingInput(
                "PROTAC-Splitter failed to run:\n    " + "\n    ".join(tail)
            )
        results.append((chunk, stdout))

    # A failed split carries its reason rather than a bare None, so a falling
    # split rate can be explained without re-running anything. The reason is
    # kept in the cache too: it is the only record of why a molecule did not
    # decompose, and discarding it meant a sparse component table could not be
    # read without a re-run that costs hours.
    for chunk, stdout in results:
        payload = None
        # The last parseable line is the payload; the library prints progress
        # before it.
        for line in reversed(stdout.strip().splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
        if payload is None:
            raise MissingInput("PROTAC-Splitter produced no parseable output.")
        for smi, entry in zip(chunk, payload):
            cache[smi] = entry
    _save_cache(model, cache)
    return [_fragments_or_none(cache.get(s)) for s in smiles_list]


def _fragments_or_none(entry) -> dict | None:
    """The three fragments, or None for anything that is not a complete split."""
    if not isinstance(entry, dict) or "failed" in entry:
        return None
    return entry if all(entry.get(r) for r in ("poi", "linker", "e3")) else None


def decompose_with_reasons(smiles_list: list, *, model: str = SPLITTER_MODEL) -> list:
    """
    `decompose`, but each failure comes back as its reason rather than as None.

    One entry per input: either the three-fragment dict, or
    `{"failed": "<reason>"}`. The per-molecule component record needs the
    reason, and the aggregate rates need only the verdict, so the two callers
    share one subprocess run and one cache.
    """
    decompose(smiles_list, model=model)
    cache = _load_cache(model)
    out: list = []
    for smi in smiles_list:
        entry = cache.get(smi)
        if entry is None:
            # Written by a cache from before the reasons were kept, or a
            # molecule the splitter was never asked about.
            out.append({"failed": "no_prediction"})
        elif isinstance(entry, dict) and "failed" in entry:
            out.append(entry)
        elif _fragments_or_none(entry) is None:
            out.append({"failed": "malformed_prediction"})
        else:
            out.append(entry)
    return out


def warhead_of(fragment_smiles: str | None) -> tuple:
    """
    The target-binding fragment as the molecule that gets fingerprinted, plus
    the status of getting there.

    Returns `(warhead_smiles_or_None, status)` with the status drawn from
    `WARHEAD_STATUSES`. The stages are kept apart because they fail for
    different reasons and the fix differs: a fragment the splitter never
    produced is a decomposition problem, one that will not parse is a splitter
    output problem, and one that will not survive capping is a chemistry
    problem with the cut it chose.

    Under `WARHEAD_REPRESENTATION = "parent_warhead"` the attachment point is
    capped with hydrogen, giving the ligand as it exists off the degrader.
    Under `"attachment_aware"` the dummy atom is kept, so the same ligand cut
    at two positions counts as two warheads.
    """
    from rdkit import Chem

    if not fragment_smiles:
        return None, "no_prediction"
    if Chem.MolFromSmiles(fragment_smiles) is None:
        return None, "parse_failed"

    if WARHEAD_REPRESENTATION == "attachment_aware":
        capped = fragment_smiles
    else:
        capped = cap_attachment_points(fragment_smiles)
        if capped is None:
            return None, "capping_failed"

    mol = Chem.MolFromSmiles(capped)
    if mol is None:
        return None, "canonicalization_failed"
    canonical = Chem.MolToSmiles(mol)
    if not canonical:
        return None, "canonicalization_failed"
    return canonical, "ok"


def warhead_records(smiles_list: list, *, model: str = SPLITTER_MODEL) -> list:
    """
    One record per input molecule, in order: what was asked of the splitter,
    what came back, and how far the fragment got.

    Each record carries `molecule`, `fragment` (the raw POI fragment with its
    attachment points), `warhead` (the processed one, or None), `status` and
    `split_status`. `results/component_records.json` is this list, and it is
    what makes a rate over a sparse sample explicable: both denominators are
    recoverable from it without re-running the splitter.
    """
    entries = decompose_with_reasons(smiles_list, model=model)
    records: list = []
    for smi, entry in zip(smiles_list, entries):
        fragments = _fragments_or_none(entry)
        if fragments is None:
            records.append(
                {
                    "molecule": smi,
                    "fragment": None,
                    "warhead": None,
                    "e3_fragment": None,
                    "split_status": entry.get("failed", "no_prediction"),
                    "status": "no_prediction",
                }
            )
            continue
        warhead, status = warhead_of(fragments["poi"])
        records.append(
            {
                "molecule": smi,
                "fragment": fragments["poi"],
                "warhead": warhead,
                "e3_fragment": fragments["e3"],
                "split_status": "ok",
                "status": status,
            }
        )
    return records


def e3_cores() -> list:
    """
    The recognized E3-binder cores as (smarts, query) pairs.

    Read from tools/data/e3_binder_cores.smarts, which carries one citation per
    core and the measured ceiling of the list on real degraders.
    """
    from rdkit import Chem

    if not E3_CORES_FILE.exists():
        raise MissingInput(
            f"e3_recognizable_rate needs the E3-binder core list at "
            f"{E3_CORES_FILE}, which is committed to this repository and is "
            "missing. It is an exact substructure test, so there is no "
            "fallback."
        )
    cores: list = []
    for line in E3_CORES_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        query = Chem.MolFromSmarts(line)
        if query is None:
            raise MissingInput(
                f"{E3_CORES_FILE} line {line!r} is not a valid SMARTS. A core "
                "that does not compile would silently never match."
            )
        cores.append((line, query))
    if not cores:
        raise MissingInput(f"{E3_CORES_FILE} declares no cores.")
    return cores


def compute_component_metrics(
    generated_smiles: list,
    n_sampled: int,
    training_warheads: list,
    *,
    isomeric: bool = False,
) -> tuple:
    """
    `split_success_rate`, `warhead_usable_rate`, the `warhead_*` recomputation
    and `e3_recognizable_rate`, with the per-molecule record beside them.

    Returns `(metrics, records)`. The records go to
    `results/component_records.json`.

    TWO DENOMINATORS, NOT ONE. `split_success_rate` is stage one only: the
    fraction of the sample PROTAC-Splitter decomposed into three fragments.
    `warhead_usable_rate` is stage two: of those, the fraction whose POI
    fragment survived capping and canonicalization into a molecule that can be
    fingerprinted. Their product is the end-to-end rate, and the two counts
    `n_split_successfully` and `n_warheads_usable` make it reconstructable.
    One combined rate hid which stage was failing, and the two stages fail for
    unrelated reasons.
    """
    from rdkit import Chem, DataStructs

    records = warhead_records(generated_smiles)
    split_ok = [r for r in records if r["split_status"] == "ok"]
    usable = [r for r in split_ok if r["status"] == "ok"]

    status_counts: dict = {}
    for record in records:
        key = record["status"] if record["split_status"] == "ok" else "no_prediction"
        status_counts[key] = status_counts.get(key, 0) + 1

    results: dict = {
        "split_success_rate": len(split_ok) / n_sampled if n_sampled else None,
        "warhead_usable_rate": (len(usable) / len(split_ok) if split_ok else None),
        "n_split_successfully": len(split_ok),
        "n_warheads_usable": len(usable),
        "warhead_representation": WARHEAD_REPRESENTATION,
        "warhead_fingerprint_radius": ECFP4_RADIUS,
        "warhead_fingerprint_bits": ECFP4_BITS,
        "split_status_counts": status_counts,
        "e3_similarity_floor": e3_similarity_floor_for_report(),
        "ecfp4_radius": ECFP4_RADIUS,
    }
    if not usable:
        results["e3_recognizable_rate"] = None
        results["e3_max_similarities"] = []
        return results, records

    # The core table recomputed over target-binding components only. The
    # whole-molecule core metrics stay whole-molecule and are reported
    # separately; these carry the warhead_ prefix CLAUDE.md tabulates.
    #
    # The denominator is the number of usable warheads, not the number that
    # split: a fragment that will not parse once capped is counted against
    # `warhead_usable_rate` and then excluded, rather than dragging down every
    # warhead_* rate as an invalid molecule and being counted twice.
    #
    # isomeric must match the generated side. Generated warheads are flat when
    # the model runs without chirality, and the reference warheads come from
    # real degraders that carry it, so the default of True would compare a
    # flattened set against a stereo-bearing one and pin warhead_novelty at
    # 1.0 -- the same asymmetry that was fixed at whole-molecule level.
    warheads = [r["warhead"] for r in usable]
    warhead_core = compute_core_metrics(
        warheads,
        n_sampled=len(warheads),
        training_smiles=training_warheads,
        isomeric=isomeric,
    )
    warhead_core.pop("_filter_unavailable", None)
    for key, value in warhead_core.items():
        results[f"warhead_{key}"] = value

    cores = e3_cores()
    known_fps = []
    for smi in reference_e3_ligands():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            known_fps.append(ecfp4(mol))

    # An exact substructure test against the recognized cores. The maximum
    # ECFP4 Tanimoto to a reference ligand comes back as a supporting field,
    # not as the criterion: a Tanimoto threshold answers "how close is this to
    # some known ligand" when the question is whether a recognized binding core
    # is present, and the two disagree in both directions -- a glutarimide
    # carrying a large unfamiliar substituent scores low while being a perfectly
    # good CRBN binder.
    recognized = 0
    similarities: list = []
    n_tested = 0
    for record in usable:
        capped, _ = warhead_of(record["e3_fragment"])
        mol = Chem.MolFromSmiles(capped) if capped else None
        if mol is None:
            continue
        n_tested += 1
        if any(mol.HasSubstructMatch(query) for _, query in cores):
            recognized += 1
        if known_fps:
            sims = DataStructs.BulkTanimotoSimilarity(ecfp4(mol), known_fps)
            similarities.append(max(sims) if sims else 0.0)

    results["e3_recognizable_rate"] = recognized / n_tested if n_tested else None
    results["e3_n_tested"] = n_tested
    results["e3_n_cores"] = len(cores)
    results["e3_max_similarities"] = [round(v, 4) for v in similarities]
    return results, records


def reference_e3_ligands() -> list:
    """
    The reference E3 ligands `e3_max_similarities` is measured against.

    One per family in tools/data/e3_binder_cores.smarts, so the supporting
    similarity is comparable with the substructure verdict rather than against
    a different set of molecules.
    """
    return [
        # thalidomide (CRBN)
        "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
        # lenalidomide (CRBN)
        "NC1=CC=CC2=C1CN(C1CCC(=O)NC1=O)C2=O",
        # VH032 (VHL)
        "CC(C)(C)[C@H](NC(=O)C)C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc1ccc(cc1)-c1scnc1C",
        # nutlin-3a (MDM2)
        "COc1ccc(cc1OC(C)C)C1=N[C@@H](c2ccc(Cl)cc2)[C@H](c2ccc(Cl)cc2)N1C(=O)N1CCNCC1",
        # LCL161 (IAP)
        "CNC(C)C(=O)N1CCCC1C(=O)Nc1cccc(c1)-c1ccccc1",
    ]


def e3_similarity_floor_for_report() -> float:
    """
    The similarity floor, kept only so `e3_max_similarities` can be read.

    It is no longer the criterion for `e3_recognizable_rate`, which is an exact
    substructure test, and nothing branches on it.
    """
    return E3_SIMILARITY_FLOOR


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
        capped, _ = warhead_of(fragments["poi"])
        scaffold = bemis_murcko(capped) if capped else None
        # Falsy, not just None: a warhead with no ring system has an empty
        # scaffold, and counting that as one inflates the tally.
        if scaffold:
            scaffolds.add(scaffold)
    results["top100_warhead_scaffold_count"] = len(scaffolds)
    return results


def compute_class_warhead_metrics(
    records: list,
    training_warheads_in_class: list,
    *,
    n_unique_valid: int,
    isomeric: bool = False,
) -> tuple:
    """
    The per-class warhead row for one requested class, plus its nulls.

    Returns `(row, unavailable)`. The row's keys are unsuffixed; the caller
    adds the `_<class>` suffix that keeps metrics.json flat.

    `warhead_novelty_in_class` is the one that carries the conditional claim.
    Global `warhead_novelty`, which is what `compute_component_metrics`
    reports, asks whether the warhead appears anywhere in the training corpus;
    a model that has learned the conditioning will score low on it, because the
    right answer for a requested class is a warhead resembling that class's
    known ligands. In-class novelty asks the question that distinguishes
    conditioning from retrieval: of the warheads generated for this class, how
    many are absent from *this class's* training warheads. Both are reported,
    and they are expected to disagree.
    """
    split_ok = [r for r in records if r["split_status"] == "ok"]
    usable = [r for r in split_ok if r["status"] == "ok"]
    warheads = [r["warhead"] for r in usable]
    unique = sorted(set(warheads))

    row: dict = {
        "n_unique_valid": n_unique_valid,
        "split_success_rate": (len(split_ok) / len(records) if records else None),
        "warhead_usable_rate": (len(usable) / len(split_ok) if split_ok else None),
        "n_split_successfully": len(split_ok),
        "n_warheads_usable": len(usable),
        "n_unique_warheads": len(unique),
        "warhead_uniqueness": (len(unique) / len(warheads) if warheads else None),
        "warhead_diversity": None,
        "warhead_novelty_in_class": None,
    }
    unavailable: dict = {}

    if not records:
        unavailable["split_success_rate"] = (
            "insufficient_sample: no molecule was generated for this class"
        )
    if not split_ok:
        unavailable["warhead_usable_rate"] = (
            "insufficient_sample: no molecule of this class decomposed, so "
            "there is no fragment to process"
        )

    # Mean pairwise Tanimoto distance over the class's unique warheads. One
    # warhead has no pair, so this is a categorized null rather than 0.0, which
    # would read as total collapse and is the number a model that generated a
    # single excellent warhead would be reported as having earned.
    if len(unique) < 2:
        unavailable["warhead_diversity"] = (
            f"insufficient_sample: {len(unique)} unique warhead(s) for this "
            "class, and a mean pairwise distance needs two"
        )
    else:
        from ._internal_diversity import compute_internal_diversity

        row["warhead_diversity"] = compute_internal_diversity(unique)[
            "internal_diversity"
        ]

    if not unique:
        unavailable["warhead_novelty_in_class"] = (
            "insufficient_sample: no usable warhead for this class"
        )
    elif not training_warheads_in_class:
        unavailable["warhead_novelty_in_class"] = (
            "insufficient_sample: the training split carries no warhead for "
            "this class, so there is nothing for a generated warhead to be "
            "novel against"
        )
    else:
        from ._core import canonical_smiles

        reference = {
            s for s in canonical_smiles(training_warheads_in_class, isomeric=isomeric)
        }
        row["warhead_novelty_in_class"] = sum(
            1 for w in unique if w not in reference
        ) / len(unique)
    return row, unavailable
