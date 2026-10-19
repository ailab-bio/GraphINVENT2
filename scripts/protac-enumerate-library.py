#!/usr/bin/env python3
"""
Enumerate a recombinant PROTAC library from decomposed TPD-DB degraders.

    python scripts/protac-enumerate-library.py --n 50000 \
        --output data/datasets/protac_enumerated

The enumerated library is the base corpus for the PROTAC transfer-learning arm.
It was recorded as blocked on a download; it is not. PROTAC-Splitter ships the
enumeration itself, so the library is built here from our own decomposed
components, which is better provenance than a download: every warhead, linker
and E3 ligand in it comes from a molecule in a dated TPD-DB snapshot whose
checksum is in `data/raw/tpddb/MANIFEST.json`.

How it works. Each real degrader is decomposed into a target-binding component,
a linker and an E3 ligand, each carrying the `[*:n]` dummy atoms that mark where
it was cut, which is what PROTAC-Splitter calls a SMILES "with direction". Their
generator reads the distribution of functional groups at those attachment points
and samples new combinations that respect it, so a generated molecule is a
recombination of real parts joined through chemistry that occurs in real
degraders rather than an arbitrary graft.

Runs in PROTAC-Splitter's own interpreter, which `tools/install.sh` builds,
because its xgboost pin and ours are mutually exclusive. The decomposition is
cached under `tools/splitter-cache/`, so re-running costs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

SPLITTER = REPO / "tools" / "PROTAC-Splitter"
SPLITTER_PYTHON = SPLITTER / ".venv" / "bin" / "python"

# Runs inside PROTAC-Splitter's interpreter. Reads the curated CSV we wrote and
# the sample count, writes the enumerated SMILES one per line.
_GENERATE = """
import json, sys
import pandas as pd
from protac_splitter.data.generation.functional_groups import (
    get_functional_groups_distributions,
)
from protac_splitter.data.generation.generation import generate_protacs

spec = json.load(sys.stdin)
df = pd.read_csv(spec["curated_csv"])
info = get_functional_groups_distributions(df, load_from_file=False, verbose=0)
generated = generate_protacs(
    poi_fg_distr=info["poi_fg_distr"],
    e3_fg_distr=info["e3_fg_distr"],
    substr_fg_2_linker=info["substr_fg_2_linker"],
    poi_fg_2_substr=info["poi_fg_2_substr"],
    e3_fg_2_substr=info["e3_fg_2_substr"],
    num_samples=spec["n"],
    random_state=spec["seed"],
    batch_size=spec["batch_size"],
    max_workers=spec["max_workers"],
    original_df=df,
)
col = "PROTAC SMILES" if "PROTAC SMILES" in generated.columns else generated.columns[0]
out = [s for s in generated[col].astype(str).tolist() if s and s != "nan"]
print("<<<GENERATED>>>")
print(json.dumps(out))
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=Path("data/datasets/tpddb_protac/tpddb_protac.smi"),
        help="real degraders to decompose into parts",
    )
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--max-sources",
        type=int,
        default=0,
        help="decompose only the first N real degraders; 0 means all. "
        "Decomposition is about 2 s per uncached molecule.",
    )
    args = ap.parse_args()

    if not SPLITTER_PYTHON.exists():
        print(
            f"PROTAC-Splitter is not installed at {SPLITTER_PYTHON}.\n"
            "Run tools/install.sh.",
            file=sys.stderr,
        )
        return 1
    if not args.source.exists():
        print(
            f"{args.source} does not exist. Build it with\n"
            "  python scripts/tpddb-create-dataset.py --modality protac "
            "--output data/datasets/tpddb_protac",
            file=sys.stderr,
        )
        return 1

    from metrics._components import decompose

    smiles = [
        ln.split("\t")[0].split()[0]
        for i, ln in enumerate(args.source.read_text().splitlines())
        if ln.strip() and not (i == 0 and ln.split()[0].upper() == "SMILES")
    ]
    if args.max_sources:
        smiles = smiles[: args.max_sources]
    print(f"* {len(smiles)} real degraders from {args.source}", flush=True)

    fragments = decompose(smiles)
    rows = [
        (smi, f["poi"], f["linker"], f["e3"]) for smi, f in zip(smiles, fragments) if f
    ]
    print(
        f"  {len(rows)} decomposed into three parts "
        f"({len(smiles) - len(rows)} could not be)",
        flush=True,
    )
    if len(rows) < 10:
        print(
            "Too few decomposed molecules to estimate an attachment-point "
            "distribution from.",
            file=sys.stderr,
        )
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    curated = args.output / "curated_parts.csv"
    import csv as _csv

    with curated.open("w", newline="") as handle:
        writer = _csv.writer(handle)
        writer.writerow(
            [
                "PROTAC SMILES",
                "POI Ligand SMILES with direction",
                "Linker SMILES with direction",
                "E3 Binder SMILES with direction",
            ]
        )
        writer.writerows(rows)
    print(f"  parts -> {curated}", flush=True)

    env = dict(os.environ)
    env["HF_HOME"] = str(REPO / "tools" / "hf-cache")
    env["PYTHONPATH"] = str(SPLITTER)
    spec = {
        # Absolute: the subprocess runs in PROTAC-Splitter's directory so
        # its package imports resolve, and a relative path would be read
        # from there instead of from the repository root.
        "curated_csv": str(curated.resolve()),
        "n": args.n,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_workers": args.max_workers,
    }
    print(
        f"* Enumerating {args.n} molecules in PROTAC-Splitter's interpreter", flush=True
    )
    probe = subprocess.run(
        [str(SPLITTER_PYTHON), "-c", _GENERATE],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        cwd=str(SPLITTER),
        env=env,
    )
    if probe.returncode != 0 or "<<<GENERATED>>>" not in probe.stdout:
        tail = (probe.stderr or probe.stdout).strip().splitlines()[-8:]
        print("Enumeration failed:\n    " + "\n    ".join(tail), file=sys.stderr)
        return 1

    payload = probe.stdout.split("<<<GENERATED>>>", 1)[1].strip().splitlines()[0]
    generated = json.loads(payload)

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    real = {
        Chem.MolToSmiles(m)
        for m in (Chem.MolFromSmiles(s) for s, *_ in rows)
        if m is not None
    }
    kept, seen = [], set()
    n_unparseable = n_echoed = 0
    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_unparseable += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        # A base corpus that contains the task corpus would make the transfer
        # arm's novelty meaningless, so anything identical to a real degrader
        # is dropped here rather than discovered later.
        if canonical in real:
            n_echoed += 1
            continue
        if canonical not in seen:
            seen.add(canonical)
            kept.append(canonical)

    name = args.output.name
    out = args.output / f"{name}.smi"
    out.write_text("\n".join(kept) + "\n")
    sizes = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in kept[:2000]]
    (args.output / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "built_by": "scripts/protac-enumerate-library.py",
                "method": "PROTAC-Splitter functional-group recombination of components "
                "decomposed from real TPD-DB degraders",
                "source_file": str(args.source),
                "n_source_degraders": len(smiles),
                "n_source_decomposed": len(rows),
                "requested": args.n,
                "returned_by_generator": len(generated),
                "dropped_unparseable": n_unparseable,
                "dropped_identical_to_a_real_degrader": n_echoed,
                "written": len(kept),
                "heavy_atoms_min": min(sizes) if sizes else None,
                "heavy_atoms_max": max(sizes) if sizes else None,
                "seed": args.seed,
                "splitter_commit_pin": "see tools/README.md",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"* Wrote {len(kept)} molecules to {out}", flush=True)
    print(
        f"  dropped {n_unparseable} unparseable, {n_echoed} identical to a "
        f"real degrader",
        flush=True,
    )
    print(f"  provenance -> {args.output / 'PROVENANCE.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
