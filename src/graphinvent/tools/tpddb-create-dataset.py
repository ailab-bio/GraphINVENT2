"""
Builds a GraphINVENT dataset of targeted protein degraders from TPDdb.

TPDdb (https://tpddb.idrblab.net) catalogues targeted protein degradation
agents: PROTACs, molecular glues, and lysosome-directed modalities.  It serves
its release as static tab-separated files, so a download is reproducible
without a browser, a login, or an API key.  See Qin et al., Nucleic Acids
Research 54(D1):D1683-D1691 (2026), doi:10.1093/nar/gkaf996.

Reproducibility is the reason this exists as a script rather than a one-off
download.  Each run records the source URL, the retrieval timestamp, and a
SHA-256 of every raw file in a ``PROVENANCE.json`` beside the dataset, so a
later run can be checked against an earlier one rather than assumed equal to
it.  Molecule selection is deterministic: candidates are sorted by heavy-atom
count and then by canonical SMILES, so ties break the same way on every run and
the same N molecules come out.

Selecting the *smallest* degraders is deliberate.  A PROTAC is a bivalent
molecule -- a target ligand and an E3-ligase ligand joined by a linker -- and
routinely has 60 to 120 heavy atoms, whereas the molecular glues in the same
database are conventional small molecules.  Since GraphINVENT allocates its
action-probability tensor as
``max_n_nodes x |atom types| x |charges| x |implicit H| x |bond types|``, and
the readout MLPs are sized from that, cost grows steeply with the largest
molecule in the set.  A small subset keeps preprocessing and training tractable
while still exercising real degrader chemistry.

Molecules are filtered to what the graph representation can encode: RDKit must
sanitize them, and they must be single-fragment, because the BFS/DFS traversal
that builds the decoding route cannot order a disconnected graph and will
reject salts and mixtures.

Usage:
    python src/graphinvent/tools/tpddb-create-dataset.py \
        --modality both --n-molecules 10 --output data/datasets/tpddb_small/
"""

import argparse
import datetime
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors

BASE_URL = "https://tpddb.idrblab.net/sites/files/tpd_download"

# TPDdb has no degrader-type column; the modality is implied by which file a
# record came from, so it is attached here at ingest.
SOURCE_FILES = {
    "protac": "PROTAC_main_table.txt",
    "glue": "MG_main_table.txt",
}

CITATION = (
    "Qin X, Zhang Y, Wang Y, et al. TPDdb: the comprehensive database of "
    "targeted protein degrader. Nucleic Acids Research 54(D1):D1683-D1691 "
    "(2026). doi:10.1093/nar/gkaf996"
)


def download(url: str, dest: Path) -> Tuple[bytes, str]:
    """
    Fetches *url* into *dest*, returning its bytes and their SHA-256.

    An existing file is reused rather than re-fetched, which keeps repeated runs
    cheap and offline-capable; delete the cache directory to force a refresh.
    """
    if dest.exists():
        raw = dest.read_bytes()
        print(f"  cached: {dest} ({len(raw)} bytes)", flush=True)
    else:
        print(f"  fetching {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f"{url} returned HTTP {response.status}")
            raw = response.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        print(f"  saved {len(raw)} bytes to {dest}", flush=True)
    return raw, hashlib.sha256(raw).hexdigest()


def parse_table(raw: bytes, modality: str) -> List[Dict[str, str]]:
    """
    Parses a TPDdb main table into records with SMILES, identifier, and modality.

    Column 4 is ``SMILES`` in every main table.  Molecular glues additionally
    carry a ``Subtype`` column separating degraders from stabilizers; only
    degraders are kept, since a stabilizer does not induce degradation.  A
    handful of rows have UniProt accessions leaked into that column, so the
    filter tests for the expected value rather than excluding the other one.
    """
    text = raw.decode("utf-8", errors="replace")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    header = lines[0].split("\t")

    try:
        smiles_col = header.index("SMILES")
    except ValueError as exc:
        raise RuntimeError(
            f"No 'SMILES' column in TPDdb table; header was {header}"
        ) from exc
    id_col = header.index("TPD ID") if "TPD ID" in header else 0
    subtype_col = header.index("Subtype") if "Subtype" in header else None

    records = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= smiles_col:
            continue
        smiles = parts[smiles_col].strip()
        if not smiles:
            continue
        if subtype_col is not None and len(parts) > subtype_col:
            if parts[subtype_col].strip() != "Degrader":
                continue
        records.append(
            {
                "smiles": smiles,
                "tpd_id": parts[id_col].strip() if len(parts) > id_col else "",
                "modality": modality,
            }
        )
    return records


def curate(
    records: List[Dict[str, str]], max_heavy_atoms: Optional[int]
) -> List[Dict[str, object]]:
    """
    Keeps records the graph representation can encode, annotating each with
    heavy-atom count, molecular weight, and Crippen logP.

    Rejects anything RDKit cannot sanitize and anything with more than one
    fragment: the decoding route is produced by a graph traversal that cannot
    order disconnected components, so salts and mixtures are unusable rather
    than merely awkward.  Duplicates are collapsed on canonical SMILES.
    """
    kept: Dict[str, Dict[str, object]] = {}
    n_unparseable = 0
    n_multifragment = 0
    n_too_large = 0

    for record in records:
        mol = Chem.MolFromSmiles(record["smiles"])
        if mol is None:
            n_unparseable += 1
            continue
        if len(Chem.GetMolFrags(mol)) > 1:
            n_multifragment += 1
            continue
        n_heavy = mol.GetNumHeavyAtoms()
        if max_heavy_atoms is not None and n_heavy > max_heavy_atoms:
            n_too_large += 1
            continue

        canonical = Chem.MolToSmiles(mol)
        if canonical in kept:
            continue
        kept[canonical] = {
            "smiles": canonical,
            "tpd_id": record["tpd_id"],
            "modality": record["modality"],
            "n_heavy_atoms": n_heavy,
            "mol_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Crippen.MolLogP(mol), 3),
        }

    print(
        f"  {len(kept)} usable, {n_unparseable} unparseable, "
        f"{n_multifragment} multi-fragment, {n_too_large} over the size cap",
        flush=True,
    )
    return list(kept.values())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--modality",
        choices=["protac", "glue", "both"],
        default="both",
        help="Which TPDdb tables to draw from.",
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=10,
        help="How many molecules to keep (the smallest by heavy-atom count).",
    )
    parser.add_argument(
        "--max-heavy-atoms",
        type=int,
        default=None,
        help="Discard molecules above this heavy-atom count before selecting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/tpddb_small/"),
        help="Directory to write the dataset into.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/tpddb/"),
        help="Where the raw TPDdb tables are cached.",
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    modalities = ["protac", "glue"] if args.modality == "both" else [args.modality]

    print(f"* Downloading TPDdb tables for: {', '.join(modalities)}", flush=True)
    all_records: List[Dict[str, str]] = []
    sources = []
    for modality in modalities:
        filename = SOURCE_FILES[modality]
        url = f"{BASE_URL}/{filename}"
        raw, digest = download(url, args.cache_dir / filename)
        records = parse_table(raw, modality)
        print(f"  {filename}: {len(records)} records with SMILES", flush=True)
        sources.append(
            {
                "modality": modality,
                "url": url,
                "filename": filename,
                "bytes": len(raw),
                "sha256": digest,
                "records_with_smiles": len(records),
            }
        )
        all_records.extend(records)

    print("* Curating molecules", flush=True)
    curated = curate(all_records, args.max_heavy_atoms)
    if not curated:
        print("No usable molecules found.", file=sys.stderr)
        return 1

    # Sorting by canonical SMILES after heavy-atom count makes ties break
    # identically on every run, so the selection is reproducible.
    curated.sort(key=lambda r: (r["n_heavy_atoms"], r["smiles"]))
    selected = curated[: args.n_molecules]

    args.output.mkdir(parents=True, exist_ok=True)
    name = args.output.name or "tpddb"
    smi_path = args.output / f"{name}.smi"
    with open(smi_path, "w") as f:
        for record in selected:
            f.write(f"{record['smiles']}\n")

    # A parallel TSV carries the identifiers and properties; the .smi stays
    # plain because that is what the preprocessing job reads.
    tsv_path = args.output / f"{name}_metadata.tsv"
    with open(tsv_path, "w") as f:
        f.write("SMILES\tTPD_ID\tmodality\tn_heavy_atoms\tmol_weight\tlogp\n")
        for r in selected:
            f.write(
                f"{r['smiles']}\t{r['tpd_id']}\t{r['modality']}\t"
                f"{r['n_heavy_atoms']}\t{r['mol_weight']}\t{r['logp']}\n"
            )

    sizes = [r["n_heavy_atoms"] for r in selected]
    logps = [r["logp"] for r in selected]
    provenance = {
        "database": "TPDdb",
        "homepage": "https://tpddb.idrblab.net",
        "citation": CITATION,
        "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": sources,
        "selection": {
            "rule": "sorted by (heavy atom count, canonical SMILES); first N kept",
            "n_molecules": len(selected),
            "modality": args.modality,
            "max_heavy_atoms": args.max_heavy_atoms,
            "n_candidates_after_curation": len(curated),
        },
        "summary": {
            "heavy_atoms_min": min(sizes),
            "heavy_atoms_max": max(sizes),
            "logp_min": min(logps),
            "logp_max": max(logps),
            "modalities": sorted({r["modality"] for r in selected}),
        },
        "regenerate_with": (
            "python src/graphinvent/tools/tpddb-create-dataset.py "
            f"--modality {args.modality} --n-molecules {args.n_molecules} "
            f"--output {args.output}/"
        ),
    }
    with open(args.output / "PROVENANCE.json", "w") as f:
        json.dump(provenance, f, indent=2)
        # json.dump writes no trailing newline, which trips the
        # end-of-file-fixer pre-commit hook every time one of these
        # generated files is regenerated and committed.
        f.write("\n")

    print(f"* Wrote {len(selected)} molecules to {smi_path}", flush=True)
    print(
        f"  heavy atoms {min(sizes)}-{max(sizes)}, " f"logP {min(logps)}-{max(logps)}",
        flush=True,
    )
    print(f"  provenance: {args.output / 'PROVENANCE.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
