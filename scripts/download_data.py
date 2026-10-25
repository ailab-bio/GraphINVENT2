"""
Download the raw data every experiment is built from, into `data/raw/`.

Idempotent and cached: a file already present with the recorded checksum is not
fetched again. Every download records its URL, the date it was retrieved and its
sha256 in a MANIFEST.json beside it, because TPD-DB is a live resource that is
updated in place and our numbers have to be traceable to a dated snapshot.

    python scripts/download_data.py                 # everything
    python scripts/download_data.py --source tpddb  # one source

A source that has moved or disappeared stops the script rather than being
substituted. Nothing here guesses at a replacement.
"""

import argparse
import datetime
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# TPD-DB bulk downloads. Plain tab-separated text, no documented schema; the
# columns we rely on are recorded in manuscript/experiment-plan.md.
TPDDB_BASE = "https://tpddb.idrblab.net/sites/files/tpd_download/"
TPDDB_FILES = [
    "PROTAC_main_table.txt",
    "MG_main_table.txt",
    "PROTAC_activity.txt",
    "MG_activity.txt",
    "TPD_Related_Diseases.txt",
    "TPD_PDB.txt",
]

# ZINC250k, for the development subset and for the PROTAC-like prior corpus.
ZINC250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)

# The synthetic PROTAC library, the base corpus for the PROTAC prior: 1,298,401
# recombinant degraders released with PROTAC-Splitter. Open access, MIT, no
# authentication. It carries POI ligand, linker and E3 binder substructures but
# NO target-protein annotation, so it is a distribution-matching corpus only and
# nothing conditional can be trained on it.
#
# This is a download and not an enumeration. scripts/protac-enumerate-library.py
# used to build a local substitute by recombining the 492 decomposed TPD-DB
# degraders, and it was removed: upstream's own generator accumulates results
# with a pd.concat().drop_duplicates() inside its result loop, which is
# quadratic in the batch count, and its ThreadPoolExecutor wraps a Python-level
# RDKit loop so it is GIL-bound and --max-workers has no effect. The measured
# rate was under 1.85 molecules/s, which is over eight days for this many
# molecules, and the parts it recombined all came from the task corpus anyway.
PROTAC_SPLITTER_ZENODO = (
    "https://zenodo.org/api/records/15797310/files/Datasets.zip/content"
)

SOURCES = {
    "tpddb": ("data/raw/tpddb", [(TPDDB_BASE + name, name) for name in TPDDB_FILES]),
    "zinc": ("data/raw/zinc", [(ZINC250K_URL, "250k_rndm_zinc_drugs_clean_3.csv")]),
    "protac_splitter": (
        "data/raw/protac_splitter",
        [(PROTAC_SPLITTER_ZENODO, "Datasets.zip")],
    ),
}

# The battery assets come from a git repository rather than from URLs, because
# one of them is a Git LFS object and another was deleted from the tip and has
# to be read out of history. A plain download cannot do either.
BATTSLM_URL = "https://github.com/Teoroo-CMC/Batt-SLM.git"
# The commit that added the datasets. KBS-409.csv and KBS-FP-174.smi were
# removed from the tip in 00ed98f ("change folder name and delete KBS"), so the
# known-solvent reference set exists only here. It is the reference set for one
# of the six terms of the battery objective, which cannot be computed without
# it.
BATTSLM_DATA_COMMIT = "f2fe0b0"
# Files read from BATTSLM_DATA_COMMIT rather than from the tip.
BATTSLM_FROM_HISTORY = [
    ("Batt-SLM/KBS-409.csv", "KBS-409.csv"),
    ("Batt-SLM/KBS-FP-174.smi", "KBS-FP-174.smi"),
]
# Files taken from the tip, where they are current.
BATTSLM_FROM_TIP = [
    ("Batt-SLM/Batt-SLM.smi", "Batt-SLM.smi"),
    ("Batt-SLM/filter_smiles.py", "filter_smiles.py"),
    ("CPI/Input/Features-NoF.csv", "Features-NoF.csv"),
    ("CPI/SolvFunc-87.csv", "SolvFunc-87.csv"),
    ("CPI/chelation_propensity_index.py", "chelation_propensity_index.py"),
    ("Redox-Pot/redox_potential.py", "redox_potential.py"),
    # Four artifacts the battery objective needs that this script used to leave
    # behind, so the code to use them was present and could not run:
    #   - the fitted CPI coefficients, without which predict_cpi() has an
    #     implementation and no model. Also committed at
    #     tools/data/cpi_logistic_coefficients.txt, because it is 405 bytes and
    #     the funnel cannot be computed without it.
    #   - RX-392 and its two input tables, the only one of the paper's five
    #     property datasets that was actually released. They are what the S_r
    #     redox term of the objective is fitted on.
    ("CPI/LogR-CPI/result.txt", "CPI-LogR-result.txt"),
    ("Redox-Pot/Redox-Ener/RX-392.csv", "RX-392.csv"),
    ("Redox-Pot/Redox-Ener/Input/EA-Red.csv", "EA-Red.csv"),
    ("Redox-Pot/Redox-Ener/Input/IE-Ox.csv", "IE-Ox.csv"),
    ("Redox-Pot/Redox-Ener/redox_free_ener.py", "redox_free_ener.py"),
    # The PiNet2 loader, needed to read the dipole checkpoints that the
    # dielectric-constant chain starts from.
    ("Batt-P30K/build_pinet2.py", "build_pinet2.py"),
    ("Batt-P30K/io/hdf5_gsds.py", "hdf5_gsds.py"),
]
# A Git LFS object, 183,771,012 bytes.
BATTSLM_LFS = [("Batt-P30K/Batt-P30K.h5", "Batt-P30K.h5")]


def _git_url(path: str, ref: str) -> str:
    """A pseudo-URL the fetcher resolves with git rather than with HTTP."""
    return f"git+{BATTSLM_URL}@{ref}#{path}"


SOURCES["battslm"] = (
    "data/raw/battslm",
    [
        _git_url(src, BATTSLM_DATA_COMMIT)
        and (_git_url(src, BATTSLM_DATA_COMMIT), name)
        for src, name in BATTSLM_FROM_HISTORY
    ]
    + [(_git_url(src, "HEAD"), name) for src, name in BATTSLM_FROM_TIP]
    + [(_git_url(src, "HEAD"), name) for src, name in BATTSLM_LFS],
)


_GIT_CLONES: dict = {}


def _git_clone(repo: str) -> Path:
    """One clone per repository per run, reused across its files."""
    import subprocess
    import tempfile

    if repo not in _GIT_CLONES:
        target = Path(tempfile.mkdtemp(prefix="graphinvent-src-")) / "repo"
        print(f"   cloning {repo}", flush=True)
        subprocess.run(["git", "clone", "--quiet", repo, str(target)], check=True)
        _GIT_CLONES[repo] = target
    return _GIT_CLONES[repo]


def fetch_from_git(url: str, dest: Path) -> bytes:
    """
    One file out of a git repository, at a named ref.

    Two of the battery assets cannot be fetched over HTTP: `Batt-P30K.h5` is a
    Git LFS object, and `KBS-409.csv` was deleted from the tip in
    00ed98f ("change folder name and delete KBS"), so the known-solvent
    reference set exists only in history. It is the reference set for one of the
    six terms of the battery objective, so without reading history that
    objective cannot be computed at all.
    """
    import shutil
    import subprocess

    repo_and_ref, path = url[len("git+") :].split("#", 1)
    repo, ref = repo_and_ref.rsplit("@", 1)
    if not shutil.which("git"):
        raise SystemExit("git is needed for this source and is not on PATH.")
    clone = _git_clone(repo)

    is_lfs = path.endswith(".h5")
    if is_lfs:
        if not shutil.which("git-lfs"):
            raise SystemExit(
                f"{path} is a Git LFS object and git-lfs is not installed.\n"
                "Install it (brew install git-lfs, or the cluster's module). "
                "Everything except the DFT-labelled subset works without it, "
                "so --source battslm can be deferred rather than skipped."
            )
        subprocess.run(
            ["git", "-C", str(clone), "lfs", "install", "--local"],
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["git", "-C", str(clone), "lfs", "pull", "--include", path], check=True
        )
        payload = (clone / path).read_bytes()
    else:
        blob = subprocess.run(
            ["git", "-C", str(clone), "show", f"{ref}:{path}"],
            capture_output=True,
            check=True,
        )
        payload = blob.stdout
    dest.write_bytes(payload)
    return payload


def fetch(url: str, dest: Path) -> bytes:
    """Download `url` to `dest`, or read it back if it is already there."""
    if dest.exists():
        return dest.read_bytes()
    if url.startswith("git+"):
        return fetch_from_git(url, dest)
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as error:
        raise SystemExit(
            f"\nFAILED to download {url}\n  {type(error).__name__}: {error}\n\n"
            "The source has moved or is unreachable. Nothing was substituted.\n"
            "Check the URL by hand before going further: every downstream number "
            "depends on which snapshot this is."
        ) from None
    dest.write_bytes(payload)
    return payload


def download(name: str) -> None:
    directory, items = SOURCES[name]
    out = REPO_ROOT / directory
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "MANIFEST.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"files": {}}
    )
    # A manifest written by an earlier run may predate this shape.
    manifest.setdefault("files", {})
    manifest["source"] = SOURCES[name][0]
    manifest.setdefault(
        "retrieved_utc",
        datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    )

    print(f"== {name} -> {directory}")
    for url, filename in items:
        dest = out / filename
        existed = dest.exists()
        payload = fetch(url, dest)
        digest = hashlib.sha256(payload).hexdigest()

        recorded = manifest["files"].get(filename, {}).get("sha256")
        if recorded and recorded != digest:
            raise SystemExit(
                f"\n{filename} on disk does not match the recorded checksum.\n"
                f"  recorded {recorded}\n  on disk  {digest}\n"
                "Either the file was edited or the snapshot changed. Resolve that "
                "before trusting any number derived from it."
            )
        manifest["files"][filename] = {
            "url": url,
            "bytes": len(payload),
            "sha256": digest,
            "retrieved_utc": manifest["files"]
            .get(filename, {})
            .get(
                "retrieved_utc",
                datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            ),
        }
        print(
            f"   {'cached ' if existed else 'fetched'} {filename:28s} "
            f"{len(payload):>10,} B  {digest[:16]}"
        )

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"   manifest: {manifest_path.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        action="append",
        help="download only this source; repeatable. Default: all of them.",
    )
    args = parser.parse_args()
    for name in args.source or sorted(SOURCES):
        download(name)
    print("\nDone. Raw files are not committed; the manifests are.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
