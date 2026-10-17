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

SOURCES = {
    "tpddb": ("data/raw/tpddb", [(TPDDB_BASE + name, name) for name in TPDDB_FILES]),
    "zinc": ("data/raw/zinc", [(ZINC250K_URL, "250k_rndm_zinc_drugs_clean_3.csv")]),
}


def fetch(url: str, dest: Path) -> bytes:
    """Download `url` to `dest`, or read it back if it is already there."""
    if dest.exists():
        return dest.read_bytes()
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
