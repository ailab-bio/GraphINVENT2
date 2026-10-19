#!/bin/bash
# Install every external tool this project's metrics and oracles need, at the
# version pinned in README.md. Idempotent: it reports what it installed and
# what was already present, and re-running it changes nothing.
#
#   bash tools/install.sh
#
# Run from anywhere; paths are resolved relative to this script.
set -euo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$TOOLS")"

# Pins. Change one here and in README.md together.
PROTAC_SPLITTER_URL="https://github.com/ribesstefano/PROTAC-Splitter.git"
PROTAC_SPLITTER_SHA="f1920122e1aa4f1d5f3643b8e92e72e99826e43a"
RETROTAC_URL="https://github.com/ribesstefano/RetroTAC.git"
RETROTAC_SHA="5ba851656753542d4a7f23438c033af280db9935"
# sha256 of release/gnn_20260828_182305_final.ckpt.gz, from its MANIFEST.txt
RETROTAC_CKPT_SHA256="83ccd2404cfffec230f7457926a95421731e56cc37268080e491b18d4da038af"
MCF_URL="https://raw.githubusercontent.com/molecularsets/moses/master/moses/metrics/mcf.csv"
MCF_SHA256="f4a2bfacb4cb8877"   # first 16 hex of sha256; see README.md

installed=()
present=()

echo "== GraphINVENT2 external tools =="
echo "   target: $TOOLS"
echo

# ---------------------------------------------------------------------------
# PROTAC-Splitter: component-level metrics (Group 5).
#
# Built into its OWN virtual environment, not the project's. Its XGBoost
# edge-classifier was pickled under xgboost<3 and fails to load under the
# xgboost>=3.2 this project needs for aarch64 GPU support, with
# "'XGBModel' object has no attribute 'feature_weights'". The two pins cannot
# coexist, so src/metrics/_components.py calls into this interpreter.
# ---------------------------------------------------------------------------
if [ -x "$TOOLS/PROTAC-Splitter/.venv/bin/python" ]; then
    present+=("PROTAC-Splitter (venv exists)")
else
    echo "-- installing PROTAC-Splitter at ${PROTAC_SPLITTER_SHA:0:8}"
    if [ ! -d "$TOOLS/PROTAC-Splitter/.git" ]; then
        git clone "$PROTAC_SPLITTER_URL" "$TOOLS/PROTAC-Splitter"
    fi
    git -C "$TOOLS/PROTAC-Splitter" fetch --quiet origin
    git -C "$TOOLS/PROTAC-Splitter" checkout --quiet "$PROTAC_SPLITTER_SHA"
    python3 -m venv "$TOOLS/PROTAC-Splitter/.venv"
    "$TOOLS/PROTAC-Splitter/.venv/bin/python" -m pip install --quiet --upgrade pip
    "$TOOLS/PROTAC-Splitter/.venv/bin/python" -m pip install --quiet \
        "$TOOLS/PROTAC-Splitter"
    installed+=("PROTAC-Splitter + its own venv")
fi

# Its XGBoost weights (~17 MB) download from the HuggingFace Hub on first use.
# Pinned here rather than left to the library's default cache: a cluster home
# directory has a small quota and a surprise download into one fails at
# midnight. src/metrics/_components.py sets HF_HOME to this path.
mkdir -p "$TOOLS/hf-cache"
if [ -n "$(find "$TOOLS/hf-cache" -name '*.json' -o -name '*.ubj' 2>/dev/null | head -1)" ]; then
    present+=("PROTAC-Splitter model weights (hf-cache populated)")
else
    echo "-- warming the PROTAC-Splitter model cache"
    HF_HOME="$TOOLS/hf-cache" "$TOOLS/PROTAC-Splitter/.venv/bin/python" - <<'PROBE' || \
        echo "   (download failed; component metrics will record null until it succeeds)"
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from protac_splitter import split_protac
split_protac("CCO", model="xgboost", verbose=0)
PROBE
    installed+=("PROTAC-Splitter model weights into tools/hf-cache")
fi

# ---------------------------------------------------------------------------
# RetroTAC: whole-PROTAC synthesizability (retrotac_mean/_median, and the
# bifunctional form of quality_filtered_top10_mean).
#
# Its OWN environment, like PROTAC-Splitter: RetroTAC pins python ==3.12.* and
# xgboost>=2.0,<3.0 against this project's 3.13 and xgboost>=3.2.
# ---------------------------------------------------------------------------
if [ -x "$TOOLS/RetroTAC/.venv/bin/python" ]; then
    present+=("RetroTAC (env exists)")
else
    echo "-- installing RetroTAC at ${RETROTAC_SHA:0:8}"
    if [ ! -d "$TOOLS/RetroTAC/.git" ]; then
        git clone "$RETROTAC_URL" "$TOOLS/RetroTAC"
    fi
    git -C "$TOOLS/RetroTAC" fetch --quiet origin
    git -C "$TOOLS/RetroTAC" checkout --quiet "$RETROTAC_SHA"

    # Python 3.12 exactly. Prefer a real python3.12; otherwise conda, which is
    # what the dev machine has. uv's `uv venv --python 3.12` is equivalent on a
    # cluster that has uv.
    if command -v python3.12 >/dev/null 2>&1; then
        python3.12 -m venv "$TOOLS/RetroTAC/.venv"
    elif command -v uv >/dev/null 2>&1; then
        # The cluster route: RUN.md Step 3 installs uv, and an Lmod module tree
        # is unlikely to carry a 3.12 interpreter alongside the 3.13 this
        # project uses.
        uv python install 3.12
        uv venv --python 3.12 "$TOOLS/RetroTAC/.venv"
    elif command -v conda >/dev/null 2>&1; then
        conda create -y -q -p "$TOOLS/RetroTAC/.venv" python=3.12
    else
        echo "!! need python3.12 (or uv, or conda) to build RetroTAC's env; skipping"
        echo "   retrotac_* will record null until this is resolved"
    fi

    if [ -x "$TOOLS/RetroTAC/.venv/bin/python" ]; then
        "$TOOLS/RetroTAC/.venv/bin/python" -m pip install --quiet --upgrade pip
        "$TOOLS/RetroTAC/.venv/bin/python" -m pip install --quiet "$TOOLS/RetroTAC"

        # The pip xgboost wheel links @rpath/libomp.dylib and looks for
        # Homebrew's copy. Without it EVERY import fails, the GNN path included,
        # because it pulls xgboost transitively. conda puts libomp in the env
        # prefix, which is on the dlopen search path. On Linux the wheels bundle
        # their own libgomp and this is a no-op.
        if ! "$TOOLS/RetroTAC/.venv/bin/python" -c "import xgboost" 2>/dev/null; then
            if command -v conda >/dev/null 2>&1; then
                echo "   adding llvm-openmp (xgboost cannot load without it)"
                conda install -y -q -p "$TOOLS/RetroTAC/.venv" llvm-openmp
            else
                echo "!! xgboost cannot load: no OpenMP runtime. On macOS: brew install libomp"
            fi
        fi
        installed+=("RetroTAC + its own python3.12 env")
    fi
fi

# Its model weights ship in the clone, gzipped under release/. Verify against
# the upstream MANIFEST before unpacking: these are 105.8 MiB of weights that
# every reported retrotac_* number depends on.
RETROTAC_CKPT="$TOOLS/RetroTAC/outputs/models/gnn_20260828_182305/gnn_20260828_182305_final.ckpt"
if [ -s "$RETROTAC_CKPT" ]; then
    present+=("RetroTAC GNN checkpoint (unpacked)")
elif [ -s "$TOOLS/RetroTAC/release/gnn_20260828_182305_final.ckpt.gz" ]; then
    echo "-- unpacking the RetroTAC GNN refit"
    if command -v sha256sum >/dev/null 2>&1; then
        got="$(sha256sum "$TOOLS/RetroTAC/release/gnn_20260828_182305_final.ckpt.gz" | cut -d' ' -f1)"
    else
        got="$(shasum -a 256 "$TOOLS/RetroTAC/release/gnn_20260828_182305_final.ckpt.gz" | cut -d' ' -f1)"
    fi
    if [ "$got" != "$RETROTAC_CKPT_SHA256" ]; then
        echo "!! RetroTAC checkpoint checksum is $got"
        echo "   expected $RETROTAC_CKPT_SHA256 -- refusing to unpack it"
        exit 1
    fi
    mkdir -p "$(dirname "$RETROTAC_CKPT")"
    gunzip -kc "$TOOLS/RetroTAC/release/gnn_20260828_182305_final.ckpt.gz" > "$RETROTAC_CKPT"
    # The XGBoost and MLP refits, 3.4 MiB, for completeness.
    tar xzf "$TOOLS/RetroTAC/release/retrotac_checkpoints.tar.gz" -C "$TOOLS/RetroTAC" || true
    installed+=("RetroTAC GNN checkpoint (105.8 MiB, checksum verified)")
fi

# ---------------------------------------------------------------------------
# MOSES MCF SMARTS: filter_pass_rate. Committed, because it is 739 bytes of
# data rather than a package, and the upstream file has no release tag.
# ---------------------------------------------------------------------------
if [ -s "$TOOLS/data/mcf.csv" ]; then
    present+=("tools/data/mcf.csv")
else
    echo "-- fetching MOSES mcf.csv"
    curl -fsSL "$MCF_URL" -o "$TOOLS/data/mcf.csv"
    installed+=("tools/data/mcf.csv")
fi
# sha256sum is the GNU/Linux name, shasum the macOS one. This script has to
# run on both, and under `set -e` a missing command aborts the whole install.
if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "$TOOLS/data/mcf.csv" | cut -c1-16)"
elif command -v shasum >/dev/null 2>&1; then
    got="$(shasum -a 256 "$TOOLS/data/mcf.csv" | cut -c1-16)"
else
    echo "!! neither sha256sum nor shasum is on PATH; cannot verify mcf.csv"
    exit 1
fi
if [ "$got" != "$MCF_SHA256" ]; then
    echo "!! tools/data/mcf.csv checksum is $got, expected $MCF_SHA256"
    echo "   The upstream file changed. Check it before trusting filter_pass_rate."
    exit 1
fi

# ---------------------------------------------------------------------------
# fcd: the fcd metric. Plain PyPI, but it downloads ChemNet weights on first
# use, so the cache goes under tools/ for the same quota reason as above.
# ---------------------------------------------------------------------------
if "$REPO/.venv/bin/python" -c "import fcd" 2>/dev/null; then
    present+=("fcd")
else
    echo "-- fcd is not installed; the fcd metric will record null"
    echo "   install with: uv pip install fcd   (then re-run this script)"
fi

echo
echo "installed:"; for x in "${installed[@]:-}"; do [ -n "$x" ] && echo "  + $x"; done
echo "already present:"; for x in "${present[@]:-}"; do [ -n "$x" ] && echo "  = $x"; done
echo
echo "Done. See tools/README.md for what records null when a tool is absent."
