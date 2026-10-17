# External tools

Everything the metrics and oracles need that is not a plain `uv pip install`
from `pyproject.toml`: packages distributed only from GitHub, data files lifted
from other projects, and model weights downloaded on first use.

`bash tools/install.sh` installs all of it at the pinned versions below. It is
idempotent and reports what it installed and what was already there.

Downloaded weights land under `tools/`, never in a library's default cache.
Cluster home directories have small quotas and a surprise download into one is
a failure waiting to happen.

**A dependency used but not listed here is a bug.** If you add one, add it here.

---

## PROTAC-Splitter — required for Group 5

| | |
|---|---|
| What | Splits a bifunctional molecule into POI warhead, linker and E3 ligand |
| Needed by | `split_success_rate`, every `warhead_*` metric, `e3_recognizable_rate` |
| Source | https://github.com/ribesstefano/PROTAC-Splitter |
| Pinned to | commit `f1920122e1aa4f1d5f3643b8e92e72e99826e43a` (version 1.5.0) |
| Citation | Ribes et al., *J. Cheminform.* 18:30 (2026), doi 10.1186/s13321-025-01135-9. Data: doi 10.5281/zenodo.15797310 |
| License | MIT |
| Install | `bash tools/install.sh` |
| Required? | Required for component-level metrics; absent, they record null |

**It runs in its own virtual environment, at `tools/PROTAC-Splitter/.venv`.**
This is not tidiness. Its XGBoost edge-classifier was pickled under
`xgboost<3`, and loading it under the `xgboost>=3.2` this project needs for
aarch64 GPU support fails with `'XGBModel' object has no attribute
'feature_weights'`. The two pins cannot coexist in one interpreter, so
`src/metrics/_components.py` calls into that interpreter as a subprocess.

Its `model="heuristic"` strategy does load under our own xgboost, and it is
**not accurate enough to use**: on a VH032-based degrader it cut inside the E3
ligand, assigning the hydroxyproline and 4-methylthiazole to the linker and
calling only the acetyl-*tert*-leucine cap the E3 ligand. The `adaptive`
default, which escalates to the XGBoost classifier, gets the same molecule
right. That is why the isolated environment is worth its cost.

Its ~17 MB model downloads from the HuggingFace Hub into `tools/hf-cache` on
first use; `install.sh` warms it so the first metrics run does not.

When absent, these record null: `split_success_rate`, `n_split_successfully`,
`warhead_*`, `e3_recognizable_rate`, `e3_max_similarities`.

---

## MOSES MCF SMARTS — required for `filter_pass_rate`

| | |
|---|---|
| What | 22 SMARTS patterns for medicinal-chemistry filters |
| Needed by | `filter_pass_rate`, and `quality_filtered_top10_mean` through it |
| Source | https://raw.githubusercontent.com/molecularsets/moses/master/moses/metrics/mcf.csv |
| Pinned to | `tools/data/mcf.csv`, committed; sha256 begins `f4a2bfacb4cb8877` |
| Citation | Polykovskiy et al., *Front. Pharmacol.* 11:565644 (2020) |
| License | MIT (MOSES) |
| Install | committed; `install.sh` verifies the checksum |
| Required? | Required for `filter_pass_rate` |

Committed rather than downloaded, because it is 739 bytes of data and upstream
publishes no release tag for it. `install.sh` checksums it, so an upstream
change is noticed rather than silently altering a published number.

**It contains 22 SMARTS, not 29.** All 22 parse under RDKit.

Two deliberate differences from MOSES's own filter, so our number is not
compared to theirs by mistake:

- We apply **MCF and PAINS only**, which is what CLAUDE.md defines. MOSES also
  requires every atom to be in an allowed element set and the molecule to be a
  single fragment.
- Our PAINS comes from **RDKit's `FilterCatalog`** (480 entries; Baell &
  Holloway, *J. Med. Chem.* 53:2719, 2010), not from MOSES's vendored
  `wehi_pains.csv`. Those are different sets.

---

## SA score — required, ships with RDKit but not in its API

| | |
|---|---|
| What | Ertl & Schuffenhauer synthetic accessibility score |
| Needed by | `sa_score_mean`, `sa_score_median`, `quality_filtered_top10_mean` |
| Source | RDKit Contrib: `$RDConfig.RDContribDir/SA_Score/sascorer.py` |
| Pinned to | whatever ships with the installed RDKit |
| Citation | Ertl & Schuffenhauer, *J. Cheminform.* 1:8 (2009) |
| License | BSD (RDKit) |
| Install | nothing to do; comes with RDKit |
| Required? | Required |

Recorded here because it is **not part of the RDKit library API**. It is a
Contrib script reached by appending `RDConfig.RDContribDir/SA_Score` to
`sys.path` and importing `sascorer`, which `src/metrics/_properties.py` does.
The path is an RDKit implementation detail and has moved before; if
`sa_score_*` ever starts failing after an RDKit upgrade, this is why.

---

## fcd — optional, downloads weights

| | |
|---|---|
| What | Fréchet ChemNet Distance |
| Needed by | `fcd` |
| Source | PyPI: `fcd` |
| Pinned to | not pinned; not currently installed |
| Citation | Preuer et al., *J. Chem. Inf. Model.* 58:1736 (2018) |
| License | LGPL |
| Install | `uv pip install fcd` |
| Required? | Optional. Absent, `fcd` records null |

Plain PyPI, but it **downloads ChemNet weights on first use**, which is why it
is here rather than only in `pyproject.toml`. Point its cache under `tools/`
before the first cluster run.

Not installed on the development machine, so `fcd` has never produced a number.

---

## AutoDock Vina and meeko — optional, not a plain install on aarch64

| | |
|---|---|
| What | Docking oracle |
| Needed by | `src/oracles/_vina.py`, when an experiment declares a docking oracle |
| Source | PyPI `vina`, `meeko`; conda-forge for aarch64 |
| Pinned to | not pinned |
| Citation | Eberhardt et al., *J. Chem. Inf. Model.* 61:3891 (2021) |
| License | Apache 2.0 |
| Install | `uv sync --extra docking` on x86; **conda-forge on aarch64** |
| Required? | Optional. No experiment declares a docking oracle yet |

`vina` publishes **no aarch64 wheel at any version**, so the `docking` extra
does not install on the cluster and the `pip` route will try to compile.
conda-forge has `linux-aarch64` builds.

---

## Reserved, not installed

These have entries so the inventory is complete and whoever adds them has
somewhere to put the pin.

| Tool | For | Status |
|---|---|---|
| **RetroTAC** | goal-directed oracle | Not installed. Reserved for the next task |
| **REINVENT4** | baseline comparison | Not installed. Reserved for the next task |
| **AiZynthFinder** | retrosynthetic accessibility; needs downloaded model files | Not installed, and **no code reaches for it**. Reserved |

---

## Reference, not a dependency

**Degrader property windows.** Hornberger & Araujo, "Physicochemical Property
Determinants of Oral Absorption for PROTAC Protein Degraders", *J. Med. Chem.*
(2023), doi 10.1021/acs.jmedchem.3c00740, derives limits from real degrader rat
PK: MW ≤ 950, cLogP 1–7, HBD ≤ 4 with unsatisfied HBD ≤ 2, HBA ≤ 15, TPSA ≤ 200
Å², rotatable bonds ≤ 14, aromatic rings ≤ 5.

This is **not** what `bRo5_pass_rate` computes. CLAUDE.md defines that as a core
metric reported for every setting, small molecules included, so a
degrader-specific window would be the wrong rule there. Recorded as the
reference for degrader property windows if a PROTAC-specific metric is ever
added.
