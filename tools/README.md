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

## fcd — required for the headline unconditional metric

| | |
|---|---|
| What | Fréchet ChemNet Distance |
| Needed by | `fcd` |
| Source | PyPI: `fcd` |
| Pinned to | 1.2.2 |
| Citation | Preuer et al., *J. Chem. Inf. Model.* 58:1736 (2018) |
| License | LGPL |
| Install | `uv pip install fcd==1.2.2` |
| Required? | Required in practice. CLAUDE.md calls `fcd` the headline number for unconditional generation, so a run without it has no headline result |

**It downloads nothing.** The ChemNet weights ship inside the wheel as
`fcd/ChemNet_v0.13_pretrained.pt` (5.6 MB), so there is no cache to redirect
and no first-use network call. An earlier note here claimed otherwise; it was
wrong.

Two things about how it is called, both load-bearing:

- `src/metrics/_unconditional.py` composes `get_predictions` and
  `calculate_frechet_distance` instead of calling `fcd.get_fcd`, because
  `get_fcd` does not expose `n_jobs` and so inherits `DataLoader(num_workers=1)`
  — a worker process for no gain. `n_jobs=0` is roughly forty times faster on a
  small sample.
- **Both sides need at least two molecules.** `np.cov` of a single row divides
  by `n - 1 == 0` and returns a 512x512 matrix of NaN, and
  `calculate_frechet_distance` hands that to `scipy.linalg.sqrtm` *before* its
  own finiteness check. `sqrtm` of a NaN matrix that size never finishes: the
  process sits at 100% CPU with no error. That was a real hang in the generate
  smoke test, whose briefly-trained model produced one valid molecule. The
  floor is enforced on our side and records a reason instead.

FCD compares two 512-dimensional Gaussians, so it is only meaningful once each
side has enough molecules to estimate a covariance. At CLAUDE.md's N = 10,000
that is fine. Below a few hundred the number is dominated by the rank
deficiency of the covariance rather than by any property of the model, so treat
a small-sample `fcd` as a smoke signal and not a result.

---

## RetroTAC — required for whole-PROTAC synthesizability

| | |
|---|---|
| What | Route-derived synthesizability of a whole PROTAC, from SMILES, with no retrosynthesis planner at inference |
| Needed by | `retrotac_mean`, `retrotac_median`, and `quality_filtered_top10_mean` on a bifunctional dataset |
| Source | https://github.com/ribesstefano/RetroTAC |
| Pinned to | commit `5ba851656753542d4a7f23438c033af280db9935` |
| Citation | Ribes, Zhu, Zhu, Mo, Hudec, Papadopoulos, Patronov & Mercado, arXiv:2610.13639 |
| License | MIT |
| Install | `bash tools/install.sh` |
| Required? | Required when `bifunctional: true`; absent, `retrotac_*` record null and the quality gate has no substitute for SA score |

**Why it exists at all.** SA score reads a PROTAC's size as difficulty.
Measured here on 500 real TPD-DB degraders, median SA is 4.71 and **58.6% fail
`sa_score < 4.5`**, against 2.87 and 6.2% for drug-like ZINC. Those degraders
were synthesized and assayed, so at the whole-PROTAC level SA score rejects the
majority of chemistry that demonstrably exists. SA score remains correct for the
**decomposed warheads**, which are drug-sized, and for molecular glues.

**It runs in its own environment at `tools/RetroTAC/.venv`**, like
PROTAC-Splitter and for the same class of reason: RetroTAC pins
`python ==3.12.*` and `xgboost>=2.0,<3.0`, against this project's 3.13 and
`xgboost>=3.2`. `src/metrics/_retrotac.py` calls into that interpreter.

Two install notes, both found the hard way on macOS arm64:

- There is **no Python 3.12 on the dev machine**, so `install.sh` builds the
  env with conda when `python3.12` is absent. On a cluster with `uv`, `uv venv
  --python 3.12` is equivalent.
- The pip `xgboost` wheel links `@rpath/libomp.dylib` and looks for Homebrew's
  copy, which is not present. Without it **every** import fails, including the
  GNN-only path, which pulls xgboost transitively. `install.sh` installs
  `llvm-openmp` into the env prefix, where the dlopen search path finds it.

**Which model is installed.** The released final **GNN refit** (held-out R²
0.615, RMSE 0.140), which ships in the clone under `release/` as a 90.7 MiB
gzipped checkpoint, SHA-256 verified by `install.sh` against
`release/MANIFEST.txt` and unpacked to 105.8 MiB. The XGBoost and MLP refits are
unpacked too. **RetroTAC proper — the 27-member Caruana ensemble, R² 0.647 —
is not installed**: its members are the per-fold cross-validation models, 1.4
GiB, which the release archives deliberately omit, so running it would mean
retraining the cross-validation. Upstream will publish it to the HuggingFace Hub
after peer review. The gap to the single GNN is 0.03 in R².

Two properties of the score worth knowing before reading one:

- **It is not clipped to [0, 1].** The authors treat an out-of-range value as a
  signal that the molecule is far from the training distribution, so nothing
  clips it. 11 of the 500 real degraders scored outside [0, 1].
- **The 0.7 threshold in the paper is a reporting convention, not a calibrated
  decision boundary**, and only 18.8% of real degraders reach it. The
  continuous score and its ranking are the primary output. The gate constant
  `QUALITY_RETROTAC_MIN` in `src/metrics/_budget.py` is therefore 0.30, chosen
  to admit ~94% of real degraders and so match what `sa_score < 4.5` admits of
  drug-like molecules. **Approved 2026-10-18.**

Measured cost: 12.2 s for 500 molecules, of which ~6 s is the one-off model
load, so ~12.5 ms per molecule after that. Everything goes over in one
subprocess call for that reason. Projected 2.2 min for 10,000 molecules and
6.3 min for three seeds. Taken on a machine with other load, so treat as an
upper bound.

When absent, these record null: `retrotac_mean`, `retrotac_median`, and the
bifunctional form of `quality_filtered_top10_mean`.

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
